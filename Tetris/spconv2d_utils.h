#ifndef CONV2D_UTILS_H_
#define CONV2D_UTILS_H_

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <cudnn.h>
#include <set>
#include <cuda_runtime.h>

#include "cuda_utils.h"

#define WARP_SIZE 32

// #define SPCONV2D_DEBUG_FLAG

constexpr int BIT_S = 3;
constexpr int BIT_R = 3;

// default img: NCHW, filter: OIRS
template<typename ValueType, typename OffsetType, typename PositionType>
struct SparseFilter {
    SparseFilter(float *filter, int in_channel, int out_channel, int kernel_size,
                 int TILE_IC = 32, bool apply_reorder = false, bool apply_vectorize = true)
        : in_channel_(in_channel), out_channel_(out_channel), kernel_size_(kernel_size), TILE_IC_(TILE_IC) {
        
        int STAGE_PER_CHANNEL = ((in_channel_ - 1) / TILE_IC_ + 1);
        nnz_ = 0;
        int stage_id = 0;
        offsets_.emplace_back(0);

        for (int o = 0; o < out_channel_; ++o)
            oc_permutation_.emplace_back(o);

        // keep sparse data as is
        if (apply_reorder == false) {
            for (int o = 0; o < out_channel_; ++o){
                for (int i = 0; i < in_channel_; ++i) {
                    if (i%TILE_IC == 0) {
                        stage_id += 1;
                        offsets_.emplace_back(offsets_.back());
                    }
                    for (int r = 0; r < kernel_size_; ++r){
                        for (int s = 0; s < kernel_size_; ++s) {
                            float value = filter[o * in_channel_ * kernel_size_ * kernel_size_ + i * kernel_size_ * kernel_size_ + r * kernel_size_ + s];
                            if (value != 0.f) {
                                nnz_ += 1;
                                #ifdef SPCONV2D_DEBUG_FLAG
                                    values_.emplace_back(1.0);
                                #else
                                    values_.emplace_back(value);
                                #endif
                                position_.emplace_back(s + (r << BIT_S) + ((i%TILE_IC) << (BIT_S + BIT_R)));
                                offsets_[offsets_.size() - 1] += 1;
                            }
                        }
                    }
                }
            }
        } else {
            // setp0: reorder alone out_channel
            for (int i = 0; i < out_channel_; ++i)oc_permutation_[i] = i;

            #ifdef REORDER_OUT_CHANNEL
            {
              ReorderOutChannel(filter);
            }
            #endif // REORDER_OUT_CHANNEL

            // setp1: according oc_permutation_, reorder the filter
            // padding the stages to the longest stage in each warp(32)
            const int pad_out_channel = (out_channel_ + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
            offsets_.resize(((out_channel_ - 1) / WARP_SIZE + 1) * STAGE_PER_CHANNEL  + 1, 0);
            stage_len_.resize(((out_channel_ - 1) / WARP_SIZE + 1) * WARP_SIZE * STAGE_PER_CHANNEL, 0);
            std::vector<ValueType> stage_values[pad_out_channel * STAGE_PER_CHANNEL  + 1];
            std::vector<PositionType> stage_positions[pad_out_channel * STAGE_PER_CHANNEL  + 1];

            #ifndef BANK_OPTIMIZE
              for (int g = 0; g < out_channel_; g += WARP_SIZE) {
                for (int i = 0; i < in_channel_; i += TILE_IC_) {
                  for (int o = g; o < std::min(g + WARP_SIZE, out_channel_); ++o) {
                    int stage_id = o * STAGE_PER_CHANNEL + i / TILE_IC_;
                    for (int j = i; j < std::min(i + TILE_IC_, in_channel_); ++j) {
                      for (int r = 0; r < kernel_size_; ++r) {
                        for (int s = 0; s < kernel_size_; ++s) {
                          float value = filter[oc_permutation_[o] * in_channel_ * kernel_size_ * kernel_size_ + j * kernel_size_ * kernel_size_ + r * kernel_size_ + s];
                          if (value != 0.f) {
                            stage_values[stage_id].emplace_back(value);
                            stage_positions[stage_id].emplace_back(s + (r << BIT_S) + ((j%TILE_IC) << (BIT_S + BIT_R)));
                          }
                        }
                      }
                    }
                  }
                }
              }
            #else
              for (int g = 0; g < out_channel_; g += WARP_SIZE) {
                for (int i = 0; i < in_channel_; i += TILE_IC_) {
                  std::vector<ValueType> temp_values[WARP_SIZE];
                  std::vector<PositionType> temp_positions[WARP_SIZE];
                  std::vector<int> channel_pad_cnt(WARP_SIZE, 0);
                  int stage_mx_nnz = 0;
                  for (int o = g; o < std::min(g + WARP_SIZE, out_channel_); ++o) {
                    for (int j = i; j < std::min(i + TILE_IC_, in_channel_); ++j) {
                        for (int r = 0; r < kernel_size_; ++r) {
                            for (int s = 0; s < kernel_size_; ++s) {
                                float value = filter[oc_permutation_[o] * in_channel_ * kernel_size_ * kernel_size_ + j * kernel_size_ * kernel_size_ + r * kernel_size_ + s];
                                if (value != 0.f) {
                                  temp_values[o - g].emplace_back(value);
                                  temp_positions[o - g].emplace_back(s + (r << BIT_S) + ((j%TILE_IC) << (BIT_S + BIT_R)));
                                }
                            }
                        }
                    }
                    stage_mx_nnz = std::max(stage_mx_nnz, (int)temp_values[o - g].size());
                  }

                  for (int o = g; o < g + WARP_SIZE; ++o) {
                    channel_pad_cnt[o - g] = stage_mx_nnz - temp_values[o - g].size();
                  }

                  for (int t = 0; t < stage_mx_nnz; ++t) {
                    std::vector<int> bank_flag(WARP_SIZE, 0);
                    std::vector<int> kernel_pos_flag(in_channel_ * kernel_size_ * kernel_size_, 0);
                    std::vector<int> wait_list;
                    for (int o = g; o < g + WARP_SIZE; ++o) {
                      int select = -1;
                      for (int j = 0; j < temp_positions[o - g].size(); ++j) {
                        int pos = temp_positions[o - g][j];
                        int ic = pos >> (BIT_S + BIT_R);
                        int y = pos & ((1 << BIT_S) - 1);
                        int x = (pos >> BIT_S) & ((1 << BIT_R) - 1);
                        if (kernel_pos_flag[ic * kernel_size_ * kernel_size_ + x * kernel_size_ + y] == 1) {
                          kernel_pos_flag[ic * kernel_size_ * kernel_size_ + x * kernel_size_ + y] = 1;
                          select = j;
                          break;
                        }
                      }

                      for (int j = 0; j < temp_positions[o - g].size(); ++j) {
                        int pos = temp_positions[o - g][j];
                        int bank_id = (pos >> (BIT_S + BIT_R)) % WARP_SIZE;
                        int ic = pos >> (BIT_S + BIT_R);
                        int y = pos & ((1 << BIT_S) - 1);
                        int x = (pos >> BIT_S) & ((1 << BIT_R) - 1);
                        if (bank_flag[bank_id] == 0 && select == -1) {
                          select = j;
                          bank_flag[bank_id] = 1;
                          kernel_pos_flag[ic * kernel_size_ * kernel_size_ + x * kernel_size_ + y] = 1;
                          break;
                        }
                      }
                      if (select != -1) {
                        int stage_id = o * STAGE_PER_CHANNEL + i / TILE_IC_;
                        stage_values[stage_id].emplace_back(temp_values[o - g][select]);
                        stage_positions[stage_id].emplace_back(temp_positions[o - g][select]);
                        temp_positions[o - g].erase(temp_positions[o - g].begin() + select);
                        temp_values[o - g].erase(temp_values[o - g].begin() + select);
                      } else {
                        wait_list.emplace_back(o);
                      }
                    }

                    for (auto o : wait_list) {
                      int stage_id = o * STAGE_PER_CHANNEL + i / TILE_IC_;
                      if (channel_pad_cnt[o - g] > 0) {
                        for (int b = 0; b < bank_flag.size(); ++b) {
                          if (bank_flag[b] == false) {
                            bank_flag[b] = true;
                            channel_pad_cnt[o - g] -= 1;
                            stage_values[stage_id].emplace_back(0.0);
                            stage_positions[stage_id].emplace_back((b << (BIT_S + BIT_R)));
                            break;
                          }
                        }
                      } else {
                        int select = -1;
                        for (int j = 0; j < temp_positions[o - g].size(); ++j) {
                          int pos = temp_positions[o - g][j];
                          int bank_id = (pos >> (BIT_S + BIT_R)) % WARP_SIZE;
                          if (select == -1 || bank_flag[select] > bank_flag[bank_id]) {
                            select = j;
                          }
                        }
                        int bank_id = (temp_positions[o - g][select] >> (BIT_S + BIT_R)) % WARP_SIZE;
                        bank_flag[bank_id] += 1;
                        stage_values[stage_id].emplace_back(temp_values[o - g][select]);
                        stage_positions[stage_id].emplace_back(temp_positions[o - g][select]);
                        temp_positions[o - g].erase(temp_positions[o - g].begin() + select);
                        temp_values[o - g].erase(temp_values[o - g].begin() + select);
                      }
                    }
                  }
                }
              }
            #endif //  BANK_OPTIMIZE

            int kVecWidth = 1;
            if (apply_vectorize) {
              kVecWidth = 4;
            }

            // step2: generate the sparse format
            for (int warp = 0; warp < out_channel_; warp += WARP_SIZE){
                for (int i = 0; i < in_channel_; i += TILE_IC_) {
                    int mx_stage_nnz = 0;
                    int stage_nnz_sum = 0;
                    for (int o = warp; o < std::min(out_channel_, warp + WARP_SIZE); o++) {
                        int stage_id = o * STAGE_PER_CHANNEL + i / TILE_IC_;
                        int cur_stage_nnz = (int)stage_values[stage_id].size();

                        cur_stage_nnz = (cur_stage_nnz + kVecWidth - 1) / kVecWidth * kVecWidth;
                        
                        mx_stage_nnz = std::max(mx_stage_nnz, cur_stage_nnz);
                        stage_nnz_sum += stage_values[stage_id].size();
                    }

                    int offset_id = warp / WARP_SIZE * STAGE_PER_CHANNEL + (i / TILE_IC_);
                    offsets_[offset_id + 1] = offsets_[offset_id] + mx_stage_nnz;
                    
                    for (int o = warp; o < std::min(out_channel_, warp + WARP_SIZE); o++) {
                        int new_stage_id = warp * STAGE_PER_CHANNEL + (i/TILE_IC_) * WARP_SIZE + o % WARP_SIZE;
                        int ori_stage_id = o * STAGE_PER_CHANNEL + i / TILE_IC_;
                        stage_len_[new_stage_id] = stage_values[ori_stage_id].size();
                    }
                    for (int t = 0; t < mx_stage_nnz; t += kVecWidth) {
                        for (int o = warp; o < warp + WARP_SIZE; o++) {
                            int stage_id = o * STAGE_PER_CHANNEL + i / TILE_IC_;
                            for (int iter = 0; iter < kVecWidth; iter++) {
                              if (t + iter < stage_values[stage_id].size()) {
                                  values_.emplace_back(stage_values[stage_id][t + iter]);
                                  position_.emplace_back(stage_positions[stage_id][t + iter]);
                              } else {
                                  values_.emplace_back(0.0);
                                  position_.emplace_back(0);
                              }
                            }
                        }
                    }
                }
            }
            
        }

        // alloc device memory
        size_t offsets_byte_size = offsets_.size() * sizeof(OffsetType);
        size_t position_byte_size = position_.size() * sizeof(PositionType);
        size_t values_byte_size = values_.size() * sizeof(ValueType);
        size_t stage_len_byte_size = stage_len_.size() * sizeof(OffsetType);
        size_t oc_permutation_byte_size = oc_permutation_.size() * sizeof(PositionType);
        checkCudaErrors(cudaMalloc(&d_offsets_, offsets_byte_size));
        checkCudaErrors(cudaMalloc(&d_position_, position_byte_size));
        checkCudaErrors(cudaMalloc(&d_values_, values_byte_size));
        checkCudaErrors(cudaMalloc(&d_stage_len_, stage_len_byte_size));
        checkCudaErrors(cudaMalloc(&d_oc_permutation_, oc_permutation_byte_size));

        checkCudaErrors(cudaMemcpy(d_offsets_, offsets_.data(), offsets_byte_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_position_, position_.data(), position_byte_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_values_, values_.data(), values_byte_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_stage_len_, stage_len_.data(), stage_len_byte_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_oc_permutation_, oc_permutation_.data(), oc_permutation_byte_size, cudaMemcpyHostToDevice));
    }

    ~SparseFilter() {
      cudaFree(d_offsets_);
      cudaFree(d_position_);
      cudaFree(d_values_);
      cudaFree(d_stage_len_);
      cudaFree(d_oc_permutation_);
    }

    void ReorderOutChannel(float *filter) {
      int STAGE_PER_CHANNEL = ((in_channel_ - 1) / TILE_IC_ + 1);
      std::vector<int> stage_cnt(out_channel_ * STAGE_PER_CHANNEL, 0);
      std::vector<std::vector<int>> stage_nz_pos(out_channel_ * STAGE_PER_CHANNEL);
      std::vector<int> oc_cnt(out_channel_, 0);
      for (int o = 0; o < out_channel_; ++o) {
        for (int i = 0; i < in_channel_; i += TILE_IC_) {
          int stage_id = o * STAGE_PER_CHANNEL + i / TILE_IC_;
          for (int j = i; j < std::min(i + TILE_IC_, in_channel_); ++j) {
            for (int r = 0; r < kernel_size_; ++r){
              for (int s = 0; s < kernel_size_; ++s) {
                float value = filter[o * in_channel_ * kernel_size_ * kernel_size_ + j * kernel_size_ * kernel_size_ + r * kernel_size_ + s];
                if (value != 0.f) {
                  stage_cnt[stage_id] += 1;
                  stage_nz_pos[stage_id].emplace_back(s + (r << BIT_S) + ((j) << (BIT_S + BIT_R)));
                  oc_cnt[o] += 1;
                }
              }
            }
          }
        }
      }
      
      oc_permutation_.clear();
      constexpr int GROUP_SIZE = 8;
      struct SubGroup {
        int l_, r_, cost_;
        SubGroup (int l, int r, int cost) : l_(l), r_(r), cost_(cost) {}
        bool operator< (const SubGroup &b) const{
          if (cost_ != b.cost_)
            return cost_ > b.cost_;
          return l_ < b.l_;
        }
      };

      std::set<SubGroup> oc_wait_list;
      for (int i = 0; i < out_channel_; i += GROUP_SIZE) {
        int cost = 0;
        int l = i, r = std::min(out_channel_, i + GROUP_SIZE);
        for (int j = 0; j < STAGE_PER_CHANNEL; ++j) {
          int stage_nnz = 0;
          for (int t = l; t < r; ++t) {
            int stage_id = t * STAGE_PER_CHANNEL + j;
            stage_nnz = std::max(stage_nnz, stage_cnt[stage_id]);
          }
          cost += stage_nnz;
        }
        
        oc_wait_list.insert(SubGroup(l, r, cost));
      }

      while(!oc_wait_list.empty()) {
        std::vector<int> current_group;
        std::vector<int> current_stage_mx(STAGE_PER_CHANNEL, 0);
        std::vector<int> current_bank_count(STAGE_PER_CHANNEL * WARP_SIZE, 0);
        std::unordered_map<int, int> pos_flag;
        int start_l= (*oc_wait_list.begin()).l_;
        int start_r= (*oc_wait_list.begin()).r_;
        oc_wait_list.erase(oc_wait_list.begin());
        for (int t = start_l; t < start_r; ++t) {
          current_group.emplace_back(t);
        }
      
        auto UpdateStageMX = [&](int oc) {
          for (int i = 0; i < STAGE_PER_CHANNEL; ++i) {
            int stage_id = oc * STAGE_PER_CHANNEL + i;
            current_stage_mx[i] = std::max(current_stage_mx[i], stage_cnt[stage_id]);   
          }
        };

        auto UpdateBankCount = [&](int l, int r) -> void {
          for (int oc = l; oc < r; ++oc) {
            for (int i = 0; i < STAGE_PER_CHANNEL; ++i) {
              int stage_id = oc * STAGE_PER_CHANNEL + i;
              for (auto pos : stage_nz_pos[stage_id]) {
                int bank_id = (pos >> (BIT_S + BIT_R)) % WARP_SIZE;
                if (!pos_flag.count(pos)) {
                  current_bank_count[i * WARP_SIZE + bank_id] += 1;
                  pos_flag[pos] = 1;
                }
              }
            }
          }
        };

        auto CalcGroupCost = [&](int l, int r) -> float {
          float group_cost = 0;
          float group_padding = 0;
          for (int i = 0; i < STAGE_PER_CHANNEL; ++i) {
            std::vector<int> stage_bank_count(WARP_SIZE, 0);
            int stage_mx_nnz = 0;
            int stage_padding = 0;
            for (int oc = l; oc < r; ++oc) {
              int stage_id = oc * STAGE_PER_CHANNEL + i;
              stage_mx_nnz = std::max(stage_mx_nnz, stage_cnt[stage_id]);
              for (auto pos : stage_nz_pos[stage_id]) {
                int bank_id = (pos >> (BIT_S + BIT_R)) % WARP_SIZE; 
                if (!pos_flag.count(pos)) {
                  stage_bank_count[bank_id] += 1;
                }
              }
            }
            stage_padding += stage_mx_nnz - current_stage_mx[i] > 0? stage_mx_nnz - current_stage_mx[i]: current_stage_mx[i] - stage_mx_nnz;
            for (int t = 0; t < WARP_SIZE; ++t) {
              stage_bank_count[t] += current_bank_count[i * WARP_SIZE + t];
            }
            int stage_mx_bank = stage_bank_count[0], stage_mn_bank = stage_bank_count[0];
            for (auto v : stage_bank_count) {
              stage_mx_bank = std::max(stage_mx_bank, v);
              stage_mn_bank = std::min(stage_mn_bank, v);

            } 
            group_padding += stage_padding;
            group_cost += std::max(stage_mx_bank - std::max(stage_mx_nnz, current_stage_mx[i]), stage_padding);
          }
          return group_cost;
        };

        for (int t = start_l; t < start_r; ++t) {
          UpdateStageMX(t);
        }

        UpdateBankCount(start_l, start_r);

        while(!oc_wait_list.empty() && current_group.size() < WARP_SIZE) {
          float min_cost = 1e18;
          SubGroup min_group(-1, -1, -1);
          for (auto iter : oc_wait_list) {
            int oc_l = iter.l_;
            int oc_r = iter.r_;
            float cur_cost = CalcGroupCost(oc_l, oc_r);
            if (cur_cost < min_cost) {
              min_cost = cur_cost;
              min_group = iter;
            }
          }
          oc_wait_list.erase(min_group);
          UpdateBankCount(min_group.l_, min_group.r_);
          for (int t = min_group.l_; t < min_group.r_; ++t) {
            UpdateStageMX(t);
            current_group.emplace_back(t);
          }
        }
        oc_permutation_.insert(oc_permutation_.end(), current_group.begin(), current_group.end());
      }
    }

    void HostSpconv2d(float *input, float *output, int batch_size,
                      int img_h, int img_w, int padding, int stride) {

        int out_h = (img_h + padding * 2 - kernel_size_) / stride + 1;
        int out_w = (img_w + padding * 2 - kernel_size_) / stride + 1;
#define output(n, c, h, w) output[(n) * out_channel_ * out_h * out_w + (c) *  out_h * out_w + (h) * out_w + (w)]
#define input(n, c, h, w) input[(n) * in_channel_ * img_h * img_w + (c) *  img_h * img_w + (h) * img_w + (w)]

        for (int n = 0; n < batch_size; ++n) {
            for (int oc = 0; oc < out_channel_; ++oc) {
                for (int h = 0; h < out_h; ++h) {
                    for (int w = 0; w < out_w; ++w) {
                        output(n, oc, h, w) = 0.f;
                        int stage_num = (in_channel_ - 1)/TILE_IC_ + 1;
                        int start_stage_id = stage_num * oc;
                        for (int s_id = start_stage_id, t = 0; s_id < stage_num * (oc + 1); ++s_id, ++t) {
                            for (int k = offsets_[s_id]; k < offsets_[s_id + 1]; ++k) {
                              int pos = position_[k];
                              int y = pos & ((1 << BIT_S) - 1);
                              int x = (pos >> BIT_S) & ((1 << BIT_R) - 1);
                              int ic = (pos >> (BIT_S + BIT_R)) + t * TILE_IC_;
                              if (h * stride + x < padding || h * stride + x >= img_h + padding) continue;
                              if (w * stride + y < padding || w * stride + y >= img_w + padding) continue;
                              output(n, oc, h, w) += input(n, ic, h * stride + x - padding, w * stride + y - padding) * values_[k];
                            }
                        }
                    }
                }
            }
        }
#undef output
#undef input

    }

    void HostReorderSpconv2d(float *input, float *output, int batch_size,
                      int img_h, int img_w, int padding, int stride) {
        int out_h = (img_h + padding * 2 - kernel_size_) / stride + 1;
        int out_w = (img_w + padding * 2 - kernel_size_) / stride + 1;
#define output(n, c, h, w) output[(n) * out_channel_ * out_h * out_w + (c) *  out_h * out_w + (h) * out_w + (w)]
#define input(n, c, h, w) input[(n) * in_channel_ * img_h * img_w + (c) *  img_h * img_w + (h) * img_w + (w)]
       
        for (int n = 0; n < batch_size; ++n) {
            for (int oc = 0; oc < out_channel_; ++oc) {
                for (int h = 0; h < out_h; ++h) {
                    for (int w = 0; w < out_w; ++w) {
                        int stage_num = (in_channel_ - 1)/TILE_IC_ + 1;
                        float sum = 0;
                        for (int j = 0; j < stage_num; ++j) {
                            int offset_id = oc / WARP_SIZE * stage_num + j;
                            int nnz = stage_len_[(oc/WARP_SIZE) * WARP_SIZE + j * WARP_SIZE + oc % WARP_SIZE];
                            for (int k = 0; k < offsets_[offset_id + 1] - offsets_[offset_id]; ++k) {
                                int start = offsets_[offset_id] * WARP_SIZE;
                                int pos = position_[start + k * WARP_SIZE + oc % WARP_SIZE];
                                float v = values_[start + k * WARP_SIZE + oc % WARP_SIZE];
                                int y = pos & ((1 << BIT_S) - 1);
                                int x = (pos >> BIT_S) & ((1 << BIT_R) - 1);
                                int ic = (pos >> (BIT_S + BIT_R)) + j * TILE_IC_;
                                if (h * stride + x < padding || h * stride + x >= img_h + padding) continue;
                                if (w * stride + y < padding || w * stride + y >= img_w + padding) continue;
                                sum += input(n, ic, h * stride + x - padding, w * stride + y - padding) * v;
                            }
                        }
                        output(n, oc_permutation_[oc], h, w) = sum;
                    }
                }
            }
        }
#undef output
#undef input

    }

    OffsetType* GetDeviceOffsets() { return d_offsets_; }
    OffsetType* GetDeviceStageLen() { return d_stage_len_; }
    ValueType* GetDeviceValues() { return d_values_; }
    PositionType* GetDevicePosition() { return d_position_; }
    PositionType* GetDeviceOCPermutation() { return d_oc_permutation_; }
    int GetTileIC() { return TILE_IC_; }

    int in_channel_, out_channel_, kernel_size_;
    int nnz_;
    int TILE_IC_;
    std::vector<OffsetType> oc_permutation_;
    std::vector<OffsetType> stage_len_;
    std::vector<OffsetType> offsets_; // stage offsets
    std::vector<PositionType> position_; // low -> high(bit): (r, s, ic);
    std::vector<ValueType> values_;
    OffsetType *d_offsets_; // offsets tensor in device
    OffsetType *d_stage_len_;  // stage_len_ tensor in device
    PositionType *d_position_; // position tensor in device
    PositionType *d_oc_permutation_;
    ValueType *d_values_; // values tensor in device
};


#endif  // CONV2D_UTILS_H_
