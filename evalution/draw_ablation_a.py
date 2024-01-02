import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import CalculateSpeedup

csfont = {'family':'Times New Roman', 'size': 16}
plt.rc('font', **csfont)

# baseline
# SPF format（SPF），
# Bank-Sensing Reorganization（Bank-SR），
# Filter Group Reorder（Filter-GR），
# parameters tuning（para）

if __name__ == "__main__":
    ablation_data_file = "./log/ablation-log/ablation_a.csv"
    pd_data = pd.read_csv(ablation_data_file, header=0)
    method_list = ["Baseline", "SPF", "Bank-SR", "Filter-GR", "Para"]
    ops_sparsity = {

    }
    ablation_data = {
        "op_name": [],
        "sparsity": []
    }

    # fix the order of kernel
    kernel_order = {
        "module3_0_conv2d_0_H_112_W_112_IC_64_OC_128_KS_3_Pad_1_S_1_G_1": 0,
        "module8_2_module2_0_conv2d_3_H_14_W_14_IC_512_OC_512_KS_3_Pad_1_S_1_G_1": 1,
        "module8_1_module3_0_conv2d_0_H_56_W_56_IC_256_OC_256_KS_3_Pad_1_S_1_G_1": 2,
        "module8_2_module2_0_conv2d_1_H_14_W_14_IC_512_OC_512_KS_3_Pad_1_S_1_G_1": 3,
        "module8_0_module3_0_conv2d_0_H_112_W_112_IC_128_OC_128_KS_3_Pad_1_S_1_G_1": 4,
        "module8_1_module2_0_conv2d_1_H_28_W_28_IC_256_OC_512_KS_3_Pad_1_S_1_G_1": 5,
        "module8_2_module2_0_conv2d_5_H_14_W_14_IC_512_OC_512_KS_3_Pad_1_S_1_G_1": 6,
        "module8_0_module2_0_conv2d_3_H_56_W_56_IC_256_OC_256_KS_3_Pad_1_S_1_G_1": 7,
        "module8_2_module3_0_conv2d_0_H_28_W_28_IC_512_OC_512_KS_3_Pad_1_S_1_G_1": 8,
        "module5_0_module3_0_conv2d_0_H_224_W_224_IC_64_OC_64_KS_3_Pad_1_S_1_G_1": 9,
        "conv2d_0_H_224_W_224_IC_3_OC_64_KS_3_Pad_1_S_1_G_1": 10,
        "module5_1_module3_0_conv2d_0_H_14_W_14_IC_512_OC_512_KS_3_Pad_1_S_1_G_1": 11,
        "module8_0_module2_0_conv2d_1_H_56_W_56_IC_128_OC_256_KS_3_Pad_1_S_1_G_1": 12,
        "module8_1_module2_0_conv2d_3_H_28_W_28_IC_512_OC_512_KS_3_Pad_1_S_1_G_1": 13,
        "module8_0_module2_0_conv2d_5_H_56_W_56_IC_256_OC_256_KS_3_Pad_1_S_1_G_1": 14,
        "module8_1_module2_0_conv2d_5_H_28_W_28_IC_512_OC_512_KS_3_Pad_1_S_1_G_1": 15,
    }

    ordered_data = [0 for i in range(len(kernel_order))]
    for index, row in pd_data.iterrows():
        cur_op = dict(row)
        op_name = cur_op["op_name"]
        ordered_data[kernel_order[op_name]] = dict(cur_op)

    for cur_op in ordered_data:
        op_name = cur_op["op_name"]
        # sparsity = cur_op["sparsity"]
        # the sparsity of filter is zero
        if "IC_3_"  in op_name:
            continue
        ablation_data["op_name"].append(op_name)
        # ablation_data["sparsity"].append(sparsity)
        for k, v in cur_op.items():
            if k not in method_list:
                continue
            if k not in ablation_data.keys():
                ablation_data[k] = []
            ablation_data[k].append(v)
        # ops_sparsity[op_name] = sparsity


    spf_mean_speedup, spf_max_speedup, _ = CalculateSpeedup(ablation_data["Baseline"], ablation_data["SPF"])
    speedup = [x * 1.0 / y for x, y in zip(ablation_data["Baseline"], ablation_data["SPF"])]
    mx_val = -1
    mx_idx = -1
    for i, val in enumerate(speedup):
        if val > mx_val:
            mx_idx = i
            mx_val = val

    print("# {} boost performance {:.2f}x, up to {:.2f}x, kernel with channel {}".format("SPF format", spf_mean_speedup, spf_max_speedup, ablation_data["op_name"][mx_idx].split("OC_")[1].split("_")[0]))
    bank_sr_mean_speedup, _, _ = CalculateSpeedup(ablation_data["SPF"], ablation_data["Bank-SR"])
    print("# {} boost {:.2f}x performance(geo mean)".format("Bank-SR", bank_sr_mean_speedup))
    _, filter_gr_max_speedup, _ = CalculateSpeedup(ablation_data["Bank-SR"], ablation_data["Filter-GR"])
    _, para_max_speedup, _ = CalculateSpeedup(ablation_data["Filter-GR"], ablation_data["Para"])
    print("# The speedup of Filter-GR and Para up to {:.2f}x and {:.2f}x respectively".format(filter_gr_max_speedup, para_max_speedup))
        
    fig, ax = plt.subplots(figsize=(20, 3.2), dpi=300)
    colors = ["#5B9BD5", "#ed7d31", "#a5a5a5", "#70ad47", "#ffc000"]
    op_num = len(ablation_data["op_name"])
    total_width = 0.7
    single_width = total_width / len(method_list)
    base_ticks = np.arange(op_num)
    for idx, method in enumerate(method_list):
        method_time = ablation_data[method]
        ax.bar(base_ticks + idx * single_width, method_time, width = single_width, label=method, edgecolor='black', color=colors[idx])
    
    ax.legend(loc="upper center", ncol=5,fontsize=20)
    ax.set_ylabel('latency (ms)',fontsize=20)

    x_label = [str(i) for i in range(1, op_num + 1)]
    ax.set_xticks(base_ticks + (len(method_list)/2 - 0.5) * single_width)
    ax.set_xticklabels(x_label, fontsize = 20)
    ax.set_xlabel("sparse convolution kernel id",fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig("./figure/fig8-a.pdf", bbox_inches='tight')
    plt.savefig("./figure/fig8-a.png", bbox_inches='tight')

