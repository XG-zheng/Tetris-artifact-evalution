from utils import  CalculateSpeedup, GetUCFCkptTimePerBatch, ReadFromText, GetBatchSize, GetSparsity, GetStandardModelName, GetModelName, GetH
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

# print(matplotlib.matplotlib_fname())
csfont = {'family':'Times New Roman'}
plt.rc('font', **csfont)

def GetCkptTimePerBatch(model_log_dir,  ops_time, ops_data, ucf_ops_data = {}, bs_list=[1, 4, 8, 16]):

    ckpt_list = os.listdir(model_log_dir)
    for ckpt in ckpt_list:
        if ".log" not in ckpt:
            continue
        path = model_log_dir + ckpt
        batch_size = GetBatchSize(ckpt)
        if batch_size not in bs_list:
            continue
        model_name = GetModelName(ckpt)
        sparsity = GetSparsity(ckpt)

        model_name = model_name + "-" + str(sparsity)

        if model_name not in ops_time.keys():
            ops_time[model_name] = {}
            ops_data[model_name] = {}
        if batch_size not in ops_time[model_name].keys():
            ops_time[model_name][batch_size] = {}
            ops_data[model_name][batch_size] = {}
     
        time_per_checkpoint, sparsity_per_op = ReadFromText(path)

        for op_name, vals in time_per_checkpoint.items():

            crush_flag = False
            for vendor, t in vals:
                if t == -1:                
                    crush_flag = True
            if crush_flag:
                continue

            if op_name not in ops_data[model_name][batch_size].keys():
                ops_data[model_name][batch_size][op_name] = {}
                ops_data[model_name][batch_size][op_name]["sparsity"] = sparsity_per_op[op_name]
            if ucf_ops_data:
                if "TACO-UCF" not in ops_time[model_name][batch_size].keys():
                    ops_time[model_name][batch_size]["TACO-UCF"] = []
                ops_time[model_name][batch_size]["TACO-UCF"].append(ucf_ops_data[model_name][batch_size][op_name])
                ops_data[model_name][batch_size][op_name]["TACO-UCF"] = ucf_ops_data[model_name][batch_size][op_name]

            for vendor, t in vals:
                if vendor not in ops_time[model_name][batch_size].keys():
                    ops_time[model_name][batch_size][vendor] = []
                ops_time[model_name][batch_size][vendor].append(t)
                ops_data[model_name][batch_size][op_name][vendor] = t


def PrintKernelGeoMean(select_op):
    method_list = ["cudnn", "TACO-UCF", "cusparse", "sputnik_wi_im2col", "spconv2d"]
    all_op_name = select_op.keys()
    our_approch = "spconv2d"
    speedup_per_baseline = {}
    for method in method_list:
        if method != our_approch:
            baseline_time = [ select_op[op_name][method] for op_name in all_op_name]
            our_approch_time = [ select_op[op_name][our_approch] for op_name in all_op_name]
            mean_speedup, max_speedup, min_speedup = CalculateSpeedup(baseline_time, our_approch_time)
            print("# Compare to {} : mean = {:.2f}x, min = {:.2f}x, max = {:.2f}x".format(method, mean_speedup, min_speedup, max_speedup))


def KernelBenchmark(ops_data):
    baseline = "cudnn"
    our_approch = "spconv2d"
    method_to_label = {
        "cudnn": "cuDNN",
        "cusparse": "cuSPARSE",
        "sputnik_wi_im2col": "Sputnik",
        "TACO-UCF": "TACO-UCF",
        "spconv2d": "Tetris",
    }
    colors = ["#5c9bd4", "#ed7c31", "#70ad47", "#ffbf01"]
    cudnn_color = "red"
    method_list = ["cudnn", "TACO-UCF", "cusparse", "sputnik_wi_im2col", "spconv2d"]
    select_op = {}

    benchmark_models = []
    for model_name in ops_data.keys():
        benchmark_models.append(model_name)
    for model_name in benchmark_models:
        for batch_size in ops_data[model_name].keys():
            for op_name in ops_data[model_name][batch_size].keys():
                sparsity = ops_data[model_name][batch_size][op_name]["sparsity"]
                if sparsity == 0:
                    continue

                config = op_name + "_sparsity_" + str(sparsity) + "_batch_" + str(batch_size)

                if config not in select_op.keys():
                    select_op[config] = ops_data[model_name][batch_size][op_name]
                    select_op[config].pop("sparsity")

    fig = plt.figure(figsize=(10.5, 4), dpi=200)

    plt.subplots_adjust(wspace = 0.1,hspace = 0.45)

    def CmpOpName(op_name):
        batch = int(op_name.split("_batch_")[1])
        sparsity = int(op_name.split("_batch_")[0].split("_sparsity_")[1])
        return batch, sparsity

    ordered_ops_name = [op_name for op_name in select_op.keys()]
    ordered_ops_name.sort(key=CmpOpName)

    # calculate speedup
    PrintKernelGeoMean(select_op)

    marks = ["-", "/", "\\", ".", "+", "o"]
    
    mn_val = 10000
    mx_val = 0

    bs_list = [1, 4, 8, 16]
    for fig_id, batch_size in enumerate(bs_list):
        ax = fig.add_subplot(2, 2, fig_id + 1) 
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        cur_batch_ops_name = []
        
        annotated_sparsity = [70, 80, 90, 95]
        annotation_pos = [0 for i in range(len(annotated_sparsity))]
        sp_idx = 0
        for i, op_name in enumerate(ordered_ops_name):
            op_batch, sparsity = CmpOpName(op_name)
            if op_batch == batch_size:
                cur_batch_ops_name.append(op_name)
            if sp_idx < len(annotated_sparsity):
                if annotated_sparsity[sp_idx] >= sparsity:
                    annotation_pos[sp_idx] = i
                else:
                    sp_idx += 1
                    if sp_idx < len(annotated_sparsity) and annotated_sparsity[sp_idx] >= sparsity:
                        annotation_pos[sp_idx] = i
        n = len(cur_batch_ops_name)
        base_ticks = np.arange(n)
        
        # cudnn
        ax.plot(base_ticks, np.ones(n), label=method_to_label[baseline], ls="dashed", c=cudnn_color, linewidth=2.5)

        idx = 0
        for method in method_list:
            # print(ordered_ops_name[721], method, select_op[ordered_ops_name[721]][method])
            if method == baseline:
                continue
            time_method = [ select_op[op_name][baseline] / select_op[op_name][method] for op_name in cur_batch_ops_name]
            ax.scatter(base_ticks, time_method, label=method_to_label[method], s=10, marker="^", c=colors[idx])
            idx += 1


        
        y_max = 101
        y_min = 0.04
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel('log(speedup)', fontsize=10)
        ax.set_yscale("log", base=10)
        ax.set_yticks([0.1, 1, 10, 100])
        ax.set_yticklabels(["0.1", "1", "10", "100"], fontsize=8)
        ax.set_xlabel('convolution sparsity', fontsize=10)
        ax.set_xlim(0)
        
        # annotated_sparsity = [59] + annotated_sparsity
        # annotation_pos = [0] + annotation_pos
        ax.set_xticks(annotation_pos)
        ax.set_xticklabels([ str(sp) + "%" for sp in annotated_sparsity], fontsize=8)
    
    bar, labels = fig.axes[0].get_legend_handles_labels()

    fig.legend(bar, labels, loc="upper center", bbox_to_anchor=(0.5,0.97), ncol = len(bar),fontsize=10)
    
    plt.savefig("./figure/fig7.pdf", bbox_inches='tight')
    plt.savefig("./figure/fig7.png", bbox_inches='tight')


if __name__ == "__main__":

    ops_data = {}
    ops_time = {}
    batch_list = [1, 4, 8, 16]
    model_list = ["vgg19", "resnet50", "yolov3", "mobilenet"]


    base_dir = "./log"
    for model_name in model_list:

        ucf_ops_data = {}
        ucf_log_dir = base_dir + "/UCF-log/" + model_name + "/"
        ucf_ops_data = GetUCFCkptTimePerBatch(ucf_log_dir, batch_list)
        model_data_log_dir = base_dir + "/other-log/" + model_name + "/"
        # update ops_data and ops_time in place.
        GetCkptTimePerBatch(model_data_log_dir, ops_time, ops_data, ucf_ops_data, batch_list)

    KernelBenchmark(ops_data)

