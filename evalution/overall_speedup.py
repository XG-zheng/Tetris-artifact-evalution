import os
from  utils import  ReadFromText, GetBatchSize, GetSparsity, GetStandardModelName, GetModelName, GetH, CalculateSpeedup, GetUCFCkptTimePerBatch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse


plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


def PrintKernelGeoMean(ops_data):
    method_list = ["cudnn", "TACO-UCF", "cusparse", "sputnik_wi_im2col", "spconv2d"]
    method_to_label = {
        "cudnn": "cuDNN",
        "cusparse": "cuSPARSE",
        "sputnik_wi_im2col": "Sputnik",
        "TACO-UCF": "TACO-UCF",
        "spconv2d": "Tetris",
    }
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

    all_op_name = select_op.keys()
    our_approch = "spconv2d"
    speedup_per_baseline = {}
    for method in method_list:
        if method != our_approch:
            baseline_time = [ select_op[op_name][method] for op_name in all_op_name]
            our_approch_time = [ select_op[op_name][our_approch] for op_name in all_op_name]
            mean_speedup, max_speedup, min_speedup = CalculateSpeedup(baseline_time, our_approch_time)
            print("# Compare to {} : Geo Mean Speedup = {:.2f}x".format(method_to_label[method], mean_speedup))


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




def ShowOverAll(ops_time, bs_list = [1, 4, 8, 16]):
    colors = ["#9e480e",  "#5b9bd5", "#ed7d31", "#70ad47", "#ffc000"]
    im2col_color = "#a5a5a5"
    baseline = "cudnn"
    our_approch = "spconv2d"
    """
    data_per_batch[batch_size]["model_name"] = [vgg, .....]
    data_per_batch[batch_size][method] = [t1, t2.....]
    
    """
    def CmpModelName(model_name):
        sparsity = int(model_name.split("-")[1])
        model_name = model_name.split("-")[0]
        prior = {
            "mobilenet":0,
            "yolov3":1,
            "vgg19":2,
            "ResNet50":3
        }
        return prior[model_name], sparsity
    ordered_model_name = []
    
    # filter unnecessary models
    for n in ops_time.keys():
        _, sparsity = CmpModelName(n)
        ordered_model_name.append(n)

    ordered_model_name.sort(key=CmpModelName)

    # speedup_per_batch[batch][model][baseline]
    speedup_per_batch = {}
    # model_time_per_batch[batch][model][baseline]
    model_time_per_batch = {}
    data_per_batch = {}
    for model_name in ordered_model_name:
        for batch_size in ops_time[model_name].keys():
            if batch_size == "sparsity":
                continue
            if batch_size not in data_per_batch.keys():
                data_per_batch[batch_size] = {}
                data_per_batch[batch_size]["model_name"] = []

            for method in ops_time[model_name][batch_size].keys():
                if method not in data_per_batch[batch_size].keys():
                    data_per_batch[batch_size][method] = []
            if "im2col" not in data_per_batch[batch_size].keys():
                data_per_batch[batch_size]["im2col"] = []

            if batch_size not in speedup_per_batch.keys():
                speedup_per_batch[batch_size] = {}
                model_time_per_batch[batch_size] = {}

            if model_name not in speedup_per_batch[batch_size].keys():
                speedup_per_batch[batch_size][model_name] = {}
                model_time_per_batch[batch_size][model_name] = {}

            data_per_batch[batch_size]["model_name"].append(model_name)
            data_per_batch[batch_size][baseline].append(1.0)
            for method in ops_time[model_name][batch_size].keys():
                if method != baseline:
                    opt_time = sum(ops_time[model_name][batch_size][method])
                    baseline_time = sum(ops_time[model_name][batch_size][baseline])
                    
                    norm_perf = opt_time / baseline_time
                    data_per_batch[batch_size][method].append(norm_perf)

                if method != our_approch:
                    our_approch_time = sum(ops_time[model_name][batch_size][our_approch])
                    cur_method_time = sum(ops_time[model_name][batch_size][method])
                    speedup_per_batch[batch_size][model_name][method] = cur_method_time / our_approch_time
                
                
                model_time_per_batch[batch_size][model_name][method] = sum(ops_time[model_name][batch_size][method])
            model_time_per_batch[batch_size][model_name]["im2col"] = sum(ops_time[model_name][batch_size]["sputnik_wi_im2col"]) - sum(ops_time[model_name][batch_size]["sputnik_wo_im2col"])
            data_per_batch[batch_size]["im2col"].append(data_per_batch[batch_size]["sputnik_wi_im2col"][-1] - data_per_batch[batch_size]["sputnik_wo_im2col"][-1])
            

    
    marks = ["-", "/", "\\", ".", "+", "o"]
    method_list = ["cudnn", "TACO-UCF", "cusparse", "sputnik_wi_im2col", "spconv2d"]
    method_to_label = {
        "cudnn": "cuDNN",
        "cusparse": "cuSPARSE",
        "sputnik_wi_im2col": "Sputnik",
        "TACO-UCF": "TACO-UCF",
        "spconv2d": "Tetris",
    }

    fig = plt.figure(figsize=(10.5, 4), dpi=200)
    # plt.grid(axis="y", c='#d2c9eb', linestyle = '--',zorder=0)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in' 
    fig.tight_layout()
    plt.subplots_adjust(wspace = 0.1,hspace = 0.45)
    number_to_letter = {
        0: "(a)",
        1: "(b)",
        2: "(c)",
        3: "(d)",
    }
    
    for fig_id, batch_size in enumerate(bs_list):
        all_model_name = data_per_batch[batch_size]["model_name"]
        x_label = []
        for model_name in all_model_name:
            sparsity = model_name.split("-")[1]
            model_name = GetStandardModelName(model_name)
            x_label.append(model_name + "\n" + "(" + sparsity + "%)")
        n = len(method_list)
        total_width = 0.7
        single_width = total_width / n

        base_ticks = np.arange(len(x_label))

        ax = fig.add_subplot(2, 2, fig_id + 1) 
        ax.axhline(y=1,c="#808080",ls="dashed", lw=1)
        y_max = 5.1
 
        idx = 0
        spmm_idx = 0
        need_text_data = []
        for method in method_list:
            if method == "model_name":
                continue
            if method != "cusparse" and "sputnik" not in method:
                ax.bar(base_ticks + idx * single_width, data_per_batch[batch_size][method], width=single_width, label=method_to_label[method], edgecolor='black', color=colors[idx], lw=0.5)
                for x, y in zip(base_ticks + idx * single_width, data_per_batch[batch_size][method]):
                    if y > y_max:
                        need_text_data.append((x, y));
            else:
                spmm_time = [ total - im2col for total, im2col in zip(data_per_batch[batch_size][method], data_per_batch[batch_size]["im2col"])]
                ax.bar(base_ticks + idx * single_width, spmm_time, width=single_width, label=method_to_label[method], edgecolor='black', color=colors[idx], lw=0.5)
                ax.bar(base_ticks + idx * single_width, data_per_batch[batch_size]["im2col"], bottom=spmm_time, width=single_width, label="Im2col", edgecolor='black', color=im2col_color, lw=0.5)
            idx += 1

        text_width = single_width
        text_height = 0.22
        for x, y in need_text_data:
            s = "{:.1f}".format(y)
            ax.text(x - text_width, y_max - text_height, str(s), fontsize=10)

        ax.set_xticks(base_ticks + idx/2 * single_width)
        ax.set_xticklabels(x_label, fontsize = 7)
        ax.set_xlim(-0.5, 7)
        ax.xaxis.set_ticks_position('none')
        ax.set_xlabel(number_to_letter[fig_id] + " batchsize = " + str(batch_size),fontsize=12)
        

        ax.set_ylabel('GM performance',fontsize=10)
        ax.set_ylim(0, y_max)
        ax.set_yticks([i for i in range(5)])


    bar, labels = fig.axes[0].get_legend_handles_labels()
    unique_bar = []
    unique_labels = []
    for b, l in zip(bar, labels):
        if l not in unique_labels:
            unique_bar.append(b)
            unique_labels.append(l)

    fig.legend(unique_bar, unique_labels, loc="upper center", bbox_to_anchor=(0.5,0.97), ncol = len(unique_bar),fontsize=10)
    

    plt.savefig("./figure/fig6.png", bbox_inches='tight')
    plt.savefig("./figure/fig6.pdf", bbox_inches='tight')

if __name__ == "__main__":

    ops_data = {}
    ops_time = {}
    batch_list = [1, 4, 8, 16]
    model_list = ["vgg19", "resnet50", "yolov3", "mobilenet"]
    parser = argparse.ArgumentParser(description='conv2d timing')

    parser.add_argument('--batch_size', type=int, default=-1, help='run specific batch size.')
    parser.add_argument('--model_name', type=str, default='all', help='run specific model.')
    args = parser.parse_args()

    if args.model_name != 'all':
        model_list = [args.model_name]
    
    if args.batch_size != -1:
        batch_list = [args.batch_size]

    base_dir = "./log"
    for model_name in model_list:
        
        ucf_ops_data = {}
        ucf_log_dir = base_dir + "/UCF-log/" + model_name + "/"
        ucf_ops_data = GetUCFCkptTimePerBatch(ucf_log_dir, batch_list)
        model_data_log_dir = base_dir + "/other-log/" + model_name + "/"
        # update ops_data and ops_time in place.
        GetCkptTimePerBatch(model_data_log_dir, ops_time, ops_data, ucf_ops_data, batch_list)
    PrintKernelGeoMean(ops_data)
    ShowOverAll(ops_time, batch_list)
