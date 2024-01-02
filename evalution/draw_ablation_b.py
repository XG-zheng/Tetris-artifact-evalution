import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import CalculateSpeedup

csfont = {'family':'Times New Roman'}
plt.rc('font', **csfont)


wavefronts_name = '\"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum\"'
request_name = "sass__inst_executed_shared_loads"

def ReadMetricFromDir(profile_dir):
    all_op_profile = os.listdir(profile_dir)
    ops_metrics = {}
    for op_profile in all_op_profile:
        op_name = op_profile[:-4]
        ops_metrics[op_name] = {}
        with open(profile_dir + op_profile, "r") as f:
            all_lines = f.readlines()
            for line in all_lines:
                if wavefronts_name in line:
                    val = int(line.split(",")[-1].strip()[1:-1])
                    ops_metrics[op_name][wavefronts_name] = val
                if request_name in line:
                    val = int(line.split(",")[-1].strip()[1:-1])
                    ops_metrics[op_name][request_name] = val

    return ops_metrics
if __name__ == "__main__":
    
    base_dir = "./log/ablation-log/"
    baseline_ops_metrics = ReadMetricFromDir(base_dir + "wo-bank-sr/")
    opt_ops_metrics = ReadMetricFromDir(base_dir + "wi-bank-sr/")
    for op_name in baseline_ops_metrics.keys():
        if baseline_ops_metrics[op_name][request_name] != opt_ops_metrics[op_name][request_name]:
            raise ValueError("requese should be equal")

        if opt_ops_metrics[op_name][request_name] > opt_ops_metrics[op_name][wavefronts_name]:
            raise ValueError("wavefront error")
        
    all_ops_name = []
    for op_name in baseline_ops_metrics.keys():
        if "IC_3_" in op_name:
            continue
        all_ops_name.append(op_name)
    theoretical_wave = []
    baseline_wave = []
    opt_wave = []
    # fix the order of kernel
    all_ops_name = [
        "module3_0_conv2d_0_H_112_W_112_IC_64_OC_128_KS_3_Pad_1_S_1_G_1",
        "module5_0_module3_0_conv2d_0_H_224_W_224_IC_64_OC_64_KS_3_Pad_1_S_1_G_1",
        "module8_2_module2_0_conv2d_5_H_14_W_14_IC_512_OC_512_KS_3_Pad_1_S_1_G_1",
        "module8_0_module2_0_conv2d_1_H_56_W_56_IC_128_OC_256_KS_3_Pad_1_S_1_G_1",
        "module8_0_module2_0_conv2d_3_H_56_W_56_IC_256_OC_256_KS_3_Pad_1_S_1_G_1",
        "module8_1_module2_0_conv2d_5_H_28_W_28_IC_512_OC_512_KS_3_Pad_1_S_1_G_1",
        "module8_0_module3_0_conv2d_0_H_112_W_112_IC_128_OC_128_KS_3_Pad_1_S_1_G_1",
        "module8_1_module2_0_conv2d_1_H_28_W_28_IC_256_OC_512_KS_3_Pad_1_S_1_G_1",
        "module8_2_module3_0_conv2d_0_H_28_W_28_IC_512_OC_512_KS_3_Pad_1_S_1_G_1",
        "module8_1_module2_0_conv2d_3_H_28_W_28_IC_512_OC_512_KS_3_Pad_1_S_1_G_1",
        "module8_1_module3_0_conv2d_0_H_56_W_56_IC_256_OC_256_KS_3_Pad_1_S_1_G_1",
        "module8_2_module2_0_conv2d_3_H_14_W_14_IC_512_OC_512_KS_3_Pad_1_S_1_G_1",
        "module8_2_module2_0_conv2d_1_H_14_W_14_IC_512_OC_512_KS_3_Pad_1_S_1_G_1",
        "module5_1_module3_0_conv2d_0_H_14_W_14_IC_512_OC_512_KS_3_Pad_1_S_1_G_1",
        "module8_0_module2_0_conv2d_5_H_56_W_56_IC_256_OC_256_KS_3_Pad_1_S_1_G_1"
    ]
    for op_name in all_ops_name:
        baseline_wave.append(baseline_ops_metrics[op_name][wavefronts_name])
        opt_wave.append(opt_ops_metrics[op_name][wavefronts_name])
        theoretical_wave.append(opt_ops_metrics[op_name][request_name])

    baseline_geo_speedup, _, _ = CalculateSpeedup(baseline_wave, theoretical_wave)

    print("# The average number of load request for Without-Bank-SR compare to Theoretical: {:.5f}x".format(baseline_geo_speedup))
    opt_geo_speedup, _, _ = CalculateSpeedup(opt_wave, theoretical_wave)
    print("# The average number of load request for With-Bank-SR compare to Theoretical: {:.5f}x".format(opt_geo_speedup))

    fig, ax = plt.subplots(figsize=(20, 3.2), dpi=300)
    colors = ["#5B9BD5", "#ed7d31", "#ffc000", "#a5a5a5", "#70ad47"]
    op_num = len(all_ops_name)
    total_width = 0.7
    single_width = total_width / 3
    base_ticks = np.arange(op_num)
    ax.bar(base_ticks, theoretical_wave, width = single_width, label="Theoretical request", edgecolor='black', color=colors[0])
    ax.bar(base_ticks + single_width, opt_wave, width = single_width, label="With-Bank-SR", edgecolor='black', color=colors[2])
    ax.bar(base_ticks + 2 * single_width, baseline_wave, width = single_width, label="Without-Bank-SR", edgecolor='black', color=colors[1])
    
    ax.legend(loc="upper center", ncol=5,fontsize=20)
    ax.set_ylabel('Number of Wavefronts')
    x_label = [str(i) for i in range(1, op_num + 1)]
    ax.set_xticks(base_ticks + single_width)
    ax.set_xticklabels(x_label, fontsize = 20)
    ax.set_yticks([0, 0.5 * 1e8, 1e8, 1.5 * 1e8])
    ax.set_xlabel("Convolution Id")
    plt.savefig("./figure/fig8-b.png", bbox_inches='tight')
    plt.savefig("./figure/fig8-b.pdf", bbox_inches='tight')
