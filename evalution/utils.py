import os
import pandas as pd
import numpy as np

def GetStandardModelName(model_name):
    if "mobilenet" in model_name:
        return "MobileNet"
    if "yolo" in model_name:
        return "YOLOv3"
    if "ResNet50" in model_name:
        return "ResNet50"
    if "vgg19" in model_name:
        return "VGG19"

def GetSparsity(ckpt):
    ckpt = ckpt[:-4]
    sparsity = int(ckpt.split("-")[-3])
    return sparsity

def GetBatchSize(ckpt):
    ckpt = ckpt[:-4]
    batch_size = int(ckpt.split("_batch_")[-1])
    return batch_size

def GetModelName(ckpt):
    ckpt = ckpt[:-4]
    model_name = ckpt.split("-")[0]
    if model_name == "madlag":
        model_name = "bert"
    return model_name

def GetH(op_name):
    last = op_name.split("_H_")[-1]
    h = int(last.split("_")[0])
    return h

def ReadFromText(log_path):
    time_per_checkpoint = {}
    sparsity_per_op = {}
    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "# Run[" in line:
                checkpoint = line.split("/")[-2]
                op_name = line.split("/")[-1].strip()
                if op_name not in time_per_checkpoint.keys():
                    time_per_checkpoint[op_name] = []

            if "{'sparsity':" in line:
                config = eval(line)
                kernel_size = config['kernel_size']
                stride = config['stride']
                sparsity = int(float(config['sparsity']) * 100)
                sparsity_per_op[op_name] = sparsity

            if "{'M':" in line:
                config = eval(line)
                M = config['M']
                K = config['K']
                nnz = config['nnz']
                sparsity = int(100 - float(nnz) / float(M) / float (K) * 100)
                sparsity_per_op[op_name] = sparsity

            if ": time = " in line:
                vendor , _= line.split(": time = ")
                t = float(_.split("ms")[0])
                time_per_checkpoint[op_name].append((vendor, t))
                

    return time_per_checkpoint, sparsity_per_op


def GetUCFCkptTimePerBatch(log_dir, bs_list=[1, 4, 8, 16]):
    ckpt_list = os.listdir(log_dir)
    """
    ucf_ops_data[model_name][batch_size][op_name] = t
    """

    ucf_ops_data = {}
    ckpt_list = os.listdir(log_dir)
    for ckpt in ckpt_list:
        if ".log" not in ckpt or ckpt.startswith("batch"):
            continue

        sparsity = int(float(ckpt.split("-")[-5]))
        model_name = ckpt.split("-")[0]
        batch_size = int(ckpt[:-4].split("-")[-1])
        if batch_size not in bs_list:
            continue
        model_name = model_name + "-" + str(sparsity)
        if model_name not in ucf_ops_data.keys():
            ucf_ops_data[model_name] = {}
        if batch_size not in ucf_ops_data[model_name].keys():
            ucf_ops_data[model_name][batch_size] = {}

        time_per_checkpoint = {}
        with open(log_dir + ckpt, "r") as f:
            all_lines = f.readlines()
            for line in all_lines:
                if "Layer c" in line:
                    config = line.split(".smtx")[1].strip().split(" ")
                    config = [int(item) for item in config]
                    batch, IH, IW, KS, _, stride, padding, IC, OC = config
                    op_name = line.split(".smtx")[0].split("/")[-1]
                if "ms" in line:
                    t = float(line.split("/")[-1].split("ms")[0])
                    time_per_checkpoint[op_name] = t
                    op_name = ""
        ucf_ops_data[model_name][batch_size] = time_per_checkpoint


    return ucf_ops_data


def CalculateSpeedup(baseline, opt):
    max_speedup = -10000
    min_speedup = 10000
    speedup_per_op = []
    for x, y in zip(baseline, opt):
        s = x * 1.0 / y
        speedup_per_op.append(s)
        max_speedup = max(max_speedup, s)
        min_speedup = min(min_speedup, s)

    mean_speedup = np.exp(np.mean(np.log(np.array(speedup_per_op))))
    return mean_speedup, max_speedup, min_speedup
