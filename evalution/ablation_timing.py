import subprocess
import os
from string import Template
import pandas as pd
import argparse




bin_dir = "../Tetris/build/"

def Log(fstream, content):
    fstream.write(str(content) + "\n")

def PefCounterForOperator(fstream, bin, file_path, batch_size, optional=''):
    cmd_template = Template("$a '$b' $c $d")
    stmt = cmd_template.substitute(
        a=bin, b=file_path, c=batch_size, d=optional)
    res = os.popen(stmt).readlines()

    op_profile = {}
    batch = -1
    cnt = 0

    op_config = {}
    pass_cnt = 0
    vendor_time = -1.0
    vendor_flops = -1.0
    for line in res:
        line = line.strip()
        if "FLOPS of" in line:
            key, flops = line.split(" :")
            vendor_flops = float(flops[:-1])
        if "Time of" in line:
            key, t = line.split(" :")
            vendor = key.split(" ")[-1]
            vendor_time = float(t[:-2])
        if "[Config]" in line:
            _, config = line.split("[Config] ")
            config = config.split(", ")
            for item in config:
                k, v = item.split(" = ")
                op_config[k] = v

        if "Pass" in line:
            Log(fstream, line)

        if "Error" in line:
            if vendor != "Sparse":
                continue
            else:
                Log(fstream, vendor)
                Log(fstream, op_config)
                raise ValueError(line)

        if "best tile size" in line:
            Log(fstream, line)

    return op_config, vendor_time, vendor_flops


def AppendDataToDict(record_dict, kernel_size, vendor_name, data):
    if kernel_size not in record_dict.keys():
        record_dict[kernel_size] = {}

    if vendor_name not in record_dict[kernel_size].keys():
        record_dict[kernel_size][vendor_name] = []

    record_dict[kernel_size][vendor_name].append(data)


def PrintOpResult(fstream, vendor, vendor_time, vendor_flops):
    Log(fstream, "{}: time = {:.5f}ms".format(vendor, vendor_time))



def PerfCounterForAllOperator(model_name, batch_size, log_dir, floder_path):
    op_list = os.listdir(floder_path)
    checkpoint = floder_path.split('/')[-2]
    log_file_path = log_dir + "ablation_a" + ".log"
    ops_time_path = log_dir + "ablation_a" + ".csv"

    if os.path.exists(ops_time_path):
        os.remove(ops_time_path)

    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    fstream = open(log_file_path, mode='a+')

    ops_time = {}  # key, value => 'kernel_size': {}
    pd_dict = {
        "op_name": [],
    }

    def _Append2PdDict(method_name, t):
        if method_name not in pd_dict.keys():
            pd_dict[method_name] = []
        pd_dict[method_name].append(t)

    SP_INDEX = 2
    if model_name != "resnet50":
        SP_INDEX = 1

    sparsity = float(floder_path.split("/")[-2].split("-")[SP_INDEX]) / 100.0
    bin_and_name = {
        "baseline_test.o": "Baseline",
        "spf_test.o": "SPF",
        "bank_sr_test.o": "Bank-SR",
        "filter_gr_test.o": "Filter-GR",
        "para_test.o": "Para"
    }
    for idx, op in enumerate(op_list):

        file_path = floder_path + op
        _Append2PdDict("op_name", op)

        Log(fstream, "# Run[{}]: {}".format(idx, file_path))

        log_config = False
        for binary, name in bin_and_name.items():
            config, bin_time, bin_flops = PefCounterForOperator(fstream, bin_dir + binary, file_path, batch_size)
            if log_config == False:
                log_config = True
                Log(fstream, config)
            kernel_size = config['kernel_size']
            AppendDataToDict(ops_time, kernel_size, name, bin_time)
            _Append2PdDict(name, bin_time)
            PrintOpResult(fstream, name, bin_time, bin_flops)

        

        Log(fstream, "-" * 84 + "\n")

    pd_data = pd.DataFrame(pd_dict)
    pd_data.to_csv(ops_time_path, float_format='%.5f')

    fstream.close()
    return ops_time

if __name__ == "__main__":
    batch_list = [4]
    model_list = ["vgg19"]

    for batch_size in batch_list:
        for model_name in model_list:
            base_dir = "./conv_weight_data/" + model_name + "/"
            log_dir = "./log/ablation-log/"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            all_checkpoint = os.listdir(base_dir)
            for checkpoint in all_checkpoint:
                ops_time = PerfCounterForAllOperator(model_name, batch_size, log_dir, base_dir + checkpoint + "/")
