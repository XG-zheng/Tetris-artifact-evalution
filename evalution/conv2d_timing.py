import subprocess
import os
from string import Template
import pandas as pd
import argparse


# model
run_cudnn_flag = True
run_sputnik_wi_im2col_flag = True
run_sputnik_wo_im2col_flag = True
run_cusparse_flag = True
run_spconv2d_flag = True     # Tetris  

# binnary
cudnn_bin = "../Tetris/build/cudnn_test.o"
sputnik_wi_im2col_bin = "../Tetris/build/sputnik_test.o"
sputnik_wo_im2col_bin = "../Tetris/build/sputnik_test.o"
cusparse_bin = "../Tetris/build/cusparse_test.o"
spconv2d_bin = "../Tetris/build/spconv2d_test.o"   # Tetris


# vendor name
vendor_cudnn_name = 'cudnn'
vendor_sputnik_wi_im2col_name = 'sputnik_wi_im2col'
vendor_sputnik_wo_im2col_name = 'sputnik_wo_im2col'
vendor_cusparse_name = 'cusparse'
vendor_spconv2d_name = 'spconv2d'   # Tetris


# optional:
# sputnik: optional = 1 -> im2col timing

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
    Log(fstream, "{}: time = {:.5f}ms, FLOPS = {:.5f}T".format(
        vendor, vendor_time, vendor_flops))



def PerfCounterForAllOperator(model_name, batch_size, log_dir, floder_path):
    op_list = os.listdir(floder_path)
    checkpoint = floder_path.split('/')[-2]
    log_file_path = log_dir + checkpoint + "_batch_" + str(batch_size) + ".log"
    ops_time_path = log_dir + checkpoint + "_batch_" + str(batch_size) + ".csv"

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
    for idx, op in enumerate(op_list):

        file_path = floder_path + op
        _Append2PdDict("op_name", op)

        Log(fstream, "# Run[{}]: {}".format(idx, file_path))

        if run_cudnn_flag:
            config, cudnn_time, cudnn_flops = PefCounterForOperator(
                fstream, cudnn_bin, file_path, batch_size)
            kernel_size = config['kernel_size']
            AppendDataToDict(ops_time, kernel_size, vendor_cudnn_name, cudnn_time)
            _Append2PdDict(vendor_cudnn_name, cudnn_time)


        if run_sputnik_wi_im2col_flag:
            config, sputnik_wi_im2col_time, sputnik_wi_im2col_flops = PefCounterForOperator(
                fstream, sputnik_wi_im2col_bin, file_path, batch_size, 1)
            kernel_size = config['kernel_size']
            AppendDataToDict(ops_time, kernel_size,
                         vendor_sputnik_wi_im2col_name, sputnik_wi_im2col_time)
            _Append2PdDict(vendor_sputnik_wi_im2col_name, sputnik_wi_im2col_time)


        if run_sputnik_wo_im2col_flag:
            config, sputnik_wo_im2col_time, sputnik_wo_im2col_flops = PefCounterForOperator(
                fstream, sputnik_wo_im2col_bin, file_path, batch_size)
            kernel_size = config['kernel_size']
            AppendDataToDict(ops_time, kernel_size,
                         vendor_sputnik_wo_im2col_name, sputnik_wo_im2col_time)
            _Append2PdDict(vendor_sputnik_wo_im2col_name, sputnik_wo_im2col_time)


        if run_cusparse_flag:
            config, cusparse_time, cusparse_flops = PefCounterForOperator(
                fstream, cusparse_bin, file_path, batch_size)
            kernel_size = config['kernel_size']
            AppendDataToDict(ops_time, kernel_size, vendor_cusparse_name, cusparse_time)
            _Append2PdDict(vendor_cusparse_name, cusparse_time)


        if run_spconv2d_flag:
            config, spconv2d_time, spconv2d_flops = PefCounterForOperator(
                fstream, spconv2d_bin, file_path, batch_size)
            kernel_size = config['kernel_size']
            AppendDataToDict(ops_time, kernel_size,
                         vendor_spconv2d_name, spconv2d_time)
            _Append2PdDict(vendor_spconv2d_name, spconv2d_time)
            

        # log
        Log(fstream, config)
        if run_cudnn_flag:
            PrintOpResult(fstream, vendor_cudnn_name, cudnn_time, cudnn_flops)

        if run_sputnik_wi_im2col_flag:
            PrintOpResult(fstream, vendor_sputnik_wi_im2col_name,
                        sputnik_wi_im2col_time, sputnik_wi_im2col_flops)
        if run_sputnik_wo_im2col_flag:        
            PrintOpResult(fstream, vendor_sputnik_wo_im2col_name,
                        sputnik_wo_im2col_time, sputnik_wo_im2col_flops)

        if run_cusparse_flag:
            PrintOpResult(fstream, vendor_cusparse_name, cusparse_time, cusparse_flops)

        if run_spconv2d_flag:
            PrintOpResult(fstream, vendor_spconv2d_name, spconv2d_time, spconv2d_flops)

        Log(fstream, "-" * 84 + "\n")

    pd_data = pd.DataFrame(pd_dict)
    pd_data.to_csv(ops_time_path, float_format='%.5f')

    fstream.close()
    return ops_time

if __name__ == "__main__":
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

    for batch_size in batch_list:
        for model_name in model_list:
            base_dir = "./conv_weight_data/" + model_name + "/"
            log_dir = "./log/other-log/" + model_name + "/"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            all_checkpoint = os.listdir(base_dir)
            for checkpoint in all_checkpoint:
                ops_time = PerfCounterForAllOperator(model_name, batch_size, log_dir, base_dir + checkpoint + "/")
