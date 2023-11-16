data_dir='./conv_weight_data/vgg19/vgg19-92-acc-71.7/'
save_dir_0='./log/ablation-log/wo-bank-sr/'
save_dir_1='./log/ablation-log/wi-bank-sr/'
codebase_dir='../Tetris/'

mkdir -p $save_dir_0
mkdir -p $save_dir_1

for file_name in $(ls $data_dir)
do
    weight_file=$data_dir$file_name
    ncu --csv -f -k SparseConv2d -c 1 --metrics group:memory__chart  $codebase_dir/build/wo_bank_sr_test.o $weight_file >$save_dir_0$file_name'.csv'
    ncu --csv -f -k SparseConv2d -c 1 --metrics group:memory__chart  $codebase_dir/build/wi_bank_sr_test.o $weight_file >$save_dir_1$file_name'.csv'
    
done
