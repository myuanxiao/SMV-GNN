#!/bin/bash
cd $(dirname $0)
cd ..

source ~/.bashrc
source ./scripts/utils.sh

root_path="$(pwd)/../../.."
# export FLAGS_fraction_of_gpu_memory_to_use=0.6
export PYTHONPATH="$root_path/":$PYTHONPATH

datasets="lipo"

compound_encoder_config="model_configs/pregnn_paper.json"
# compound_encoder_config="model_configs/pregnn_feat_l8_readavg.json"
model_config="model_configs/down_linear.json"
# model_config="model_configs/down_mlp3.json"

init_model="$root_path/output/pretrain_gnns/pregnn_paper-pre_Attrmask-pre_Supervised/epoch40/compound_encoder.pdparams"

count=0
for dataset in $datasets; do
	echo $dataset
	data_path="$root_path/data/chem_dataset/$dataset"
	# log_file="log_paper/finetune-pregnn_paper-down_linear-$dataset.txt"
	log_file="log_paper/pre_Attrmask_Supervised-pregnn_paper-down_linear-$dataset.txt"
	for time in $(seq 1 4); do
	for batch_size in 16 32 64; do
	for encoder_lr in 1e-2 1e-3 1e-4; do
	for head_lr in 1e-2 1e-3 1e-4; do
	for dropout_rate in 0.1 0.2 0.3; do
		cuda_id=$(($count % 8))
		CUDA_VISIBLE_DEVICES=$cuda_id paddle2.0 finetune.py \
				--batch_size=$batch_size \
				--max_epoch=100 \
				--dataset_name=$dataset \
				--data_path=$data_path \
				--split_type=scaffold \
				--compound_encoder_config=$compound_encoder_config \
				--model_config=$model_config \
				--init_model=$init_model \
				--model_dir=$root_path/output/pretrain_gnns/finetune/$dataset \
				--encoder_lr=$encoder_lr \
				--head_lr= $head_lr \
				--dropout_rate=$dropout_rate >> $log_file 2>&1 &
		let count+=1
		if [[ $(($count % 16)) -eq 0 ]]; then
			wait
		fi
	done
	done
	done
	done
	done
done
wait

