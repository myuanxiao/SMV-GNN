#!/usr/bin/env bash

task_list=(lipo)
running_index_list=(0 1 2 3 4)
model_list=(n_gram_rf n_gram_xgb)
max_depth_list=(5 10 20)
learning_rate_list=(0.01 0.1 1)
n_estimators_list=(50 100 200)
reg_alpha_list=(0 0.3 1)
reg_lambda_list=(0 0.3 1)

for task in "${task_list[@]}"; do
    for model in "${model_list[@]}"; do
        for running_index in "${running_index_list[@]}"; do
        for max_depth in "${max_depth_list[@]}"; do
        for learning_rate in "${learning_rate_list[@]}"; do
        for n_estimators in "${n_estimators_list[@]}"; do
        for reg_alpha in "${reg_alpha_list[@]}"; do
        for reg_lambda in "${reg_lambda_list[@]}"; do
            mkdir -p ../output/"$model"/"$running_index"
            json_file="../hyper/"$model"/"$task".json"
            jq ".max_depth = ${max_depth}" "$json_file" > temp.json && mv temp.json "$json_file"
            jq ".learning_rate = ${learning_rate}" "$json_file" > temp.json && mv temp.json "$json_file"
            jq ".n_estimators = ${n_estimators}" "$json_file" > temp.json && mv temp.json "$json_file"
            jq ".reg_alpha = ${reg_alpha}" "$json_file" > temp.json && mv temp.json "$json_file"
            jq ".reg_lambda = ${reg_lambda}" "$json_file" > temp.json && mv temp.json "$json_file"
            
            python main_regression.py \
            --task="$task" \
            --config_json_file=../config/"$model"/"$task".json \
            --weight_file=temp.pt \
            --running_index="$running_index" \
            --model="$model" > ../output/"$model"/"$running_index"/"$task".out
        done
        done
        done
        done
        done
        done
    done
done