yaml_file="./config_finetune.yaml"


batch_size_list=(32 64 128)
weight_decay_list=(1e-4 1e-6 -1e-8)           
init_lr_list=(0.01 0.001 0.001)
init_base_lr_list=(0.01 0.001 0.001)
num_layer_list=(3 5 7)
feat_dim_list=(128 256 512)
drop_ratio_list=(0.1 0.3 0.5)
emb_dim_list=(100 200 300)

for batch_size in batch_size_list
do
    for weight_decay in weight_decay_list
    do
        for init_lr in init_lr_list 
        do
            for drop_ratio  in drop_ratio_list 
            do
            for num_layer in num_layer_list 
            do
                sed -i "s/\(batch_size:\).*/\1 $batch_size/" "$yaml_file"
                sed -i "s/\(weight_decay:\).*/\1 $weight_decay/" "$yaml_file"
                sed -i "s/\(init_lr:\).*/\1 $init_lr/" "$yaml_file"
                sed -i "s/\(num_layer:\).*/\1 $num_layer/" "$yaml_file"
                sed -i "s/\(drop_ratio:\).*/\1 $drop_ratio/" "$yaml_file"
                python finetune.py
            done
            done
        done
    done
done
