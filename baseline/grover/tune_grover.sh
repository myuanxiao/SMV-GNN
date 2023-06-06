for epochs in 10 20 30
do
    for batch_size in 16 32 64
    do
        for ffn_hidden_size in 100 200 400
        do
            for init_lr in 0.00045 0.00015 0.00005
            do
                python main.py finetune --data_path ../train_lipo.csv \
                            --features_path exampledata/finetune/train_lipo.npz \
                            --save_dir model/finetune/lipo/ \
                            --checkpoint_path model/tryout/model.ep3 \
                            --dataset_type regression \
                            --ensemble_size 1 \
                            --no_features_scaling \
                            --ffn_hidden_size 200 \
                            --batch_size 32 \
                            --epochs 10 \
                            --init_lr 0.00015
            done
        done
    done
done


