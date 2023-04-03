for mu in 0.01 0.003
do
    for eta in 0.0001 0.0003
    do
        for knn in 60
        do
            CUDA_VISIBLE_DEVICES=0 python main_bbbp_nossl.py \
            --eta $eta --mu $mu  \
            --lr 0.0003 --knn $knn --weight_decay 1e-18 --exp_name "nossl"
        done
    done
done
