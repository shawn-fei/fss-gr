#!/usr/bin/env bash
# train on shrec
#改1：dataset_name 改2：log_dir 改5：nb_points（原来是2048） 思路1/2/3都是16，SCOOP4是22
#log_dir:训练生成模型所在地
#nb_epochs 100,batch_size_train 16
python train_scoop.py --dataset_name FlowNet3D_shrec --nb_points 16 \
    --batch_size_train 16 --batch_size_val 1 --nb_epochs 1 --nb_workers 8 \
    --backward_dist_weight 0.0 --use_corr_conf 1 --corr_conf_loss_weight 0.1 \
    --add_model_suff 1 --save_model_epoch 50 --log_dir shrec_62720_examples_bs16_e1_p16_scoop3_2