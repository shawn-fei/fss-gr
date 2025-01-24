#!/usr/bin/env bash
# train on kitti_v
python train_scoop.py --dataset_name FlowNet3D_kitti --nb_points 2048 \
    --batch_size_train 1 --batch_size_val 1 --nb_epochs 1 --nb_workers 8 \
    --backward_dist_weight 0.0 --use_corr_conf 1 --corr_conf_loss_weight 0.1 \
    --add_model_suff 1 --save_model_epoch 1 --log_dir kitti_v_100_examples
