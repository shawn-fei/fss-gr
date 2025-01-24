#!/usr/bin/env bash
# evaluate on shrec
python evaluate_scoop.py --dataset_name FlowNet3D_shrec --mode all --nb_points 16 --all_points 0 --all_candidates 0 \
    --path2ckpt ./../experiments/shrec_62720_examples_bs16_e1_p16_scoop3_2/model_e001.tar \
    --use_test_time_refinement 1 --test_time_num_step 1000 --test_time_update_rate 0.05 \
    --log_fname log_evaluation_scoop3_2_shrec.txt

