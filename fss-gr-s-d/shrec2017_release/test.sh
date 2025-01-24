
command_file=`basename "$0"`
gpu=0
model=model_cls_dynamic_2_4a2_3
num_point=128
num_frame=32
batch_size=8 #原来：8
num_epoch=250 #自己加的
learning_rate=0.001
#model_path=trained_dynamic_model/model.ckpt
#下行代码是对model_path的修改，原来:model_path=trained_dynamic_model/model.ckpt
#model_path=./log_model_cls_dynamic_frames32_batch_size8/model0249.ckpt #dynamic,batchsize=8,epoch=250
model_path=./log_model_cls_dynamic_2_4a2_3_K6_7_exp2_3_base249_frames32_batch_size8_epoch250/model0229.ckpt  #dynamic,batchsize=1,epoch=1
#上行代码是对model_path的修改
log_dir=testlog_exp6_7_base229_${model}_frames${num_frame}_batch_size${batch_size}_epoch${num_epoch} #最后加了epoch

python main_2.py  --phase=test --work-dir=${log_dir}/results/ \
    --gpu $gpu \
    --network_file $model \
    --learning_rate $learning_rate \
    --model_path $model_path \
    --log_dir $log_dir \
    --num_point $num_point \
    --num_frame $num_frame \
    --batch_size $batch_size \
    --num-epoch $num_epoch \
    --command_file $command_file \
    #> $log_dir.txt 2>&1 &
    tee $log_dir.txt 2>&1 &

