
command_file=`basename "$0"`
gpu=0
model=model_cls_static
num_point=128
num_frame=32
num_epoch=1  #自己加的,point.yaml中赋值epoch。原来是250
batch_size=4 #原来是8
learning_rate=0.001
log_dir=log_${model}_frames${num_frame}_batch_size${batch_size}_epoch${num_epoch}  #最后加了epoch
modality=static

#model_path=pretrained_on_modelnet/model.ckpt
#python main.py --phase=train --work-dir=${log_dir}/results/ --model_path model_path \
python  main_orig_perf.py  --phase=train --work-dir=${log_dir}/results/ \
    --modality $modality \
    --gpu $gpu \
    --network_file $model \
    --learning_rate $learning_rate \
    --log_dir $log_dir \
    --num_point $num_point \
    --num_frame $num_frame \
    --batch_size $batch_size \
    --num-epoch $num_epoch \
    --command_file $command_file \
    #> $log_dir.txt 2>&1 &
    tee $log_dir.txt 2>&1 &

#15:27开始
#epoch 391 1:45