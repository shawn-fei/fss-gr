
command_file=`basename "$0"`
gpu=0
model=model_cls_dynamic_2_4a2_3  #动态分支模型所在的python文件
num_point=128
num_frame=32
batch_size=8 #二分支为4，三分支为8。 同时注意修改shrec2017_release/pointlstm.yaml中size，要统一
num_epoch=250 ##原来是250，测成本时为1
learning_rate=0.001
decay_step=200000 #默认200000，查之后认为这个值=训练样本数1960/batch_size8=245。即一个epoch后lr=0.0015*decay_rate。如果设置为245*2.那就是2个epoch后变化
decay_rate=0.7 #默认0.7。我希望是loss降到<1后减速 ，比如20个epoch后变为0.0016,40个后学习率再减小
#model_path=trained_static_model/model.ckpt
log_dir=log_${model}_K6_7_exp2_3_base249_frames${num_frame}_batch_size${batch_size}_epoch${num_epoch} #最后加了epoch
#下行代码是对model_path的修改，原来:model_path=trained_static_model/model.ckpt
#model_cls_static5_2
#model_path=./log_model_cls_static_frames32_batch_size8/model249.ckpt #num_epoch=250,batch_size=8
model_path=./log_model_cls_static_frames32_batch_size4_epoch250/model249.ckpt #static,epoch=1,batch_size=1

#上行代码是对model_path的修改
modality=dynamic

python -u main_2.py 2>&1 --phase=train --work-dir=${log_dir}/results/ \
    --modality $modality \
    --gpu $gpu \
    --network_file $model \
    --learning_rate $learning_rate \
    --model_path $model_path \
    --log_dir $log_dir \
    --num_point $num_point \
    --num_frame $num_frame \
    --batch_size $batch_size \
    --num-epoch $num_epoch \
    --decay_step $decay_step \
    --decay_rate $decay_rate \
    --command_file $command_file \

    tee > $log_dir.txt 2>&1 &
#3:04
#python -u main.py 2>&1 --phase=train --work-dir=${log_dir}/results/ \
#pythonmain.py--phase=train--work-dir=${log_dir}/results/\
#tee $log_dir.txt 2>&1 &
#> $log_dir.txt 2>&1 &