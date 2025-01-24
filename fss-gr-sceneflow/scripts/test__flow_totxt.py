#测试将训练集和测试集经过scoop模型输出的recon_flow输出到txt
import os
from data.FlowNet3D.SHREC2017.utils.record import Recorder
import numpy as np

# 记录recon_flow文件路径
path_log_recon_flow = "../test_recon_flow/"  # 日志文件写到：SCOOP/recon_flow/log.txt
try:  # 下面是可能会产生异常的情况
    os.makedirs(path_log_recon_flow)  # 创建多层目录
except FileExistsError:  # 若创建多层目录时已存在，则触发此异常
    pass  # 异常处理，pass：忽略错误继续运行

recon_flow=np.random.randint(0,100,size=[3,3,2]) #recon_flow.shape[0]=3:batch_size=3
# 调用Recorder类方法print_log，将flow输入写入txt
record = Recorder(work_dir=path_log_recon_flow, print_log=True)
print("!!!请问之前生成的{}log.txt删除了吗".format(path_log_recon_flow ))

# 将recon_flow（batch_size,22,3）转换为（batch_size,66），然后按batch_size逐行写入txt
new_recon_flow = recon_flow.reshape(recon_flow.shape[0],
                                    (recon_flow.shape[1] * recon_flow.shape[2]))  # （batch_size,22,3）转换为（batch_size,66）
len_train_list=9   #类比：训练集样本个数2800
len_val_list=9     #类比：训练集样本个数：840
len_data=len_train_list+len_val_list  #类比：2800
batch_Size=3
for it in range(int(len_train_list/batch_Size)):   #it:和第几个batch有关
    for i in range(new_recon_flow.shape[0]):  # 逐行写入txt。 i:batch内第几个
        list_recon_flow = new_recon_flow[i].tolist()  # tolist:array(66)转为list
        list_recon_flow=map(str,list_recon_flow)    #map将list_recon_flow中每个元素转为str
        str_recon_flow = ' '.join(list_recon_flow)  # join:list_recon_flow每个元素中间用空格隔开，转为str
        record.print_log(str_recon_flow,print_time=False)  # 写入日志，但不输出时间
        idx_sample = it * new_recon_flow.shape[0] + i + 1  # 第几个样本
        print("第{}/{}行写入txt".format(idx_sample,len_data)) #不写入日志，只输出

####下面是测试集
val_recon_flow=np.random.randint(100,150,size=[3,3,2])   #假设这是和val_dataset有关recon_flow
# 将recon_flow（batch_size,22,3）转换为（batch_size,66），然后按batch_size逐行写入txt
new_val_recon_flow = val_recon_flow.reshape(val_recon_flow.shape[0],
                                    (val_recon_flow.shape[1] * val_recon_flow.shape[2]))  # （batch_size,22,3）转换为（batch_size,66）
for it in range(int(len_val_list/batch_Size)):
    for i in range(new_val_recon_flow.shape[0]):  # 逐行写入txt
        list_recon_flow = new_val_recon_flow[i].tolist()  # tolist:array(66)转为list
        list_recon_flow=map(str,list_recon_flow)    #map将list_recon_flow中每个元素转为str
        str_recon_flow = ' '.join(list_recon_flow)  # join:list_recon_flow每个元素中间用空格隔开，转为str
        record.print_log(str_recon_flow,print_time=False)  # 写入日志，但不输出时间
        idx_sample = len_train_list+(it * new_val_recon_flow.shape[0] + i + 1)  # 第几个样本:开始要加上训练集总条数len_train_list
        print("第{}/{}行写入txt".format(idx_sample,len_data))
print("!!!请问之前生成的{}log.txt删除了吗".format(path_log_recon_flow ))
print("recon_flow保存至{}log.txt".format(path_log_recon_flow ))