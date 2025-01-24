#参考process_recon_flow.py
#该文件新建，目标将89600条流数据(est_flow，测试阶段优化后场景流)，变为2800个txt文件。一个txt里是同一个样本的32条（16*3）数据。这32条按时间先后排。
import os
import re
import sys
# add path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)
from data.FlowNet3D.SHREC2017.utils.record import Recorder

if __name__ == "__main__":
    # 1.Path to current file,datasource,est_flow
    path_current = project_dir  # SCOOP
    path_datasource = os.path.join(path_current, "data_source_est_flow")#SCOOP/data_source_est_flow
    path_estflow = os.path.join(path_current, "est_flow")
    path_dataset=os.path.join(path_current, "data","FlowNet3D","SHREC2017")

    #2.读进数据来源，estflow_list和train_list/test_list
    #prefix = dataset_prefix + "/gesture_{}/finger_{}/subject_{}/essai_{}"
    datasource_list = open(path_datasource + "/log.txt").readlines()#SCOOP\data_source_est_flow\log.txt
    estflow_list = open(path_estflow + "/log.txt").readlines()#SCOOP\est_flow\log.txt
    train_list = open(path_dataset + "/train_gestures.txt").readlines()  #data/FlowNet3D/SHREC2017/train_gestures.txt
    test_list = open(path_dataset + "/test_gestures.txt").readlines()  # data/FlowNet3D/SHREC2017/test_gestures.txt
    input_list = train_list + test_list

    r = re.compile('[ \t\n\r]+')
    for idx, line in enumerate(input_list):#按照input_list前4个数据依次查找
        # traget:查找数据源，' 1 1 2 1\n'
        splitLine = r.split(line)#['1', '1', '2', '1', '1', '1', '77', '']
        idx_gesture, idx_finger, idx_subject, idx_esaai = splitLine[0], splitLine[1], splitLine[2], splitLine[3]
        ds_l=[idx_gesture, idx_finger, idx_subject, idx_esaai]
        target=" ".join(ds_l)
        target_= "_".join(ds_l)
        target=" "+target+"\n"#' 1 1 2 1\n' 注意：前加空格，不然"11 1 2 1\n"也会被选出来
        print("查找数据源：",target)

        #第一，根据后4个数字（target）查找出32条数据来源（对应索引idx32_datasource_list）
        idx32_datasource_list = [idx_datasource for idx_datasource,s in enumerate(datasource_list) if target in s]
        sub32_datasource_list=[datasource_list[index] for index in idx32_datasource_list]#根据32个索引，取出对应32行数据来源
        #print("32条流对应的索引(在89600条数据中):",idx32_datasource_list)

        #第二，然后根据32条数据来源的第一个数排序（按帧排，升序），返回排序后32个索引
        str_idx32_89600=[]#放了32条数据来源的第一个数(字符串形式)
        for line_32 in sub32_datasource_list:
            splitLine_32 = r.split(line_32)
            idx_89600=splitLine_32[0]#数据来源的第一个数,也就是89600个npz里的第几个文件
            str_idx32_89600.append(idx_89600)
        idx32_89600 = list(map(int, str_idx32_89600))#放了32条数据来源的第一个数(字符串形式)
        sorted_id32=sorted(range(len(idx32_89600)), key=lambda k: idx32_89600[k])#idx32_89600升序排序后返回值的索引
        ordered_idx32_89600 = sorted(idx32_89600)#idx32_89600升序后的结果
        ordered_idx32_datasource_list=[idx32_datasource_list[i] for i in sorted_id32]#返回排序后32个索引
        #print("按帧排序的32条流数据在89600条流数据中索引:",ordered_idx32_datasource_list)

        #第三，按排序后32个索引ordered_idx32_datasource_list，从SCOOP\est_flow\log.txt取出32行数据（每行数据16*3），然后保存到SCOOP\est_flow\8_2_2_3.txt
        est_flow_video=[]
        # 记录est_flow_video文件路径
        path_log_est_flow = path_estflow
        try:  # 下面是可能会产生异常的情况
            os.makedirs(path_log_est_flow)  # 创建多层目录
        except FileExistsError:  # 若创建多层目录时已存在，则触发此异常
            pass  # 异常处理，pass：忽略错误继续运行
        record = Recorder(work_dir=path_log_est_flow, print_log=True)
        record.log_path = '{}/log_{}.txt'.format(path_log_est_flow,target_)  # 日志文件写到：SCOOP/est_flow/log_1_1_2_1.txt

        #将32行数据取出来写入
        print("开始写第{}/2800个样本对应的32条流数据(estimated flow:est_flow)".format(idx+1))

        for i,j in enumerate(ordered_idx32_datasource_list):
            est_flow_line=estflow_list[j]#待写入的一行数据(16*3一行数据，格式是字符串)
            est_flow_line=est_flow_line.replace('\n', '')#去掉一行最后的换行符
            est_flow_video.append(est_flow_line)#最终est_flow_video是包含32行est_flow的list,每行是字符串
            record.print_log(est_flow_line,print_time=False)  # 写入日志，但不输出时间
            #print("第{}/32条流数据已写入".format(i+1))
        print("第{}/2800个样本对应的32条流数据写入路径：\n{}".format(idx+1,record.log_path))
        print("************************")




