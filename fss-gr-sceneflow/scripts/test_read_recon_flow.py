import os
import re
import torch
import numpy

if __name__ == "__main__":
    target_='1_1_2_1'
    r = re.compile('[ \t\n\r]+')
    path_current = os.path.dirname(__file__)  # 去掉文件名后的绝对路径
    path_recon_flow_2800 = os.path.join(path_current, "..", "recon_flow")#/recon_flow
    txt_recon_flow='/log_'+target_+'.txt'#"/log_1_1_2_1.txt"
    recon_flow = open(path_recon_flow_2800 + txt_recon_flow).readlines()#shape(32,16*3)

    list_video_flow=[[] for i in range(32)]#新建二维列表，32行。最终是list,shape(32,16*3)
    for j, str_frame_flow in enumerate(recon_flow):  # 共32行
        str_frame_flow = r.split(str_frame_flow)  # ['p1_x', 'p1_y', 'p1_z', ……,'p16_x', 'p16_y', 'p16_z', '']
        if str_frame_flow[-1] == '' or "\n":
            str_frame_flow = str_frame_flow[0:-1]  # 去掉换行符['p1_x', 'p1_y', 'p1_z', ……,'p16_x', 'p16_y', 'p16_z']
        list_frame_flow = [eval(x) for x in str_frame_flow]  # 字符外引号去掉变为数值，没用map（可以更换类型为float32）
        # [p1_x, p1_y, p1_z, ……,p16_x, p16_y, p16_z] shape(16*3)
        list_video_flow[j] = list_frame_flow#这是batch中一条数据，也就是一个视频对应的流数据shape（32,48）
    tensor_video_flow=torch.Tensor(list_video_flow)
    array_video_flow=tensor_video_flow.detach().numpy()

    print("finished")