import os
import sys
import numpy as np
import re
import datetime
sys.path.append("../..")
from utils.record import Recorder
nbpoint=22
#修改4：生成sherec_npz文件 +该文件运行需要用到record.py，复制到data/FlowNet3D/SHREC2017/utils（utils新建）
class G_SHREC_NPZ(object):
    def __init__(self):
        self.path_log_npz = "../log_npz/" #日志文件在哪:self.path_log_npz/log.txt
        #Recorder类对象
        self.recoder = Recorder(work_dir=self.path_log_npz,print_log=True)

    def print_time(self):
        #输出当前的时间
        now_time = datetime.datetime.now()
        now_time_str = datetime.datetime.strftime(now_time, '%Y-%m-%d %H:%M:%S')
        self.recoder.print_log("["+now_time_str+"]")

    def frame_generate_pc(self,pc_oned):
        #将长度为66的ndarray变为(22,3)。每行是一个关节点的3维坐标
        #输入：pc_oned为shape：66的一维ndarray
        #输出：pc为shape=（22,3）的二维ndarry
        pc = np.zeros((22, 3))
        k = 0
        for j in range(66):
            if j % 3 == 0:
                pc[k][0] = pc_oned[j]
            if j % 3 == 1:
                pc[k][1] = pc_oned[j]
            if j % 3 == 2:
                pc[k][2] = pc_oned[j]
                if k <= 20:
                    k = k + 1
        return pc


    def video_npz(self,idx_video,idx_gesture,idx_finger,idx_subject ,idx_essai,root_datase,project_dir):
        #Input:
        #idx_video：该视频在整个数据集中的序号
        #idx_gesture ,idx_subject ,idx_finger,idx_essai:四个参数定位到某个视频
        #root_datase:'SCOOP/data/FlowNet3D/SHREC2017' 原数据集所在文件夹
        #project_dir:SCOOP/data/FlowNet3D：原数据集，生成的文件夹都在这个下面

        #output:
        #生成npz，并保存。
        #每个npz文件里需要['gt', 'pos2', 'pos1']
        #保存路径：path_npz_kinet，path_npz_scoop。
        #path_npz_kinet：SCOOP/data/FlowNet3D/Process_NPZ_kinet_SHREC//gesture_{}/finger_{}/subject_{}/essai_{}
        #path_npz_scoop：SCOOP/data/FlowNet3D/Process_NPZ_scoop_SHREC

        #return:
        # return scoop_npzfile,kinet_npzfile

        prefix = '/gesture_{}/finger_{}/subject_{}/essai_{}'.format(idx_gesture,idx_finger,idx_subject, idx_essai)
        path_gesture = root_datase + prefix
        #path_gesture = root_datase + '/gesture_' + str(idx_gesture) + '/finger_'   \
                    #+ str(idx_finger) + '/subject_' + str(idx_subject) + '/essai_' + str(idx_essai)+'/'


        path_skeletons_world = path_gesture+'/'+ 'skeletons_world.txt' #数据来源：3维坐标
        skeletons_world = np.loadtxt(path_skeletons_world)#ndarray:(帧数，66)，其中66=22*3.比如（95,66）:95帧每帧22个点的3维坐标
        self.recoder.print_log("******\n导入第{}/2800个样本的关节点坐标，样本来自：{}".format(idx_video+1,prefix))
        #step1:从视频中选两帧，生成对应的点云。
        # 1.索引是id1,id2。  2.选的规则：人选，1/3处和2/3。   3.id1对应的点云是pos1 ndarray(22,3),id2对应的点云是pos2 ndarray(22,3)。
        len_video=len(skeletons_world) #视频的帧数：95
        id1=int(len_video/3)         #比如：32
        id2=int((2*len_video)/3)     #比如：63
        self.recoder.print_log("该视频帧数：{}，选的第一帧：{}，第二帧：{}".format(len_video,id1,id2))

        pos1=skeletons_world[id1-1]  #第一帧的所有点坐标pos1:ndarray(66,)
        pos2=skeletons_world[id2-1]  #第二帧的所有点坐标pos2:ndarray(66,)
        pos1=self.frame_generate_pc(pos1)  #ndaarray:66变为nbpoint*3 ,nbpoint可以是22，也可以是16（此时就是从22个点中选16个点）
        self.recoder.print_log("pos1生成")
        pos2 = self.frame_generate_pc(pos2) #ndaarray:66变为nbpoint*3
        self.recoder.print_log("pos2生成")
        gt = np.zeros((nbpoint, 3))#gt和流有关，应该存放的是真实的流数据。但是现在不需要这个数据，不过为了和原代码统一，就保留了。
        #gt=np.zeros((22, 3))
        self.recoder.print_log("gt生成")
        #step2:新建npz:每个npz文件里需要['gt', 'pos2', 'pos1']。
        #path_npz_kinet:适应于kinet的npz的路径,按照原数据集分级目录保存：gesture-finger-subjrct-essai
        #path_npz_scoop:适应于scoop的npz的路径,按照样本在数据集中序号idx_video命名，都保存在Process_NPZ_scoop_SHREC。
        #关于kinet:隐藏第1，共5
        #path_npz_kinet=project_dir+'/Process_NPZ_kinet_SHREC'+ prefix #SCOOP/data/FlowNet3D/Process_NPZ_kinet_SHREC/gesture_{}/finger_{}/subject_{}/essai_{}
        path_npz_scoop = project_dir + '/Process_NPZ_scoop_SHREC' #SCOOP/data/FlowNet3D/Process_NPZ_scoop_SHREC
        try:  # 下面是可能会产生异常的情况

            #关于kinet:隐藏第2，共5
            #os.makedirs(path_npz_kinet)  # 创建多层目录
            os.makedirs(path_npz_scoop)
        except FileExistsError:  # 若创建多层目录时已存在，则触发此异常
            pass  # 异常处理，pass：忽略错误继续运行
        # 关于kinet:隐藏第3，共5
        #kinet_npzfile=path_npz_kinet+'/'+'flow_two_frame.npz'
        scoop_npzfile=path_npz_scoop+'/'+str(idx_video)+".npz"
        # 关于kinet:隐藏第4，共5
        #np.savez(kinet_npzfile,gt=gt,pos2=pos2,pos1=pos1)
        np.savez(scoop_npzfile,gt=gt,pos2=pos2,pos1=pos1)

        # 关于kinet:隐藏第5，共5
        #return scoop_npzfile,kinet_npzfile
        self.recoder.print_log("npz保存路径:{}".format(scoop_npzfile))
        return scoop_npzfile


    def load_shrec_npz(self):
        #将shrec每个视频转化为npz文件，保存在SCOOP/data/FlowNet3D下，目录是Process_NPZ_kinet_SHREC，Process_NPZ_scoop_SHREC
        #project_dir:'/mnt/e/software/Pycharm/PyCharm-20220203/PythonProject/SCOOP/data/FlowNet3D'
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_dir not in sys.path:
            sys.path.append(project_dir)

        root_datase = os.path.dirname(__file__)#去掉文件名后的绝对路径
        #root_datase = '/mnt/e/software/Pycharm/PyCharm-20220203/PythonProject/SCOOP/data/FlowNet3D/SHREC2017'


        r = re.compile('[ \t\n\r]+')
        train_list = open(root_datase + "/train_gestures.txt").readlines() #train_list{list:1960} train_list[0]={str}'1 1 2 1 1 1 77\n'
        test_list = open(root_datase + "/test_gestures.txt").readlines()#test_list{list:840}
        input_list = train_list + test_list#input_list={list:2800}
        for idx, line in enumerate(input_list):
            # Loading dataset
            splitLine = r.split(line)
            #idx_video=idx
            #idx_gesture,idx_finger,idx_subject ,idx_essai=splitLine[0], splitLine[1], splitLine[2], splitLine[3]
            scoop_npzfile=self.video_npz(idx_video=idx,idx_gesture=splitLine[0],idx_finger=splitLine[1],idx_subject=splitLine[2],idx_essai=splitLine[3],root_datase=root_datase,project_dir=project_dir)
            self.recoder.print_log("第{}/2800个样本对应npz['gt', 'pos2', 'pos1']已生成。".format(idx+1))


    def start(self):
        #创建日志文件路径
        try:  # 下面是可能会产生异常的情况
            os.makedirs(self.path_log_npz)# 创建多层目录
        except FileExistsError:  # 若创建多层目录时已存在，则触发此异常
            pass  # 异常处理，pass：忽略错误继续运行

        #print("开始")
        self.recoder.print_log("开始")
        self.print_time()
        #核心语句：将sherec中2800样本对应的npz生成（保存到data/FlowNet3D/Process_NPZ_scoop_SHREC，隐藏了适应kinet的npz生成相关代码）
        self.load_shrec_npz()

        self.print_time()
        self.recoder.print_log("结束")


generate_shrec_npz=G_SHREC_NPZ()
generate_shrec_npz.start()
