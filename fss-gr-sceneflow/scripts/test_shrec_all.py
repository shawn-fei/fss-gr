#参考test_all.py
import os
import sys
import time
import numpy as np
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# add path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from datasets.generic import Batch
from models.scoop import SCOOP
from tools.seed import seed_everything
from tools.losses import compute_loss_unsupervised
from tools.utils import log_string

#测试1：scoop模型输入，1个batch是怎么样的？

pathroot = os.path.dirname(__file__)#去掉文件名后的绝对路径
path2data = os.path.join(pathroot, "..", "data", "FlowNet3D")
path2data = os.path.join(path2data, "Process_NPZ_scoop_SHREC")  # SHREC_npz数据集所在路径


#from datasets.kitti_flownet3d import Kitti
from datasets.shrec_flownet3d import Shrec
# train_dataset = Kitti(root_dir=path2data, nb_points=2048, all_points=False,
#                       same_v_t_split=1, mode="train")
train_dataset = Shrec(root_dir=path2data, nb_points=22, all_points=False,
                                   same_v_t_split=1, mode="train")
# val_dataset = Shrec(root_dir=path2data, nb_points=22, all_points=False,
#                     same_v_t_split=1, mode="val")
#train_dataset属性filename:包含了train/test/val各数据集中examples的文件（npz）路径。每个npz文件中包含3个数组，名字分别是['gt', 'pos2', 'pos1']。利用这些数组，每个npz对应一个dict={"sequence":List [pc1, pc2] ,"ground_truth":List [mask, flow]}

train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=True,
        num_workers=0,#原本是8（多线程），调试时改为0
        collate_fn=Batch,
        drop_last=True,
    )

# Validation data
# val_dataloader = DataLoader(
#     val_dataset,
#     batch_size=1,
#     pin_memory=True,
#     shuffle=False,
#     num_workers=0,
#     collate_fn=Batch,
#     drop_last=False,
# )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#一个batch:{dict:3}.三个键至对是{"sequence":List [pc1, pc2] ,"ground_truth":List [mask, flow]，“orig_size”:List[example_pc_orig_size,example_pc_orig_size]
#pc1,pc2,flow:{Tensor:(batch_size,nb_points,3)}   mask{Tensor:(batch_size,nb_points,1)}
#example_pc_orig_size:{Tensor:(batch_size,1)}
#batch_size:一个batch中有几个sample.(对应几个npz文件)
for it, batch in enumerate(train_dataloader):
# for it, batch in enumerate(val_dataloader):
    # Send data to GPU
    batch = batch.to(device, non_blocking=True)
    batch["sequence"]
    print(it)


#**************KITTI类对象的方法load_sequence（）*****************
# Load data:KITTI类对象的方法load_sequence（） （datasets/kitti_flownet3d.py）
# idx=0
# train_dataset.filename_curr = train_dataset.filenames[idx]
# with np.load(train_dataset.filename_curr) as data:
#     sequence = [data["pos1"][:, (1, 2, 0)], data["pos2"][:, (1, 2, 0)]]
#     ground_truth = [
#         np.ones_like(data["pos1"][:, 0:1]),
#         data["gt"][:, (1, 2, 0)],
#     ]
#
# # Restrict to 35m
# loc = sequence[0][:, 2] < 35
# sequence[0] = sequence[0][loc]
# ground_truth[0] = ground_truth[0][loc]
# ground_truth[1] = ground_truth[1][loc]
# loc = sequence[1][:, 2] < 35
# sequence[1] = sequence[1][loc]

#**************KITTI类对象的方法load_sequence（）*****************
#以上测试1：scoop模型输入，1个batch是怎么样的？

