#参考datasets/kitti_flownet3d.py建
#参考datasets/shrec_flownet32_2.py建：该文件是一个视频只用2帧生成流。
#89600个文件
import os
import glob
import numpy as np
from .generic import SceneFlowDataset


class Shrec(SceneFlowDataset):
    def __init__(self, root_dir, nb_points, all_points, same_v_t_split, mode):
        """
        Construct the KITTI scene flow datatset as in:
        Liu, X., Qi, C.R., Guibas, L.J.: FlowNet3D: Learning scene ﬂow in 3D
        point clouds. IEEE Conf. Computer Vision and Pattern Recognition
        (CVPR). pp. 529–537 (2019)

        Parameters
        ----------
        root_dir : str
            Path to root directory containing the datasets.
        nb_points : int
            Maximum number of points in point clouds.
        all_points : bool
            Whether to use all point in the point cloud (in chucks of nb_points) or only nb_points.
        same_v_t_split: bool
            Whether to use the same validation and test split.
        mode : str
            'train': training dataset.

            'val': validation dataset.

            'test': test dataset

            'all': all dataset

        """
        #下： 改6
        #super(Kitti, self).__init__(nb_points, all_points)
        super(Shrec, self).__init__(nb_points, all_points)
        # 上： 改6
        self.mode = mode
        self.root_dir = root_dir
        self.same_v_t_split = same_v_t_split
        self.filenames = self.make_dataset()
        self.filename_curr = ""

    def __len__(self):

        return len(self.filenames)

    def make_dataset(self):
        """
        Find and filter out paths to all examples in the dataset.

        """
        # filename:包含了train/test/val各数据集中examples的文件路径,从filenames_all中筛选。
        #下改7：数据集长度
        #len_dataset = 150
        #第二次修改：修改3，数据集共有2600*32
        len_dataset = 89600
        # 上第二次修改：修改3
        #上改7：数据集长度

        #root_dir，也是"Project_root/data/FlowNet3D/Process_NPZ_scoop_SHREC"
        filenames_all = glob.glob(os.path.join(self.root_dir, "*.npz"))
        # print("filenames_all",filenames_all)

        #下改8：test_list:后840条（1960-2799）
        #test_list = [1, 5, 7, 8, 10, 12, 15, 17, 20, 21, 24, 25, 29, 30, 31, 32, 34, 35, 36, 39, 40, 44, 45, 47, 48,
                     #49, 50, 51, 53, 55, 56, 58, 59, 60, 70, 71, 72, 74, 76, 77, 78, 79, 81, 82, 88, 91, 93, 94, 95, 98]
        #第二次修改，改4
        test_list =[i for i in range(62720,89600)] #  #test_list对应序号62720-89599
        # 上第二次修改，改4
        val_list = [4, 54, 73, 101, 102, 104, 115, 130, 136, 147]

        if self.same_v_t_split:
            val_list = test_list
        # 下改9：train_list对应序号1-1959
        #train_list = [i for i in range(len_dataset) if
                      #i not in test_list and i not in val_list]  # 除了test_list和val_list之外的其他example
        # 第二次修改，改5
        train_list=[i for i in range(62720)]   #train_list对应序号0-62719
        # 上改9：train_list对应序号1-1959
        #上第二次修改，改5
        if self.mode == "train":
            #第二次修改：改6
            #原来是89599.npz现在是89599_2799_14_2_28_10.npz。现在os.path.split(fn)[1].split(".")[0]是"89599_2799_14_2_28_10"。
            # 所以现在"89599_2799_14_2_28_10".split("_")[0]
            filenames_train = [fn for fn in filenames_all if int(os.path.split(fn)[1].split(".")[0].split("_")[0]) in train_list]
            ##上第二次修改：改6
            #下改10：train_size
            #train_size = 100 if self.same_v_t_split else 90
            #第二次修改7：1960*32
            train_size = 62720#1960
            #上第二次修改7：1960*32
            # 上改10：train_size
            print("train_size:", train_size)
            print("len(filenames_train):", len(filenames_train))
            assert len(filenames_train) == train_size, "Problem with size of kitti train dataset"
            filenames = filenames_train

        elif self.mode == "val":
            filenames_val = [fn for fn in filenames_all if int(os.path.split(fn)[1].split(".")[0].split("_")[0]) in val_list]
            #下改11：val_size
            #val_size = 50 if self.same_v_t_split else 10
            val_size = 26880#840*32
            #上改11：val_size

            assert len(filenames_val) == val_size, "Problem with size of kitti validation dataset"
            filenames = filenames_val

        elif self.mode == "test":
            filenames_test = [fn for fn in filenames_all if int(os.path.split(fn)[1].split(".")[0].split("_")[0]) in test_list]
            #下改12：len(filenames_test)
            #assert len(filenames_test) == 50, "Problem with size of kitti test dataset"
            assert len(filenames_test) == 26880, "Problem with size of kitti test dataset"
            #上改12：len(filenames_test)
            filenames = filenames_test

        elif self.mode == "all":
            ##下改13：len(filenames_all)
            #assert len(filenames_all) == 150, "Problem with size of kitti dataset"
            assert len(filenames_all) == 89600, "Problem with size of kitti dataset"
            filenames = filenames_all
            ##上改13：len(filenames_all)
        else:
            raise ValueError("Mode " + str(self.mode) + "unknown.")

        return filenames

    def load_sequence(self, idx):
        """
        Load a sequence of point clouds.

        Parameters
        ----------
        idx : int
            Index of the sequence to load.

        Returns
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size n x 3 and pc2 has size m x 3.

        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size n x 1 and pc1 has size n x 3.
            flow is the ground truth scene flow between pc1 and pc2. mask is
            binary with zeros indicating where the flow is not valid/occluded.

        """
        # 每个npz文件中包含3个数组，名字分别是['gt', 'pos2', 'pos1','idx_89600_sample']。
        # 利用这些数组，每个npz对应一个dict={"sequence":List [pc1, pc2] ,"ground_truth":List [mask, flow],"idx_89600_sample":str}。(# Restrict to 35m)
        # Load data
        self.filename_curr = self.filenames[idx]
        with np.load(self.filename_curr) as data:
            sequence = [data["pos1"][:, (1, 2, 0)], data["pos2"][:, (1, 2, 0)]]
            ground_truth = [
                np.ones_like(data["pos1"][:, 0:1]),
                data["gt"][:, (1, 2, 0)],
            ]
            #第二次修改，改14：新增键值对idx_89600_sample
            idx_89600_sample=data["idx_89600_sample"]#比如："89599_2799_14_2_28_10"
            #上第二次修改，改14：新增键值对idx_89600_sample
        #下改14：代码隐藏，取消限制35m(之前限制是相当于去除极端数据，异常)
        # # Restrict to 35m
        # loc = sequence[0][:, 2] < 35  # 用于筛选data['pos1']中第一列<35的点。
        # sequence[0] = sequence[0][loc]
        # ground_truth[0] = ground_truth[0][loc]
        # ground_truth[1] = ground_truth[1][loc]
        # loc = sequence[1][:, 2] < 35  ##用于筛选data['pos2']中第一列<35的点
        # sequence[1] = sequence[1][loc]
        # 上改14：代码隐藏，取消限制35m(之前限制是相当于去除极端数据，异常)
        return sequence, ground_truth,idx_89600_sample
