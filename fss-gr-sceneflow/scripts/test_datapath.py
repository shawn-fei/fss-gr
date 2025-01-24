import os
import sys

# add path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)

if __name__ == "__main__":
    dataset_name="FlowNet3D_kitti"
    # Path to current file
    pathroot = os.path.dirname(__file__)
    print("pathroot:",pathroot)
    path2data = os.path.join(pathroot, "..", "data", "FlowNet3D")
    print("path2data:", path2data)
    path2data = os.path.join(path2data, "kitti_rm_ground")
    print("path2data:", path2data)
    print("finished")
    '''
    from datasets.kitti_flownet3d import Kitti

    # datasets
    nb_points=2048
    same_val_test_kitti=1
    train_dataset = Kitti(root_dir=path2data, nb_points=nb_points, all_points=False,
                          same_v_t_split=same_val_test_kitti, mode="train")
    val_dataset = Kitti(root_dir=path2data, nb_points=nb_points, all_points=False,
                        same_v_t_split=same_val_test_kitti, mode="val")
    '''
