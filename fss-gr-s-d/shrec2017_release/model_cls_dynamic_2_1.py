"""
    Compared with model_baseline, do not use correlation output for skip link
    Compared to model_baseline_fixed, added return values to test whether nsample is set reasonably.
"""
#T-FSS-GR
#参考原先shrec2017_release/model_cls_dynamic.py（现改名为model_cls_dynamic_orig）
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/sampling'))
from net_utils import *
from tf_util import *

def placeholder_inputs(batch_size, num_point, num_frames, input_dim):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point * num_frames, input_dim))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    #修改1：修增占位为数据来源   对应SHRECLoader类对象getitem方法返回参数3：self.inputs_list[index]:字符串
    datasource_pl = tf.placeholder(tf.int32, shape=(batch_size,7))#batch_size中每条数据对应一条
    reconflow_pl = tf.placeholder(tf.float32, shape=(batch_size, num_frames, 48))#num_frames=32
    #上修改1
    return pointclouds_pl, labels_pl, datasource_pl,reconflow_pl

#改7：get_model新增输入参数 datasource,reconflow
def get_model(point_cloud, datasource,reconflow,num_frames, is_training, bn_decay=None, CLS_COUNT=-1):
    """ Input:
            point_cloud: [batch_size, num_point * num_frames, 3]
            datasource:[batch_size]  #新增输入。里面包含了batch中每条数据的数据来源及标签，比如其中一条{str}'1 1 2 1 1 1 77\n'
            reconflow:array[batch_size,32,16*3] #point_cloud对应的几个样本的场景流数据#新增输入：32*16正好对应函数内第二次调用dynamic_module是输入参数npoint=32 * 16
        Output:
            net: [batch_size, num_class] """
    end_points = {}
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value // num_frames

    l0_xyz = point_cloud[:, :, 0:3]
    l0_time = tf.concat([tf.ones([batch_size, num_point, 1]) * i for i in range(num_frames)], \
                        axis=-2)  # l0_time:Tensor,Shape(batch_size,num_point*num_frame,1)。含义（batch,frame/point,index_frame）
    l0_points = None  # l0_time

    RADIUS1 = np.linspace(0.5, 0.6, num_frames, dtype='float32')  # 0.5开始（第0个元素），0.6结束，共num_frames多个数。这些数均匀分布
    RADIUS2 = RADIUS1 * 2  # 半径扩大：每个数值*2

    print("**********************l0_xyz:", l0_xyz.shape)
    l1_xyz, l1_time, l1_points, l1_indices, flow1_list, w1 = dynamic_module(l0_xyz, l0_time, l0_points,
                                                                            npoint=32 * 32, radius=RADIUS1, nsample=64,
                                                                            mlp=[64, 128], mlp2=None,
                                                                            group_all=False, knn=False,
                                                                            is_training=is_training, bn_decay=bn_decay,
                                                                            scope='layer1', delta_t=2, flow_dim=63)
    print("**********************l1_xyz:", l1_xyz.shape)

    l2_xyz, l2_time, l2_points, l2_indices, flow2_list, w2 = dynamic_module(l1_xyz, l1_time, l1_points,
                                                                            npoint=32 * 16, radius=RADIUS2, nsample=64,
                                                                            mlp=[128, 256], mlp2=None,
                                                                            group_all=False, knn=False,
                                                                            is_training=is_training, bn_decay=bn_decay,
                                                                            scope='layer2', delta_t=2, flow_dim=33,
                                                                            last_W=w1)  # last_flow=flow1 , sample_idx=sample_idx

    print("**********************l2_xyz:", l2_xyz.shape)
    # pointnet_sa_module:PointNet Set Abstraction (SA) Module
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None,
                                                       mlp=[512, 1024], mlp2=None, group_all=True,
                                                       is_training=False, freeze_bn=True, bn_decay=bn_decay,
                                                       scope='layer4')
    # l4_xyz:(batch_size,1,3) 因为group_all=True，每个batch只有一个searchpoint,就是(0,0,0)
    # l4_points:shape(batch_size,1,1024) 每个batch中search point（0,0,0） 有1024个特征 （最初的l4_points是l2_xyz和l2_points相连，特征数3+256，通过两个tf_util.conv2d）
    # l4_indices:shape(batch_size,1,32*16)。把l2_xyz中所有search point作为新search point(0,0,0)的邻居。每个batch的一个search point对应nsample个邻居的索引是range(nsample)

    # Fully connected layers
    net = tf.reshape(l4_points, [batch_size, -1])  # shape(batch_size,1024)
    static_net = tf_util.fully_connected(net, CLS_COUNT, activation_fn=None, scope='fc3', is_training=False)
    # 本质static_net=net*w+b 。 (二维矩阵net)
    # static_net:shape(batch_size,CLS_COUNT)
    # cls_count：num_classes
    # net:shape(batch_size,1024) w:shape(1024,CLS_COUNT) b:shape[CLS_COUNT]

    flow1, flow2 = flow_merge_across_res(flow1_list, flow2_list, l1_indices, l2_indices,
                                         l1_time, l2_time, l1_xyz, l2_xyz,
                                         scope="agg_dynamic", is_training=is_training,
                                         bn_decay=bn_decay, bn=False)
    # flow2:(batch_size, 32*16, 43*(3+1))。

    # 改8：
    # 第一，将reconflow: [batch_size, 32, 16 * 3]reshape为[batch_size,32*16,3]
    #reconflow = reconflow.reshape(batch_size, -1, 3)
    reconflow_re=tf.reshape(reconflow,[batch_size, -1, 3])
    #reconflow [batch_size,32*16,3]
    #第二，reconflow和flow2经过conv1d,得到flow2_new，再融合l2_time得到flow2_new2 shape和原来最终的flow的shape一致
    flow2_channel=flow2.get_shape()[2].value#最后一个维度
    #flow2_channel2=flow2.get_shape()[1].value#32*16
    reconflow_re_new = tf_util.conv1d(reconflow_re,flow2_channel, 1, padding='VALID', stride=1,
                               bn=True, is_training=is_training,  # activation_fn=tf.nn.leaky_relu,
                               scope='reconflow_toflow2_channel_dynamic', bn_decay=bn_decay,
                               data_format='NHWC')#reconflow_re最后一个通道变得和flow2一样（NHWC）

    reconflow_re_new = tf.reshape(reconflow_re_new, [batch_size, 32,-1,flow2_channel])#4维
    flow2_new0 = tf.reshape(flow2, [batch_size, 32, -1, flow2_channel])  # 4维
    numpoint_frame=flow2_new0.get_shape()[2].value#16
    cat_rec_flow2=tf.concat([flow2_new0, reconflow_re_new], 2)#[bs,32,16+16,flow2_channel]
    cat_rec_flow2_trans=tf.transpose(cat_rec_flow2, [0, 2, 1, 3])#变为NCHW形式#[bs,16+16,32,flow2_channel]

    flow2_new=tf_util.conv2d(cat_rec_flow2_trans, numpoint_frame,[1, 1],
                   padding='VALID', stride=[1, 1],
                   bn=False, is_training=is_training,
                   scope='normal_add_recon_flow_dynamic', bn_decay=bn_decay,
                   data_format='NCHW')#变为NCHW形式#[bs,16,32,flow2_channel]
    flow2_new=tf.transpose(flow2_new, [0, 2, 1, 3])##[bs,32,16,flow2_channel]
    flow2_new=tf.reshape(flow2_new, [batch_size, -1, flow2_channel])#变回3维##[bs,32*16,flow2_channel]和flow2同型
    flow2_new2=tf_util.conv1d(tf.concat([flow2_new, l2_time], -1),
                   flow2_channel, 1, padding='VALID', stride=1,
                   bn=False, is_training=is_training,  # activation_fn=tf.nn.leaky_relu,
                   scope='flow2_new_addtime_dynamic' , bn_decay=bn_decay,
                   data_format='NHWC')

    #flow = flow2 #原flow
    flow = flow2_new2
    #上改8
    #改8的结果就是下面pointnet_sa_module输入参数flow变

    # 思路2.1改1 ：共6步
    # 总目标就是改ointnet_sa_module输入参数l2_xyz
    # 1/6.reconflow和l2_xyz:4d形式  [bs,32,16,3]
    reconflow_4d = tf.reshape(reconflow_re, [batch_size, 32, -1, 3])
    l2_xyz_4d = tf.reshape(l2_xyz, [batch_size, 32, -1, 3])

    # 2/6.连接reconflow_4d和l2_xyz，然后变为NCHW形式
    cat_l2xyz_recon = tf.concat([l2_xyz_4d, reconflow_4d], 2)  # [bs,32,16+16,3]
    cat_l2xyz_recon_trans = tf.transpose(cat_l2xyz_recon, [0, 2, 1, 3])  # 变为NCHW形式#[bs,16+16,32,3]

    # 3/6.tf_util.conv2d: 卷积：(1, 1), sride = (1, 1), outchannel = numpoint_frame = 16。 +BN+激活函数relu
    # [bs,16,32,3]
    # 含义：16个特征点，不是之前选出的16个点。最后的3是融合了坐标和场景流信息
    l2_xyz_new = tf_util.conv2d(cat_l2xyz_recon_trans, numpoint_frame, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=True, is_training=is_training,
                                scope='xyz_add_recon_flow_dynamic', bn_decay=bn_decay,
                                data_format='NCHW')  # 变为NCHW形式#[bs,16,32,flow2_channel]
    l2_xyz_new = tf.transpose(l2_xyz_new, [0, 2, 1, 3])  ##[bs,32,16,3]

    # 4/6. 变回3维[bs,32*16,3]和l2_xyz同型
    l2_xyz_new = tf.reshape(l2_xyz_new, [batch_size, -1, 3])

    # 5/6. tf_util.conv1d
    # 卷积(1,1),stride=(1,1)：输入l2_xyz_new的特征加上一个时间特征[bs,32*16,3+1],输出特征数3
    # BN,有激活函数relu
    l2_xyz_new2 = tf_util.conv1d(tf.concat([l2_xyz_new, l2_time], -1),
                                 3, 1, padding='VALID', stride=1,
                                 bn=True, is_training=is_training,  # activation_fn=tf.nn.leaky_relu,
                                 scope='l2xyz_new_addtime_dynamic', bn_decay=bn_decay,
                                 data_format='NHWC')
    # 6/6.l2_xyz_new2送入SA模块
    l2_xyz = l2_xyz_new2

    # 上思路2.1改1 ：总目标就是改ointnet_sa_module输入参数l2_xyz

    flow_xyz, flow, flow_indices = pointnet_sa_module(l2_xyz, flow, npoint=None, radius=None, nsample=None,
                                                      mlp=[512, 1024],
                                                      mlp2=None, group_all=True, is_training=is_training,
                                                      bn=True, bn_decay=bn_decay, scope='layer4_dynamic')
    # flow和l4_points处理过程和输出shape一样，不同的是最开始的值。
    # 最开始的flow:l2_xyz和输入参数flow相连，特征数3+flow的特征数，然后通过两个tf_util.conv2d，输出特征数1024
    # flow_xyz同l4_xyz，flow_indices同l4_indices

    # Fully connected layers
    flow_net = tf.reshape(flow, [batch_size, -1])  # shape(batch_size,1024)
    flow_net = tf_util.fully_connected(flow_net, CLS_COUNT, activation_fn=None, scope='fc3_dynamic')
    # 本质flow_net=flow_net*w+b 。 (二维矩阵flow_net)
    # flow_net:shape(batch_size,CLS_COUNT)
    return static_net, flow_net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024 * 2, 6))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
