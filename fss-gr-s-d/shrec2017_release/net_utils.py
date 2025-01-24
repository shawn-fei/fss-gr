""" PointNet++ Layers

Author: Charles R. Qi
Modified by Xingyu Liu
Date: November 2019

Modified by Jiaxing Zhong
Date: November 2021
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
sys.path.append(os.path.join(ROOT_DIR))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, query_ball_point_var_rad, query_ball_point_var_rad_var_seed, group_point, knn_point, select_top_k
from tf_interpolate import three_nn, three_interpolate
# import tensorflow._api.v2.compat.v1 as tf
# # tf.disable_v2_behavior()
import tensorflow as tf
import numpy as np
import tf_util

def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    sample_idx = farthest_point_sample(npoint, xyz)
    new_xyz = gather_point(xyz, sample_idx) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, sample_idx, grouped_xyz

def sample_and_group_by_index(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True, sample_idx=None):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    if sample_idx is None:
        sample_idx = farthest_point_sample(npoint, xyz)
    new_xyz = gather_point(xyz, sample_idx) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, sample_idx, grouped_xyz

def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)#含义每个batch只有一个search point,就是(0,0,0)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))#(batch_size,1,nsample)#每个batch的一个search point对应nsample个邻居的索引是range(nsample)。其实就是把xyz中所有search point作为新search point(0,0,0)的邻居
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)#xyz的3+points的256。其实就是把坐标和256个通道信息拼接在一起。
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False, freeze_bn=False, sample_idx=None, trainable=True):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor  若group_all=True:npoint=1 (每个batch的search point只有一个（0,0,0）)
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions  若group_all=True:nsample=ndataset
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:#就是每个batch只有一个search point(0,0,0)，把所有点分到一组。
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
            #new_xyz:(batch_size, 1, 3) 每个batch只有一个search point,就是(0,0,0)。npoint=1
            #idx： shape(batch_size,1,nsample)。nsample是xyz第二个维度的size，比如32*16.#每个batch的一个search point对应nsample个邻居的索引是range(nsample)。把xyz中所有search point作为新search point(0,0,0)的邻居。
            #grouped_xyz(batch_size, npoint=1, nsample, 3) ：xyz-reshpe而来。
            #new_points: # (batch_size, 1, nsample, 259)#xyz的3+points的256。其实就是把坐标和256个通道信息拼接在一起。
        elif sample_idx is not None:
            new_xyz, new_points, idx, sample_idx, grouped_xyz = sample_and_group_by_index(npoint, radius, nsample, xyz, points, knn, use_xyz, sample_idx)
        else:
            new_xyz, new_points, idx, sample_idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2]) ## group_all:npoint=1,new_points:shape(batch_size ,259,1, nsample)
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training, trainable=trainable,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format, freeze_bn=freeze_bn)
            # data_format = 'NCHW' ，因为kernel_size[1,1], padding='VALID', stride=[1,1]
            # 所以，shape中改变的只有C。
            #new_points:shape(batch_size,num_out_channel,1, nsample)
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])
        #new_points:shape(batch_size,npoint, nsample,mlp[-1])
        #比如mlp=[512, 1024],shape(batch_size,npoint, nsample,1024)

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keepdims=True, name='maxpool')
            #shape(batch_size,npoint, 1,mlp[-1])
            #比如mlp=[512, 1024],group_all=True,shape(batch_size,1, 1,1024)
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keepdims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keepdims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keepdims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keepdims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keepdims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keepdims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training, trainable=trainable,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format, freeze_bn=freeze_bn)
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        ##比如mlp=[512, 1024],group_all=True,shape(batch_size,1, 1024)
        if sample_idx is not None:
            return new_xyz, new_points, (idx, sample_idx)
        else:
            return new_xyz, new_points, idx


#新增函数changed_pointnet_sa_module（）：points经过多层tf_util.conv2d后，maxpool,最终return new_points
def changed_pointnet_sa_module(points, npoint, mlp, mlp2,mlp3, is_training, bn_decay, scope, bn=True, pooling='max',
                               knn=False, use_xyz=True, use_nchw=False, freeze_bn=False, trainable=True):
    ''' changed PointNet Set Abstraction (SA) Module:参考pointnet_sa_module写
        Input:

            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling

            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region

            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'  # NHWC
    with tf.variable_scope(scope) as sc:
        new_points = tf.expand_dims(points, 1)  # (batch_size, 1, 32*16, 3)#points:(batch_size,32*16, 3)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training, trainable=trainable,
                                        scope='conv%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format, freeze_bn=freeze_bn)
            # data_format = 'NHWC' ，因为kernel_size[1,1], padding='VALID', stride=[1,1]
            # 所以，shape中改变的只有C。
            # new_points:shape(batch_size,1,32*16,mlp[-1])
        if use_nchw: new_points = tf.transpose(new_points, [0, 2, 3, 1])

        # Pooling in Local Regions
        if pooling == 'max':
            new_points = tf.reduce_max(new_points, axis=[2], keepdims=True, name='maxpool')
            # shape(batch_size,1, 1,mlp[-1])
            # 比如mlp=[512, 1024],group_all=True,shape(batch_size,1, 1,1024)
        elif pooling == 'avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keepdims=True, name='avgpool')
        elif pooling == 'weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz, axis=-1, ord=2, keepdims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists / tf.reduce_sum(exp_dists, axis=2,
                                                    keepdims=True)  # (batch_size, npoint, nsample, 1)
                new_points *= weights  # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keepdims=True)
        elif pooling == 'max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keepdims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keepdims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1, 1],
                                            bn=bn, is_training=is_training, trainable=trainable,
                                            scope='conv_post_%d' % (i), bn_decay=bn_decay,
                                            data_format=data_format, freeze_bn=freeze_bn)
            if use_nchw: new_points = tf.transpose(new_points, [0, 2, 3, 1])

        #新增mlp3
        if mlp3 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])
            for i, num_out_channel in enumerate(mlp3):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1, 1],
                                            bn=bn, is_training=is_training, trainable=trainable,
                                            scope='conv_post2_%d' % (i), bn_decay=bn_decay,
                                            data_format=data_format, freeze_bn=freeze_bn)
            if use_nchw: new_points = tf.transpose(new_points, [0, 2, 3, 1])

        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
        ##比如mlp=[512, 1024],group_all=True,shape(batch_size,1, 1024)

        return new_points

def static_module(xyz, time, points, npoint, radius, nsample, mlp, mlp2, group_all,
                  is_training, bn_decay, scope, module_type='ind_without_time', fps=True, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    '''
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            time: (batch_size, ndataset, 1) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radiuses in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            module_type: 'ind' or 'rel' -- the type of meteor module
            fps: whether to do farthest point sampling; Requires npoint == xyz.get_shape()[1].value, when fps=False
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    assert(module_type is 'ind_without_time') # The time is only used for sampling points.
    data_format = 'NCHW' if use_nchw else 'NHWC'
    sample_idx = None
    batch_size = xyz.get_shape()[0].value
    with tf.variable_scope(scope) as sc:

        if fps:
            ##### sample and group with variable radius
            sample_idx = farthest_point_sample(npoint, xyz)
        else:
            ##### no sampling at all
            sample_idx = tf.tile(tf.expand_dims(tf.range(npoint, dtype=tf.int32), 0), [batch_size, 1])

        new_xyz = gather_point(xyz, sample_idx)  # (batch_size, npoint, 3)
        new_time = gather_point(time, sample_idx)  # (batch_size, npoint, 1)
        time_ = tf.reshape(time, [batch_size, 1, -1])  # (batch_size, 1, ndataset)
        new_time_ = tf.abs(new_time - time_)  # (batch_size, npoint, ndataset)
        radius_ = tf.gather(radius, tf.cast(new_time_, tf.int32))  # (batch_size, npoint, ndataset)
        idx, pts_cnt = query_ball_point_var_rad(radius_, nsample, xyz, new_xyz)

        grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization        
        FRAME_CNT = 32
        grouped_time = group_point(time, idx)  # (batch_size, npoint, nsample * 3, channel)
        grouped_time -= tf.tile(tf.expand_dims(new_time, 2), [1, 1, nsample, 1])  # time-shift normalization
        grouped_time += FRAME_CNT

        if points is not None:
            new_points = gather_point(points, sample_idx)  # (batch_size, npoint, channel)
            grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
            grouped_time = group_point(time, idx) # (batch_size, npoint, nsample, channel)
            if use_xyz:
                if module_type == 'ind':
                    new_points = tf.concat([grouped_xyz, grouped_time, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+1+channel)
                elif module_type == 'ind_without_time':
                    new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+1+channel)
                else:
                    new_points_expand = tf.tile(tf.expand_dims(new_points, 2), [1,1,nsample,1])
                    new_points = tf.concat([grouped_xyz, grouped_time, grouped_points, new_points_expand], axis=-1) # (batch_size, npoint, nample, 3+1+channel+channel)
            else:
                new_points = grouped_points
        else:
            new_points = grouped_xyz

        # Point Feature Embedding
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)
                                        

        new_points = tf.reduce_max(new_points, axis=[2], name='maxpool')
        return new_xyz, new_time, new_points, idx


def get_4D_inversion(AtA, group_size, batch_size, npoint, flow_dim, EPS):
    #返回逆矩阵inv_AtA, shape[batch_size, npoint, flow_dim // group_size, group_size + 1, group_size + 1]
    assert(group_size + 1 == 4)
    AtA = tf.reshape(AtA, [batch_size, npoint, flow_dim // group_size, -1]) #最后一个维度的shape是（group_size+1）*（group_size+1），就是16
    AtA0, AtA1, AtA2, AtA3, \
    AtA4, AtA5, AtA6, AtA7, \
    AtA8, AtA9, AtA10, AtA11, \
    AtA12, AtA13, AtA14, AtA15 = tf.unstack(AtA, axis=3)
    #AtAi：shape(batch_size, npoint, flow_dim // group_size)。
    #含义： 就是每个search point(共npoint) 11个组在第i/16个通道上的值。其中11是flow_dim // group_size

    inv0 = AtA5 * AtA10 * AtA15 - \
           AtA5 * AtA11 * AtA14 - \
           AtA9 * AtA6 * AtA15 + \
           AtA9 * AtA7 * AtA14 + \
           AtA13 * AtA6 * AtA11 - \
           AtA13 * AtA7 * AtA10
    inv4 = -AtA4 * AtA10 * AtA15 + \
           AtA4 * AtA11 * AtA14 + \
           AtA8 * AtA6 * AtA15 - \
           AtA8 * AtA7 * AtA14 - \
           AtA12 * AtA6 * AtA11 + \
           AtA12 * AtA7 * AtA10
    inv8 = AtA4 * AtA9 * AtA15 - \
           AtA4 * AtA11 * AtA13 - \
           AtA8 * AtA5 * AtA15 + \
           AtA8 * AtA7 * AtA13 + \
           AtA12 * AtA5 * AtA11 - \
           AtA12 * AtA7 * AtA9
    inv12 = -AtA4 * AtA9 * AtA14 + \
            AtA4 * AtA10 * AtA13 + \
            AtA8 * AtA5 * AtA14 - \
            AtA8 * AtA6 * AtA13 - \
            AtA12 * AtA5 * AtA10 + \
            AtA12 * AtA6 * AtA9
    inv1 = -AtA1 * AtA10 * AtA15 + \
           AtA1 * AtA11 * AtA14 + \
           AtA9 * AtA2 * AtA15 - \
           AtA9 * AtA3 * AtA14 - \
           AtA13 * AtA2 * AtA11 + \
           AtA13 * AtA3 * AtA10
    inv5 = AtA0 * AtA10 * AtA15 - \
           AtA0 * AtA11 * AtA14 - \
           AtA8 * AtA2 * AtA15 + \
           AtA8 * AtA3 * AtA14 + \
           AtA12 * AtA2 * AtA11 - \
           AtA12 * AtA3 * AtA10
    inv9 = -AtA0 * AtA9 * AtA15 + \
           AtA0 * AtA11 * AtA13 + \
           AtA8 * AtA1 * AtA15 - \
           AtA8 * AtA3 * AtA13 - \
           AtA12 * AtA1 * AtA11 + \
           AtA12 * AtA3 * AtA9
    inv13 = AtA0 * AtA9 * AtA14 - \
            AtA0 * AtA10 * AtA13 - \
            AtA8 * AtA1 * AtA14 + \
            AtA8 * AtA2 * AtA13 + \
            AtA12 * AtA1 * AtA10 - \
            AtA12 * AtA2 * AtA9
    inv2 = AtA1 * AtA6 * AtA15 - \
           AtA1 * AtA7 * AtA14 - \
           AtA5 * AtA2 * AtA15 + \
           AtA5 * AtA3 * AtA14 + \
           AtA13 * AtA2 * AtA7 - \
           AtA13 * AtA3 * AtA6
    inv6 = -AtA0 * AtA6 * AtA15 + \
           AtA0 * AtA7 * AtA14 + \
           AtA4 * AtA2 * AtA15 - \
           AtA4 * AtA3 * AtA14 - \
           AtA12 * AtA2 * AtA7 + \
           AtA12 * AtA3 * AtA6
    inv10 = AtA0 * AtA5 * AtA15 - \
            AtA0 * AtA7 * AtA13 - \
            AtA4 * AtA1 * AtA15 + \
            AtA4 * AtA3 * AtA13 + \
            AtA12 * AtA1 * AtA7 - \
            AtA12 * AtA3 * AtA5
    inv14 = -AtA0 * AtA5 * AtA14 + \
            AtA0 * AtA6 * AtA13 + \
            AtA4 * AtA1 * AtA14 - \
            AtA4 * AtA2 * AtA13 - \
            AtA12 * AtA1 * AtA6 + \
            AtA12 * AtA2 * AtA5
    inv3 = -AtA1 * AtA6 * AtA11 + \
           AtA1 * AtA7 * AtA10 + \
           AtA5 * AtA2 * AtA11 - \
           AtA5 * AtA3 * AtA10 - \
           AtA9 * AtA2 * AtA7 + \
           AtA9 * AtA3 * AtA6
    inv7 = AtA0 * AtA6 * AtA11 - \
           AtA0 * AtA7 * AtA10 - \
           AtA4 * AtA2 * AtA11 + \
           AtA4 * AtA3 * AtA10 + \
           AtA8 * AtA2 * AtA7 - \
           AtA8 * AtA3 * AtA6
    inv11 = -AtA0 * AtA5 * AtA11 + \
            AtA0 * AtA7 * AtA9 + \
            AtA4 * AtA1 * AtA11 - \
            AtA4 * AtA3 * AtA9 - \
            AtA8 * AtA1 * AtA7 + \
            AtA8 * AtA3 * AtA5
    inv15 = AtA0 * AtA5 * AtA10 - \
            AtA0 * AtA6 * AtA9 - \
            AtA4 * AtA1 * AtA10 + \
            AtA4 * AtA2 * AtA9 + \
            AtA8 * AtA1 * AtA6 - \
            AtA8 * AtA2 * AtA5
    D = AtA0 * inv0 + AtA1 * inv4 \
        + AtA2 * inv8 + AtA3 * inv12
    D = tf.where(tf.abs(D) < EPS, EPS * tf.ones_like(D), D)
    D = tf.expand_dims(D, -1)
    inv0 = tf.expand_dims(inv0, -1)
    inv1 = tf.expand_dims(inv1, -1)
    inv2 = tf.expand_dims(inv2, -1)
    inv3 = tf.expand_dims(inv3, -1)
    inv4 = tf.expand_dims(inv4, -1)
    inv5 = tf.expand_dims(inv5, -1)
    inv6 = tf.expand_dims(inv6, -1)
    inv7 = tf.expand_dims(inv7, -1)
    inv8 = tf.expand_dims(inv8, -1)
    inv9 = tf.expand_dims(inv9, -1)
    inv10 = tf.expand_dims(inv10, -1)
    inv11 = tf.expand_dims(inv11, -1)
    inv12 = tf.expand_dims(inv12, -1)
    inv13 = tf.expand_dims(inv13, -1)
    inv14 = tf.expand_dims(inv14, -1)
    inv15 = tf.expand_dims(inv15, -1)
    inv_AtA = tf.concat([inv0, inv1, inv2, inv3,
                         inv4, inv5, inv6, inv7,
                         inv8, inv9, inv10, inv11,
                         inv12, inv13, inv14, inv15], -1)
    inv_AtA = inv_AtA / D  #逆矩阵
    return tf.reshape(inv_AtA, [batch_size, npoint, flow_dim // group_size, group_size + 1, group_size + 1])


def get_st_surface(GROUP_SIZE, batch_size, duplicate_grouped_time, flow_dim, new_points_, npoint, nsample, W=None):
    #返回n_new_points_：根据F(new_points_)，时间t(duplicate_grouped_time),权重矩阵W得到时空曲面st_surface相关参数A,b
    #整个过程对应论文中公式（6）
    #shape(batch_size,npoint,组数，4,1)
    EPS = 0.000001
    new_points_ = tf.concat([new_points_, tf.ones(shape=[batch_size, npoint, flow_dim // GROUP_SIZE, nsample, 1], dtype=new_points_.dtype)], -1)
    #new_points_:the feature matrix .search point P(i,t)第k个分组的特征,对应shape是(nsample,GROUP_SIZE+1)。
    # nsample是每个Search point邻居点个数。GROUP_SIZE就是论文中的d。GROUP_SIZE+1中的1的那一列值是1.
    if W is None:#layer1,i=0
        AtA = tf.matmul(new_points_, new_points_, transpose_a=True)#矩阵相乘：new_points_的转置乘new_points_ F转置*F
        inv_AtA = get_4D_inversion(AtA, GROUP_SIZE, batch_size, npoint, flow_dim, EPS) #AtA的逆矩阵
        n_new_points_ = tf.matmul(new_points_, duplicate_grouped_time, transpose_a=True)
        n_new_points_ = tf.matmul(inv_AtA, n_new_points_) ##公式(6) shape(batch_size,npoint,组数，4,1)
    else:#layer1,i=1
        AtA = W * new_points_#W*F
        AtA = tf.matmul(new_points_, AtA, transpose_a=True)#F转置*（W*F）
        inv_AtA = get_4D_inversion(AtA, GROUP_SIZE, batch_size, npoint, flow_dim, EPS)#上行的逆矩阵
        n_new_points_ = W * duplicate_grouped_time
        n_new_points_ = tf.matmul(new_points_, n_new_points_, transpose_a=True)#F转置*(W*时间)
        n_new_points_ = tf.matmul(inv_AtA, n_new_points_)#公式(6) shape(batch_size,npoint,组数，4,1)
    return n_new_points_
    
def dynamic_module(xyz, time, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope,
                   module_type='ind_without_time', fps=True, bn=True, pooling='max', knn=False, use_xyz=True,
                   use_nchw=False, delta_t=1, flow_dim=2, last_flow=None, last_W=None, sample_idx=None):
    data_format = 'NCHW' if use_nchw else 'NHWC'
    batch_size = xyz.get_shape()[0].value
    with tf.variable_scope(scope) as sc:
        if sample_idx is None:
            if fps:#最远点采样
                sample_idx = farthest_point_sample(npoint, xyz)
                #从xyz（b_s,num_point*num_frame,3）中选出npoint（32*32）个样本点
                # 返回这些点索引sample_idx: shape（b_s，npoint）
            else:
                sample_idx = tf.tile(tf.expand_dims(tf.range(npoint, dtype=tf.int32), 0), [batch_size, 1])

        radius[delta_t:] = 0  #radius本是array，32个元素，从下标为2的元素开始全部变为0。
        #函数dynamic_module返回参数1,2：new_xyz,new_time
        new_xyz = gather_point(xyz, sample_idx)  # (batch_size, npoint, 3)  比如：npoint（32*32）就是32*32个采样点的3维坐标
        new_time = gather_point(time, sample_idx)  # (batch_size, npoint, 1)  比如：一个batch,npoint（32*32）就是一列：值是32*32个点对应的帧是第几帧.
        time_ = tf.reshape(time, [batch_size, 1, -1])  # (batch_size, 1, ndataset) #比如：一个batch,ndataset（32帧*128点）就是一行：128个0（帧time索引）,128个1,128个2，……，128个31.
        new_time_ = tf.abs(time_ - new_time)  # (batch_size, npoint, ndataset) #比如：shape(batch_size,（32*32），(32*128))。
        # 感性理解这个new_time_记录的是：对于batch中每个样本（1个动态手势），采样出了32*32个点，这些中心点和点云序列中所有点（32*128）的时间差
        radius_ = tf.gather(radius, tf.cast(new_time_, tf.int32))  # (batch_size, npoint, ndataset)
                  # #cast:将new_time_中每个数值转化为tf.int32类型
                 #radius_:shape同tf.cast的结果。radius_[i][j][k]=radius[m][j][k],其中m=new_t[i][j][k],new_t是tf.cast的结果

        #函数dynamic_module返回参数4：idx
        idx, pts_cnt = query_ball_point_var_rad(radius_, nsample, xyz, new_xyz) #以new_xyz中npoint个点为查找点，在原始xyz的ndataset中找nsample个点。查找范围是球形搜索区域（查找点为球心，radius为半径）
        #idx: (batch_size, npoint, nsample) int32 array, indices to input points(xyz:32*128中索引)
        grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)  比如(batch_size,32*32,64,3)
        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
        # tf.expand_dims(new_xyz, 2):new_xyz的shape从(batch_size, npoint, 3)到 (batch_size, npoint, 1,3)
        # tf.tile(*, [1, 1, nsample, 1]):shape从(batch_size, npoint, 1,3)变为(batch_size, npoint, 1*nsample,3) 比如(batch_size,32*32,1*64,3)
        #grouped_xyz：(batch_size, npoint, nsample,3) 比如grouped_xyz：(batch_size,32*32,64,3)。
        # 含义针对每个search point(共npoint)，nsample(64)个邻居和该search_point的3维坐标差。
        grouped_time = group_point(time, idx)
        # 我觉得是 (batch_size, npoint, nsample,1)这是根据函数定义中注释得出
        # 原注释是(batch_size, npoint, nsample * 3, channel)

        '''
        FRAME_CNT = 32
        grouped_time -= tf.tile(tf.expand_dims(new_time, 2), [1, 1, nsample, 1])  # time-shift normalization
        grouped_time += FRAME_CNT
        '''
        if points is not None:  #layer1中points=l0_points是None.layer2中points是l1_points，和layer1输出有关
            new_points = gather_point(points, sample_idx)  # (batch_size, npoint, channel)
            grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample * 3, channel)
            if use_xyz:#True
                if module_type == 'ind_without_time':#走这里
                    new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)
                elif module_type == 'ind':
                    new_points = tf.concat([grouped_xyz, grouped_time, grouped_points],
                                           axis=-1)  # (batch_size, npoint, nample * 3, 3+1+channel)
                else:
                    points_expand = tf.tile(tf.expand_dims(new_points, 2), [1, 1, nsample, 1])
                    new_points = tf.concat([grouped_xyz, grouped_time, grouped_points, points_expand],
                                           axis=-1)  # (batch_size, npoint, nsample * 3, 3+1+channel+channel)
            else:
                new_points = grouped_points
        else:
            new_points = grouped_xyz
        # '''
        flow_list = []
        # Point Feature Embedding
        for i, num_out_channel in enumerate(mlp):  #layer1：mlp=[64, 128]。layer2:mlp=[128, 256],
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=False, trainable=False, freeze_bn=True,  # activation_fn=tf.nn.swish,
                                        scope='conv%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format)
            #new_points(NHWC)经过二维卷积、加bias,BN,RELU得到输出
            # 输出的shape和num_out_channel、padding，kenel,stride有关，由于函数赋值，kernel=[1,1],padding='VALID', stride=[1, 1]。(NHWC)中HW的值不变
            #layer1，i=0：scope:layer1/conv0,new_points的shape:(batch_size,npoints,nsample,num_out_channel)即（batch_size,32*32,64,64）
            #layer1，i=1：scope:layer1/conv1,new_points的shape:即（batch_size,32*32,64,128）

            GROUP_SIZE = 3
            flow_dim = num_out_channel // 2 // GROUP_SIZE * GROUP_SIZE + GROUP_SIZE  #num_out_channel=64,flow_dim=33。（128->66,256->129）
            new_points_ = tf_util.conv2d(tf.concat([new_points, grouped_time], -1),
                                         flow_dim - 3, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=False, is_training=is_training, #activation_fn=tf.nn.leaky_relu,
                                         scope='conv%d_dynamic' % i, bn_decay=bn_decay,
                                         data_format=data_format)
            #shape:(batch_size,npoints,nsample,flow_dim - 3)
            #layer1，i=0：scope:layer1/conv0_dynamic,new_points的shape:（batch_size,32*32,64,30）
            #layer1，i=1：scope:layer1/conv1_dynamic,new_points的shape:即（batch_size,32*32,64,63）

            new_points_ = tf.concat([new_points_, grouped_xyz], -1)#shape:(batch_size,npoints,nsample,flow_dim)#layer1，i=0/1：(batch_size,32*32,64,33/66）
            new_points_ = tf.Print(new_points_, [new_points_.shape], "new_points_: ", summarize=2333, first_n=1)
            #tf.Print的第一个参数是tf.Print的返回值，该代码将返回值赋值给new_points_。但是这个语句本身主要功能是打印输出，并没有对变量有影响
            new_points_ = tf.reshape(new_points_, [batch_size, npoint, nsample, flow_dim // GROUP_SIZE, GROUP_SIZE])
            #特征分组：#shape:(batch_size,npoints,nsample,组数，每组特征数)
            # layer1，i=0/1： (batch_size,32*32,64,11/22,3)

            duplicate_grouped_time = tf.tile(grouped_time, [1, 1, 1, flow_dim // GROUP_SIZE])#shape:(batch_size, npoint, nsample,flow_dim // GROUP_SIZE) flow_dim // GROUP_SIZE是组数（就是把flow分为几组）
            #duplicate_grouped_time:grouped_time(batch_size, npoint, nsample,1)最后一个维度（邻居点对应时间）复制flow_dim // GROUP_SIZE次。
            duplicate_grouped_time = tf.reshape(duplicate_grouped_time, [batch_size, npoint, nsample, flow_dim // GROUP_SIZE, 1])
            #shape:(batch_size, npoint, nsample,flow_dim // GROUP_SIZE,1) 第4个维度：比如每个查找点的邻居对应的时间复制11次
            new_points_ = tf.transpose(new_points_, [0, 1, 3, 2, 4])##转置：shape:(batch_size,npoints,特征组数，nsample，每组包含特征数GROUP_SIZE)
            duplicate_grouped_time = tf.transpose(duplicate_grouped_time, [0, 1, 3, 2, 4])#(batch_size, npoint, flow_dim // GROUP_SIZE,nsample,1)
            #new_points_:每个search point(共npoints)的nasmple个邻居的33(flow_dim)个特征，然后将这33个特征分为11组，每组3(GROUP_SIZE)个特征。最后就是（nsample个邻居的3个特征  \n nsample个邻居的另3个特征 …… ）
            #duplicate_grouped_time:每个search point对应nsample个邻居的时间[t_n1,t_n2,t_n3,……,t_n64]，然后沿着dim=0复制11次[t_n1,t_n2,t_n3,……,t_n64]。


            if i == 0 and last_W is None:    
                n_new_points_ = get_st_surface(GROUP_SIZE, batch_size, duplicate_grouped_time, flow_dim, new_points_, npoint, nsample)
                # 返回n_new_points_：根据F(new_points_)，时间t(duplicate_grouped_time),权重矩阵W得到时空曲面st_surface相关参数A,b
                # 整个过程对应论文中公式（6）
                #shape(batch_size,npoint,flow_dim // group_size,GROUP_SIZE+1,1) 比如（batch_szie,32*32,11,,4,1）
            else:
                if i == 0:#last_W 非空。
                    # #layer1,i=0
                    last_W_shape = last_W.shape
                    W = last_W[:,:,0,:] #tf.reduce_max(last_W, 2)
                    W = tf.reshape(W, [batch_size, last_W_shape[0] // batch_size, -1])
                    W = group_point(W, idx)
                    W = tf.reshape(W, [batch_size * npoint, nsample, last_W_shape[1], last_W_shape[3]])
                    W = tf.transpose(W, [0, 2, 1, 3])
                #i==1 或者   i==0且last_W 非空
                with tf.variable_scope("pointwise_attention_dynamic", reuse=tf.AUTO_REUSE):
                    W = tf_util.conv2d(W, flow_dim // GROUP_SIZE, [1, 1],
                                   padding='VALID', stride=[1, 1],
                                   bn=False, is_training=is_training, data_format="NCHW",
                                   scope='W_%d_dynamic' % i, bn_decay=bn_decay) #scope:layer1/pointwise_attention_dynamic/W_1_dynamic
                    #data_format="NCHW",输出W(batch_size*npoint,flow_dim // GROUP_SIZE,nsample,GROUP_SIZE+1)  flow_dim :33/66/129
                    W = tf_util.conv2d(W, GROUP_SIZE * 2, [1, 1],
                                       padding='VALID', stride=[1, 1],
                                       bn=False, is_training=is_training, data_format="NHWC", 
                                       scope='W1_%d_dynamic' % i, bn_decay=bn_decay)#scope:layer1/pointwise_attention_dynamic/W1_1_dynamic
                    #data_format="NHWC",输出W(batch_size*npoint,flow_dim // GROUP_SIZE,nsample,GROUP_SIZE * 2)  flow_dim :33/66/129
                    W = tf_util.conv2d(W, 1, [1, 1],
                                       padding='VALID', stride=[1, 1], data_format="NHWC",
                                       bn=False, is_training=is_training, activation_fn=None,
                                       scope='W2_%d_dynamic' % i, bn_decay=bn_decay)#scope:layer1/pointwise_attention_dynamic/W2_1_dynamic
                    # data_format="NHWC",输出W(batch_size*npoint,flow_dim // GROUP_SIZE,nsample,1)  flow_dim :33/66/129
                W = tf.sigmoid(W) #压缩到(0,1)
                W = tf.reshape(W, [batch_size, npoint, flow_dim // GROUP_SIZE, nsample, 1])
                n_new_points_ = get_st_surface(GROUP_SIZE, batch_size, duplicate_grouped_time, flow_dim, new_points_, npoint, nsample, W)
                #shape(batch_size,npoint,组数，4,1)
                # layer1,i=1:(batch_size,32*32,22，4,1)
            #以下是flow_list相关，normals_feat是list中元素
            #normals_feat:看论文公式(7),normal vector
            normals_feat = tf.reshape(n_new_points_, [batch_size, npoint, flow_dim // GROUP_SIZE, GROUP_SIZE + 1])#比如(batch_size,32*32,11,4)
            normals_feat = normals_feat[:, :, :, 0:GROUP_SIZE]#公式（7）中A_star.比如(batch_size,32*32,11,3)
            normals_feat = tf.concat([normals_feat, -tf.ones(shape=[batch_size, npoint, flow_dim // GROUP_SIZE, 1], dtype=normals_feat.dtype)], -1)##比如(batch_size,32*32,11,3+1),3+1中1那列都是-1
            normals_feat = tf.nn.l2_normalize(normals_feat, -1)  #normal vector:对应论文公式（7）。shape(batch_size,32*32,11，3+1)
            normals_feat = tf.reshape(normals_feat, [batch_size, npoint, -1])
            #比如shape(batch_size,32*32,11*(3+1))  11*(3+1)是组数*（GROUP_SIZE+1）  GROUP_SIZE+1中1那列都是-1
            #含义：32*32(npoint)个search point 44个特征。

            #函数dynamic_module返回参数5：flow_list
            flow_list.append(normals_feat)

            #以下是W相关：
            new_points_normals = tf.reshape(n_new_points_, [batch_size, npoint, flow_dim // GROUP_SIZE, 1, GROUP_SIZE + 1]) * tf.concat([new_points_, tf.ones(shape=[batch_size, npoint, flow_dim // GROUP_SIZE, nsample, 1], dtype=new_points_.dtype)], -1)
            #[A,b]*[f,1]=Af+b.
            #shape:(batch_size, npoint, flow_dim // GROUP_SIZE, nsample, GROUP_SIZE + 1)
            #比如，layer1,i=0,(N,32*32,11,64,4)
            new_points_normals = tf.reduce_sum(new_points_normals, -1, keep_dims=True) - duplicate_grouped_time
            #公式（5）相减那部分，参考公式（4）
            #Af+b-t
            #shape(batch_size, npoint, flow_dim // GROUP_SIZE, nsample, 1)
            #比如，layer1,i=0,(N,32*32,11,64,1)
            new_points_normals = tf.reshape(new_points_normals, [batch_size, npoint, flow_dim // GROUP_SIZE, nsample])
            ##Af+b-t
            #new_points_normals.shape:[batch_size, npoint, flow_dim // GROUP_SIZE, nsample]
            ##比如，layer1,i=0,(N,32*32,11,64)

            n_new_points_ = tf.reshape(n_new_points_, [batch_size, npoint, flow_dim // GROUP_SIZE, GROUP_SIZE + 1])
            #[A,b]
            #shape(batch_size,npoint,flow_dim // group_size,GROUP_SIZE+1) 比如（batch_szie,32*32,11,4）
            n_new_points_ = n_new_points_[:, :, :, 0:GROUP_SIZE]#A_star,公式（6） 比如shape（batch_szie,32*32,11,3） A每组3个参数(ak0,ak1,ak2)，针对每组3个特征
            n_new_points_ = tf.concat([n_new_points_, -tf.ones(shape=[batch_size, npoint, flow_dim // GROUP_SIZE, 1], dtype=n_new_points_.dtype)], -1)
            #(A_star,-1)
            #shape(batch_size,npoint,flow_dim // group_size,GROUP_SIZE+1)  比如（batch_szie,32*32,11,3+1）
            normalization_factor = tf.square(n_new_points_)
            #(A_star,-1)中每个元素取平方
            normalization_factor = tf.reduce_sum(normalization_factor, 3, keep_dims=True)
            #(A_star,-1)中每个元素取平方，然后每组4个数求和。->(ak0^2+ak1^2+ak2^2+1)    A每组3个参数，针对每组3个特征。A中3个数平方然后求和，最后加上(-1)的平方。
            #shape(batch_size,npoint,flow_dim // group_size,1)  比如（batch_szie,32*32,11,1）
            t = new_points_normals / normalization_factor
            t = tf.reshape(t, [batch_size, npoint, flow_dim // GROUP_SIZE, nsample, 1])
            #归一化的差距。 Af+b-t是真实和预测的差距。每组对应一组A，b。64个邻居点(f,t)特征被分为11组。所以每组差距的归一化，分母和A，b有关。
            # shape(batch_size, npoint, flow_dim // GROUP_SIZE, nsample, 1)
            # 比如，layer1,i=0,(N,32*32,11,64,1)
            n_new_points_ = tf.reshape(n_new_points_, [batch_size, npoint, flow_dim // GROUP_SIZE, 1, GROUP_SIZE + 1])
            W = n_new_points_ * t
            # batch_size, npoint, flow_dim // GROUP_SIZE, nsample, GROUP_SIZE
            W = tf.reshape(W, [batch_size * npoint, flow_dim // GROUP_SIZE, nsample, GROUP_SIZE + 1])
            #每个search point,每组每个邻居的W(k,j)=[ak0,ak1,ak2,-1]*归一化的差距。这样每组每个邻居有独特W
            #k:组号。j:邻居编号。   组数：flow_dim // GROUP_SIZE   邻居数：nsample
            #函数dynamic_module返回参数6W：shape:[batch_size * npoint, flow_dim // GROUP_SIZE, nsample, GROUP_SIZE + 1]
        #函数dynamic_module返回参数3：new_points
        new_points = tf.reduce_max(new_points, axis=[2], name='maxpool')
        #layer1:shape(batch_size ,npoint,128)
        # new_xyz = tf.Print(new_xyz, [new_xyz.shape], "new_xyz: ", summarize=2333, first_n=1)
        # new_time = tf.Print(new_time, [new_time.shape], "new_time: ", summarize=2333, first_n=1)
        # new_points = tf.Print(new_points, [new_points.shape], "new_points: ", summarize=2333, first_n=1)
        # idx = tf.Print(idx, [idx.shape], "idx: ", summarize=2333, first_n=1)

        return new_xyz, new_time, new_points, idx, flow_list, W


def flow_merge_across_res(flow1_list, flow2_list, flow1_idx, flow2_idx,
                          flow1_time, flow2_time, l1_xyz, l2_xyz,
                          scope, is_training, bn_decay,
                          bn=False, data_format='NHWC'):
    it_num = max(len(flow1_list), len(flow2_list))#it_num=2
    for i in range(it_num):
        if i < len(flow1_list):
            flow1_current = flow1_list[i]  #flow1_list[0]:(b_s,32*32,11*(3+1))  flow1_list[1]:(b_s,32*32,22*(3+1))
            flow1_channel = flow1_current.shape[-1]#i=0,值为11*(3+1)。i=1时，值为22*(3+1)
            flow1 = tf_util.conv1d(tf.concat([flow1_current, flow1_time], -1),
                                   flow1_channel, 1, padding='VALID', stride=1,
                                   bn=bn, is_training=is_training, #activation_fn=tf.nn.leaky_relu,
                                   scope='cur1_%d_dynamic' % i, bn_decay=bn_decay,
                                   data_format=data_format)
            #shape(b_s,32*32,flow1_channel)
            #比如，i=0时,卷积网络输入是将flow1_current和flow1_time连接，就是特征11*(3+1)+1(时间)。
            #因为kernel:1, padding='VALID', stride=1，所以输出flow1的shape只有最后一个维度变为flow1_channel
            flow1 = flow1 + tf_util.conv1d(tf.concat([last_flow1, flow1_time], -1),
                                         flow1_channel, 1,
                                         padding='VALID', stride=1,
                                         bn=bn, is_training=is_training, #activation_fn=tf.nn.leaky_relu,
                                         scope='last_conv1%d_dynamic' % i, bn_decay=bn_decay,
                                         data_format=data_format) if i > 0 else flow1
            #shape(b_s,32*32,flow1_channel)
            #卷积网络输入是将last_flow1和flow1_time连接
            last_flow1 = flow1#shape(b_s,32*32,flow1_channel)

        if i < len(flow2_list):
            flow2_current = flow2_list[i]        
            flow2_channel = flow2_current.shape[-1]#i=0,值为22*(3+1)。i=1时，值为43*(3+1)
            flow2 = tf_util.conv1d(tf.concat([flow2_current, flow2_time], -1),
                                   flow2_channel, 1, padding='VALID', stride=1,
                                   bn=bn, is_training=is_training, #activation_fn=tf.nn.leaky_relu,
                                   scope='cur2_%d_dynamic' % i, bn_decay=bn_decay,
                                   data_format=data_format)
            flow2 = flow2 + tf_util.conv1d(tf.concat([last_flow2, flow2_time], -1),
                                         flow2_channel, 1,
                                         padding='VALID', stride=1,
                                         bn=bn, is_training=is_training, #activation_fn=tf.nn.leaky_relu,
                                         scope='last2_%d_dynamic' % i, bn_decay=bn_decay,
                                         data_format=data_format) if i > 0 else flow2
            last_flow2 = flow2#shape(b_s,32*16,flow1_channel)

        flow2_to_1 = pointnet_fp_module(l1_xyz, l2_xyz, tf.concat([flow1, flow1_time], -1),
                                        tf.concat([flow2, flow2_time], -1), (flow1_channel, ),
                                        is_training=is_training, bn_decay=bn_decay, scope="2to1_%d_dynamic" % i, bn=bn)
        last_flow1 = last_flow1 + flow2_to_1

        flow1_to_2 = pointnet_fp_module(l2_xyz, l1_xyz, tf.concat([flow2, flow2_time], -1),
                                        tf.concat([flow1, flow1_time], -1), (flow2_channel, ),
                                        is_training=is_training, bn_decay=bn_decay, scope="1to2_%d_dynamic" % i, bn=bn)
        #flow1_to_2:(batch_size, 32*16, flow2_channel)
        last_flow2 = last_flow2 + flow1_to_2
        # flow1_to_2:(batch_size, 32*16, flow2_channel)
        #flow2_channel:i=0,值为22*(3+1)。i=1时，值为43*(3+1)

    return last_flow1, last_flow2

def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2, keepdims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1

