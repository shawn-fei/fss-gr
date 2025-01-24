#T-FSS-GR:main
#main_2:参考main.py。原先main.py命名为main_orig.py
#863行,733行，593行附近修改 path_recon_flow_2800：用recon_flow还是用est_flow
#加了profile,性能测试

import os
import sys

from gesture_utils import get_parser, import_class, Recorder, Stat, RandomState
import torch

import yaml
import numpy as np

# import tensorflow._api.v2.compat.v1 as tf
import tensorflow as tf
# tf.disable_v2_behavior()

import importlib

#为了测试模型性能新增profiler 1：
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from tensorflow.python import pywrap_tensorflow
#上：为了测试模型性能新增profiler 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))

sparser = get_parser()
FLAGS = sparser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_FRAME = FLAGS.num_frames
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
COMMAND_FILE = FLAGS.command_file

MODALITY = FLAGS.modality
MODEL_PATH = FLAGS.model_path

MODEL = importlib.import_module(FLAGS.network_file)  # import network module
print("MODEL:", MODEL)
MODEL_FILE = os.path.join(FLAGS.network_file + '.py')
print("MODEL_FILE", MODEL_FILE)
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp *_dataloader*.py %s' % (LOG_DIR))  # bkp of data loader
os.system('cp main.py %s' % (LOG_DIR))  # bkp of train procedure
os.system('cp net_utils.py %s' % (LOG_DIR))  # bkp of net_utils
os.system('cp %s %s' % (COMMAND_FILE, LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
INPUT_DIM = 7


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):#一个函数，用于修改学习率随时间变化的方式
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.200000
        DECAY_RATE,  # Decay rate.0.7
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = RandomState(seed=self.arg.random_seed)
        self.recoder = Recorder(self.arg.work_dir, self.arg.print_log)
        #模型评估性能 profiler 7: 定义recoder2
        self.path_performance = os.path.join(LOG_DIR, "performance")  # 性能相关文件输出文件夹## 输出文件夹必须存在
        if not os.path.exists(self.path_performance):
            os.makedirs(self.path_performance)

        self.recoder2 = Recorder(work_dir=self.path_performance, print_log=self.arg.print_log) #这个recoder2用来记录性能评估器profiler的输出,work_dir是log_dir下results子文件夹，默认输出txt就是results里的log.txt
        #上新增：定义recoder2

        # self.device = GpuDataParallel()
        self.data_loader = {}
        self.topk = (1, 5)
        self.stat = Stat(self.arg.model_args['num_classes'], self.topk)
        # self.model, self.optimizer = self.Loading()
        self.load_data()
        self.loss = self.criterion()

    def criterion(self):
        loss = torch.nn.CrossEntropyLoss(reduction="none")
        return loss  # self.device.criterion_to_device(loss)

    def load_data(self):
        print("Loading data")
        Feeder = import_class(self.arg.dataloader)
        self.data_loader = dict()
        if self.arg.train_loader_args != {}:
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_loader_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.arg.num_worker,
            )
        if self.arg.valid_loader_args != {}:
            self.data_loader['valid'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.valid_loader_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.arg.num_worker,
            )
        if self.arg.test_loader_args != {}:
            test_dataset = Feeder(**self.arg.test_loader_args)
            self.stat.test_size = len(test_dataset)
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.arg.num_worker,
            )
        print("Loading data finished.")

    def start(self):
        if self.arg.phase == 'train' and MODALITY == "static":
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            with tf.Graph().as_default():
                # '''
                with tf.device('/gpu:' + str(GPU_INDEX)):
                    pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FRAME,
                                                                         INPUT_DIM)  # 模型（点云和标签）占位（placeholder），分配必要的内存
                    is_training_pl = tf.placeholder(tf.bool, shape=())

                    # Note the global_step=batch parameter to minimize.
                    # That tells the optimizer to helpfully increment the 'batch' parameter
                    # for you every time it trains.
                    batch = tf.get_variable('batch', [],
                                            initializer=tf.constant_initializer(0),
                                            trainable=False)  # 用于获取已存在的变量,不存在就新建一个
                    bn_decay = get_bn_decay(batch)
                    tf.summary.scalar('bn_decay', bn_decay)

                    # Get model and loss
                    pred, end_points = MODEL.get_model(pointclouds_pl, NUM_FRAME, is_training_pl,
                                                       bn_decay=bn_decay, CLS_COUNT=self.arg.model_args['num_classes'])
                    MODEL.get_loss(pred, labels_pl, end_points)
                    losses = tf.get_collection('losses')
                    total_loss = tf.add_n(losses, name='total_loss')
                    tf.summary.scalar('total_loss', total_loss)
                    for l in losses:
                        tf.summary.scalar(l.op.name, l)
                    correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
                    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
                    tf.summary.scalar('accuracy', accuracy)

                    print("--- Get training operator")
                    # Get training operator
                    learning_rate = get_learning_rate(batch)#一个函数，用于修改学习率随时间变化的方式
                    tf.summary.scalar('learning_rate', learning_rate)
                    if OPTIMIZER == 'momentum':
                        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                    elif OPTIMIZER == 'adam':
                        optimizer = tf.train.AdamOptimizer(learning_rate)
                    train_op = optimizer.minimize(total_loss, global_step=batch)

                    # Add ops to save and restore all the variables.
                    self.saver = tf.train.Saver()
                # '''
                # Create a session
                tf_config = tf.ConfigProto()
                tf_config.gpu_options.allow_growth = False
                tf_config.allow_soft_placement = True
                tf_config.log_device_placement = False
                sess = tf.Session(config=tf_config)

                # Add summary writers
                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
                test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)
                # Init variables
                init = tf.global_variables_initializer()
                sess.run(init)

                pretrained = MODEL_PATH
                if pretrained is not '':
                    variables = tf.contrib.framework.get_variables_to_restore()
                    variables_to_restore = [v for v in variables if 'fc3' not in v.name and "batch:" not in v.name]
                    if OPTIMIZER == 'adam':
                        variables_to_restore = [v for v in variables_to_restore if "_power" not in v.name]
                    elif OPTIMIZER == 'momentum':
                        variables_to_restore = [v for v in variables_to_restore if "Momentum" not in v.name]
                    print("variables_to_restore", variables_to_restore)
                    loading_saver = tf.train.Saver(variables_to_restore)
                    pretrained_model_path = pretrained  # "pretrained_on_modelnet/model1464.ckpt"
                    loading_saver.restore(sess, pretrained_model_path)
                    print("The model has been loaded !!!!!!!!!!!!!")

                ops = {'pointclouds_pl': pointclouds_pl,
                       'labels_pl': labels_pl,
                       'is_training_pl': is_training_pl,
                       'pred': pred,
                       'loss': total_loss,
                       'accuracy': accuracy,
                       'train_op': train_op,
                       'merged': merged,
                       'step': batch,
                       'end_points': end_points}
                for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                    save_model = ((epoch + 1) % self.arg.save_interval == 0) or \
                                 (epoch + 1 == self.arg.num_epoch)
                    eval_model = ((epoch + 1) % self.arg.eval_interval == 0) or \
                                 (epoch + 1 == self.arg.num_epoch)
                    self.train(epoch, sess, ops, train_writer)
                    if eval_model:
                        if self.arg.valid_loader_args != {}:
                            self.stat.reset_statistic()
                            self.eval(loader_name=['valid'], sess=sess, ops=ops)
                        if self.arg.test_loader_args != {}:
                            self.stat.reset_statistic()
                            self.eval(loader_name=['test'], sess=sess, ops=ops)
                    if save_model:
                        save_path = self.saver.save(sess, os.path.join(LOG_DIR, "model%03d.ckpt" % epoch))

                        log_string("Model saved in file: %s" % save_path)

        if self.arg.phase == 'train' and MODALITY == "dynamic":
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            with tf.Graph().as_default():
                # '''
                with tf.device('/gpu:' + str(GPU_INDEX)):
                    #改2：返回参数新增datasource_pl,reconflow_pl
                    #pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FRAME, INPUT_DIM)
                    pointclouds_pl, labels_pl,datasource_pl,reconflow_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FRAME, INPUT_DIM)
                    #改2：返回参数新增datasource_pl,reconflow_pl
                    is_training_pl = tf.placeholder(tf.bool, shape=())

                    # Note the global_step=batch parameter to minimize.
                    # That tells the optimizer to helpfully increment the 'batch' parameter
                    # for you every time it trains.
                    batch = tf.get_variable('batch', [],
                                            initializer=tf.constant_initializer(0), trainable=False)
                    bn_decay = get_bn_decay(batch)
                    tf.summary.scalar('bn_decay', bn_decay)#记录数据bn_decay到'此标签'bn_decay'下

                    # Get model and loss
                    #改三：输入参数新增datasource_pl,reconflow_pl
                    static_pred, flow_pred, end_points = MODEL.get_model(pointclouds_pl,datasource_pl,reconflow_pl,NUM_FRAME, is_training_pl,
                                                                         bn_decay=bn_decay,
                                                                         CLS_COUNT=self.arg.model_args['num_classes'])
                    #改三：输入参数新增datasource_pl,reconflow_pl
                    pred = flow_pred
                    MODEL.get_loss(pred, labels_pl, end_points)
                    losses = tf.get_collection('losses')#返回所有放入'losses'的变量的列表
                    total_loss = tf.add_n(losses, name='total_loss')
                    tf.summary.scalar('total_loss', total_loss)
                    for l in losses:
                        tf.summary.scalar(l.op.name, l)##记录数据l到'此标签l.op.name下
                    correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
                    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
                    tf.summary.scalar('accuracy', accuracy)

                    print("--- Get training operator")
                    # Get training operator
                    learning_rate = get_learning_rate(batch)#一个函数，用于修改学习率随时间变化的方式
                    tf.summary.scalar('learning_rate', learning_rate)
                    if OPTIMIZER == 'momentum':
                        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                    elif OPTIMIZER == 'adam':
                        optimizer = tf.train.AdamOptimizer(learning_rate)
                    train_op = optimizer.minimize(total_loss, global_step=batch)

                    # Add ops to save and restore all the variables.
                    self.saver = tf.train.Saver()#一个类，提供了变量、模型(也称图Graph)的保存和恢复模型方法。
                    # #tf.train.Saver()类初始化时，用于保存和恢复的save和restore operator会被加入Graph。所以，下列类初始化操作应在搭建Graph时完成

                # '''
                # Create a session
                tf_config = tf.ConfigProto()
                tf_config.gpu_options.allow_growth = False
                tf_config.allow_soft_placement = True
                tf_config.log_device_placement = False
                sess = tf.Session(config=tf_config)

                # Add summary writers
                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)#指定一个文件用来保存图
                test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)
                # Init variables
                init = tf.global_variables_initializer()
                sess.run(init)

                #改：试模型性能新增profiler 2
                profiler = model_analyzer.Profiler(graph=sess.graph)   #创建tfprofiler实例，作为记录、处理和显示数据的主体
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) #定义trace level为FULL_TRACE，这样我们才能搜集到包括GPU硬件在内的最全统计数据
                run_metadata = tf.RunMetadata() #创建RunMetadata，用于在每次session.Run()时汇总统计数据
                #上改：试模型性能新增profiler 2

                if MODEL_PATH is not '':
                    # 下行代码修改：tf.contrib.framework.get_variables_to_restore()用tf.Variable（）代替
                    variables = tf.contrib.framework.get_variables_to_restore()  # 得到需要restore的变量
                    # 上行代码修改：tf.contrib.framework.get_variables_to_restore()用tf.Variable（）代替
                    variables_to_restore = [v for v in variables if '_dynamic' not in v.name and "batch:" not in v.name]
                    if OPTIMIZER == 'adam':
                        variables_to_restore = [v for v in variables_to_restore if "_power" not in v.name]
                    elif OPTIMIZER == 'momentum':
                        variables_to_restore = [v for v in variables_to_restore if "Momentum" not in v.name]
                    print("variables_to_restore", variables_to_restore)  # 需要restore的变量。

                    loading_saver = tf.train.Saver(variables_to_restore)
                    pretrained_model_path = MODEL_PATH
                    loading_saver.restore(sess, pretrained_model_path)
                    print("The model has been loaded !!!!!!!!!!!!!")

                #改4：ops新增键值对'datasource_pl':datasource_pl,'reconflow_pl':reconflow_pl,
                ops = {'pointclouds_pl': pointclouds_pl,
                       'labels_pl': labels_pl,
                       'datasource_pl':datasource_pl,
                       'reconflow_pl':reconflow_pl,
                       'is_training_pl': is_training_pl,
                       'pred': tf.tuple([static_pred, flow_pred]),
                       'loss': total_loss,
                       'accuracy': accuracy,
                       'train_op': train_op,
                       'merged': merged,
                       'step': batch,
                       'end_points': end_points}
                ##上改4：ops新增键值对'datasource_pl':datasource_pl
                for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                    save_model = ((epoch + 1) % self.arg.save_interval == 0) or \
                                 (epoch + 1 == self.arg.num_epoch)
                    eval_model = ((epoch + 1) % self.arg.eval_interval == 0) or \
                                 (epoch + 1 == self.arg.num_epoch)
                    #改：模型性能分析profile 3:新增三个参数输入
                    self.train(epoch, sess, ops, train_writer,profiler=profiler,run_options=run_options,run_metadata=run_metadata)
                    #self.train(epoch, sess, ops, train_writer)
                    # 改：模型性能分析profile 3
                    if eval_model:
                        if self.arg.valid_loader_args != {}:
                            self.stat.reset_statistic()
                            self.eval_when_training(loader_name=['valid'], sess=sess, ops=ops)
                        if self.arg.test_loader_args != {}:
                            self.stat.reset_statistic()
                            self.eval_when_training(loader_name=['test'], sess=sess, ops=ops)
                    if save_model:
                        save_path = self.saver.save(sess, os.path.join(LOG_DIR, "model%04d.ckpt" % epoch))

                        log_string("Model saved in file: %s" % save_path)

                # 6.2 性能测试proflier：scope view显示模型中的参数数量分布:通过这种方式，查看各个layer中参数的总数，以控制模型的大小和参数分布
                # 统计内容为所有trainable Variable Op

                profile_scope_opt_builder = option_builder.ProfileOptionBuilder(
                    option_builder.ProfileOptionBuilder.trainable_variables_parameter())
                # 显示的嵌套深度为4
                profile_scope_opt_builder.with_max_depth(5)
                # 显示字段是params，即参数
                profile_scope_opt_builder.select(['params'])
                # 根据params数量进行显示结果排序
                profile_scope_opt_builder.order_by('params')
                # 显示视图为scope view

                checkpoint_path=os.path.join(LOG_DIR, "model%04d.ckpt" % epoch)

                params=profiler.profile_name_scope(options=profile_scope_opt_builder.build())#这条语句显示Options和Model Analysis Report
                flops = tf.profiler.profile(graph=sess.graph, run_meta=run_metadata,cmd='op',options=tf.profiler.ProfileOptionBuilder.float_operation())
                total_params = self.count_total_params(checkpoint_path=checkpoint_path)

                # print('FLOPs: {}'.format(flops.total_float_ops))
                # print('Trainable Params: {}'.format(params.total_parameters))
                # print("total_parameters:",total_params) #总参数个数
                self.recoder2.print_log('Model saved in file: {}'.format(checkpoint_path), print_time=False)
                self.recoder2.print_log('FLOPs: {}'.format(flops.total_float_ops),print_time=False)
                self.recoder2.print_log('GFLOPs: {} G'.format(flops.total_float_ops/(10**9)), print_time=False)
                self.recoder2.print_log('Trainable Params: {}'.format(params.total_parameters),print_time=False)
                self.recoder2.print_log('Trainable Params: {}M'.format(params.total_parameters/(10**6)), print_time=False)
                self.recoder2.print_log("total_parameters:{}M".format(total_params/(10**6)),print_time=False)
                #print("************6.2:")



        elif self.arg.phase == 'test':
            with tf.Graph().as_default():
                with tf.device('/gpu:' + str(GPU_INDEX)):
                    pointclouds_pl, labels_pl,datasource_pl,reconflow_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FRAME, INPUT_DIM)
                    is_training_pl = tf.placeholder(tf.bool, shape=())

                    # Note the global_step=batch parameter to minimize.
                    # That tells the optimizer to helpfully increment the 'batch' parameter
                    # for you every time it trains.
                    batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
                    bn_decay = get_bn_decay(batch)
                    static_pred, flow_pred, end_points = MODEL.get_model(pointclouds_pl, datasource_pl,reconflow_pl,NUM_FRAME, is_training_pl,
                                                                         bn_decay=bn_decay,
                                                                         CLS_COUNT=self.arg.model_args['num_classes'])

                # Create a session
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True
                config.log_device_placement = False
                sess = tf.Session(config=config)
                # Init variables
                loading_saver = tf.train.Saver()
                loading_saver.restore(sess, MODEL_PATH)
                print("The model has been loaded !!!!!!!!!!!!!")
                # Add summary writers
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'sess'), sess.graph)

                ops = {'pointclouds_pl': pointclouds_pl,
                       'labels_pl': labels_pl,
                       'datasource_pl': datasource_pl,
                       'reconflow_pl': reconflow_pl,
                       'is_training_pl': is_training_pl,
                       'static_pred': static_pred,
                       'flow_pred': flow_pred,
                       'merged': merged,
                       'step': batch}

                self.recoder.print_log('Model: {}.'.format(MODEL_PATH))
                result_list = []
                if self.arg.valid_loader_args != {}:
                    static_ratio = 1.0
                    self.stat.reset_statistic()
                    self.mixed_eval(['valid'], sess, ops, writer, static_ratio=static_ratio)
                    result_list.append((static_ratio, self.print_inf_log(self.arg.optimizer_args['start_epoch'],
                                                                         "Valid (static_ratio: %.2f)" % static_ratio)))

                    static_ratio = 0.0
                    self.stat.reset_statistic()
                    self.mixed_eval(['valid'], sess, ops, writer, static_ratio=static_ratio)
                    result_list.append((static_ratio, self.print_inf_log(self.arg.optimizer_args['start_epoch'],
                                                                         "Valid (static_ratio: %.2f)" % static_ratio)))

                    static_ratio = 0.5
                    self.stat.reset_statistic()
                    self.mixed_eval(['valid'], sess, ops, writer, static_ratio=static_ratio)
                    result_list.append((static_ratio, self.print_inf_log(self.arg.optimizer_args['start_epoch'],
                                                                         "Valid (static_ratio: %.2f)" % static_ratio)))

                if self.arg.test_loader_args != {}:
                    static_ratio = 1.0
                    self.stat.reset_statistic()
                    self.mixed_eval(['test'], sess, ops, writer, static_ratio=static_ratio)
                    result_list.append((static_ratio, self.print_inf_log(self.arg.optimizer_args['start_epoch'],
                                                                         "Test (static_ratio: %.2f)" % static_ratio)))

                    static_ratio = 0.0
                    self.stat.reset_statistic()
                    self.mixed_eval(['test'], sess, ops, writer, static_ratio=static_ratio)
                    result_list.append((static_ratio, self.print_inf_log(self.arg.optimizer_args['start_epoch'],
                                                                         "Test (static_ratio: %.2f)" % static_ratio)))

                    static_ratio = 0.5
                    self.stat.reset_statistic()
                    self.mixed_eval(['test'], sess, ops, writer, static_ratio=static_ratio)
                    result_list.append((static_ratio, self.print_inf_log(self.arg.optimizer_args['start_epoch'],
                                                                         "Test (static_ratio: %.2f)" % static_ratio)))

            self.recoder.print_log('Evaluation Done.\n')
            for r in result_list:
                self.recoder.print_log("static ratio %.2f: acc=%.6f" % r)

    def print_inf_log(self, epoch, mode):
        stati = self.stat.show_accuracy('{}/{}_confusion_mat'.format(self.arg.work_dir, mode))
        prec1 = stati[str(self.topk[0])] / self.stat.test_size * 100
        prec5 = stati[str(self.topk[1])] / self.stat.test_size * 100
        self.recoder.print_log("Epoch {}, {}, Evaluation: prec1 {:.4f}, prec5 {:.4f}".
                               format(epoch, mode, prec1, prec5),
                               '{}/{}.txt'.format(self.arg.work_dir, self.arg.phase))
        return prec1

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    #模型性能分析：函数count_total_params(checkpoint_path)：根据ckpt，计算总参数量。
    def count_total_params(self,checkpoint_path):
        # checkpoint_path:ckpt的路径，比如"./log_model_cls_static_frames32_batch_size4_epoch250/model249.ckpt"
        # return :总参数量
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        total_parameters = 0
        for key in var_to_shape_map:  # list the keys of the model
            # print(key)
            # print(reader.get_tensor(key))
            shape = np.shape(reader.get_tensor(key))  # get the shape of the tensor in the model
            shape = list(shape)
            # print(shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                # print(dim)
                variable_parameters *= dim
            # print(variable_parameters)
            total_parameters += variable_parameters

        return total_parameters
    # 模型性能分析profile: 函数count_total_params(checkpoint_path)：根据ckpt，计算总参数量。

    #改： 模型性能分析profile 4 ：新增三个形参 profiler,run_options,run_metadata
    def train(self, epoch, sess, ops, tf_writer,profiler,run_options,run_metadata):
    #def train(self, epoch, sess, ops, tf_writer):
    #上改： 模型性能分析profile 4
        """ ops: dict mapping from string to tf ops """
        is_training = True
        self.recoder.print_log('Training epoch: {}'.format(epoch))
        loader = self.data_loader['train']
        loss_value = []
        self.recoder.timer_reset()
        # current_learning_rate = [group['lr'] for group in self.optimizer.optimizer.param_groups]

        batch_num = 1960 / self.arg.batch_size
        self.recoder2.print_log("一个epoch中共{}个batch,每35步统计一次数据".format(batch_num),
                            print_time=False)  # (1960/batch_size=1960/8=245 1960/4=490)
        for batch_idx, data in enumerate(loader):
            self.recoder.record_timer("dataloader")
            batch_data = data[0].detach().numpy()
            batch_label = data[1].detach().numpy()

            # 改5：只有dynamic时才运行
            if self.arg.phase == 'train' and MODALITY == "dynamic":
                #改5：   data[2]对应SHRECLoader类对象getitem方法返回参数3：self.inputs_list[index]:字符串
                #print("data[2]",data[2]) #比如batch_size为2时('1 1 2 1 1 1 77\n', '1 1 2 2 1 1 117\n')
                #print("长度data[2]",len(data[2]))
                # for i,s in enumerate(data[2]):
                #     print("第{}个数据：".foramt(i,s))
                #data[2]=torch.tensor(data[2])#tuple转tensor
                import re
                r = re.compile('[ \t\n\r]+')
                list_data2=[[] for i in range(BATCH_SIZE)]#二维列表(Batch_size,7)
                #data[2]:比如batch_size为2时['1 1 2 1 1 1 77\n', '1 1 2 2 1 1 117\n']
                for i,str_d in enumerate(data[2]):#'1 1 2 1 1 1 77\n'
                    l = r.split(str_d)#['1', '1', '2', '1', '1', '1', '77', '']
                    if l[-1] == '' or "\n":
                        l = l[0:-1]##['1', '1', '2', '1', '1', '1', '77']
                    l = [int(x) for x in l]  # 字符外引号去掉变为数值 [1, 1, 2, 1, 1, 1, 77]
                    list_data2[i]=l
                #print("list_data2", list_data2)
                #list_data2=list(map(int,list_data2))
                data2_tensor=torch.Tensor(list_data2)
                batch_datasource = data2_tensor.detach().numpy()
                batch_datasource = batch_datasource.astype(int)  # 每个数值变为整数
                # print("batch_datasource", batch_datasource)
                # print("typebatch_datasource", type(batch_datasource))#ndaaray
                # print("shape,batch_datasource", batch_datasource.shape)

                batch_reconflow_list=[[[0] * 48 for i in range(32)] for j in range(BATCH_SIZE)]#创建(BATCH_SIZE,32,48)的3维列表
                for k,str_datasource in enumerate(data[2]):#btach_size是几，就会循环几次
                    # 1.获得kinet这边数据来源splitLine->target_  #比如'1_1_2_1'
                    splitLine = r.split(str_datasource)#['1', '1', '2', '1', '1', '1', '77', '']
                    idx_gesture, idx_finger, idx_subject, idx_esaai = splitLine[0], splitLine[1], splitLine[2], splitLine[3]
                    ds_l = [idx_gesture, idx_finger, idx_subject, idx_esaai]
                    #target = " ".join(ds_l)#'1 1 2 1'
                    target_ = "_".join(ds_l)#'1_1_2_1'
                    #target = " " + target + "\n"  # ' 1 1 2 1\n' 注意：前加空格，不然"11 1 2 1\n"也会被选出来
                    #2.获得target_对应recon_flow,shape(32,16*3) 这只是一条视频（batch中一条数据）对应的flow
                    # Path to current file
                    path_current = os.path.dirname(__file__)  # 去掉文件名后的绝对路径
                    #path_recon_flow_2800 = os.path.join(path_current, "..", "dataset","est_flow")  # dataset/est_flow
                    path_recon_flow_2800 = os.path.join(path_current, "..", "dataset", "recon_flow")#dataset/recon_flow
                    txt_recon_flow='/log_'+target_+'.txt'#"/log_1_1_2_1.txt"
                    recon_flow = open(path_recon_flow_2800 + txt_recon_flow).readlines()#list:32.每个元素是str。str中数据16*3

                    list_video_flow=[[] for i in range(32)]#新建二维列表，多行48列。最终是list,shape(32,16*3)
                    for j,str_frame_flow  in enumerate(recon_flow):#共32行
                        str_frame_flow = r.split(str_frame_flow)  # ['p1_x', 'p1_y', 'p1_z', ……,'p16_x', 'p16_y', 'p16_z', '']
                        if str_frame_flow[-1]=='' or "\n":
                            str_frame_flow=str_frame_flow[0:-1] #去掉换行符['p1_x', 'p1_y', 'p1_z', ……,'p16_x', 'p16_y', 'p16_z']
                        list_frame_flow=[eval(x) for x in str_frame_flow]#字符外引号去掉变为数值，没用map（可以更换类型为float32）
                        # [p1_x, p1_y, p1_z, ……,p16_x, p16_y, p16_z] shape(16*3)
                        list_video_flow[j]=list_frame_flow
                batch_reconflow_list[k]=list_video_flow
                #3.得到整个batch的recon_flow数据。shape(BATCH_SIZE,32,48)
                tensor_batch_reconflow = torch.tensor(batch_reconflow_list)#tensor,shape(BATCH_SIZE,32,48)
                batch_reconflow = tensor_batch_reconflow.detach().numpy()#转为numpy(),,shape(BATCH_SIZE,32,48)

                #改5

            self.recoder.record_timer("device")
            #改6：feed_dict喂数据：新增键值对ops['datasource_pl']:batch_datasource , ops['reconflow_pl']:batch_reconflow,
            if self.arg.phase == 'train' and MODALITY == "dynamic":
                feed_dict = {ops['pointclouds_pl']: batch_data.reshape(batch_data.shape[0], -1, batch_data.shape[-1]),
                             ops['labels_pl']: batch_label,
                             ops['datasource_pl']:batch_datasource,
                             ops['reconflow_pl']:batch_reconflow,
                             ops['is_training_pl']: is_training}
            else:#dynamic
                feed_dict = {ops['pointclouds_pl']: batch_data.reshape(batch_data.shape[0], -1, batch_data.shape[-1]),
                             ops['labels_pl']: batch_label,
                             ops['is_training_pl']: is_training}
            #上改6

            # 模型性能分析profile 5 原先只有else分支，现在新增if支线;每50步统计一次数据

            if (batch_idx+1)%35 == 0:  #每35步(每35个batch)统计一次数据(1960/batch_size=1960/8=245 1960/4=490)

                self.recoder2.print_log(str="\n=========性能评估profiler: batch_idx={}============\n".format(batch_idx),print_time=False)
                summary, step, _, loss_val, pred_val, acc_val = sess.run([ops['merged'], ops['step'],
                                                                      ops['train_op'], ops['loss'], ops['pred'],
                                                                      ops['accuracy']], feed_dict=feed_dict,
                                                                      options=run_options,run_metadata=run_metadata)
                #将本步（第35个batch,第70个batch,……）搜集的统计数据添加到tfprofiler实例中
                profiler.add_step(step=batch_idx, run_meta=run_metadata)



                # 模型性能分析profile 6:显示统计视图
                # 6.1 统计内容为每个graph node的运行时间和占用内存
                self.recoder2.print_log(str="=========================timeline=============================",print_time=False)
                profile_graph_opts_builder = option_builder.ProfileOptionBuilder(
                    option_builder.ProfileOptionBuilder.time_and_memory())

                # 输出方式为timeline
                filename="shrec_profiler"+str(batch_idx)+".json"
                timeline_file = os.path.join(self.path_performance, filename)  # timeline_file

                profile_graph_opts_builder.with_timeline_output(timeline_file=timeline_file)
                # 定义显示sess.Run() 第batch_idx步的统计数据
                profile_graph_opts_builder.with_step(batch_idx)
                # 显示视图为graph view
                profiler.profile_graph(options=profile_graph_opts_builder.build())  #这条命令输出timeline file在哪
                self.recoder2.print_log("=======================End:timeline=============================\n",print_time=False)
                # 模型性能分析profile 6:显示统计视图
            else:
                summary, step, _, loss_val, pred_val, acc_val = sess.run([ops['merged'], ops['step'],
                                                                      ops['train_op'], ops['loss'], ops['pred'],
                                                                      ops['accuracy']], feed_dict=feed_dict)

            # 模型性能分析profile 5
            tf_writer.add_summary(summary, step)
            self.recoder.record_timer("forward")
            loss = torch.from_numpy(np.array(loss_val))
            loss_value.append(loss.item())
            if batch_idx % self.arg.log_interval == 0:
                # self.viz.append_loss(epoch * len(loader) + batch_idx, loss.item())
                self.recoder.print_log(
                    '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}'
                    .format(epoch, batch_idx, len(loader), loss.item()))
                self.recoder.print_time_statistics()
                # 下面两行新增
                print('\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}'
                      .format(epoch, batch_idx, len(loader), loss.item()))
                # 上面两行新增
        self.recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))

    def mixed_eval(self, loader_name, sess, ops, writer, static_ratio):
        is_training = False
        for l_name in loader_name:
            loader = self.data_loader[l_name]
            for batch_idx, data in enumerate(loader):
                cur_batch_data = data[0]  # self.device.data_to_device(data[0])
                cur_batch_label = data[1]  # self.device.data_to_device(data[1])

                # 改5：mixed_eval只有在self.arg.phase == 'test' or 'valid'时才会被调用
                if self.arg.phase == 'test' or 'valid':
                    # 改5：   data[2]对应SHRECLoader类对象getitem方法返回参数3：self.inputs_list[index]:字符串
                    # print("data[2]",data[2]) #比如batch_size为2时('1 1 2 1 1 1 77\n', '1 1 2 2 1 1 117\n')
                    # print("长度data[2]",len(data[2]))
                    # for i,s in enumerate(data[2]):
                    #     print("第{}个数据：".foramt(i,s))
                    # data[2]=torch.tensor(data[2])#tuple转tensor
                    import re
                    r = re.compile('[ \t\n\r]+')
                    list_data2 = [[] for i in range(BATCH_SIZE)]  # 二维列表(Batch_size,7)
                    # data[2]:比如batch_size为2时['1 1 2 1 1 1 77\n', '1 1 2 2 1 1 117\n']
                    for i, str_d in enumerate(data[2]):  # '1 1 2 1 1 1 77\n'
                        l = r.split(str_d)  # ['1', '1', '2', '1', '1', '1', '77', '']
                        if l[-1] == '' or "\n":
                            l = l[0:-1]  ##['1', '1', '2', '1', '1', '1', '77']
                        l = [int(x) for x in l]  # 字符外引号去掉变为数值 [1, 1, 2, 1, 1, 1, 77]
                        list_data2[i] = l
                    # print("list_data2", list_data2)
                    # list_data2=list(map(int,list_data2))
                    data2_tensor = torch.Tensor(list_data2)
                    batch_datasource = data2_tensor.detach().numpy()
                    batch_datasource = batch_datasource.astype(int)  # 每个数值变为整数
                    # print("batch_datasource", batch_datasource)
                    # print("typebatch_datasource", type(batch_datasource))  # ndaaray
                    # print("shape,batch_datasource", batch_datasource.shape)

                    batch_reconflow_list = [[[0] * 48 for i in range(32)] for j in
                                            range(BATCH_SIZE)]  # 创建(BATCH_SIZE,32,48)的3维列表
                    for k, str_datasource in enumerate(data[2]):  # btach_size是几，就会循环几次
                        # 1.获得kinet这边数据来源splitLine->target_  #比如'1_1_2_1'
                        splitLine = r.split(str_datasource)  # ['1', '1', '2', '1', '1', '1', '77', '']
                        idx_gesture, idx_finger, idx_subject, idx_esaai = splitLine[0], splitLine[1], splitLine[2], \
                                                                          splitLine[3]
                        ds_l = [idx_gesture, idx_finger, idx_subject, idx_esaai]
                        # target = " ".join(ds_l)#'1 1 2 1'
                        target_ = "_".join(ds_l)  # '1_1_2_1'
                        # target = " " + target + "\n"  # ' 1 1 2 1\n' 注意：前加空格，不然"11 1 2 1\n"也会被选出来
                        # 2.获得target_对应recon_flow,shape(32,16*3) 这只是一条视频（batch中一条数据）对应的flow
                        # Path to current file
                        path_current = os.path.dirname(__file__)  # 去掉文件名后的绝对路径
                        path_recon_flow_2800 = os.path.join(path_current, "..", "dataset","recon_flow")  # dataset/recon_flow
                        #path_recon_flow_2800 = os.path.join(path_current, "..", "dataset","est_flow")  # dataset/est_flow
                        txt_recon_flow = '/log_' + target_ + '.txt'  # "/log_1_1_2_1.txt"
                        recon_flow = open(
                            path_recon_flow_2800 + txt_recon_flow).readlines()  # list:32.每个元素是str。str中数据16*3

                        list_video_flow = [[] for i in range(32)]  # 新建二维列表，多行48列。最终是list,shape(32,16*3)
                        for j, str_frame_flow in enumerate(recon_flow):  # 共32行
                            str_frame_flow = r.split(
                                str_frame_flow)  # ['p1_x', 'p1_y', 'p1_z', ……,'p16_x', 'p16_y', 'p16_z', '']
                            if str_frame_flow[-1] == '' or "\n":
                                str_frame_flow = str_frame_flow[
                                                 0:-1]  # 去掉换行符['p1_x', 'p1_y', 'p1_z', ……,'p16_x', 'p16_y', 'p16_z']
                            list_frame_flow = [eval(x) for x in str_frame_flow]  # 字符外引号去掉变为数值，没用map（可以更换类型为float32）
                            # [p1_x, p1_y, p1_z, ……,p16_x, p16_y, p16_z] shape(16*3)
                            list_video_flow[j] = list_frame_flow
                    batch_reconflow_list[k] = list_video_flow
                    # 3.得到整个batch的recon_flow数据。shape(BATCH_SIZE,32,48)
                    tensor_batch_reconflow = torch.tensor(batch_reconflow_list)  # tensor,shape(BATCH_SIZE,32,48)
                    batch_reconflow = tensor_batch_reconflow.detach().numpy()  # 转为numpy(),,shape(BATCH_SIZE,32,48)

                    # 改5

                if self.arg.phase == 'test' or 'valid':#mixed_eval只有在self.arg.phase == 'test' or 'valid'时才会被调用
                    feed_dict = {
                        ops['pointclouds_pl']: cur_batch_data.reshape(cur_batch_data.shape[0], -1, cur_batch_data.shape[-1]),
                        ops['labels_pl']: cur_batch_label,
                        ops['datasource_pl']: batch_datasource,
                        ops['reconflow_pl']: batch_reconflow,
                        ops['is_training_pl']: is_training}
                else:  # static
                    feed_dict = {ops['pointclouds_pl']: cur_batch_data.detach().numpy().reshape(cur_batch_data.shape[0], -1,
                                                                                                cur_batch_data.shape[-1]),
                                 ops['labels_pl']: cur_batch_label.detach().numpy(),
                                 ops['is_training_pl']: is_training}
                pred_val_static, pred_val_flow = sess.run([ops['static_pred'], ops['flow_pred']], feed_dict=feed_dict)
                # summary, step, pred_val_static, pred_val_flow = sess.run([ops['merged'], ops['step'], ops['static_pred'], ops['flow_pred']], feed_dict=feed_dict)
                '''
                pred_prob_static = self.softmax(pred_val_static)
                pred_prob_flow = self.softmax(pred_val_flow)
                '''
                pred_prob_static = pred_val_static
                pred_prob_flow = pred_val_flow

                pred_val = pred_prob_static * static_ratio + pred_prob_flow * (1.0 - static_ratio)
                pred_val = self.softmax(pred_val)

                # writer.add_summary(summary, step)
                output = torch.from_numpy(pred_val)
                self.stat.update_accuracy(output.data, cur_batch_label, topk=self.topk)

    def softmax(self, arr_list):
        ret_list = []
        for arr in arr_list:
            arr = arr - np.amax(arr)
            exp_arr = np.exp(arr)
            ret_list.append(exp_arr / np.sum(exp_arr))
        return np.array(ret_list)

    def eval(self, loader_name, sess, ops):
        is_training = False
        for l_name in loader_name:
            loader = self.data_loader[l_name]
            loss_mean = []
            for batch_idx, data in enumerate(loader):
                cur_batch_data = data[0]  # self.device.data_to_device(data[0])
                cur_batch_label = data[1]  # self.device.data_to_device(data[1])

                feed_dict = {ops['pointclouds_pl']: cur_batch_data.detach().numpy().reshape(cur_batch_data.shape[0], -1,
                                                                                            cur_batch_data.shape[-1]),
                             ops['labels_pl']: cur_batch_label.detach().numpy(),
                             ops['is_training_pl']: is_training}
                summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                              ops['loss'], ops['pred']], feed_dict=feed_dict)

                output = torch.from_numpy(pred_val)
                loss_mean += np.array(loss_val).flatten().tolist()
                self.stat.update_accuracy(output.data, cur_batch_label, topk=self.topk)
            self.recoder.print_log('mean loss: ' + str(np.mean(loss_mean)))

    def eval_when_training(self, loader_name, sess, ops):
        is_training = False
        for l_name in loader_name:
            loader = self.data_loader[l_name]
            loss_mean = []
            for batch_idx, data in enumerate(loader):
                cur_batch_data = data[0]  # self.device.data_to_device(data[0])
                cur_batch_label = data[1]  # self.device.data_to_device(data[1])

                # 改5：只有dynamic时才运行
                if self.arg.phase == 'train' and MODALITY == "dynamic":
                    # 改5：   data[2]对应SHRECLoader类对象getitem方法返回参数3：self.inputs_list[index]:字符串
                    # print("data[2]",data[2]) #比如batch_size为2时('1 1 2 1 1 1 77\n', '1 1 2 2 1 1 117\n')
                    # print("长度data[2]",len(data[2]))
                    # for i,s in enumerate(data[2]):
                    #     print("第{}个数据：".foramt(i,s))
                    # data[2]=torch.tensor(data[2])#tuple转tensor
                    import re
                    r = re.compile('[ \t\n\r]+')
                    list_data2 = [[] for i in range(BATCH_SIZE)]  # 二维列表(Batch_size,7)
                    # data[2]:比如batch_size为2时['1 1 2 1 1 1 77\n', '1 1 2 2 1 1 117\n']
                    for i, str_d in enumerate(data[2]):  # '1 1 2 1 1 1 77\n'
                        l = r.split(str_d)  # ['1', '1', '2', '1', '1', '1', '77', '']
                        if l[-1] == '' or "\n":
                            l = l[0:-1]  ##['1', '1', '2', '1', '1', '1', '77']
                        l = [int(x) for x in l]  # 字符外引号去掉变为数值 [1, 1, 2, 1, 1, 1, 77]
                        list_data2[i] = l
                    #print("list_data2", list_data2)
                    # list_data2=list(map(int,list_data2))
                    data2_tensor = torch.Tensor(list_data2)
                    batch_datasource = data2_tensor.detach().numpy()
                    batch_datasource = batch_datasource.astype(int)  # 每个数值变为整数
                    # print("batch_datasource", batch_datasource)
                    # print("typebatch_datasource", type(batch_datasource))  # ndaaray
                    # print("shape,batch_datasource", batch_datasource.shape)

                    batch_reconflow_list = [[[0] * 48 for i in range(32)] for j in
                                            range(BATCH_SIZE)]  # 创建(BATCH_SIZE,32,48)的3维列表
                    for k, str_datasource in enumerate(data[2]):  # btach_size是几，就会循环几次
                        # 1.获得kinet这边数据来源splitLine->target_  #比如'1_1_2_1'
                        splitLine = r.split(str_datasource)  # ['1', '1', '2', '1', '1', '1', '77', '']
                        idx_gesture, idx_finger, idx_subject, idx_esaai = splitLine[0], splitLine[1], splitLine[2], \
                                                                          splitLine[3]
                        ds_l = [idx_gesture, idx_finger, idx_subject, idx_esaai]
                        # target = " ".join(ds_l)#'1 1 2 1'
                        target_ = "_".join(ds_l)  # '1_1_2_1'
                        # target = " " + target + "\n"  # ' 1 1 2 1\n' 注意：前加空格，不然"11 1 2 1\n"也会被选出来
                        # 2.获得target_对应recon_flow,shape(32,16*3) 这只是一条视频（batch中一条数据）对应的flow
                        # Path to current file
                        path_current = os.path.dirname(__file__)  # 去掉文件名后的绝对路径
                        path_recon_flow_2800 = os.path.join(path_current, "..", "dataset","recon_flow")  # dataset/recon_flow
                        #path_recon_flow_2800 = os.path.join(path_current, "..", "dataset","est_flow")  # dataset/est_flow
                        txt_recon_flow = '/log_' + target_ + '.txt'  # "/log_1_1_2_1.txt"
                        recon_flow = open(
                            path_recon_flow_2800 + txt_recon_flow).readlines()  # list:32.每个元素是str。str中数据16*3

                        list_video_flow = [[] for i in range(32)]  # 新建二维列表，多行48列。最终是list,shape(32,16*3)
                        for j, str_frame_flow in enumerate(recon_flow):  # 共32行
                            str_frame_flow = r.split(
                                str_frame_flow)  # ['p1_x', 'p1_y', 'p1_z', ……,'p16_x', 'p16_y', 'p16_z', '']
                            if str_frame_flow[-1] == '' or "\n":
                                str_frame_flow = str_frame_flow[
                                                 0:-1]  # 去掉换行符['p1_x', 'p1_y', 'p1_z', ……,'p16_x', 'p16_y', 'p16_z']
                            list_frame_flow = [eval(x) for x in str_frame_flow]  # 字符外引号去掉变为数值，没用map（可以更换类型为float32）
                            # [p1_x, p1_y, p1_z, ……,p16_x, p16_y, p16_z] shape(16*3)
                            list_video_flow[j] = list_frame_flow
                    batch_reconflow_list[k] = list_video_flow
                    # 3.得到整个batch的recon_flow数据。shape(BATCH_SIZE,32,48)
                    tensor_batch_reconflow = torch.tensor(batch_reconflow_list)  # tensor,shape(BATCH_SIZE,32,48)
                    batch_reconflow = tensor_batch_reconflow.detach().numpy()  # 转为numpy(),,shape(BATCH_SIZE,32,48)

                    # 改5
                if self.arg.phase == 'train' and MODALITY == "dynamic":
                    feed_dict = {
                        ops['pointclouds_pl']: cur_batch_data.reshape(cur_batch_data.shape[0], -1, cur_batch_data.shape[-1]),
                        ops['labels_pl']: cur_batch_label,
                        ops['datasource_pl']: batch_datasource,
                        ops['reconflow_pl']: batch_reconflow,
                        ops['is_training_pl']: is_training}
                else:  # static

                    feed_dict = {ops['pointclouds_pl']: cur_batch_data.detach().numpy().reshape(cur_batch_data.shape[0], -1,
                                                                                                cur_batch_data.shape[-1]),
                                 ops['labels_pl']: cur_batch_label.detach().numpy(),
                                 ops['is_training_pl']: is_training}
                summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                              ops['loss'], ops['pred']], feed_dict=feed_dict)
                weight = 0.5
                output_static = torch.from_numpy(pred_val[0])
                output_flow = torch.from_numpy(pred_val[1])
                output = weight * output_static + (1.0 - weight) * output_flow

                loss_mean += np.array(loss_val).flatten().tolist()
                self.stat.update_accuracy(output.data, cur_batch_label, topk=self.topk)
            self.recoder.print_log('mean loss: ' + str(np.mean(loss_mean)))


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    if FLAGS.config is not None:
        with open(FLAGS.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(FLAGS).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    print("args.network_file:", args.network_file)
    print("args.model_path:", args.model_path)
    print("args.command_file:", args.command_file)

    # args.gpu=0
    # args.num_point=128
    # args.model='model_cls_static'
    # args.num_frame=32
    # args.batch_size = 2
    # args.learning_rate = 0.001
    # args.log_dir='log_model_cls_static_frames32_batch_size2'
    # args.modality='static'
    # args.phase='train'


    processor = Processor(args)
    processor.start()
    LOG_FOUT.close()
    print("Finished!")

