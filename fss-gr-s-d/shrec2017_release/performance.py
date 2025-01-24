from tensorflow.python import pywrap_tensorflow

import numpy as np

# model_dir = "models_pretrained/"
# checkpoint_path = os.path.join(model_dir, "model.ckpt-82798")
#checkpoint_path ="./log_model_cls_dynamic_exp5_12_base249_frames32_batch_size4_epoch250/model0249.ckpt"

#函数count_total_params(checkpoint_path)：根据ckpt，计算总参数量。
def count_total_params(checkpoint_path):
    #checkpoint_path:ckpt的路径，比如"./log_model_cls_static_frames32_batch_size4_epoch250/model249.ckpt"
    #return :总参数量
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

checkpoint_path ="log2puttoT_model_cls_dynamic_exp5_12_base249_frames32_batch_size4_epoch1/model0000.ckpt"
total_parameters=count_total_params(checkpoint_path) #调用函数

print("total_parameters:",total_parameters) #参数个数