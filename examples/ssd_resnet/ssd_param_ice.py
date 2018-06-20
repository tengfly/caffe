import math
import os
import os.path as osp
from time import gmtime, strftime

import caffe
from caffe.model_libs import *
from caffe import params as P
from caffe import layers as L


def make_dir_if_not_exists(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def GeneratePriorSizeQuad(input_height, input_width, min_ratio, max_ratio, feature_layer_numer, prior_num_per_anchor):
    # The size value is fit to a quadratic curve, y = a*x^2 + b, one point is: (0, min_ratio*max(resize_height, resize_width)), the other point is: (feature_layer_numer*prior_num_per_anchor, max_ratio*max(resize_height, resize_width))
    dim = max(input_height, input_width)
    xmin = 0
    size_min = min_ratio * dim / 100
    xmax = feature_layer_numer * prior_num_per_anchor - 1
    size_max = max_ratio * dim / 100
    b = size_min
    a = (size_max - size_min) * 1.0 / (xmax ** 2)

    prior_sizes = []
    for i in xrange(0, feature_layer_numer, 1):
        size_list = []
        for j in xrange(0, prior_num_per_anchor, 1):
            size_list.append(a * (i * prior_num_per_anchor + j) ** 2 + b)
        prior_sizes.append(size_list)
    return prior_sizes


def GeneratePrioSizeLinearity(input_height, input_width, min_ratio, max_ratio, feature_layer_numer):
    step = int(math.floor((max_ratio - min_ratio) / (feature_layer_numer - 2)))
    min_sizes = []
    max_sizes = []
    min_dim = max(input_height, input_width)
    for ratio in xrange(min_ratio, max_ratio + 1, step):
        min_sizes.append(min_dim * ratio / 100.)
        max_sizes.append(min_dim * (ratio + step) / 100.)

    min_sizes = [min_dim * ((min_ratio) / 2) / 100.] + min_sizes
    max_sizes = [min_dim * min_ratio / 100.] + max_sizes

    return [min_sizes, max_sizes]


current_dir = os.getcwd()
caffe_root = current_dir  # osp.join(current_dir, '..')
data_root = r'D:\OpenImages\rtsDevKit'
dev_root = r'E:\code\personal\RTS\OpenImages'
PRIORBOX_SIZE_LINEARITY = 'linear'
PRIORBOX_SIZE_QUAD = 'quad'
#######################Custom-Configured Data#####################
gpus = "0"
# experiment param
min_ratio = 10
max_ratio = 100
nms_thresh = 0.45
batch_size = 4
pretrain_fromraw = False

job_id = '-'.join(gpus.split(','))
resize_height = 300
resize_width = 300
use_batchnorm = True

accum_batch_size = 16
train_on_diff_gt = True
neg_pos_ratio = 3.0

loc_weight = 1

flip = True
clip = True

priorbox_size_mode = PRIORBOX_SIZE_QUAD
prior_num_per_anchor = 2

test_batch_size = 1
##################################################################
# rts data
num_classes = 52
if num_classes == 52:
    train_data = '{}/rts2018/lmdb/rts2018_trainval_lmdb'.format(data_root)
    test_data = '{}/rts2018/lmdb/rts2018_test_lmdb'.format(data_root)
    if pretrain_fromraw:
        pretrain_model = "{}/models/ResNet/init/ResNet-101-model.caffemodel".format(caffe_root)
    else:
        pretrain_model = "{}/../modelzoo/SSD/ResNet/300/OpenImages52/ResNet_SSD_rts_300x300_2018-06-05_1_iter_100000.caffemodel".format(
            caffe_root)

    label_map_file = '{}/data/labelmap_oi4rts.prototxt'.format(dev_root)
    num_test_image = 11869
elif num_classes == 33:
    train_data = '{}/rts2018/lmdb/rts2018_trainval-small_lmdb'.format(data_root)
    test_data = '{}/rts2018/lmdb/rts2018_test-small_lmdb'.format(data_root)
    if pretrain_fromraw:
        pretrain_model = "{}/models/ResNet/init/ResNet-101-model.caffemodel".format(caffe_root)
    else:
        pretrain_model = "{}/../modelzoo/SSD/ResNet/300/OpenImages33/ResNet_SSD_rts_300x300_2018-06-10_0_iter_100000.caffemodel".format(
            caffe_root)

    label_map_file = '{}/data/labelmap_oi4rts_33.prototxt'.format(dev_root)
    num_test_image = 3624
elif num_classes == 15:
    train_data = '{}/rts2018/lmdb/rts2018_trainval-mvp_lmdb'.format(data_root)
    test_data = '{}/rts2018/lmdb/rts2018_test-mvp_lmdb'.format(data_root)
    if pretrain_fromraw:
        pretrain_model = "{}/models/ResNet/init/ResNet-101-model.caffemodel".format(caffe_root)
    else:
        pretrain_model = "{}/../modelzoo/SSD/ResNet/300/OpenImages33/ResNet_SSD_rts_300x300_2018-06-10_0_iter_100000.caffemodel".format(
            caffe_root)

    label_map_file = '{}/data/labelmap_oi4rts_mvp.prototxt'.format(dev_root)
    num_test_image = 4047

output_result_dir = "{}/result/ssd_ResNet".format(caffe_root)
timestr = strftime("%Y-%m-%d", gmtime())
job_name = "SSD_rts_{}x{}_{}_{}".format(resize_width, resize_height, timestr, job_id)
snapshot_dir = "{}/models/ResNet/{}".format(caffe_root, job_name)
make_dir_if_not_exists(snapshot_dir)

job_dir = "{}/jobs/ResNet/{}".format(caffe_root, job_name)
make_dir_if_not_exists(job_dir)

model_name = "ResNet_{}".format(job_name)
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
save_dir = "{}/models/ResNet/{}".format(caffe_root, job_name)
make_dir_if_not_exists(save_dir)

name_size_file = "{}/data/test_name_size.txt".format(dev_root)
resume_training = True
mbox_source_layers = ['res3b3_relu', 'res5c_relu', 'res5c_relu/conv1_2', 'res5c_relu/conv2_2', 'res5c_relu/conv3_2',
                      'pool6']

# min_ratio = 15
# max_ratio = 90
# step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
# min_sizes = []
# max_sizes = []
# min_dim = max(resize_height, resize_width)
# for ratio in xrange(min_ratio, max_ratio + 1, step):
#     min_sizes.append(min_dim * ratio / 100.)
#     max_sizes.append(min_dim * (ratio + step) / 100.)
#
# # if min_ratio == 15:
# #     min_sizes = [min_dim * 7 / 100.] + min_sizes
# #     max_sizes = [min_dim * 15 / 100.] + max_sizes
# # else:
# #     min_sizes = [min_dim * 10 / 100.] + min_sizes
# #     max_sizes = [min_dim * 20 / 100.] + max_sizes
#
# min_sizes = [min_dim * ((min_ratio)/2) / 100.] + min_sizes
# max_sizes = [min_dim * min_ratio / 100.] + max_sizes

min_sizes = []
max_sizes = []
if priorbox_size_mode == PRIORBOX_SIZE_LINEARITY:
    min_sizes, max_sizes = GeneratePrioSizeLinearity(resize_height, resize_width, min_ratio, max_ratio,
                                                     len(mbox_source_layers))
elif priorbox_size_mode == PRIORBOX_SIZE_QUAD:
    min_sizes = GeneratePriorSizeQuad(resize_height, resize_width, min_ratio, max_ratio, len(mbox_source_layers),
                                      prior_num_per_anchor)

aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
aspect_ratios_flip = True
prior_variance = [0.1, 0.1, 0.2, 0.2]

share_location = True
background_label_id = 0
code_type = P.PriorBox.CENTER_SIZE
normalization_mode = P.Loss.VALID
ignore_cross_boundary_bbox = False

multibox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'loc_weight': loc_weight,
    'num_classes': num_classes,
    'share_location': share_location,
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.5,
    'use_prior_for_matching': True,
    'background_label_id': background_label_id,
    'use_difficult_gt': train_on_diff_gt,
    'mining_type': P.MultiBoxLoss.MAX_NEGATIVE,  # .HARD_EXAMPLE,
    'neg_pos_ratio': neg_pos_ratio,
    'neg_overlap': 0.5,
    'code_type': code_type,
    'ignore_cross_boundary_bbox': ignore_cross_boundary_bbox,
}
loss_param = {
    'normalization': normalization_mode,
}

# det_out_param = {
# 'num_classes': num_classes,
# 'share_location': share_location,
# 'background_label_id': background_label_id,
# 'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
# 'save_output_param': {
#     'output_directory': '/home/chunyuwang',
#     'output_name_prefix': "comp4_det_test_",
#     'output_format': "VOC",
#     'label_map_file': label_map_file,
#     },
# 'keep_top_k': 200,
# 'confidence_threshold': 0.01,
# 'code_type': code_type,
# }

train_transform_param = {
    'mirror': True,
    'mean_value': [104, 117, 123],
    'force_color': True,
    'resize_param': {
        'prob': 1,
        'resize_mode': P.Resize.WARP,
        'height': resize_height,
        'width': resize_width,
        'interp_mode': [
            P.Resize.LINEAR,
            P.Resize.AREA,
            P.Resize.NEAREST,
            P.Resize.CUBIC,
            P.Resize.LANCZOS4,
        ],
    },
    'distort_param': {
        'brightness_prob': 0.5,
        'brightness_delta': 32,
        'contrast_prob': 0.5,
        'contrast_lower': 0.5,
        'contrast_upper': 1.5,
        'hue_prob': 0.5,
        'hue_delta': 18,
        'saturation_prob': 0.5,
        'saturation_lower': 0.5,
        'saturation_upper': 1.5,
        'random_order_prob': 0.0,
    },
    'expand_param': {
        'prob': 0.5,
        'max_expand_ratio': 2.0,
    },
    'emit_constraint': {
        'emit_type': caffe_pb2.EmitConstraint.MIN_OVERLAP,
        'emit_overlap': 0.3,
    }
}

batch_sampler = [
    {
        'sampler': {
        },
        'max_trials': 1,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.1,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.1,
        },

        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.1,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.3,
        },

        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.1,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.5,
        },

        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.1,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.7,
        },

        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.1,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.9,
        },

        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.1,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'max_jaccard_overlap': 1.0,
        },

        'max_trials': 50,
        'max_sample': 1,
    },
]

test_transform_param = {
    'crop_size': resize_height,
    'mean_value': [104, 117, 123],
    'force_color': True,
    'resize_param': {
        'prob': 1,
        'resize_mode': P.Resize.WARP,
        'height': resize_height,
        'width': resize_width,
        'interp_mode': [P.Resize.LINEAR],
    },
}

gpulist = gpus.split(",")
num_gpus = len(gpulist)
run_soon = True
remove_old_models = True

# Divide the mini-batch to different GPUs.
lr_mult = 1
if use_batchnorm:
    base_lr = 0.001
else:
    base_lr = 0.0001

iter_size = accum_batch_size / batch_size
solver_mode = P.Solver.GPU
device_id = 0
batch_size_per_device = batch_size
if num_gpus > 0:
    batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
    iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
    solver_mode = P.Solver.GPU
    device_id = int(gpulist[0])

if normalization_mode == P.Loss.NONE:
    base_lr /= batch_size_per_device
elif normalization_mode == P.Loss.VALID:
    base_lr *= 10  # 25. / loc_weight
elif normalization_mode == P.Loss.FULL:
    # Roughly there are 2000 prior bboxes per image.
    # TODO(weiliu89): Estimate the exact # of priors.
    base_lr *= 2000.

# Evaluate on whole test set.

# Ideally test_batch_size should be divisible by num_test_image,
# otherwise mAP will be slightly off the true value.
test_iter = int(math.ceil(float(num_test_image) / test_batch_size))

solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': 0.0005,
    'lr_policy': "multistep",
    'stepvalue': [40000, 60000, 80000],
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': iter_size,
    'max_iter': 200000,
    'snapshot': 10000,
    'display': 100,
    'average_loss': 100,
    'type': "SGD",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': 10000,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': False,
    'show_per_class_result': True
}

# parameters for generating detection output.
det_out_param = {
    'num_classes': num_classes,
    'share_location': share_location,
    'background_label_id': background_label_id,
    'nms_param': {'nms_threshold': nms_thresh, 'top_k': 400},
    'save_output_param': {
        'output_directory': output_result_dir,
        'output_name_prefix': "comp4_det_test_",
        'output_format': "VOC",
        'label_map_file': label_map_file,
        'name_size_file': name_size_file,
        'num_test_image': num_test_image,
    },
    'keep_top_k': 200,
    'confidence_threshold': 0.01,
    'code_type': code_type,
}

# parameters for evaluating detection results.
det_eval_param = {
    'num_classes': num_classes,
    'background_label_id': background_label_id,
    'overlap_threshold': 0.5,
    'evaluate_difficult_gt': False,
    'name_size_file': name_size_file,
}


# Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
def AddExtraLayers(net, use_batchnorm=True):
    use_relu = True

    # Add additional convolutional layers.
    # 19 x 19
    last_layer = net.keys()[-1]

    # 10 x 10
    from_layer = last_layer
    out_layer = "{}/conv1_1".format(last_layer)
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1)
    from_layer = out_layer

    out_layer = "{}/conv1_2".format(last_layer)
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2)
    from_layer = out_layer

    for i in xrange(2, 4):
        out_layer = "{}/conv{}_1".format(last_layer, i)
        ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1)
        from_layer = out_layer

        out_layer = "{}/conv{}_2".format(last_layer, i)
        ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2)
        from_layer = out_layer

    # Add global pooling layer.
    name = net.keys()[-1]
    net.pool6 = L.Pooling(net[name], pool=P.Pooling.AVE, global_pooling=True)

    return net
