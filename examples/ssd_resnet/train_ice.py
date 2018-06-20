from __future__ import print_function
import shutil
import stat
import subprocess
import sys
from subprocess import Popen
import os.path as osp

#import _init_paths
import caffe
from caffe.model_libs import *
from caffe import params as P
from caffe import layers as L
from google.protobuf import text_format


from ssd_param_ice import *


# We assume this script is executed in the main directory

if __name__=='__main__':
    # model definition files.
    train_net_file = "{}/train.prototxt".format(save_dir)
    test_net_file = "{}/test.prototxt".format(save_dir)
    deploy_net_file = "{}/deploy.prototxt".format(save_dir)
    solver_file = "{}/solver.prototxt".format(save_dir)

    # Create train net.
    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,
                                                   train=True, output_label=True, label_map_file=label_map_file,
                                                   transform_param=train_transform_param, batch_sampler=batch_sampler)

    ResNet101Body(net, from_layer='data', use_pool5=False, use_dilation_conv5=True)

    # Use batch norm for the newly added layers.
    AddExtraLayers(net, use_batchnorm=True)

    # Don't use batch norm for location/confidence prediction layers.
    mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
                                     use_batchnorm=False, min_sizes=min_sizes, max_sizes=max_sizes,
                                     aspect_ratios=aspect_ratios, num_classes=num_classes,
                                     share_location=share_location,
                                     conf_postfix='_{}_{}'.format(priorbox_size_mode, num_classes),
                                     loc_postfix='_{}_{}'.format(priorbox_size_mode, num_classes),
                                     flip=flip, clip=clip, prior_variance=prior_variance, kernel_size=3, pad=1)

    # Create the MultiBoxLossLayer.
    name = "mbox_loss"
    mbox_layers.append(net.label)
    net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
                               loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                               propagate_down=[True, True, False, False])

    with open(train_net_file, 'w') as f:
        print('name: "{}_train"'.format(model_name), file=f)
        print(net.to_proto(), file=f)
    shutil.copy(train_net_file, job_dir)

    # Create test net.
    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
                                                   train=False, output_label=True, label_map_file=label_map_file,
                                                   transform_param=test_transform_param)

    ResNet101Body(net, from_layer='data', use_pool5=False, use_dilation_conv5=True)

    # Use batch norm for the newly added layers.
    AddExtraLayers(net, use_batchnorm=True)

    # Don't use batch norm for location/confidence prediction layers.
    mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
                                     use_batchnorm=False, min_sizes=min_sizes, max_sizes=max_sizes,
                                     aspect_ratios=aspect_ratios, num_classes=num_classes,
                                     share_location=share_location,
                                     conf_postfix='_{}_{}'.format(priorbox_size_mode, num_classes),
                                     loc_postfix='_{}_{}'.format(priorbox_size_mode, num_classes),
                                     flip=flip, clip=clip, prior_variance=prior_variance, kernel_size=3, pad=1)

    conf_name = "mbox_conf"
    if multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX:
        reshape_name = "{}_reshape".format(conf_name)
        net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
        softmax_name = "{}_softmax".format(conf_name)
        net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
        flatten_name = "{}_flatten".format(conf_name)
        net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
        mbox_layers[1] = net[flatten_name]
    elif multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
        sigmoid_name = "{}_sigmoid".format(conf_name)
        net[sigmoid_name] = L.Sigmoid(net[conf_name])
        mbox_layers[1] = net[sigmoid_name]

    net.detection_out = L.DetectionOutput(*mbox_layers,
                                          detection_output_param=det_out_param,
                                          include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
                                             detection_evaluate_param=det_eval_param,
                                             include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    with open(test_net_file, 'w') as f:
        print('name: "{}_test"'.format(model_name), file=f)
        print(net.to_proto(), file=f)
    shutil.copy(test_net_file, job_dir)

    # Create deploy net.
    # Remove the first and last layer from test net.
    deploy_net = net
    with open(deploy_net_file, 'w') as f:
        net_param = deploy_net.to_proto()
        # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
        del net_param.layer[0]
        del net_param.layer[-1]
        net_param.name = '{}_deploy'.format(model_name)
        net_param.input.extend(['data'])
        net_param.input_shape.extend([
            caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
        print(net_param, file=f)
    shutil.copy(deploy_net_file, job_dir)

    # Create solver.
    solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        test_net=[test_net_file],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

    with open(solver_file, 'w') as f:
        print(solver, file=f)
    shutil.copy(solver_file, job_dir)

    # Find most recent snapshot.
    max_iter = 0
    max_iter_snapshot_prefix = snapshot_prefix
    for file in os.listdir(snapshot_dir):
      if file.endswith(".solverstate"):
        basename = os.path.splitext(file)[0]
        #iter = int(basename.split("{}_iter_".format(model_name))[1])
        iter = int(basename.split("_iter_")[1])
        if iter > max_iter:
          max_iter = iter
          max_iter_snapshot_prefix = basename.split("_iter_")[0]

    train_src_param = '--weights="{}" '.format(pretrain_model)
    if resume_training:
      if max_iter > 0:
        train_src_param = '--snapshot="{}/{}_iter_{}.solverstate" '.format(snapshot_dir, max_iter_snapshot_prefix, max_iter)

    if remove_old_models:
        # Remove any snapshots smaller than max_iter.
        for file in os.listdir(snapshot_dir):
            if file.endswith(".solverstate"):
                basename = os.path.splitext(file)[0]
                iter = int(basename.split("_iter_")[1])
                if max_iter > iter:
                    os.remove("{}/{}".format(snapshot_dir, file))
            if file.endswith(".caffemodel"):
                basename = os.path.splitext(file)[0]
                iter = int(basename.split("_iter_")[1])
                if max_iter > iter:
                    os.remove("{}/{}".format(snapshot_dir, file))

    # Create job file.
    job_file = '{}/{}.ps1'.format(job_dir, model_name)
    with open(job_file, 'w') as f:
      f.write('windows\\tools\\release\\caffe.exe train ')
      f.write('--solver="{}" '.format(solver_file))
      f.write(train_src_param)
      # if solver_param['solver_mode'] == P.Solver.GPU:
      #   f.write('--gpu {} 2>&1 1>{}/{}.log'.format(gpus, job_dir, model_name))
      # else:
      #   f.write('2>&1 1>{}/{}.log'.format(job_dir, model_name))
      if solver_param['solver_mode'] == P.Solver.GPU:
          f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, job_dir, model_name))
      else:
          f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))
      f.close()

    # copy parameter file for reference
    shutil.copyfile('examples/ssd_resnet/ssd_param_ice.py', osp.join(snapshot_dir, 'ssd_param_ice.py'))
    # Run the job
    if run_soon:
      # p = Popen(job_file)
      # stdout, stderr = p.communicate()
      print('job_file: {}'.format(job_file))
      subprocess.call(["C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe", job_file], shell=True)