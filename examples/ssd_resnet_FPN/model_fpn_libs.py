import os

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def UnpackVariable(var, num):
  assert len > 0
  if type(var) is list and len(var) == num:
    return var
  else:
    ret = []
    if type(var) is list:
      assert len(var) == 1
      for i in xrange(0, num):
        ret.append(var[0])
    else:
      for i in xrange(0, num):
        ret.append(var)
    return ret


def ConvLayer(net, from_layer, out_layer, num_output,
                kernel_size, pad, stride, lr_mult=1,
                conv_prefix='', conv_postfix='',
                group=1):
    kwargs = {
        'param': [
            dict(lr_mult=1),
            dict(lr_mult=2 * lr_mult)],
        'weight_filler': dict(type='gaussian', std=0.001),
        'bias_filler': dict(type='constant', value=0)
    }

    conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
    [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
    [pad_h, pad_w] = UnpackVariable(pad, 2)
    [stride_h, stride_w] = UnpackVariable(stride, 2)
    if kernel_h == kernel_w:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
                                         kernel_size=kernel_h, pad=pad_h, stride=stride_h, group=group, **kwargs)
    else:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
                                         kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
                                         stride_h=stride_h, stride_w=stride_w, group=group, **kwargs)

def DeConvLayer(net, from_layer, out_layer, num_output,
    kernel_size, pad, stride, lr_mult=1,
    conv_prefix='', conv_postfix='',
    group = 256):

  kwargs = {
    'param': [
        dict(lr_mult=0, decay_mult=0)
        ]
    # 'weight_filler': dict(type='bilinear'),
    # 'bias_term': False
    }

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    net[conv_name] = L.Deconvolution(net[from_layer], convolution_param=dict(num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, group=group, weight_filler=dict(type='bilinear'),
        bias_term=False), **kwargs)
  else:
    net[conv_name] = L.Deconvolution(net[from_layer], convolution_param=dict(num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, weight_filler=dict(type='bilinear'), bias_term=False), **kwargs)

def EltWiseLyaer(net, from_layers, out_layer, operation = 'sum',
    ew_prefix='', ew_postfix=''):
    ewargs = {
        'operation': 1
    }
    eltw_name = '{}{}{}'.format(ew_prefix, out_layer, ew_postfix)
    net[eltw_name] = L.Eltwise(net[from_layers[0]], net[from_layers[1]], **ewargs)