import caffe
from caffe import params as P
from caffe import layers as L

class InceptionV2:
	""" This class builds the InceptionV2 body structure
	"""
	def InceptionNetBody(self, net, from_layer, freeze_layers=[]):
		assert from_layer in net.keys()
		cn_params = {'param':[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]}
		
		net.conv1_7x7_s2 = L.Convolution(net[from_layer], num_output=64, pad=3, kernel_size=7, stride=2, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.conv1_7x7_s2_bn = L.BatchNorm(net.conv1_7x7_s2, name='conv1_7x7_s2_bn', use_global_stats=False, in_place=True)
		net.conv1_7x7_relu = L.ReLU(net.conv1_7x7_s2, in_place=True)
		net.pool1_3x3_s2 = L.Pooling(net.conv1_7x7_s2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
		
		net.conv2_3x3_reduce = L.Convolution(net.pool1_3x3_s2, num_output=64, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.conv2_3x3_reduce_bn = L.BatchNorm(net.conv2_3x3_reduce, name='conv2_3x3_reduce_bn', use_global_stats=False, in_place=True)
		net.conv2_3x3_reduce_relu = L.ReLU(net.conv2_3x3_reduce, in_place=True)
		
		net.conv2_3x3 = L.Convolution(net.conv2_3x3_reduce, num_output=192, pad=1, kernel_size=3, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.conv2_3x3_bn = L.BatchNorm(net.conv2_3x3, use_global_stats=False, in_place=True)
		net.conv2_3x3_relu = L.ReLU(net.conv2_3x3, in_place=True)
		net.pool2_3x3_s2 = L.Pooling(net.conv2_3x3, pool=P.Pooling.MAX, kernel_size=3, stride=2)
		
		#build first tower
		net.inception_3a_1x1 = L.Convolution(net.pool2_3x3_s2, num_output=64, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_3a_1x1_bn = L.BatchNorm(net.inception_3a_1x1, in_place=True, use_global_stats=False)
		net.inception_3a_relu_1x1 = L.ReLU(net.inception_3a_1x1, in_place=True)
		
		net.inception_3a_3x3_reduce = L.Convolution(net.pool2_3x3_s2, num_output=96, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_3a_3x3_reduce_bn = L.BatchNorm(net.inception_3a_3x3_reduce, in_place=True, use_global_stats=False)
		net.inception_3a_relu_3x3_reduce = L.ReLU(net.inception_3a_3x3_reduce, in_place=True)
		net.inception_3a_3x3 = L.Convolution(net.inception_3a_relu_3x3_reduce, num_output=128, pad=1, kernel_size=3, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_3a_3x3_bn = L.BatchNorm(net.inception_3a_3x3, in_place=True, use_global_stats=False)
		net.inception_3a_relu_3x3 = L.ReLU(net.inception_3a_3x3, in_place=True)  
		
		net.inception_3a_5x5_reduce = L.Convolution(net.pool2_3x3_s2, num_output=16, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_3a_5x5_reduce_bn = L.BatchNorm(net.inception_3a_5x5_reduce, in_place=True, use_global_stats=False)
		net.inception_3a_relu_5x5_reduce = L.ReLU(net.inception_3a_5x5_reduce, in_place=True)
		net.inception_3a_5x5 = L.Convolution(net.inception_3a_relu_5x5_reduce, num_output=32, pad=2, kernel_size=5, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_3a_5x5_bn = L.BatchNorm(net.inception_3a_5x5, in_place=True, use_global_stats=False)
		net.inception_3a_relu_5x5 = L.ReLU(net.inception_3a_5x5, in_place=True)
		
		net.inception_3a_pool = L.Pooling(net.pool2_3x3_s2, pool=P.Pooling.MAX, kernel_size=3, pad=1, stride=1)
		net.inception_3a_pool_proj = L.Convolution(net.inception_3a_pool, num_output=32, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_3a_pool_proj_bn = L.BatchNorm(net.inception_3a_pool_proj, in_place=True, use_global_stats=False)
		net.inception_3a_relu_pool_proj = L.ReLU(net.inception_3a_pool_proj, in_place=True)
		bottom_layers = [net.inception_3a_1x1, net.inception_3a_3x3, net.inception_3a_5x5, net.inception_3a_pool_proj]
		net.inception_3a_output = L.Concat(*bottom_layers)
		
		# build second tower
		net.inception_3b_1x1 = L.Convolution(net.inception_3a_output, num_output=128, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_3b_1x1_bn = L.BatchNorm(net.inception_3b_1x1, in_place=True, use_global_stats=False)
		net.inception_3b_relu_1x1 = L.ReLU(net.inception_3b_1x1, in_place=True)
		
		net.inception_3b_3x3_reduce = L.Convolution(net.inception_3a_output, num_output=128, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_3b_3x3_reduce_bn = L.BatchNorm(net.inception_3b_3x3_reduce, in_place=True, use_global_stats=False)
		net.inception_3b_relu_3x3_reduce = L.ReLU(net.inception_3b_3x3_reduce, in_place=True)
		net.inception_3b_3x3 = L.Convolution(net.inception_3b_relu_3x3_reduce, num_output=192, pad=1, kernel_size=3, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_3b_3x3_bn = L.BatchNorm(net.inception_3b_3x3, in_place=True, use_global_stats=False)
		net.inception_3b_relu_3x3 = L.ReLU(net.inception_3b_3x3, in_place=True)  
		
		net.inception_3b_5x5_reduce = L.Convolution(net.inception_3a_output, num_output=32, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_3b_5x5_reduce_bn = L.BatchNorm(net.inception_3b_5x5_reduce, in_place=True, use_global_stats=False)
		net.inception_3b_relu_5x5_reduce = L.ReLU(net.inception_3b_5x5_reduce, in_place=True)
		net.inception_3b_5x5 = L.Convolution(net.inception_3b_relu_5x5_reduce, num_output=96, pad=2, kernel_size=5, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_3b_5x5_bn = L.BatchNorm(net.inception_3b_5x5, in_place=True, use_global_stats=False)
		net.inception_3b_relu_5x5 = L.ReLU(net.inception_3b_5x5, in_place=True)
		
		net.inception_3b_pool = L.Pooling(net.inception_3a_output, pool=P.Pooling.MAX, kernel_size=3, pad=1, stride=1)
		net.inception_3b_pool_proj = L.Convolution(net.inception_3b_pool, num_output=64, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_3b_pool_proj_bn = L.BatchNorm(net.inception_3b_pool_proj, in_place=True, use_global_stats=False)
		net.inception_3b_relu_pool_proj = L.ReLU(net.inception_3b_pool_proj, in_place=True)
		bottom_layers = [net.inception_3b_1x1, net.inception_3b_3x3, net.inception_3b_5x5, net.inception_3b_pool_proj]
		net.inception_3b_output = L.Concat(*bottom_layers)
		net.pool3_3x3_s2 = L.Pooling(net.inception_3b_output, pool=P.Pooling.MAX, kernel_size=3, stride=2)
		
		# build third tower
		net.inception_4a_1x1 = L.Convolution(net.pool3_3x3_s2, num_output=192, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4a_1x1_bn = L.BatchNorm(net.inception_4a_1x1, in_place=True, use_global_stats=False)
		net.inception_4a_relu_1x1 = L.ReLU(net.inception_4a_1x1, in_place=True)
		
		net.inception_4a_3x3_reduce = L.Convolution(net.pool3_3x3_s2, num_output=96, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4a_3x3_reduce_bn = L.BatchNorm(net.inception_4a_3x3_reduce, in_place=True, use_global_stats=False)
		net.inception_4a_relu_3x3_reduce = L.ReLU(net.inception_4a_3x3_reduce, in_place=True)
		net.inception_4a_3x3 = L.Convolution(net.inception_4a_relu_3x3_reduce, num_output=208, pad=1, kernel_size=3, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4a_3x3_bn = L.BatchNorm(net.inception_4a_3x3, in_place=True, use_global_stats=False)
		net.inception_4a_relu_3x3 = L.ReLU(net.inception_4a_3x3, in_place=True)  
		
		net.inception_4a_5x5_reduce = L.Convolution(net.pool3_3x3_s2, num_output=16, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4a_5x5_reduce_bn = L.BatchNorm(net.inception_4a_5x5_reduce, in_place=True, use_global_stats=False)
		net.inception_4a_relu_5x5_reduce = L.ReLU(net.inception_4a_5x5_reduce, in_place=True)
		net.inception_4a_5x5 = L.Convolution(net.inception_4a_relu_5x5_reduce, num_output=48, pad=2, kernel_size=5, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4a_5x5_bn = L.BatchNorm(net.inception_4a_5x5, in_place=True, use_global_stats=False)
		net.inception_4a_relu_5x5 = L.ReLU(net.inception_4a_5x5, in_place=True)
		
		net.inception_4a_pool = L.Pooling(net.pool3_3x3_s2, pool=P.Pooling.MAX, kernel_size=3, pad=1, stride=1)
		net.inception_4a_pool_proj = L.Convolution(net.inception_4a_pool, num_output=64, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4a_pool_proj_bn = L.BatchNorm(net.inception_4a_pool_proj, in_place=True, use_global_stats=False)
		net.inception_4a_relu_pool_proj = L.ReLU(net.inception_4a_pool_proj, in_place=True)
		bottom_layers = [net.inception_4a_1x1, net.inception_4a_3x3, net.inception_4a_5x5, net.inception_4a_pool_proj]
		net.inception_4a_output = L.Concat(*bottom_layers)
		
		# build fourth tower
		net.inception_4b_1x1 = L.Convolution(net.inception_4a_output, num_output=160, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4b_1x1_bn = L.BatchNorm(net.inception_4b_1x1, in_place=True, use_global_stats=False)
		net.inception_4b_relu_1x1 = L.ReLU(net.inception_4b_1x1, in_place=True)
		
		net.inception_4b_3x3_reduce = L.Convolution(net.inception_4a_output, num_output=112, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4b_3x3_reduce_bn = L.BatchNorm(net.inception_4b_3x3_reduce, in_place=True, use_global_stats=False)
		net.inception_4b_relu_3x3_reduce = L.ReLU(net.inception_4b_3x3_reduce, in_place=True)
		net.inception_4b_3x3 = L.Convolution(net.inception_4b_relu_3x3_reduce, num_output=224, pad=1, kernel_size=3, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4b_3x3_bn = L.BatchNorm(net.inception_4b_3x3, in_place=True, use_global_stats=False)
		net.inception_4b_relu_3x3 = L.ReLU(net.inception_4b_3x3, in_place=True)  
		
		net.inception_4b_5x5_reduce = L.Convolution(net.inception_4a_output, num_output=24, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4b_5x5_reduce_bn = L.BatchNorm(net.inception_4b_5x5_reduce, in_place=True, use_global_stats=False)
		net.inception_4b_relu_5x5_reduce = L.ReLU(net.inception_4b_5x5_reduce, in_place=True)
		net.inception_4b_5x5 = L.Convolution(net.inception_4b_relu_5x5_reduce, num_output=64, pad=2, kernel_size=5, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4b_5x5_bn = L.BatchNorm(net.inception_4b_5x5, in_place=True, use_global_stats=False)
		net.inception_4b_relu_5x5 = L.ReLU(net.inception_4b_5x5, in_place=True)
		
		net.inception_4b_pool = L.Pooling(net.inception_4a_output, pool=P.Pooling.MAX, kernel_size=3, pad=1, stride=1)
		net.inception_4b_pool_proj = L.Convolution(net.inception_4b_pool, num_output=64, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4b_pool_proj_bn = L.BatchNorm(net.inception_4b_pool_proj, in_place=True, use_global_stats=False)
		net.inception_4b_relu_pool_proj = L.ReLU(net.inception_4b_pool_proj, in_place=True)
		bottom_layers = [net.inception_4b_1x1, net.inception_4b_3x3, net.inception_4b_5x5, net.inception_4b_pool_proj]
		net.inception_4b_output = L.Concat(*bottom_layers)
		bottom_layers = [net.inception_4b_1x1, net.inception_4b_3x3, net.inception_4b_5x5, net.inception_4b_pool_proj]	
		net.inception_4b_output = L.Concat(*bottom_layers)
		
		# build fifth tower
		net.inception_4c_1x1 = L.Convolution(net.inception_4b_output, num_output=128, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4c_1x1_bn = L.BatchNorm(net.inception_4c_1x1, in_place=True, use_global_stats=False)
		net.inception_4c_relu_1x1 = L.ReLU(net.inception_4c_1x1, in_place=True)
		
		net.inception_4c_3x3_reduce = L.Convolution(net.inception_4b_output, num_output=128, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4c_3x3_reduce_bn = L.BatchNorm(net.inception_4c_3x3_reduce, in_place=True, use_global_stats=False)
		net.inception_4c_relu_3x3_reduce = L.ReLU(net.inception_4c_3x3_reduce, in_place=True)
		net.inception_4c_3x3 = L.Convolution(net.inception_4c_relu_3x3_reduce, num_output=256, pad=1, kernel_size=3, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4c_3x3_bn = L.BatchNorm(net.inception_4c_3x3, in_place=True, use_global_stats=False)
		net.inception_4c_relu_3x3 = L.ReLU(net.inception_4c_3x3, in_place=True)  
		
		net.inception_4c_5x5_reduce = L.Convolution(net.inception_4b_output, num_output=24, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4c_5x5_reduce_bn = L.BatchNorm(net.inception_4c_5x5_reduce, in_place=True, use_global_stats=False)
		net.inception_4c_relu_5x5_reduce = L.ReLU(net.inception_4c_5x5_reduce, in_place=True)
		net.inception_4c_5x5 = L.Convolution(net.inception_4c_relu_5x5_reduce, num_output=64, pad=2, kernel_size=5, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4c_5x5_bn = L.BatchNorm(net.inception_4c_5x5, in_place=True, use_global_stats=False)
		net.inception_4c_relu_5x5 = L.ReLU(net.inception_4c_5x5, in_place=True)
		
		net.inception_4c_pool = L.Pooling(net.inception_4b_output, pool=P.Pooling.MAX, kernel_size=3, pad=1, stride=1)
		net.inception_4c_pool_proj = L.Convolution(net.inception_4c_pool, num_output=64, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4c_pool_proj_bn = L.BatchNorm(net.inception_4c_pool_proj, in_place=True, use_global_stats=False)
		net.inception_4c_relu_pool_proj = L.ReLU(net.inception_4c_pool_proj, in_place=True)
		bottom_layers = [net.inception_4c_1x1, net.inception_4c_3x3, net.inception_4c_5x5, net.inception_4c_pool_proj]
		net.inception_4c_output = L.Concat(*bottom_layers)
		bottom_layers = [net.inception_4c_1x1, net.inception_4c_3x3, net.inception_4c_5x5, net.inception_4c_pool_proj]	
		net.inception_4c_output = L.Concat(*bottom_layers)
		
		# build sixth tower
		net.inception_4d_1x1 = L.Convolution(net.inception_4c_output, num_output=112, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4d_1x1_bn = L.BatchNorm(net.inception_4d_1x1, in_place=True, use_global_stats=False)
		net.inception_4d_relu_1x1 = L.ReLU(net.inception_4d_1x1, in_place=True)
		
		net.inception_4d_3x3_reduce = L.Convolution(net.inception_4c_output, num_output=144, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4d_3x3_reduce_bn = L.BatchNorm(net.inception_4d_3x3_reduce, in_place=True, use_global_stats=False)
		net.inception_4d_relu_3x3_reduce = L.ReLU(net.inception_4d_3x3_reduce, in_place=True)
		net.inception_4d_3x3 = L.Convolution(net.inception_4d_relu_3x3_reduce, num_output=288, pad=1, kernel_size=3, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4d_3x3_bn = L.BatchNorm(net.inception_4d_3x3, in_place=True, use_global_stats=False)
		net.inception_4d_relu_3x3 = L.ReLU(net.inception_4d_3x3, in_place=True)  
		
		net.inception_4d_5x5_reduce = L.Convolution(net.inception_4c_output, num_output=32, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4d_5x5_reduce_bn = L.BatchNorm(net.inception_4d_5x5_reduce, in_place=True, use_global_stats=False)
		net.inception_4d_relu_5x5_reduce = L.ReLU(net.inception_4d_5x5_reduce, in_place=True)
		net.inception_4d_5x5 = L.Convolution(net.inception_4d_relu_5x5_reduce, num_output=64, pad=2, kernel_size=5, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4d_5x5_bn = L.BatchNorm(net.inception_4d_5x5, in_place=True, use_global_stats=False)
		net.inception_4d_relu_5x5 = L.ReLU(net.inception_4d_5x5, in_place=True)
		
		net.inception_4d_pool = L.Pooling(net.inception_4c_output, pool=P.Pooling.MAX, kernel_size=3, pad=1, stride=1)
		net.inception_4d_pool_proj = L.Convolution(net.inception_4d_pool, num_output=64, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4d_pool_proj_bn = L.BatchNorm(net.inception_4d_pool_proj, in_place=True, use_global_stats=False)
		net.inception_4d_relu_pool_proj = L.ReLU(net.inception_4d_pool_proj, in_place=True)
		bottom_layers = [net.inception_4d_1x1, net.inception_4d_3x3, net.inception_4d_5x5, net.inception_4d_pool_proj]
		net.inception_4d_output = L.Concat(*bottom_layers)
		bottom_layers = [net.inception_4d_1x1, net.inception_4d_3x3, net.inception_4d_5x5, net.inception_4d_pool_proj]	
		net.inception_4d_output = L.Concat(*bottom_layers)

		# build seventh tower
		net.inception_4e_1x1 = L.Convolution(net.inception_4d_output, num_output=256, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4e_1x1_bn = L.BatchNorm(net.inception_4e_1x1, in_place=True, use_global_stats=False)
		net.inception_4e_relu_1x1 = L.ReLU(net.inception_4e_1x1, in_place=True)
		
		net.inception_4e_3x3_reduce = L.Convolution(net.inception_4d_output, num_output=160, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4e_3x3_reduce_bn = L.BatchNorm(net.inception_4e_3x3_reduce, in_place=True, use_global_stats=False)
		net.inception_4e_relu_3x3_reduce = L.ReLU(net.inception_4e_3x3_reduce, in_place=True)
		net.inception_4e_3x3 = L.Convolution(net.inception_4e_relu_3x3_reduce, num_output=320, pad=1, kernel_size=3, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4e_3x3_bn = L.BatchNorm(net.inception_4e_3x3, in_place=True, use_global_stats=False)
		net.inception_4e_relu_3x3 = L.ReLU(net.inception_4e_3x3, in_place=True)  
		
		net.inception_4e_5x5_reduce = L.Convolution(net.inception_4d_output, num_output=32, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4e_5x5_reduce_bn = L.BatchNorm(net.inception_4e_5x5_reduce, in_place=True, use_global_stats=False)
		net.inception_4e_relu_5x5_reduce = L.ReLU(net.inception_4e_5x5_reduce, in_place=True)
		net.inception_4e_5x5 = L.Convolution(net.inception_4e_relu_5x5_reduce, num_output=128, pad=2, kernel_size=5, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4e_5x5_bn = L.BatchNorm(net.inception_4e_5x5, in_place=True, use_global_stats=False)
		net.inception_4e_relu_5x5 = L.ReLU(net.inception_4e_5x5, in_place=True)
		
		net.inception_4e_pool = L.Pooling(net.inception_4d_output, pool=P.Pooling.MAX, kernel_size=3, pad=1, stride=1)
		net.inception_4e_pool_proj = L.Convolution(net.inception_4e_pool, num_output=128, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_4e_pool_proj_bn = L.BatchNorm(net.inception_4e_pool_proj, in_place=True, use_global_stats=False)
		net.inception_4e_relu_pool_proj = L.ReLU(net.inception_4e_pool_proj, in_place=True)
		bottom_layers = [net.inception_4e_1x1, net.inception_4e_3x3, net.inception_4e_5x5, net.inception_4e_pool_proj]
		net.inception_4e_output = L.Concat(*bottom_layers)
		bottom_layers = [net.inception_4e_1x1, net.inception_4e_3x3, net.inception_4e_5x5, net.inception_4e_pool_proj]	
		net.inception_4e_output = L.Concat(*bottom_layers)	
		
		# build eight tower
		net.pool4_3x3_s2 = L.Pooling(net.inception_4e_output, pool=P.Pooling.MAX, kernel_size=3, stride=2)
		net.inception_5a_1x1 = L.Convolution(net.pool4_3x3_s2, num_output=256, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_5a_1x1_bn = L.BatchNorm(net.inception_5a_1x1, in_place=True, use_global_stats=False)
		net.inception_5a_relu_1x1 = L.ReLU(net.inception_5a_1x1, in_place=True)
		
		net.inception_5a_3x3_reduce = L.Convolution(net.pool4_3x3_s2, num_output=160, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_5a_3x3_reduce_bn = L.BatchNorm(net.inception_5a_3x3_reduce, in_place=True, use_global_stats=False)
		net.inception_5a_relu_3x3_reduce = L.ReLU(net.inception_5a_3x3_reduce, in_place=True)
		net.inception_5a_3x3 = L.Convolution(net.inception_5a_relu_3x3_reduce, num_output=320, pad=1, kernel_size=3, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_5a_3x3_bn = L.BatchNorm(net.inception_5a_3x3, in_place=True, use_global_stats=False)
		net.inception_5a_relu_3x3 = L.ReLU(net.inception_5a_3x3, in_place=True)  
		
		net.inception_5a_5x5_reduce = L.Convolution(net.pool4_3x3_s2, num_output=32, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_5a_5x5_reduce_bn = L.BatchNorm(net.inception_5a_5x5_reduce, in_place=True, use_global_stats=False)
		net.inception_5a_relu_5x5_reduce = L.ReLU(net.inception_5a_5x5_reduce, in_place=True)
		net.inception_5a_5x5 = L.Convolution(net.inception_5a_relu_5x5_reduce, num_output=128, pad=2, kernel_size=5, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_5a_5x5_bn = L.BatchNorm(net.inception_5a_5x5, in_place=True, use_global_stats=False)
		net.inception_5a_relu_5x5 = L.ReLU(net.inception_5a_5x5, in_place=True)
		
		net.inception_5a_pool = L.Pooling(net.pool4_3x3_s2, pool=P.Pooling.MAX, kernel_size=3, pad=1, stride=1)
		net.inception_5a_pool_proj = L.Convolution(net.inception_5a_pool, num_output=128, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_5a_pool_proj_bn = L.BatchNorm(net.inception_5a_pool_proj, in_place=True, use_global_stats=False)
		net.inception_5a_relu_pool_proj = L.ReLU(net.inception_5a_pool_proj, in_place=True)
		bottom_layers = [net.inception_5a_1x1, net.inception_5a_3x3, net.inception_5a_5x5, net.inception_5a_pool_proj]
		net.inception_5a_output = L.Concat(*bottom_layers)
		bottom_layers = [net.inception_5a_1x1, net.inception_5a_3x3, net.inception_5a_5x5, net.inception_5a_pool_proj]	
		net.inception_5a_output = L.Concat(*bottom_layers)

		# build ninth tower
		net.inception_5b_1x1 = L.Convolution(net.inception_5a_output, num_output=384, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_5b_1x1_bn = L.BatchNorm(net.inception_5b_1x1, in_place=True, use_global_stats=False)
		net.inception_5b_relu_1x1 = L.ReLU(net.inception_5b_1x1, in_place=True)
		
		net.inception_5b_3x3_reduce = L.Convolution(net.inception_5a_output, num_output=192, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_5b_3x3_reduce_bn = L.BatchNorm(net.inception_5b_3x3_reduce, in_place=True, use_global_stats=False)
		net.inception_5b_relu_3x3_reduce = L.ReLU(net.inception_5b_3x3_reduce, in_place=True)
		net.inception_5b_3x3 = L.Convolution(net.inception_5b_relu_3x3_reduce, num_output=384, pad=1, kernel_size=3, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_5b_3x3_bn = L.BatchNorm(net.inception_5b_3x3, in_place=True, use_global_stats=False)
		net.inception_5b_relu_3x3 = L.ReLU(net.inception_5b_3x3, in_place=True)  
		
		net.inception_5b_5x5_reduce = L.Convolution(net.inception_5a_output, num_output=48, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_5b_5x5_reduce_bn = L.BatchNorm(net.inception_5b_5x5_reduce, in_place=True, use_global_stats=False)
		net.inception_5b_relu_5x5_reduce = L.ReLU(net.inception_5b_5x5_reduce, in_place=True)
		net.inception_5b_5x5 = L.Convolution(net.inception_5b_relu_5x5_reduce, num_output=128, pad=2, kernel_size=5, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_5b_5x5_bn = L.BatchNorm(net.inception_5b_5x5, in_place=True, use_global_stats=False)
		net.inception_5b_relu_5x5 = L.ReLU(net.inception_5b_5x5, in_place=True)
		
		net.inception_5b_pool = L.Pooling(net.inception_5a_output, pool=P.Pooling.MAX, kernel_size=3, pad=1, stride=1)
		net.inception_5b_pool_proj = L.Convolution(net.inception_5b_pool, num_output=128, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_5b_pool_proj_bn = L.BatchNorm(net.inception_5b_pool_proj, in_place=True, use_global_stats=False)
		net.inception_5b_relu_pool_proj = L.ReLU(net.inception_5b_pool_proj, in_place=True)
		bottom_layers = [net.inception_5b_1x1, net.inception_5b_3x3, net.inception_5b_5x5, net.inception_5b_pool_proj]
		net.inception_5b_output = L.Concat(*bottom_layers)

		# build tenth tower
		net.pool5_3x3_s2 = L.Pooling(net.inception_5b_output, pool=P.Pooling.MAX, kernel_size=3, stride=2)
		net.inception_6a_1x1 = L.Convolution(net.pool5_3x3_s2, num_output=384, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_6a_1x1_bn = L.BatchNorm(net.inception_6a_1x1, in_place=True, use_global_stats=False)
		net.inception_6a_relu_1x1 = L.ReLU(net.inception_6a_1x1, in_place=True)
		
		net.inception_6a_3x3_reduce = L.Convolution(net.pool5_3x3_s2, num_output=192, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_6a_3x3_reduce_bn = L.BatchNorm(net.inception_6a_3x3_reduce, in_place=True, use_global_stats=False)
		net.inception_6a_relu_3x3_reduce = L.ReLU(net.inception_6a_3x3_reduce, in_place=True)
		net.inception_6a_3x3 = L.Convolution(net.inception_6a_relu_3x3_reduce, num_output=384, pad=1, kernel_size=3, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_6a_3x3_bn = L.BatchNorm(net.inception_6a_3x3, in_place=True, use_global_stats=False)
		net.inception_6a_relu_3x3 = L.ReLU(net.inception_6a_3x3, in_place=True)  
		
		net.inception_6a_5x5_reduce = L.Convolution(net.pool5_3x3_s2, num_output=48, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_6a_5x5_reduce_bn = L.BatchNorm(net.inception_6a_5x5_reduce, in_place=True, use_global_stats=False)
		net.inception_6a_relu_5x5_reduce = L.ReLU(net.inception_6a_5x5_reduce, in_place=True)
		net.inception_6a_5x5 = L.Convolution(net.inception_6a_relu_5x5_reduce, num_output=128, pad=2, kernel_size=5, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_6a_5x5_bn = L.BatchNorm(net.inception_6a_5x5, in_place=True, use_global_stats=False)
		net.inception_6a_relu_5x5 = L.ReLU(net.inception_6a_5x5, in_place=True)

		net.inception_6a_pool = L.Pooling(net.pool5_3x3_s2, pool=P.Pooling.MAX, kernel_size=3, pad=1, stride=1)
		net.inception_6a_pool_proj = L.Convolution(net.inception_6a_pool, num_output=128, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_6a_pool_proj_bn = L.BatchNorm(net.inception_6a_pool_proj, in_place=True, use_global_stats=False)
		net.inception_6a_relu_pool_proj = L.ReLU(net.inception_6a_pool_proj, in_place=True)

		bottom_layers = [net.inception_6a_1x1, net.inception_6a_3x3, net.inception_6a_5x5, net.inception_6a_pool_proj]
		net.inception_6a_output = L.Concat(*bottom_layers)		

		# build eleventh tower
		net.pool6_3x3_s2 = L.Pooling(net.inception_6a_output, pool=P.Pooling.MAX, kernel_size=3, stride=2)
		net.inception_7a_1x1 = L.Convolution(net.pool6_3x3_s2, num_output=384, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_7a_1x1_bn = L.BatchNorm(net.inception_7a_1x1, in_place=True, use_global_stats=False)
		net.inception_7a_relu_1x1 = L.ReLU(net.inception_7a_1x1, in_place=True)
		
		net.inception_7a_3x3_reduce = L.Convolution(net.pool6_3x3_s2, num_output=192, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_7a_3x3_reduce_bn = L.BatchNorm(net.inception_7a_3x3_reduce, in_place=True, use_global_stats=False)
		net.inception_7a_relu_3x3_reduce = L.ReLU(net.inception_7a_3x3_reduce, in_place=True)
		net.inception_7a_3x3 = L.Convolution(net.inception_7a_relu_3x3_reduce, num_output=384, pad=1, kernel_size=3, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_7a_3x3_bn = L.BatchNorm(net.inception_7a_3x3, in_place=True, use_global_stats=False)
		net.inception_7a_relu_3x3 = L.ReLU(net.inception_7a_3x3, in_place=True)  
		
		net.inception_7a_5x5_reduce = L.Convolution(net.pool6_3x3_s2, num_output=48, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_7a_5x5_reduce_bn = L.BatchNorm(net.inception_7a_5x5_reduce, in_place=True, use_global_stats=False)
		net.inception_7a_relu_5x5_reduce = L.ReLU(net.inception_7a_5x5_reduce, in_place=True)
		net.inception_7a_5x5 = L.Convolution(net.inception_7a_relu_5x5_reduce, num_output=128, pad=2, kernel_size=5, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_7a_5x5_bn = L.BatchNorm(net.inception_7a_5x5, in_place=True, use_global_stats=False)
		net.inception_7a_relu_5x5 = L.ReLU(net.inception_7a_5x5, in_place=True)

		net.inception_7a_pool = L.Pooling(net.pool6_3x3_s2, pool=P.Pooling.MAX, kernel_size=3, pad=1, stride=1)
		net.inception_7a_pool_proj = L.Convolution(net.inception_7a_pool, num_output=128, pad=0, kernel_size=1, stride=1, 
											weight_filler=dict(type='xavier', std=1), bias_filler=dict(type='constant', value=0.2), **cn_params)
		net.inception_7a_pool_proj_bn = L.BatchNorm(net.inception_7a_pool_proj, in_place=True, use_global_stats=False)
		net.inception_7a_relu_pool_proj = L.ReLU(net.inception_7a_pool_proj, in_place=True)		
		bottom_layers = [net.inception_7a_1x1, net.inception_7a_3x3, net.inception_7a_5x5, net.inception_7a_pool_proj]
		net.inception_7a_output = L.Concat(*bottom_layers)	
		return net
