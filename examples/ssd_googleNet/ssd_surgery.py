import caffe
from caffe.model_libs import *
#import _init_paths

def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        normalizations=[], min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], share_location=True, flip=True, clip=True,
        inter_layer_depth=64, kernel_size=1, pad=0, conf_postfix='', loc_postfix=''):

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth > 0:
            inter_name = "{}_inter".format(from_layer)
            ConvBNLayer(net, from_layer, inter_name, use_bn=True, use_relu=False,
                num_output=inter_layer_depth, kernel_size=3, pad=1, stride=1)
            from_layer = inter_name
            inter_name = "{}_inter_2".format(from_layer)
            ConvBNLayer(net, from_layer, inter_name, use_bn=True, use_relu=False,
                num_output=inter_layer_depth, kernel_size=3, pad=1, stride=1)
            from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        if max_sizes and max_sizes[i]:
            num_priors_per_location = 2 + len(aspect_ratio)
        else:
            num_priors_per_location = 1 + len(aspect_ratio)
        if flip:
            num_priors_per_location += len(aspect_ratio)

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4
        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer(net, from_layer, name, use_bn=False, use_relu=False,
            num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes;
        ConvBNLayer(net, from_layer, name, use_bn=False, use_relu=False,
            num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        if max_sizes and max_sizes[i]:
            if aspect_ratio:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i], max_size=max_sizes[i],
                    aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
            else:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i], max_size=max_sizes[i],
                    clip=clip, variance=prior_variance)
        else:
            if aspect_ratio:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i],
                    aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
            else:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i],
                    clip=clip, variance=prior_variance)
        priorbox_layers.append(net[name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    return mbox_layers
	
