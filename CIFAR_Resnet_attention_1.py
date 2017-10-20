from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe

# helper function for building ResNet block structures
# The function below does computations: bottom--->conv--->BatchNorm


def conv_factory(bottom, ks, n_out, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=n_out, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1),
                                dict(lr_mult=2, decay_mult=0)],
                         bias_filler=dict(type='constant', value=0), weight_filler=dict(type='gaussian', std=0.01))
    batch_norm = L.BatchNorm(conv, in_place=True,
                             param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    return scale

# bottom--->conv--->BatchNorm--->ReLU


def conv_factory_relu(bottom, ks, n_out, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=n_out, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1),
                                dict(lr_mult=2, decay_mult=0)],
                         bias_filler=dict(type='constant', value=0), weight_filler=dict(type='gaussian', std=0.01))
    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(
        lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return relu


def deconv_factory_relu(bottom, n_out, ks=2, stride=2, pad=0):
    deconv = L.Deconvolution(bottom, convolution_param=dict(kernel_size=ks, stride=stride, num_output=n_out, pad=pad,
                             bias_term=False, weight_filler=dict(type='bilinear')),
                             param=[dict(lr_mult=0, decay_mult=0)])
    batch_norm = L.BatchNorm(deconv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(
        lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return relu

 #  Residual building block! Implements option (A) from Section 3.3. The input
 #  is passed through two 3x3 convolution layers. Currently this block only supports
 #  stride == 1 or stride == 2. When stride is 2, the block actually does pooling.
 #  Instead of simply doing pooling which may cause representational bottlneck as
 #  described in inception v3, here we use 2 parallel branches P && C and add them
 #  together. Note pooling branch may has less channels than convolution branch so we
 #  need to do zero-padding along channel dimension. And to the best knowledge of
 #  ours, we haven't found current caffe implementation that supports this operation.
 #  So later I'll give implementation in C++ and CUDA.


def residual_block(bottom, num_filters, stride=1):
    if stride == 1:
        conv1 = conv_factory_relu(bottom, 3, num_filters, 1, 1)
        conv2 = conv_factory(conv1, 3, num_filters, 1, 1)
        add = L.Eltwise(bottom, conv2, operation=P.Eltwise.SUM)
        return add
    elif stride == 2:
        conv1 = conv_factory_relu(bottom, 3, num_filters, 2, 1)
        conv2 = conv_factory(conv1, 3, num_filters, 1, 1)
        pool = L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=2, stride=2)
        pad = L.PadChannel(pool, num_channels_to_pad=num_filters / 2)
        add = L.Eltwise(conv2, pad, operation=P.Eltwise.SUM)
        return add
    else:
        raise Exception('Currently, stride must be either 1 or 2.')

# Generate resnet cifar10 train && test prototxt. n_size control number of layers.
# The total number of layers is  6 * n_size + 2. Here I don't know any of implementation
# which can contain simultaneously TRAIN && TEST phase.
# ==========================Note here==============================
# !!! SO YOU have to include TRAIN && TEST by your own AFTER you use the script to generate the prototxt !!!


def attention_module_1(data, res, num_filters=32):
    p_res = residual_block(res, num_filters)
    # --------> trunck branch
    t_res_1 = residual_block(p_res, num_filters)
    t_res_2 = residual_block(t_res_1, num_filters)

    # --------> soft mask branch
    # 1/2
    r_res_1 = residual_block(p_res, num_filters * 2, 2)
    r_res_2 = residual_block(r_res_1, num_filters * 2)
    # 1/4
    r_res_3 = residual_block(r_res_2, num_filters * 4, 2)
    r_res_4 = residual_block(r_res_3, num_filters * 4)
    r_res_5 = residual_block(r_res_4, num_filters * 4)
    # 1/2
    d_conv_1 = deconv_factory_relu(r_res_5, num_filters * 2)

    elt_1 = L.Eltwise(d_conv_1, r_res_2, operation=P.Eltwise.SUM)

    d_conv_2 = residual_block(elt_1, num_filters * 2)
    d_conv_3 = deconv_factory_relu(d_conv_2, num_filters)

    conv_1 = conv_factory_relu(d_conv_3, 1, num_filters)
    conv_2 = conv_factory_relu(conv_1, 1, num_filters)
    key_region = L.KeyPoint(
        data, conv_2, region_height=9, region_width=9, data_height=224, 
        data_width=224, key_scale=1.1)
    # key_region = conv_2
    sig = L.Sigmoid(key_region)

    # merge
    mul = L.Eltwise(sig, t_res_2, operation=P.Eltwise.PROD)
    add = L.Eltwise(mul, t_res_2, operation=P.Eltwise.SUM)

    add_1 = residual_block(add, num_filters)
    return add_1


def attention_module_2(data, res, num_filters=64):
    p_res = residual_block(res, num_filters)
    # --------> trunck branch
    t_res_1 = residual_block(p_res, num_filters)
    t_res_2 = residual_block(t_res_1, num_filters)

    # --------> soft mask branch
    # 1/2
    r_res_1 = residual_block(p_res, num_filters * 2, 2)
    r_res_2 = residual_block(r_res_1, num_filters * 2)
    r_res_3 = residual_block(r_res_2, num_filters * 2)
    d_conv_1 = deconv_factory_relu(r_res_3, num_filters * 2)

    conv_1 = conv_factory_relu(d_conv_1, 1, num_filters)
    conv_2 = conv_factory_relu(conv_1, 1, num_filters)
    key_region = L.KeyPoint(
        data, conv_2, region_height=5, region_width=5, data_height=224, 
        data_width=224, key_scale=1.2)
    # key_region = conv_2
    sig = L.Sigmoid(key_region)

    # merge
    mul = L.Eltwise(sig, t_res_2, operation=P.Eltwise.PROD)
    add = L.Eltwise(mul, t_res_2, operation=P.Eltwise.SUM)

    add_1 = residual_block(add, num_filters)
    return add_1


def resnet_cifar(train_lmdb, test_lmdb, mean_file, batch_size=100, n_size=3):
    data, label = L.Data(source=test_lmdb, backend=P.Data.LMDB, 
                         batch_size=batch_size, ntop=2,
                         transform_param=dict(mean_file=mean_file, crop_size=28), 
                         include=dict(phase=getattr(caffe_pb2, 'TEST')))

    # --------------> 16, 224, 224
    residual = conv_factory_relu(data, 3, 16, 1, 1)
    # --------------> 16, 224, 224    1st group
    for i in xrange(n_size):
        residual = residual_block(residual, 16)

    # start attention module
    # --------------> 32, 112, 112    2nd group
    residual = residual_block(residual, 32, 2)
    residual = attention_module_1(label, residual, 32)
    # --------------> 64, 56, 56
    residual = residual_block(residual, 64, 2)
    residual = attention_module_2(label, residual, 64)
    # end attention module

    # --------------> 128, 28, 28    3nd group
    residual = residual_block(residual, 128, 2)
    for i in xrange(n_size - 1):
        residual = residual_block(residual, 128)

    # -------------> end of residual
    global_pool = L.Pooling(residual, pool=P.Pooling.AVE, global_pooling=True)
    fc = L.InnerProduct(global_pool, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)], num_output=2,
                        bias_filler=dict(type='constant', value=0), weight_filler=dict(type='gaussian', std=0.01))
    loss = L.SoftmaxWithLoss(fc, label)
    acc = L.Accuracy(fc, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    return to_proto(loss, acc)


def make_net(tgt_file):
    with open(tgt_file, 'w') as f:
        print('name: "resnet_cifar10"', file=f)
        print(resnet_cifar('dataset/cifar10_train_lmdb', 'dataset/cifar10_test_lmdb',
                           'dataset/mean.proto', n_size=3), file=f)


if __name__ == '__main__':
    tgt_file = 'D:/MyNet/livdet2015/resnet_attention_train_test_1.prototxt'
    make_net(tgt_file)
