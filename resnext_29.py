from __future__ import print_function
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

# helper function for building ResNet block structures 
# The function below does computations: bottom--->conv--->BatchNorm
def conv_factory(bottom, ks, n_out, stride=1, pad=0,group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=n_out, pad=pad,group=group,
                         param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         bias_filler=dict(type='constant', value=0), weight_filler=dict(type='gaussian', std=0.01))
    batch_norm = L.BatchNorm(conv, in_place=True,
                             param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    return scale

# bottom--->conv--->BatchNorm--->ReLU
def conv_factory_relu(bottom, ks, n_out, stride=1, pad=0,group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=n_out, pad=pad,group=group,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         bias_filler=dict(type='constant', value=0), weight_filler=dict(type='gaussian', std=0.01))
    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
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
        conv1 = conv_factory_relu(bottom, 1, num_filters, 1, 0)
        conv2 = conv_factory_relu(conv1, 3, num_filters, 1, 1, 8)
        conv3 = conv_factory(conv2, 1, num_filters*4, 1, 0)
        add = L.Eltwise(bottom, conv3, operation=P.Eltwise.SUM)
        return add
    elif stride == 2:
        conv1 = conv_factory_relu(bottom, 1, num_filters, 1, 0)
        conv2 = conv_factory_relu(conv1, 3, num_filters, 2, 1, 8)
        conv3 = conv_factory(conv2, 1, num_filters*4, 1, 0)
        pool = L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=2, stride=2)
        pad = L.PadChannel(pool, num_channels_to_pad=2*num_filters)
        add = L.Eltwise(conv3, pad, operation=P.Eltwise.SUM)
        return add
    else:
        raise Exception('Currently, stride must be either 1 or 2.')

# Generate resnet cifar10 train && test prototxt. n_size control number of layers.
# The total number of layers is  9 * n_size + 2. Here I don't know any of implementation 
# which can contain simultaneously TRAIN && TEST phase. 
# ==========================Note here==============================
# !!! SO YOU have to include TRAIN && TEST by your own AFTER you use the script to generate the prototxt !!!
def resnet_cifar(train_lmdb, test_lmdb, mean_file, batch_size=100, n_size=3):
    data, label = L.Data(source=test_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
        transform_param=dict(mean_file=mean_file, crop_size=28), include=dict(phase=getattr(caffe_pb2, 'TEST')))
    residual = conv_factory_relu(data, 3, 64, 1, 1)
    # --------------> 16, 16, 64    1st group
    for i in xrange(n_size):
        residual = residual_block(residual, 16)

    # --------------> 32, 32, 128    2nd group
    residual = residual_block(residual, 32, 2)
    for i in xrange(n_size - 1):
        residual = residual_block(residual, 32)

    # --------------> 64, 64, 256      3rd group
    residual = residual_block(residual, 64, 2)
    for i in xrange(n_size - 1):
        residual = residual_block(residual, 64)

    # -------------> end of residual
    global_pool = L.Pooling(residual, pool=P.Pooling.AVE, global_pooling=True)
    fc = L.InnerProduct(global_pool, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)],num_output=2,
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
    tgt_file='D:/MyNet/livdet2015/resnext_29_train_test.prototxt'
    make_net(tgt_file)
