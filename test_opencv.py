# -*-coding:utf-8-*-
from __future__ import print_function
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2


def example_network(batch_size):
    n = caffe.NetSpec()

    n.data, n.label = L.DummyData(name="dummy", shape=[dict(dim=[1, 3, 
                                                                 112, 112]),
                                                       dict(dim=[1])],
                                  data_filler=[dict(type="msra"),
                                               dict(type="constant", value=0)],
                                  transform_param=dict(scale=1.0 / 255.0,
                                                       mirror=True,
                                                       crop_size=64,
                                                       mean_value=[11, 123, 
                                                                   212]),
                                  ntop=2,
                                  include=dict(phase=getattr(caffe_pb2, 
                                                             "TEST"))
                                  )
    n.data, n.label = L.DummyData(name="dummy", shape=[dict(dim=[1, 3, 
                                                                 112, 112]),
                                                       dict(dim=[1])],
                                  data_filler=[dict(type="msra"),
                                               dict(type="constant", value=0)],
                                  transform_param=dict(scale=1.0 / 255.0,
                                                       mirror=True,
                                                       crop_size=64,
                                                       mean_value=[11, 123, 
                                                                   212]),
                                  ntop=2,
                                  include=dict(phase=getattr(caffe_pb2, 
                                                             "TRAIN"))
                                  )
    n.conv1 = L.Convolution(n.data, name="Convolution1", kernel_size=3,
                            stride=1, pad=1, num_output=16,
                            param=[dict(lr_mult=1, decay_mult=1),
                                   dict(lr_mult=2, decay_mult=0)],
                            bias_filler=dict(type="constant", value=0),
                            weight_filler=dict(type="gaussian", std=0.01),
                            engine=P.Convolution.CUDNN, ntop=1)
    n.pool1 = L.Pooling(n.conv1, name="Pooling1",
                        pool=P.Pooling.AVE, global_pooling=True,
                        engine=P.Pooling.CUDNN, ntop=1)
    n.loss = L.Python(n.pool1, n.label, name="Loss1",
                      python_param=dict(
                          module='python_accuracy',
                          layer='PythonAccuracy',
                          param_str='{ "param_name": param_value }'),
                      ntop=1,)

    return n.to_proto()


def make_net(tgt_file):
    with open(tgt_file, 'w') as f:
        print('name: "resnet_cifar10"', file=f)
        print(example_network(10), file=f)


if __name__ == '__main__':
    tgt_file = 'D:/MyNet/test_1.prototxt'
    make_net(tgt_file)
    import django
    print(django.VERSION)
    