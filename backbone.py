import sys
sys.path.insert(0,'../incubator-mxnet/python')
from mxnet import nd
from mxnet.gluon import nn

def conv_block(num_filters, loops):
    "define conv block"
    out = nn.Sequential()
    for _ in range(loops):
        out.add(nn.Conv2D(num_filters, 3, strides=1, padding=1))
        out.add(nn.BatchNorm(in_channels=num_filters))
        out.add(nn.Activation('relu'))
    out.add(nn.MaxPool2D(2))
    return out

def backbone(filters):
    "VGG style backbond"
    out = nn.Sequential()
    for filt in filters:
        out.add(conv_block(filt[0], filt[1]))
    return out

if __name__ == '__main__':
    blk = conv_block(10, 2)
    blk.initialize()
    x = nd.zeros((2, 3, 20, 20))
    print('Before', x.shape, 'after', blk(x).shape)
    print blk
    print '*' * 20
    layer_setup = [
        (64, 2),
        (128, 2),
        (256, 3),
        (512, 3),
        (512, 3),
        ]
    x = nd.zeros((2, 3,224, 224))
    bb = backbone(layer_setup)
    bb.initialize()
    print('Before', x.shape, 'after', bb(x).shape)
    print bb

