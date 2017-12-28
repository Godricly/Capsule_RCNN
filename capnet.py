import sys
sys.path.insert(0,'../incubator-mxnet/python')
from mxnet import init
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import initializer
from mxnet.contrib.ndarray import MultiBoxPrior
from backbone import backbone, conv_block
from conv_cap import PrimeConvCap, AdvConvCap

def model(num_anchors, num_classes, reg_dim, num_caps, num_filters):
    "define training net"
    layer_setup = [
        (16, 2),
        (32, 2),
        (64, 3),
        ]
    net = backbone(layer_setup)

    down_samples = nn.Sequential()
    down_samples.add(conv_block(128,2))
    down_samples.add(conv_block(128,2))
    down_samples.add(conv_block(128,2))

    class_preds = nn.Sequential()
    box_preds = nn.Sequential()
    cap_transforms = nn.Sequential()
    for scale in range(5):
        # cap_transforms.add(cap_transform(num_caps, num_filters))
        # class_preds.add(class_predictor(num_anchors, num_classes, num_caps, num_filters))
        # box_preds.add(box_predictor(num_anchors, reg_dim, num_caps, num_filters))
        box_preds.add(box_predictor(num_anchors))
        class_preds.add(class_predictor(num_anchors, num_classes))
    return net, down_samples, class_preds, box_preds, cap_transforms

def box_cap_predictor(num_anchors, dim, num_cap_in, num_filter_in):
    return AdvConvCap(num_anchors, dim, num_cap_in=num_cap_in, num_filter_in=num_filter_in)

def box_predictor(num_anchors):
    """return a layer to predict delta locations"""
    return nn.Conv2D(num_anchors * 4, 3, padding=1)

def class_predictor(num_anchors, num_classes):
    """return a layer to predict classes"""
    return nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)

def class_cap_predictor(num_anchors, num_class, num_cap_in, num_filter_in):
    return AdvConvCap(num_anchors, num_class+1, num_cap_in=num_cap_in, num_filter_in=num_filter_in)

def cap_transform(num_cap, num_filter):
    return PrimeConvCap(num_cap,num_filter)

def model_forward(x, net, down_samples, class_preds, box_preds, cap_transforms, sizes, ratios):
    # extract feature with the body network
    x = net(x)

    # for each scale, add anchors, box and class predictions,
    # then compute the input to next scale
    default_anchors = []
    predicted_boxes = []
    predicted_classes = []
    
    for i in range(5):
        default_anchors.append(MultiBoxPrior(x, sizes=sizes[i], ratios=ratios[i]))
        # prime_out = cap_transforms[i](x)
        box_pred = nd.flatten(nd.transpose(box_preds[i](x), (0,2,3,1)))
        # class_pred = nd.flatten(nd.transpose(class_preds[i](prime_out), (0,1,3,4,2)))
        class_pred = nd.flatten(nd.transpose(class_preds[i](x), (0,2,3,1)))
        predicted_boxes.append(box_pred)
        predicted_classes.append(class_pred)
        if i < 3:
            x = down_samples[i](x)
        elif i == 3:
            # simply use the pooling layer
            x = nd.Pooling(x, global_pool=True, pool_type='max', kernel=(4, 4))
    return default_anchors, predicted_classes, predicted_boxes

class SSD(gluon.Block):
    def __init__(self, num_classes, reg_dim,
                num_caps, num_filters, **kwargs):
        super(SSD, self).__init__(**kwargs)
        # anchor box sizes for 4 feature scales
        self.anchor_sizes = [[.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        # anchor box ratios for 4 feature scales
        self.anchor_ratios = [[1, 2, .5]] * 5
        self.num_classes = num_classes
        self.reg_dim = reg_dim
        with self.name_scope():
            self.body, self.downsamples, self.class_preds, self.box_preds, self.cap_transforms = model(4, num_classes, reg_dim, num_caps, num_filters)

    def forward(self, x):
        default_anchors, predicted_classes, predicted_boxes = model_forward(x, self.body, self.downsamples,
            self.class_preds, self.box_preds, self.cap_transforms, self.anchor_sizes, self.anchor_ratios)
        anchors = nd.concat(*default_anchors, dim=1)
        box_preds = nd.concat(*predicted_boxes, dim=1)#.reshape((0,-1,self.reg_dim))
        class_preds = nd.concat(*predicted_classes, dim=1).reshape((0,-1,self.num_classes+1))
        return anchors, class_preds, box_preds


if __name__ == '__main__':
    ssd = SSD(2,4,8,16)
    ssd.initialize()
    x = nd.zeros((1, 3, 256, 256))
    default_anchors, class_predictions, box_predictions = ssd(x)
    print default_anchors.shape
    print class_predictions.shape
    print box_predictions.shape


    '''
    net = train_net()
    net.initialize()
    print('Before', x.shape, 'after', net(x).shape)
    '''

