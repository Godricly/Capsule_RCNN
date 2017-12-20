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
        (64, 2),
        (128, 2),
        (256, 3),
        ]
    net = backbone(layer_setup)

    down_samples = nn.Sequential()
    down_samples.add(conv_block(256,2))
    down_samples.add(conv_block(256,2))
    down_samples.add(conv_block(256,2))

    class_preds = nn.Sequential()
    box_preds = nn.Sequential()
    cap_transforms = nn.Sequential()
    for scale in range(5):
        cap_transforms.add(cap_transform(num_caps, num_filters)) 
        class_preds.add(class_predictor(num_anchors, num_classes, num_caps, num_filters))
        box_preds.add(box_predictor(num_anchors, reg_dim, num_caps, num_filters))
    return net, down_samples, class_preds, box_preds, cap_transforms

def box_predictor(num_anchors, dim, num_cap_in, num_filter_in):
    return AdvConvCap(num_anchors, dim, num_cap_in=num_cap_in, num_filter_in=num_filter_in)

def class_predictor(num_anchors, num_class, num_cap_in, num_filter_in):
    return AdvConvCap(num_anchors, num_class, num_cap_in=num_cap_in, num_filter_in=num_filter_in)

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
        prime_out = cap_transforms[i](x)
        box_pred = box_preds[i](prime_out)
        class_pred =  class_preds[i](prime_out)
        predicted_boxes.append(box_pred)
        predicted_classes.append(class_pred)
        if i < 3:
            x = down_samples[i](x)
        elif i == 3:
            # simply use the pooling layer
            x = nd.Pooling(x, global_pool=True, pool_type='max', kernel=(4, 4))
    return default_anchors, predicted_boxes, predicted_classes

class SSD(gluon.Block):
    def __init__(self, num_anchors, num_classes, reg_dim,
                num_caps, num_filters, **kwargs):
        super(SSD, self).__init__(**kwargs)
        # anchor box sizes for 4 feature scales
        self.anchor_sizes = [[.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        # anchor box ratios for 4 feature scales
        self.anchor_ratios = [[1, 2, .5]] * 5
        self.num_classes = num_classes
        with self.name_scope():
            self.body, self.downsamples, self.class_preds, self.box_preds, self.cap_transforms = model(4, num_classes, reg_dim, num_caps, num_filters)

    def forward(self, x):
        default_anchors, predicted_classes, predicted_boxes = model_forward(x, self.body, self.downsamples,
            self.class_preds, self.box_preds, self.cap_transforms, self.anchor_sizes, self.anchor_ratios)
        return default_anchors, predicted_classes, predicted_boxes


if __name__ == '__main__':
    ssd = SSD(5,2,4,32,16)
    ssd.initialize()
    x = nd.zeros((1, 3, 256, 256))
    default_anchors, class_predictions, box_predictions = ssd(x)
    print box_predictions


    '''
    net = train_net()
    net.initialize()
    print('Before', x.shape, 'after', net(x).shape)
    '''

