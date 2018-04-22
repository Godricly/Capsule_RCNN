import sys
sys.path.insert(0,'../incubator-mxnet/python')
import time
import mxnet as mx
import numpy as np
from mxnet import autograd as ag
from mxnet import init
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import initializer
from capnet import SSD
from data import get_iterators
from mxnet.contrib.ndarray import MultiBoxTarget
def training_targets(default_anchors, class_predicts, labels):
    class_predicts = nd.transpose(class_predicts, axes=(0, 2, 1))
    z = MultiBoxTarget(*[default_anchors, labels, class_predicts])
    box_target = z[0]  # box offset target for (x, y, width, height)
    box_mask = z[1]  # mask is used to ignore box offsets we don't want to penalize, e.g. negative samples
    cls_target = z[2]  # cls_target is an array of labels for all anchors boxes
    return box_target, box_mask, cls_target

class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def forward(self, output, label):
        output = nd.softmax(output)
        pt = nd.pick(output, label, axis=self._axis, keepdims=True)
        # print output.asnumpy()[np.where(label.asnumpy() > 0)]
        loss = -self._alpha * ((1 - pt) ** self._gamma) * nd.log(pt)
        # loss = - nd.log(pt)
        return nd.mean(loss, axis=self._batch_axis, exclude=True)

class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def forward(self, output, label, mask):
        loss = nd.smooth_l1((output - label) * mask, scalar=1.0)
        return nd.mean(loss, self._batch_axis, exclude=True)


if __name__ == '__main__':
    data_shape = 256
    batch_size = 4
    train_data, test_data, class_names, num_class = get_iterators(data_shape, batch_size)
    cls_loss = FocalLoss()
    box_loss = SmoothL1Loss()
    cls_metric = mx.metric.Accuracy()
    box_metric = mx.metric.MAE() 

    ### Set context for training
    ctx = mx.gpu(0)  # it may takes too long to train using CPU
    try:
        _ = nd.zeros(1, ctx=ctx)
        # pad label for cuda implementation
        train_data.reshape(label_shape=(3, 5))
        train_data = test_data.sync_label_shape(train_data)
    except mx.base.MXNetError as err:
        print('No GPU enabled, fall back to CPU, sit back and be patient...')
        ctx = mx.cpu()

    net = SSD(num_class,4,32,16)
    net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(),'adam', {'learning_rate': 0.001})

    epochs = 150  # set larger to get better performance
    log_interval = 20
    start_epoch = 0

    for epoch in range(start_epoch, epochs):
        # reset iterator and tick
        train_data.reset()
        cls_metric.reset()
        box_metric.reset()
        tic = time.time()
        # iterate through all batch
        for i, batch in enumerate(train_data):
            btic = time.time()
            # record gradients
            with ag.record():
                x = batch.data[0].as_in_context(ctx)
                y = batch.label[0].as_in_context(ctx)
                default_anchors, class_predictions, box_predictions = net(x)
                box_target, box_mask, cls_target = training_targets(default_anchors, class_predictions, y)
                # box_predictions.attach_grad()
                # losses
                loss1 = cls_loss(class_predictions, cls_target)
                loss2 = box_loss(box_predictions, box_target, box_mask)
                # sum all losses
                loss = loss1 + loss2
                # backpropagate
                # loss.backward(retain_graph=True)
                loss.backward()
            # g_box = ag.grad(loss,[box_predictions], create_graph=True)
            # print g_box
            # apply
            trainer.step(batch_size,ignore_stale_grad=True)
            # update metrics
            cls_metric.update([cls_target], [nd.transpose(class_predictions, (0, 2, 1))])
            box_metric.update([box_target], [box_predictions * box_mask])
            if (i + 1) % log_interval == 0:
                name1, val1 = cls_metric.get()
                name2, val2 = box_metric.get()
                print('[Epoch %d Batch %d] speed: %f samples/s, training: %s=%f, %s=%f'
                      %(epoch ,i, batch_size/(time.time()-btic), name1, val1, name2, val2))
        # end of epoch logging
        name1, val1 = cls_metric.get()
        name2, val2 = box_metric.get()
        print('[Epoch %d] training: %s=%f, %s=%f'%(epoch, name1, val1, name2, val2))
        print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
    
    # we can save the trained parameters to disk
    net.save_params('ssd_%d.params' % epochs)
