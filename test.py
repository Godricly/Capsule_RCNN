import sys
sys.path.insert(0,'../incubator-mxnet/python')
import cv2
import numpy as np
import mxnet as mx
from capnet import SSD
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxDetection
from data import get_iterators
import matplotlib as mpl
mpl.use('Agg')
import random
import matplotlib.pyplot as plt
def preprocess(image):
    """Takes an image and apply preprocess"""
    # resize to data_shape
    image = cv2.resize(image, (data_shape, data_shape))
    # swap BGR to RGB
    image = image[:, :, (2, 1, 0)]
    # convert to float before subtracting mean
    image = image.astype(np.float32)
    # subtract mean
    image -= np.array([123, 117, 104])
    # organize as [batch-channel-height-width]
    image = np.transpose(image, (2, 0, 1))
    image = image[np.newaxis, :]
    # convert to ndarray
    image = nd.array(image)
    return image


def reverse(img):
    image = nd.transpose(img[0],(1,2,0)).asnumpy() + np.array([123, 117, 104])
    image = image[:,:,(2,1,0)].astype(np.uint8)
    return image

def display(img, out, thresh=0.5):
    mpl.rcParams['figure.figsize'] = (10,10)
    pens = dict()
    plt.clf()
    plt.imshow(img)
    for det in out:
        cid = int(det[0])
        if cid < 0:
            continue
        score = det[1]
        if score < thresh:
            continue
        if cid not in pens:
            pens[cid] = (random.random(), random.random(), random.random())
        print det
        scales = [img.shape[1], img.shape[0]] * 2
        xmin, ymin, xmax, ymax = [int(p * s) for p, s in zip(det[2:6].tolist(), scales)]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, 
                             edgecolor=pens[cid], linewidth=3)
        plt.gca().add_patch(rect)
        text = class_names[cid]
        plt.gca().text(xmin, ymin-2, '{:s} {:.3f}'.format(text, score),
                       bbox=dict(facecolor=pens[cid], alpha=0.5),
                       fontsize=12, color='white')
    plt.savefig('result.jpg')

if __name__ == '__main__':

    # get test data
    data_shape = 256
    batch_size = 1
    _, test_data, class_names, num_class = get_iterators(data_shape, batch_size)
    x = test_data.next().data[0]

    img = reverse(x)
    cv2.imwrite('test.jpg', img)
    '''
    image = cv2.imread('img/pikachu.jpg')
    x = preprocess(image)
    '''
    print('x', x.shape)
    ctx = mx.gpu()  # it may takes too long to train using CPU
    try:
        _ = nd.zeros(1, ctx=ctx)
    except mx.base.MXNetError as err:
        print('No GPU enabled, fall back to CPU, sit back and be patient...')
        ctx = mx.cpu()

    net = SSD(num_class,4,32,16)
    net.load_params('ssd_300.params', ctx)
    anchors, cls_preds, box_preds = net(x.as_in_context(ctx))
    print('anchors', anchors)
    print('class predictions', cls_preds)
    print('box delta predictions', box_preds)

    # convert predictions to probabilities using softmax
    cls_probs = nd.SoftmaxActivation(nd.transpose(cls_preds, (0, 2, 1)), mode='channel')
    # apply shifts to anchors boxes, non-maximum-suppression, etc...
    output = MultiBoxDetection(*[cls_probs, box_preds, anchors], force_suppress=True, clip=False)
    print(output)

    display(img[:, :, (2, 1, 0)], output[0].asnumpy(), thresh=0.195)



