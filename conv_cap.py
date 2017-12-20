import mxnet as mx
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn
from mxnet import initializer

def squash(x, axis):
    s_squared_norm = nd.sum(nd.square(x), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / nd.sqrt(s_squared_norm + 1e-5)
    return scale * x

class PrimeConvCap(nn.Block):
    def __init__(self, num_cap, num_filter, **kwargs):
        super(PrimeConvCap, self).__init__(**kwargs)
        self.num_cap = num_cap
        self.cap = nn.Conv2D(channels=(num_cap*num_filter), kernel_size=(3,3),
                             padding=(1,1))

    def forward(self, x):
        conv_out = nd.expand_dims(self.cap(x), axis=2)
        conv_out = conv_out.reshape((0,-1,self.num_cap,0,0))
        conv_out = squash(conv_out, 1)
        return conv_out


class AdvConvCap(nn.Block):
    def __init__(self, num_cap, num_filter,
                 num_cap_in, num_filter_in,
                 route_num=3, **kwargs):
        super(AdvConvCap, self).__init__(**kwargs)
        self.num_cap = num_cap
        self.num_filter = num_filter
        self.route_num = route_num
        self.num_cap_in = num_cap_in
        # num_filter_in * num_cap_in filters divided in num_cap_in groups
        # with each group output size as num_cap * num_filter
        self.cap = nn.Conv2D(channels=(num_cap * num_filter * num_cap_in), kernel_size=(3,3),# strides=(2,2),
                             padding=(1,1), groups= num_cap_in)

    def forward(self, x):
        x_reshape = x.reshape((x.shape[0], -1, x.shape[3], x.shape[4]))
        cap_out = self.cap(x_reshape)
        cap_out = cap_out.reshape((cap_out.shape[0], self.num_cap_in, self.num_filter,
                             self.num_cap, cap_out.shape[2], cap_out.shape[3]))
        return self.route(cap_out)

    def route(self, x):
        b_mat = nd.zeros((x.shape[0], self.num_cap_in, 1, self.num_cap, x.shape[4], x.shape[5]), ctx=x.context)
        u = x
        u_no_gradient = nd.stop_gradient(x)
        for i in range(self.route_num):
            c_mat = nd.softmax(b_mat, axis=3)
            if i == self.route_num -1:
                s = nd.sum(u * c_mat, axis=1)
            else:
                s = nd.sum(u_no_gradient * c_mat, axis=1)
            v = squash(s, 1)
            v1 = nd.expand_dims(v, axis=1)
            if i != self.route_num - 1:
                update_term = nd.sum(u_no_gradient*v1, axis=2, keepdims=True)
                b_mat = b_mat + update_term
        return v
