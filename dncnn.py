"""
resnet denoiser
"""

from . import nn
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from tf_slim import arg_scope

class dncnn():

    def __init__(self, chns=2, filter_size=[3,3], nr_layers=5, nr_filters=32, is_training=True):
        self.filter_size = filter_size
        self.nr_layers   = nr_layers
        self.nr_filters  = nr_filters
        self.is_training = is_training
        self.chns        = chns   
        self.counters    = {}
        self.model       = tf.make_template('model', self.forward)

    def body(self, x, init=False):
        lys = []
        with arg_scope([nn.conv2d], init=init, counters = self.counters, nonlinearity=tf.nn.relu):

            # pre-layer
            lys.append(nn.conv2d(x, self.nr_filters, self.filter_size)) 

            #
            for _ in range(self.nr_layers):
                lys.append(nn.conv2d(lys[-1], self.nr_filters, self.filter_size))

            # post-layer
            lys.append(nn.conv2d(lys[-1], self.chns, self.filter_size, nonlinearity=None))

            return x - lys[-1]
    
    def forward(self, x, init=False):
        out = self.body(x, init)
        self.counters = {}
        return out