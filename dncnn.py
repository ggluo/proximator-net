"""
resnet denoiser
"""

import nn
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from tf_slim import arg_scope

class dncnn():
    def __init__(self, chns=2, filter_size=[3,3], nr_layers=5, nr_filters=32, nr_classes=10, nonlinearity=tf.nn.relu):
        self.filter_size = filter_size
        self.nr_layers   = nr_layers
        self.nr_filters  = nr_filters
        self.is_training = is_training
        self.chns        = chns
        self.nr_classes  = nr_classes
        self.nonlinearity= nonlinearity
        self.counters    = {}
        self.forward     = tf.make_template('model', self.main)

    def body(self, x, h, init=False):
        lys = []
        with arg_scope([nn.conv2d, nn.cond_instance_norm_plus], init=init, counters = self.counters):

            # pre-layer
            ######
            l_o = nn.conv2d(x, self.nr_filters, self.filter_size, nonlinearity=None)
            l_o = nn.cond_instance_norm_plus(l_o, h, self.nr_classes)
            l_o = nonlinearity(l_o)
            ######
            lys.append(l_o)
            #
            for _ in range(self.nr_layers):
                lys.append(nn.conv2d(lys[-1], self.nr_filters, self.filter_size))
            # post-layer
            lys.append(nn.conv2d(lys[-1], self.chns, self.filter_size, nonlinearity=None))
            return x - lys[-1]
    
    def main(self, x, h, init=False):
        out = self.body(x, h, init)
        self.counters = {}
        return out
