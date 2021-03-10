import tensorflow.compat.v1 as tf
import numpy as np
from tf_slim import add_arg_scope
import math
from tf_slim.layers import layer_norm
import tensorflow as tfw

def int_shape(x):
    return list(map(int, x.get_shape()))

def reshape_4d_to_3d(x):
  """Reshape input from 4D to 3D if necessary."""
  x_shape = int_shape(x)
  is_4d = False
  if len(x_shape) == 4:
    x = tf.reshape(x, [x_shape[0], x_shape[1]*x_shape[2], x_shape[3]])
    is_4d = True
  return x, x_shape, is_4d

def crop_and_concat(x1, x2):
    x1_shape = int_shape(x1)
    x2_shape = int_shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)

def conv_cond_concat(x, y):
    """
    concatanate conditioning vector on feature map axis
    """
    x_shape = int_shape(x)
    y_shape = int_shape(y)
    return tf.concat([x, y * tf.ones(x_shape[0], x_shape[1], x_shape[2], y_shape[3])])


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape())-1
    return tf.nn.elu(tf.concat([x, -x], axis))

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis, keepdims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keepdims=True))

def get_var_maybe_avg(var_name, ema=None, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v


def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    ''' Adam optimizer '''
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1>0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1*v + (1. - mom1)*g
            v_hat = v_t / (1. - tf.pow(mom1,t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2*mg + (1. - mom2)*tf.square(g)
        mg_hat = mg_t / (1. - tf.pow(mom2,t))
        g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

@add_arg_scope
def tf_layer_norm(x, counters={}, **kwargs):

    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('layer_norm', counters)

    if 'debug' in kwargs.keys():
        if kwargs['debug']:
            print(name)
    
    with tf.variable_scope(name):
        return layer_norm(x, scope=name)


@add_arg_scope
def nn_layer_norm(x,
               filters=None,
               epsilon=1e-6,
               counters={},
               ema=None,
               **kwargs):
    if filters is None:
        filters = int_shape(x)[-1]

    
    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('layer_norm', counters)

    if 'debug' in kwargs.keys():
        if kwargs['debug']:
            print(name)

    with tf.variable_scope(name):
        
        scale = get_var_maybe_avg('scale', shape=[filters], dtype=tf.float32,
                                  initializer=tf.ones_initializer(), trainable=True)
        bias = get_var_maybe_avg('bias', shape=[filters], dtype=tf.float32,
                                  initializer=tf.zeros_initializer(), trainable=True)

        epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]

        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(
            tf.squared_difference(x, mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)

        output = norm_x * scale + bias

    return output

@add_arg_scope
def batch_normalization(x, is_training=True, counters={}, **kwargs):
    
    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('batch_norm', counters)

    if 'debug' in kwargs.keys():
        if kwargs['debug']:
            print(name)

    with tf.variable_scope(name):
        return tf.layers.batch_normalization(x, is_training)


@add_arg_scope
def tf_dense(x, units, counters={}, **kwargs):
    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('tf_dense', counters)

    if 'debug' in kwargs.keys():
        if kwargs['debug']:
            print(name)

    with tf.variable_scope(name):
        return tf.layers.dense(x, units, use_bias=False)

@add_arg_scope
def dense(x_, num_units, nonlinearity=None, init_scale=1., counters={}, init=False, use_bias=True, ema=None, **kwargs):
    ''' fully connected layer '''
    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('dense', counters)

    if 'debug' in kwargs.keys():
        if kwargs['debug']:
            print(name)

    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', shape=[int(x_.get_shape()[1]),num_units], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        if use_bias:
            b = get_var_maybe_avg('b', shape=[num_units], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        x = tf.matmul(x_, V)

        scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))

        if use_bias:
            x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])

        if init: # normalize x
            m_init, v_init = tf.nn.moments(x, [0])
            scale_init = init_scale/tf.sqrt(v_init + 1e-10)
            if use_bias:
                with tf.control_dependencies([g.assign(g*scale_init), b.assign_add(-m_init*scale_init)]):
                    x = tf.matmul(x_, V)
                    scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
                    x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])
            else:
                with tf.control_dependencies([g.assign(g*scale_init)]):
                    x = tf.matmul(x_, V)
                    scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
                    x = tf.reshape(scaler, [1, num_units]) * x

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x


def sele_attention(inputs, attention_size):
    hidden_size = inputs.shape[2].value
    w_omega = tfw.Variable(tfw.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tfw.Variable(tfw.random_normal([attention_size], stddev=0.1))
    u_omega = tfw.Variable(tfw.random_normal([attention_size], stddev=0.1))
    with tfw.name_scope('v'):
        v = tfw.tanh(tfw.tensordot(inputs, w_omega, axes=1) + b_omega)
    vu = tfw.tensordot(v, u_omega, axes=1, name='vu')  
    alphas = tfw.nn.softmax(vu, name='alphas')  
    output = tfw.reduce_sum(inputs * tfw.expand_dims(alphas, -1), 1)
    return output

@add_arg_scope
def conv2d(x_, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
    ''' convolutional layer '''
    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters) # this is the scope defined by args
    else:
        name = get_name('conv2d', counters) # this is default scope named with conv2d
    
    if 'debug' in kwargs.keys():
        if kwargs['debug']:
            print(name)

    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', shape=filter_size+[int(x_.get_shape()[-1]),num_filters], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])
        
        # calculate convolutional layer output
        x = tf.nn.bias_add(tf.nn.conv2d(x_, W, [1] + stride + [1], pad), b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0,1,2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                # x = tf.identity(x)
                W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])
                x = tf.nn.bias_add(tf.nn.conv2d(x_, W, [1] + stride + [1], pad), b)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x

@add_arg_scope
def deconv2d(x_, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
    ''' transposed convolutional layer '''
    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('deconv2d', counters)

    if 'debug' in kwargs.keys():
        if kwargs['debug']:
            print(name)

    xs = int_shape(x_)
    if pad=='SAME':
        target_shape = [xs[0], xs[1]*stride[0], xs[2]*stride[1], num_filters]
    else:
        target_shape = [xs[0], xs[1]*stride[0] + filter_size[0]-1, xs[2]*stride[1] + filter_size[1]-1, num_filters]
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', shape=filter_size+[num_filters,int(x_.get_shape()[-1])], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', shape=[num_filters], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 3])

        # calculate convolutional layer output
        x = tf.nn.conv2d_transpose(x_, W, target_shape, [1] + stride + [1], padding=pad)
        x = tf.nn.bias_add(x, b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0,1,2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                # x = tf.identity(x)
                W = tf.reshape(g, [1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 3])
                x = tf.nn.conv2d_transpose(x_, W, target_shape, [1] + stride + [1], padding=pad)
                x = tf.nn.bias_add(x, b)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x

@add_arg_scope
def nin(x, num_units, use_bias=True, nonlinearity=None, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]),s[-1]])
    x = dense(x, num_units, use_bias=use_bias, **kwargs)
    if nonlinearity is not None:
        x = nonlinearity(x)
    return tf.reshape(x, s[:-1]+[num_units])

''' meta-layer consisting of multiple base layers '''

@add_arg_scope
def gated_resnet(x, a=None, h=None, nonlinearity=concat_elu, conv=conv2d, init=False, counters={}, ema=None, dropout_p=0., **kwargs):
    xs = int_shape(x)
    num_filters = xs[-1]
    
    c1 = conv(nonlinearity(x), num_filters)
    if a is not None: # add short-cut connection if auxiliary input 'a' is given
        c1 += nin(nonlinearity(a), num_filters)
    c1 = nonlinearity(c1)
    if dropout_p > 0:
        c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
    
    c2 = conv(c1, num_filters * 2, init_scale=0.1)

    # add projection of h vector if included: conditional generation
    
    if h is not None:
        with tf.variable_scope(get_name('conditional_weights', counters)):
            hw = get_var_maybe_avg('hw', ema, shape=[int_shape(h)[-1], 2 * num_filters], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        if init:
            hw = hw.initialized_value()
        c2 += tf.reshape(tf.matmul(h, hw), [xs[0], 1, 1, 2 * num_filters])
    

    a, b = tf.split(c2, 2, 3)
    c3 = a * tf.nn.sigmoid(b)
    
    return x + c3

''' utilities for shifting the image around, efficient alternative to masking convolutions '''

def down_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],1,xs[2],xs[3]]), x[:,:xs[1]-1,:,:]],1)

def right_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],xs[1],1,xs[3]]), x[:,:,:xs[2]-1,:]],2)

@add_arg_scope
def down_shifted_conv2d(x, num_filters, filter_size=[2,3], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1,0], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)

@add_arg_scope
def down_shifted_deconv2d(x, num_filters, filter_size=[2,3], stride=[1,1], **kwargs):
    x = deconv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)
    xs = int_shape(x)
    return x[:,:(xs[1]-filter_size[0]+1),int((filter_size[1]-1)/2):(xs[2]-int((filter_size[1]-1)/2)),:]

@add_arg_scope
def down_right_shifted_conv2d(x, num_filters, filter_size=[2,2], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1, 0], [filter_size[1]-1, 0],[0,0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)

@add_arg_scope
def down_right_shifted_deconv2d(x, num_filters, filter_size=[2,2], stride=[1,1], **kwargs):
    x = deconv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)
    xs = int_shape(x)
    return x[:,:(xs[1]-filter_size[0]+1):,:(xs[2]-filter_size[1]+1),:]

@add_arg_scope
def self_attention(x, qk_chns, v_chns, **kwargs):
    shape = int_shape(x)
    query_conv = tf.reshape(nin(x, qk_chns, nonlinearity=None, scope='global_attention'), (shape[0], shape[1]*shape[2], -1))
    key_conv = tf.reshape(nin(x, qk_chns, nonlinearity=None, scope='global_attention'), (shape[0], shape[1]*shape[2], -1))
    value_conv = tf.reshape(nin(x, v_chns, nonlinearity=None, scope='global_attention'), (shape[0], shape[1]*shape[2], -1))
    energy = tf.einsum("bnf,bjf->bnj", query_conv, key_conv)
    attention_map = tf.nn.softmax(energy, axis=-1)
    out = tf.einsum("bnf,bnj->bjf", value_conv, attention_map)

    if shape[-1] != v_chns:
        x = nin(x, v_chns, nonlinearity=None, scope='global_attention')
        
    return tf.reshape(out, shape[:-1]+[v_chns]) + x
