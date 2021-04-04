import os
import sys

from proximator import proximator
from dncnn import dncnn
import nn
import utils
from utils import create_dataloader_procs, terminate_procs

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import time
import numpy as np
import argparse

from datetime import datetime
from multiprocessing import Queue

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml')

args = parser.parse_args()
config = utils.load_config(args.config)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']

log_path = os.path.join('./', 'logs', config['model']) + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists(log_path):
    os.mkdir(log_path)
log_writer = tf.summary.FileWriter(log_path)

save_path = os.path.join('./', 'save', config['model'])
if not os.path.exists(save_path):
    os.mkdir(save_path)
utils.save_config(config, save_path)

# noisy images as inputs
xs = [tf.placeholder(tf.float32, 
                   shape=[config['batch_size']]+config['slice_shape']
                   ) for _ in range(config['nr_gpu'])]
hs = [tf.placeholder(tf.int32, 
                   shape=[config['batch_size']]
                   ) for _ in range(config['nr_gpu'])]

# clean images as labels
xs_clean = [tf.placeholder(tf.float32, 
                   shape=[config['batch_size']]+config['slice_shape']
                   ) for _ in range(config['nr_gpu'])]

tf_lr = tf.placeholder(tf.float32, shape=[])
sigmas = np.linspace(config['sigma_0'], config['sigma_1'], config['sigma_steps'])
ins_net = dncnn(chns=2, nr_layers=config['nr_layers'], nr_filters=config['nr_filters'], nr_classes=len(sigmas))
init_pass = ins_net.forward(xs[0], hs[0], init=True)
ins_proximator = proximator(ins_net, chns=2, iteration=config['iteration'])

all_params = tf.trainable_variables()

loss = []
grads = []
loss_test = []
logits = []
logits_test = []
l2_reg = []


# create tower
for i in range(config['nr_gpu']):
    with tf.device('/gpu:%d'%i):
        # TODO h, nr_classes, nonlinear
        # train
        logits.append(ins_proximator.forward(xs[i], hs[i], config['nr_classes']))
        l2_reg.append(tf.concat(tf.gradients(logits[-1], xs[i]), axis=0))
        loss.append(tf.reduce_mean(tf.math.square(logits[-1]-xs_clean[i]))+config['sigma']*config['sigma']*tf.reduce_mean(tf.math.square(tf.stop_gradient(l2_reg[-1]))))
        grads.append(tf.gradients(loss[-1], all_params, colocate_gradients_with_ops=True))

        # test
        logits_test.append(ins_proximator.forward(xs[i], hs[i], config['nr_classes'], nonlinearity))
        loss_test.append(tf.reduce_mean(tf.math.square(logits_test[-1]-xs_clean[i])))

# average loss
with tf.device('/gpu:0'):    
    for i in range(1, config['nr_gpu']):
        loss[0] += loss[i]
        loss_test[0] += loss_test[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    
    optimizer = nn.adam_updates(all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995)

f_loss = loss[0]
f_loss_test = loss_test[0]


### create summary
xs_summary = tf.reshape(tf.abs(tf.complex(xs[0][..., 0], xs[0][..., 1])), [config['batch_size'], config['slice_shape'][0], config['slice_shape'][1], 1])
tf.summary.image("input_noisy_image_train", xs_summary, max_outputs=5, collections=['train'])

xs_summary = tf.reshape(tf.abs(tf.complex(xs[0][..., 0], xs[0][..., 1])), [config['batch_size'], config['slice_shape'][0], config['slice_shape'][1], 1])
tf.summary.image("input_noisy_image_test", xs_summary, max_outputs=5, collections=['test'])

xs_clean_summary = tf.reshape(tf.abs(tf.complex(xs_clean[0][..., 0], xs_clean[0][..., 1])), [config['batch_size'], config['slice_shape'][0], config['slice_shape'][1], 1])
tf.summary.image("input_clean_labels_train", xs_clean_summary, max_outputs=5, collections=['train'])

xs_clean_summary = tf.reshape(tf.abs(tf.complex(xs_clean[0][..., 0], xs_clean[0][..., 1])), [config['batch_size'], config['slice_shape'][0], config['slice_shape'][1], 1])
tf.summary.image("input_clean_labels_test", xs_clean_summary, max_outputs=5, collections=['test'])

out_summary_train = tf.reshape(tf.abs(tf.complex(logits[0][..., 0], logits[0][..., 1])), [config['batch_size'], config['slice_shape'][0], config['slice_shape'][1], 1])
tf.summary.image("denoised_train", out_summary_train, max_outputs=5, collections=['train'])

out_summary_test = tf.reshape(tf.abs(tf.complex(logits_test[0][..., 0], logits_test[0][..., 1])), [config['batch_size'], config['slice_shape'][0], config['slice_shape'][1], 1])
tf.summary.image("denoised_test", out_summary_test, max_outputs=5, collections=['test'])

tf.summary.scalar("loss_train", f_loss, collections=['train'])
tf.summary.scalar("loss_test", f_loss_test, collections=['test'])

train_summary_op = tf.summary.merge_all(key='train')
test_summary_op = tf.summary.merge_all(key='test')


### prepare data
train_files = utils.list_data_sys(config['training_data_path'], config['dataset_suffix'], ex_pattern='resam')
test_files = utils.list_data_sys(config['testing_data_path'], config['dataset_suffix'], ex_pattern='resam')

print("Number of training files: %d"%len(train_files))
print("Number of testing files: %d"%len(test_files))

train_queue = Queue(config['num_prepare'])
epoch_sign = Queue(1)
train_procs = []

test_queue = Queue(config['num_prepare'])
test_sign = Queue(1)
test_procs = []


create_dataloader_procs(datalist = train_files,
                   train_queue= train_queue,
                   sign = epoch_sign,
                   n_thread = config['nr_thread'],
                   train_procs = train_procs,
                   batch_size = config['batch_size']*config['nr_gpu'],
                   input_shape = config['slice_shape'],
                   normalize_func = utils.normalize_with_max,
                   fileloader=utils.mat_loader,
                   rand_scalor=config['rand_scalor'])

create_dataloader_procs(datalist = test_files,
                   train_queue= test_queue,
                   sign = test_sign,
                   n_thread = config['nr_thread'],
                   train_procs = train_procs,
                   batch_size = config['batch_size']*config['nr_gpu'],
                   input_shape = config['slice_shape'],
                   normalize_func = utils.normalize_with_max,
                   fileloader=utils.mat_loader,
                   rand_scalor=config['rand_scalor'])


###
print("Training begins ....")

init_op = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep = config['max_keep'])
gpu_config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=gpu_config)
sess.run(init_op)

global_step = 0
epochs = 0
begin = time.time()

while True:

    ls = config['lr']

    batch = train_queue.get()
    labels = np.random.randint(0, len(sigmas), (batch.shape[0]), dtype='int32')
    
    xc = np.split(batch, config['nr_gpu'])
    # sigma=[]
    noisy_batch = utils.noise(shape=batch.shape)*sigmas[labels] + batch
    xn = np.split(noisy_batch, config['nr_gpu'])
    labels_l = np.split(labels, config['nr_gpu'])
    
    feed_dict = {xs[i]: xn[i] for i in range(config['nr_gpu'])}
    feed_dict.update({xs_clean[i]: xc[i] for i in range(config['nr_gpu'])})
    feed_dict.update({hs[i]: labels_l[i] for i in range(config['nr_gpu'])})
    feed_dict.update({tf_lr: config['lr']})
    
    
    l, _, sums_train = sess.run([f_loss, optimizer, train_summary_op], feed_dict=feed_dict)
    
    if epoch_sign.full() and epoch_sign.get():
        epochs = epochs + 1
        log_writer.add_summary(sums_train, epochs)
        print("Epoch-> %d"%epochs)
        while True:

            test_batch = test_queue.get()
            xc = np.split(test_batch, config['nr_gpu'])
            labels = np.random.randint(0, len(sigmas), (batch.shape[0]), dtype='int32')
            
            test_batch = utils.noise(shape=test_batch.shape)*sigmas[labels] + test_batch
            xn = np.split(test_batch, config['nr_gpu'])
            labels_l = np.split(labels, config['nr_gpu'])
            
            test_dict = {xs[i]:xn[i] for i in range(config['nr_gpu'])}
            test_dict.update({xs_clean[i]:xc[i]  for i in range(config['nr_gpu'])})
            test_dict.update({hs[i]: labels_l[i] for i in range(config['nr_gpu'])})
            
            l, sums_test = sess.run([f_loss_test, test_summary_op], feed_dict=feed_dict)
            

            if test_sign.full() and test_sign.get():
                log_writer.add_summary(sums_test, epochs)
                break
        
        if epochs % config['save_interval'] == 0:
            saver.save(sess, os.path.join(save_path, config['model']+'_'+str(epochs)))

        if epochs == config['max_epochs']:
            terminate_procs(train_procs)
            terminate_procs(test_procs)
            utils.color_print("TRAINING ENDED! CHECK UP !!!!")
            break
