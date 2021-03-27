import py_bart

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import utils
from recon_pipe import pipe
import ops
from proximator import proximator
from dncnn import dncnn

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from scipy.linalg import norm
from scipy.io import loadmat
from tqdm import tqdm
recon_config = utils.load_config('recon_config.yaml')
model_config = utils.load_config('config.yaml')
fig_path = '/home/jason/proximator-net/figures/convergence_curve'

utils.save_config(recon_config, fig_path)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = recon_config['gpu_id']

xs = tf.placeholder(tf.float32, shape=[1]+model_config['slice_shape']) 

ins_net        = dncnn(chns=2, nr_layers=model_config['nr_layers'], nr_filters=model_config['nr_filters'])
init_pass      = ins_net.model(xs, init=True)
ins_proximator = proximator(ins_net, chns=2, iteration=model_config['iteration'])
op_prox        = ins_proximator.forward(xs)

saver = tf.train.Saver()
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth=True
sess = tf.Session(config=config_proto)
saver.restore(sess, os.path.join(recon_config['model_path'], recon_config['ckpt_name']))
kspace_path = '/media/jason/brain_mat/brain_mat/test/LI, Dahui_T2S_1.mat'
kspace = loadmat(kspace_path)['kspace']
nx, ny, nr_coils = kspace.shape
dims = (nx, ny)

### calculate ground truth
coilsen1 = np.squeeze(utils.bart(1, 'ecalib -m1 -r20 -c0.0001', kspace[np.newaxis, ...]))
coilsen = np.squeeze(utils.bart(1, 'caldir 20', kspace[np.newaxis, ...]))

img_coils = np.asfortranarray(ops.mifft2(kspace, dims), dtype='complex64')
rss = np.asfortranarray(np.sum(np.multiply(img_coils, np.conj(coilsen)), axis=2), dtype='complex64')
utils.save_img(abs(rss), fig_path+'/rss', np.min(abs(rss)), np.max(abs(rss)))

dims = (nx, ny, nr_coils)
traj_opts = ('traj -D -r -x%d -y%d -s%d -G'%(nx, recon_config['spokes'], recon_config['GA']))
traj = utils.bart(1, traj_opts)

radial_ksp, nufft_op = py_bart.nufft(grid_size=(nx, ny, 1), 
                                     adjoint=False,
                                     traj=traj,
                                     inp=img_coils.reshape(nx, ny, 1, nr_coils),
                                     init=True)

w = np.squeeze(utils.gen_weights(recon_config['spokes'], N=nx))
weight = np.asfortranarray(np.stack([w for _ in range(nr_coils)], axis=-1), dtype='complex64')

nop = py_bart.ops(init=1,
                o_size=(nx, ny, 1),
                traj=traj,
                inp=radial_ksp,
                wgts = weight, # must be fortran order, complex64
                maps=coilsen,
                chns=recon_config['nr_comb_coils'],
                slice=1)

x_ = py_bart.ops(init = 0, type=2, o_size=(nx, ny, 1), inp = radial_ksp, op= nop, slice = 1) # adjoint
utils.save_img(abs(np.squeeze(x_)), os.path.join(fig_path, 'zero_filled'), np.min(abs(x_)), np.max(abs(x_)))


def sense_kernel_non_cart(x_, img_k):
    tmp = py_bart.ops(init = 0,
                        type=1,
                        o_size=(nx, ny, 1),
                        inp = np.asfortranarray(img_k, dtype='complex64'),
                        op= nop,
                        slice = 1)
    img_k = img_k + np.squeeze(x_) - np.squeeze(tmp)
    return img_k



rss = abs(rss)/norm(rss)
def recon_la(laa):
    img_k = np.squeeze(x_)
    curve_net = []
    for i in range(100):
        img_k = sense_kernel_non_cart(x_, img_k)
        
        img_k, scalar = utils.scale_down(img_k)
        
        img_k = img_k*(1-laa) + laa*np.squeeze(utils.float2cplx(sess.run(op_prox, {xs: utils.cplx2float(img_k[np.newaxis, ...])})))
        img_k = utils.scale_up(img_k, scalar)
        psnr_net_radial = utils.psnr(abs(img_k)/norm(abs(img_k)), rss)
        ssim_net_radial = utils.ssim(abs(img_k)/norm(abs(img_k)), rss)
        curve_net.append([psnr_net_radial, ssim_net_radial])
    return img_k, np.array(curve_net)

img_k_1, curve_1 = recon_la(0.0)
img_k_2, curve_2 = recon_la(0.1)
img_k_3, curve_3 = recon_la(0.3)
img_k_4, curve_4 = recon_la(1.)

utils.save_img(abs(np.squeeze(img_k_1)), os.path.join(fig_path, 'img_k_1'), np.min(abs(img_k_1)), np.max(abs(img_k_1)))
utils.save_img(abs(np.squeeze(img_k_2)), os.path.join(fig_path, 'img_k_2'), np.min(abs(img_k_2)), np.max(abs(img_k_2)))
utils.save_img(abs(np.squeeze(img_k_3)), os.path.join(fig_path, 'img_k_3'), np.min(abs(img_k_3)), np.max(abs(img_k_3)))
utils.save_img(abs(np.squeeze(img_k_4)), os.path.join(fig_path, 'img_k_4'), np.min(abs(img_k_4)), np.max(abs(img_k_4)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), gridspec_kw={'width_ratios': [1, 1]})

title_size = 14
label_size = 12
itera = 100

ax1.plot(np.arange(0,itera), curve_1[:,0], label="$\lambda=0.0$")
ax1.plot(np.arange(0,itera), curve_2[:,0], label="$\lambda=0.1$")
ax1.plot(np.arange(0,itera), curve_3[:,0], label="$\lambda=0.3$")
ax1.plot(np.arange(0,itera), curve_4[:,0], label="$\lambda=1.0$")

ax1.set_xlabel('iteration', fontsize=label_size)
ax1.set_ylabel('PSNR', fontsize=label_size)
ax1.grid(color='gray', linestyle='dashed')

ax2.plot(np.arange(0,itera), curve_1[:,1], label="$\lambda=0.0$")
ax2.plot(np.arange(0,itera), curve_2[:,1], label="$\lambda=0.1$")
ax2.plot(np.arange(0,itera), curve_3[:,1], label="$\lambda=0.3$")
ax2.plot(np.arange(0,itera), curve_4[:,1], label="$\lambda=1.0$")

ax2.set_xlabel('iteration', fontsize=label_size)
ax2.set_ylabel('SSIM', fontsize=label_size)
ax2.grid(color='gray', linestyle='dashed')

fig.suptitle('The evolution of SSIM and PSNR during interations', fontsize=title_size)
plt.legend()

plt.savefig(fig_path+'/convergence_c.pdf')
plt.savefig(fig_path+'/convergence_c.png')

