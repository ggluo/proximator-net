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
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from scipy.linalg import norm
from scipy.io import loadmat
from tqdm import tqdm
recon_config = utils.load_config('recon_config.yaml')
model_config = utils.load_config('config.yaml')
fig_path = '/home/jason/proximator-net/figures/fig1_radial'

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
coilsen1 = cp.squeeze(utils.bart(1, 'ecalib -m1 -r20 -c0.0001', kspace[cp.newaxis, ...]))
coilsen = cp.squeeze(utils.bart(1, 'caldir 20', kspace[cp.newaxis, ...]))

img_coils = cp.asfortranarray(ops.mifft2(kspace, dims), dtype='complex64')
rss = cp.asfortranarray(cp.sum(cp.multiply(img_coils, cp.conj(coilsen)), axis=2), dtype='complex64')
utils.save_img(abs(rss), fig_path+'/rss', cp.min(abs(rss)), cp.max(abs(rss)))

dims = (nx, ny, nr_coils)
traj_opts = ('traj -D -r -x%d -y%d -s%d -G'%(nx, recon_config['spokes'], recon_config['GA']))
traj = utils.bart(1, traj_opts)

radial_ksp, nufft_op = py_bart.nufft(grid_size=(nx, ny, 1), 
                                     adjoint=False,
                                     traj=traj,
                                     inp=img_coils.reshape(nx, ny, 1, nr_coils),
                                     init=True)

w = cp.squeeze(utils.gen_weights(recon_config['spokes'], N=nx))
weight = cp.asfortranarray(cp.stack([w for _ in range(nr_coils)], axis=-1), dtype='complex64')

nop = py_bart.ops(init=1,
                o_size=(nx, ny, 1),
                traj=traj,
                inp=radial_ksp,
                wgts = weight, # must be fortran order, complex64
                maps=coilsen,
                chns=recon_config['nr_comb_coils'],
                slice=1)

x_ = py_bart.ops(init = 0, type=2, o_size=(nx, ny, 1), inp = radial_ksp, op= nop, slice = 1) # adjoint
utils.save_img(abs(cp.squeeze(x_)), os.path.join(fig_path, 'zero_filled'), cp.min(abs(x_)), cp.max(abs(x_)))

l1_recon = utils.bart(1, 'pics -R W:3:0:%f -d5 -i100 -e -t'%(recon_config['bart_l1_lambda']), traj, radial_ksp, coilsen1[:,:,cp.newaxis,:])
utils.save_img(abs(cp.squeeze(l1_recon)), os.path.join(fig_path, 'l1_recon'), cp.min(abs(l1_recon)), cp.max(abs(l1_recon)))

def sense_kernel_non_cart(x_, img_k):
    tmp = py_bart.ops(init = 0,
                        type=1,
                        o_size=(nx, ny, 1),
                        inp = cp.asfortranarray(img_k, dtype='complex64'),
                        op= nop,
                        slice = 1)
    img_k = img_k + cp.squeeze(x_) - cp.squeeze(tmp)
    return img_k

img_k = cp.squeeze(x_)

for i in range(recon_config['sense_iter']):
    img_k = sense_kernel_non_cart(x_, img_k)
    
    img_k, scalar = utils.scale_down(img_k)
    
    img_k = img_k*0.9 + 0.1*cp.squeeze(utils.float2cplx(sess.run(op_prox, {xs: utils.cplx2float(img_k[cp.newaxis, ...])})))
    
    img_k = utils.scale_up(img_k, scalar)
    utils.save_img(abs(img_k), fig_path+'/img_k_'+str(i), cp.min(abs(img_k)), cp.max(abs(img_k)))



rss = abs(rss)/norm(rss)
x_ = cp.squeeze(x_)
psnr_zero_radial= utils.psnr(abs(x_)/norm(abs(x_)), rss)
psnr_l1_radial= utils.psnr(abs(l1_recon)/norm(abs(l1_recon)), rss)
psnr_net_radial = utils.psnr(abs(img_k)/norm(abs(img_k)), rss)
ssim_zero_radial = utils.ssim(abs(x_)/norm(abs(x_)), rss)
ssim_l1_radial = utils.ssim(abs(l1_recon)/norm(abs(l1_recon)), rss)
ssim_net_radial = utils.ssim(abs(img_k)/norm(abs(img_k)), rss)

metrics = {'psnr_zero_radial': psnr_zero_radial, 
           'psnr_l1_radial': psnr_l1_radial,
           'psnr_net_radial': psnr_net_radial,
           'ssim_zero_radial': ssim_zero_radial,
           'ssim_l1_radial': ssim_l1_radial,
           'ssim_net_radial': ssim_net_radial}

cp.savez(fig_path+'/results_metrics', metrics)

text_file = open(fig_path+"/metrics.txt", "w")
for item in metrics.keys():
    strs = item+' %f'%metrics[item]
    text_file.write(strs+'\n')
    print(strs)
text_file.close()

img_k = x_
curve_net = []
la = 0.1
for i in range(100):
    img_k = sense_kernel_non_cart(x_, img_k)
    
    img_k, scalar = utils.scale_down(img_k)
    
    img_k = img_k*(1-la) + la*cp.squeeze(utils.float2cplx(sess.run(op_prox, {xs: utils.cplx2float(img_k[cp.newaxis, ...])})))
    img_k = utils.scale_up(img_k, scalar)
    psnr_net_radial = utils.psnr(abs(img_k)/norm(abs(img_k)), rss)
    ssim_net_radial = utils.ssim(abs(img_k)/norm(abs(img_k)), rss)
    curve_net.append([psnr_net_radial, ssim_net_radial])


img_k = x_
curve_s = []
la = 0.
for i in range(100):
    img_k = sense_kernel_non_cart(x_, img_k)
    
    img_k, scalar = utils.scale_down(img_k)
    
    img_k = img_k*(1-la) + la*cp.squeeze(utils.float2cplx(sess.run(op_prox, {xs: utils.cplx2float(img_k[cp.newaxis, ...])})))
    img_k = utils.scale_up(img_k, scalar)
    psnr_net_radial = utils.psnr(abs(img_k)/norm(abs(img_k)), rss)
    ssim_net_radial = utils.ssim(abs(img_k)/norm(abs(img_k)), rss)
    curve_s.append([psnr_net_radial, ssim_net_radial])

curve_s = cp.array(curve_s)
curve_net = cp.array(curve_net)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), gridspec_kw={'width_ratios': [1, 1]})

title_size = 14
label_size = 12
itera = 100

ax1.plot(cp.arange(0,itera), curve_net[:,0], label='lamda=0.1')
ax1.plot(cp.arange(0,itera), curve_s[:,0], label='lamda=0')
ax1.set_xlabel('iteration', fontsize=label_size)
ax1.set_ylabel('PSNR', fontsize=label_size)
ax1.grid(color='gray', linestyle='dashed')

ax2.plot(cp.arange(0,itera), curve_net[:,1], label="lamda=0.1")
ax2.plot(cp.arange(0,itera), curve_s[:,1], label='lamda=0')
ax2.set_xlabel('iteration', fontsize=label_size)
ax2.set_ylabel('SSIM', fontsize=label_size)
ax2.grid(color='gray', linestyle='dashed')

fig.suptitle('The evolution of SSIM and PSNR during interations', fontsize=title_size)
plt.legend()

plt.savefig(fig_path+'/convergence_c.pdf')
plt.savefig(fig_path+'/convergence_c.png')

