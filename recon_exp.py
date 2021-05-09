import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import utils
from recon_pipe import pipe
import ops
from proximator import proximator
from dncnn import dncnn

import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.io import loadmat
from tqdm import tqdm
recon_config = utils.load_config('recon_config.yaml')
model_config = utils.load_config('config.yaml')
save_path = os.path.join('./results', recon_config['expr_name'])
if not os.path.exists(save_path):
    os.mkdir(save_path)
utils.save_config(recon_config, save_path)

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


def recon(kspace_path, mask):
    ksp = loadmat(kspace_path)['kspace']
    rss = loadmat(kspace_path)['rss']

    nx = recon_config['nx']
    ny = recon_config['ny']

    nr_original_coils = recon_config['nr_original_coils']
    nr_comb_coils = recon_config['nr_comb_coils']

    mask = utils.gen_mask_1D(ratio=0.3, center=22)
    #mask = utils.mask2d(256, 256, 20, 0.2)
    und_ksp = mask[..., cp.newaxis] * ksp
    #utils.save_img(abs(mask), save_path+'/mask', 0, 1)

    #coilsen = utils.bart(1, 'ecalib -m1 -r20 -c0.001', und_ksp[cp.newaxis, ...])
    coilsen = utils.bart(1, 'caldir 20', und_ksp[cp.newaxis, ...])
    coilsen = cp.squeeze(coilsen)
    l1_recon = utils.bart(1, 'pics -R W:3:0:%f -d5 -i100 -e'%(recon_config['bart_l1_lambda']),  und_ksp[:,:,cp.newaxis,...], coilsen[:,:,cp.newaxis,:]) 
    utils.save_img(abs(l1_recon), save_path+'/l1_recon', cp.min(abs(l1_recon)), cp.max(abs(l1_recon)))
    utils.save_img(abs(rss), save_path+'/rss', cp.min(abs(rss)), cp.max(abs(rss)))
    def sense_kernel_cart(x_, img_k):
        tmp = ops.A_cart(img_k, coilsen, mask, [nx, ny])
        tmp = ops.AT_cart(tmp, coilsen, mask, [nx, ny])
        img_k = img_k + cp.squeeze(x_) - cp.squeeze(tmp)
        return img_k

    zero_filled = ops.AT_cart(und_ksp, coilsen, mask, [nx, ny])

    utils.save_img(abs(zero_filled), save_path+'/zero_filled', cp.min(abs(zero_filled)), cp.max(abs(zero_filled)))

    img_k = zero_filled
    rss = abs(rss)/norm(abs(rss))
    for i in range(50):
        img_k = sense_kernel_cart(zero_filled, img_k)
        
        img_k, scalar = utils.scale_down(img_k)
        img_k = img_k+ 0.001*cp.random.randn(256, 256)
        tmp = img_k
        img_k = tmp - 0.01*(cp.squeeze(utils.float2cplx(sess.run(op_prox, {xs: utils.cplx2float(img_k[cp.newaxis, ...])})))- tmp)
        img_k = img_k*0.7 + 0.3*cp.squeeze(utils.float2cplx(sess.run(op_prox, {xs: utils.cplx2float(img_k[cp.newaxis, ...])})))

        img_k = utils.scale_up(img_k, scalar)
        #utils.save_img(abs(img_k), save_path+'/img_k_'+str(i), cp.min(abs(img_k)), cp.max(abs(img_k)))
        #psnr_net = utils.psnr(abs(img_k)/norm(abs(img_k)), rss)
        #print("psnr_net %fdB"% psnr_net)
        
    psnr_l1 = utils.psnr(abs(l1_recon)/norm(abs(l1_recon)), rss)
    psnr_net = utils.psnr(abs(img_k)/norm(abs(img_k)), rss)
    ssim_l1 = utils.ssim(abs(l1_recon), abs(rss))
    ssim_net = utils.ssim(abs(img_k), abs(rss))
    print("psnr_l1 %fdB"% psnr_l1)
    print("psnr_net %fdB"% psnr_net)
    return psnr_l1, psnr_net, ssim_l1, ssim_net

compute_metrics = False
if compute_metrics:
    test_files = utils.list_data_sys(model_config['testing_data_path'], model_config['dataset_suffix'], ex_pattern='resam')
    metrics = []
    for path in tqdm(test_files):
        mask = utils.gen_mask_1D(ratio=0.3, center=22)
        metrics.append(recon(path, mask))

mask = utils.gen_mask_1D(ratio=0.3, center=22)
kspace_path = '/media/jason/brain_mat/brain_mat/test/LI, Dahui_T2S_1.mat'
a = recon(kspace_path, mask)

#a = recon(kspace_path = '/media/jason/brain_mat/brain_mat/test/LI, Dahui_T2S_1.mat')
#print("psnr_l1 %fdB"% psnr_l1)
#print("psnr_net %fdB"% psnr_net)
#utils.save_img(abs(mask), save_path+'/mask', 0, 1)