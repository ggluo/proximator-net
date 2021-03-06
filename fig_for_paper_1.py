import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import utils
from recon_pipe import pipe
import ops
from proximator import proximator
from dncnn import dncnn

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.io import loadmat
from tqdm import tqdm
recon_config = utils.load_config('recon_config.yaml')
model_config = utils.load_config('config.yaml')
fig_path = '/home/jason/proximator-net/figures/fig1'

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


def recon(kspace_path, mask):
    ksp = loadmat(kspace_path)['kspace']
    rss = loadmat(kspace_path)['rss']

    nx = recon_config['nx']
    ny = recon_config['ny']

    nr_original_coils = recon_config['nr_original_coils']
    nr_comb_coils = recon_config['nr_comb_coils']

    und_ksp = mask[..., np.newaxis] * ksp
    
    #coilsen = utils.bart(1, 'ecalib -m1 -r20 -c0.001', und_ksp[np.newaxis, ...])
    coilsen = utils.bart(1, 'caldir 20', und_ksp[np.newaxis, ...])
    coilsen = np.squeeze(coilsen)
    l1_recon = utils.bart(1, 'pics -R W:3:0:%f -d5 -i100 -e'%(recon_config['bart_l1_lambda']),  und_ksp[:,:,np.newaxis,...], coilsen[:,:,np.newaxis,:]) 
    
    def sense_kernel_cart(x_, img_k):
        tmp = ops.A_cart(img_k, coilsen, mask, [nx, ny])
        tmp = ops.AT_cart(tmp, coilsen, mask, [nx, ny])
        img_k = img_k + np.squeeze(x_) - np.squeeze(tmp)
        return img_k

    zero_filled = ops.AT_cart(und_ksp, coilsen, mask, [nx, ny])

    img_k = zero_filled
    rss = abs(rss)/norm(abs(rss))
    for i in range(recon_config['sense_iter']):
        img_k = sense_kernel_cart(zero_filled, img_k)
        
        img_k, scalar = utils.scale_down(img_k)
        img_k = img_k+ 0.001*np.random.randn(256, 256)
        img_k = img_k*0.8 + 0.2*np.squeeze(utils.float2cplx(sess.run(op_prox, {xs: utils.cplx2float(img_k[np.newaxis, ...])})))

        img_k = utils.scale_up(img_k, scalar)
        
    return zero_filled, l1_recon, img_k, rss

kspace_path = '/media/jason/brain_mat/brain_mat/test/LI, Dahui_T2S_1.mat'

mask_1D = utils.gen_mask_1D(ratio=0.3, center=22)
zero_filled_1D, l1_recon_1D, prox_recon_1D, rss = recon(kspace_path, mask_1D)

mask_2D = utils.mask2d(256, 256, 22, 0.2)
zero_filled_2D, l1_recon_2D, prox_recon_2D, rss = recon(kspace_path, mask_2D)

# save results
rss = abs(rss)/norm(rss)

psnr_zero_1D= utils.psnr(abs(zero_filled_1D)/norm(abs(zero_filled_1D)), rss)
psnr_l1_1D= utils.psnr(abs(l1_recon_1D)/norm(abs(l1_recon_1D)), rss)
psnr_net_1D = utils.psnr(abs(prox_recon_1D)/norm(abs(prox_recon_1D)), rss)
ssim_zero_1D = utils.ssim(abs(zero_filled_1D)/norm(abs(zero_filled_1D)), rss)
ssim_l1_1D = utils.ssim(abs(l1_recon_1D)/norm(abs(l1_recon_1D)), rss)
ssim_net_1D = utils.ssim(abs(prox_recon_1D)/norm(abs(prox_recon_1D)), rss)

psnr_zero_2D= utils.psnr(abs(zero_filled_2D)/norm(abs(zero_filled_2D)), rss)
psnr_l1_2D= utils.psnr(abs(l1_recon_2D)/norm(abs(l1_recon_2D)), rss)
psnr_net_2D = utils.psnr(abs(prox_recon_2D)/norm(abs(prox_recon_2D)), rss)
ssim_zero_2D = utils.ssim(abs(zero_filled_2D)/norm(abs(zero_filled_2D)), rss)
ssim_l1_2D = utils.ssim(abs(l1_recon_2D)/norm(abs(l1_recon_2D)), rss)
ssim_net_2D = utils.ssim(abs(prox_recon_2D)/norm(abs(prox_recon_2D)), rss)


results = {'zero_filled_1D': zero_filled_1D, 
           'l1_recon_1D': l1_recon_1D,
           'prox_recon_1D': prox_recon_1D,
           'zero_filled_2D': zero_filled_2D,
           'l1_recon_2D': l1_recon_2D,
           'prox_recon_2D': prox_recon_2D,
           'mask_1D': mask_1D,
           'mask_2D': mask_2D,
           'rss': rss}
np.savez(fig_path+'/results_images', results)

metrics = { 'psnr_zero_1D':psnr_zero_1D,
            'ssim_zero_1D':ssim_zero_1D,
            'psnr_zero_2D':psnr_zero_2D,
            'ssim_zero_2D':ssim_zero_2D,
            'psnr_l1_1D':psnr_l1_1D,
            'psnr_net_1D':psnr_net_1D,
            'ssim_l1_1D':ssim_l1_1D,
            'ssim_net_1D':ssim_net_1D,
            'psnr_l1_2D':psnr_l1_2D,
            'psnr_net_2D':psnr_net_2D,
            'ssim_l1_2D':ssim_l1_2D,
            'ssim_net_2D':ssim_net_2D}
np.savez(fig_path+'/results_metrics', metrics)

for item in results.keys():
    tmp = results[item]
    utils.save_img(abs(tmp), fig_path+'/'+item, np.min(abs(tmp)), np.max(abs(tmp)))

text_file = open(fig_path+"/metrics.txt", "w")
for item in metrics.keys():
    strs = item+' %f'%metrics[item]
    text_file.write(strs+'\n')
    print(strs)
text_file.close()