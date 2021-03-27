import os
import sys

import tensorflow.compat.v1 as tf
from sklearn.preprocessing import MultiLabelBinarizer

import math
import numpy as np
from numpy.matlib import repmat
from scipy.io import loadmat
from scipy.linalg import norm
import scipy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import measure, io
from skvideo.io import FFmpegWriter

import subprocess as sp
import tempfile as tmp
from multiprocessing import Process, Queue

import yaml
import _pickle as cPickle
import h5py
from termcolor import colored

def list_data_sys(path, suffix, ex_pattern=None):
    """
    list all the files under path with given suffix

    Args:

    path: the folder to be listed
    suffix: suffix of file to be listed
    ex_pattern: pattern for excluding files

    Returns:
    
    array of the files' path
    """

    cmd = 'ls '
    cmd = cmd + path + '/*.' + suffix

    if ex_pattern is not None:
        cmd = cmd + ' | ' + 'grep -v ' + ex_pattern
    
    strs = sp.check_output(cmd, shell=True).decode()
    files = strs.split('\n')[:-1]

    return files

def list_data(path, suffix):
    """
    return the list of all the file with the given suffix 

    Args:

    path: the folder to be listed
    suffix: suffix of file to be listed

    Returns:

    array of the files' path
    """
    
    file_list = []
    suffix = '.' + suffix
    for r, _, f in os.walk(path):
        for file in f:
            if suffix in file:
                file_list.append(os.path.join(r, file))
    return file_list

def get_labelizer(labels_arr):
    mlb = MultiLabelBinarizer()
    mlb.fit(labels_arr)
    return mlb

def float2cplx(float_in):
     return np.asfortranarray(float_in[...,0]+1.0j*float_in[...,1], dtype='complex64')

def cplx2float(cplx_in):
     return np.asfortranarray(np.stack((cplx_in.real, cplx_in.imag), axis=-1), dtype='float32')

def gen_weights(nSpokes, N):
    """
    generate radial compensation weight
    """
    rho = np.linspace(-0.5,0.5,N).astype('float32')
    w = abs(rho)/0.5
    w = np.transpose(repmat(w, nSpokes, 1), [1, 0])
    w = np.reshape(w, [1, N, nSpokes])
    return w


def readcfl(name):
    """
    read cfl file
    """

    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline() # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split( )]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n)
    d.close()
    return a.reshape(dims, order='F') # column-major

def writecfl(name, array):
    """
    write cfl file
    """

    h = open(name + ".hdr", "w")
    h.write('# Dimensions\n')
    for i in (array.shape):
            h.write("%d " % i)
    h.write('\n')
    h.close()
    d = open(name + ".cfl", "w")
    array.T.astype(np.complex64).tofile(d) # tranpose for column-major order
    d.close()

def noise(mu, sigma, shape):
    """
    generate gaussian noise
    """
    return np.random.normal(mu, sigma, shape).astype(np.float32)

def bart(nargout, cmd, return_str=False, *args):
    """
    call bart from system command line
    """
    if type(nargout) != int or nargout < 0:
        print("Usage: bart(<nargout>, <command>, <arguements...>)")
        return None

    name = tmp.NamedTemporaryFile().name

    nargin = len(args)
    infiles = [name + 'in' + str(idx) for idx in range(nargin)]
    in_str = ' '.join(infiles)

    for idx in range(nargin):
        writecfl(infiles[idx], args[idx])

    outfiles = [name + 'out' + str(idx) for idx in range(nargout)]
    out_str = ' '.join(outfiles)

    shell_str = 'bart ' + cmd + ' ' + in_str + ' ' + out_str
    print(shell_str)
    if not return_str:
        ERR = os.system(shell_str)
    else:
        try:
            strs = sp.check_output(shell_str, shell=True).decode()
            return strs
        except:
            ERR = True


    for elm in infiles:
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    output = []
    for idx in range(nargout):
        elm = outfiles[idx]
        if not ERR:
            output.append(readcfl(elm))
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    if ERR:
        raise Exception("Command exited with an error.")

    if nargout == 1:
        output = output[0]

    return output

def psnr(img1, img2):
    """
    calculate peak SNR 
    """
    mse = np.mean((img1-img2)**2)
    pixel_max = np.max(img2)
    return 20*math.log10(pixel_max/math.sqrt(mse))

def norm_to_uint8(inp):
    maximum = np.max(inp)
    out = inp/maximum*255.0
    return out.astype(np.uint8)

def ssim(img1, img2):
    """
    calcualte similarity index between img1 and img2
    """
    img1 = abs(img1)/norm(img1)
    img2 = abs(img2)/norm(img2)
    return measure.compare_ssim(norm_to_uint8(img1), norm_to_uint8(img2))

def save_img(img, path, vmin=0., vmax=1.):
    """
    print images to pdf and png without white margin

    Args:
    img: image arrays
    path: saving path
    """
    figure=plt.imshow(img, cmap='gray', interpolation='none', vmin=vmin, vmax=vmax)
    figure.axes.get_xaxis().set_visible(False)
    figure.axes.get_yaxis().set_visible(False)
    plt.savefig(path, bbox_inches='tight', pad_inches = 0)
    plt.savefig(path+'.pdf', bbox_inches='tight', pad_inches = 0)
    plt.close()


def gen_mask_1D(ratio=0.1,center=20, ph=256, fe=256):
    """
    generate undersampling mask along 1 dimension
    Args:

    ratio: sampling ratio
    center: center lines retained
    fe: frequency encoding
    ph: phase encoding lines

    Returns:
    mask
    """
    k = int(round(ph*ratio)/2.0)
    ma = np.zeros(ph)
    ri = np.random.choice(int(ph/2-center/2), k, replace=False)
    ma[ri] = 1
    ri = np.random.choice(int(ph/2-center/2), k, replace=False)
    ma[ri+int(ph/2+center/2)] = 1
    ma[int(ph/2-center/2): int(ph/2+center/2)] = 1
    mask = np.tile(ma, [fe, 1])
    return mask
     
#
def load_config(path):
    """
    load configuration defined with yaml file
    """

    with open(path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config

def mat_loader(filename, key='rss', labelled=False):
    tmp = loadmat(filename)
    if key not in tmp.keys():
        raise Exception("Images loading failed, key doesn't match or exist!")
    else:
        image = tmp[key]

    if labelled:
        if 'label' not in tmp.keys():
            raise Exception("labels loading failed, key doesn't match or exist!")
        else:
            label = tmp['label']
            return image, label
    else:
        return image

def h5_loader(filename, key='reconstruction_rss', labelled=False):
    with h5py.File(filename, "r") as fs:
        data = list(fs[key])
        return data

def npz_loader(filename, key='rss', labelled=False):
    tmp = np.load(filename)
    if key not in tmp.keys():
        raise Exception("Images loading failed, key doesn't match or exist!")
    else:
        image = tmp[key]

    if labelled:
        if 'label' not in tmp.keys():
            raise Exception("labels loading failed, key doesn't match or exist!")
        else:
            label = tmp['label']
            return image, label
    else:
        return image

def jpg_loader(filename, key='rss', labelled=False, as_gray=True):
    img = io.imread(filename, as_gray)
    img = img.astype('float') * 256.
    return img

def slice_image(inp, shape):
    """
    slice image into pieces

    Args:
    inp: (nx, ny, _)
    shape: 

    Returns:
    
    """
    if len(inp.shape) == 3:
        nx, ny, _ = inp.shape
    if len(inp.shape) == 2:
        nx, ny = inp.shape

    if len(shape) == 3:
        sx, sy, _ = shape
    if len(shape) == 2:
        sx, sy = shape

    steps_x = int(np.ceil(float(nx)/sx))
    steps_y = int(np.ceil(float(ny)/sy))

    total = steps_x*steps_y
    pieces = np.zeros([total] + shape, dtype=inp.dtype)
     
    for x in range(steps_x):
        
        if x == (steps_x-1):
            bx = nx-sx
            ex = nx
        else:
            bx = x*sx
            ex = bx + sx

        for y in range(steps_y):

            if y == (steps_y-1):
                by = ny-sy
                ey = ny
            else:
                by = y*sy
                ey = by + sy

            pieces[x*steps_y+y, ...] = np.reshape(inp[bx:ex, by:ey], shape)
    return pieces

def stitch_image(inp, out_shape):
    """
    stitch pieces back to complete image

    Args:
        inp: (slices, sx, sy, (chns))
        out_shape: (nx, ny)

    Returns:
        out: (nx, ny, (chns))
    """

    if len(inp.shape) == 4:
        slices, sx, sy, chns = inp.shape
        img = np.zeros(out_shape+[chns], dtype=inp.dtype)

    if len(inp.shape) == 3:
        slices, sx, sy = inp.shape
        img = np.zeros(out_shape, dtype=inp.dtype)
    
    nx, ny  = out_shape

    steps_x = int(np.ceil(float(nx)/sx))
    steps_y = int(np.ceil(float(ny)/sy))

    for x in range(steps_x):
        
        if x == (steps_x-1):
            bx = nx-sx
            ex = nx
        else:
            bx = x*sx
            ex = bx + sx

        for y in range(steps_y):

            if y == (steps_y-1):
                by = ny-sy
                ey = ny
            else:
                by = y*sy
                ey = by + sy

            img[bx:ex, by:ey, ...] = inp[x*steps_y+y, ...]
    return img

def slice_image_center(inp, shape, width=256, height=256):
    """
    crop center out then slice it into pieces

    Args:
    inp: array with shape (nx, ny)
    shape: 
    
    Returns:
    """
    nx, ny, _ = inp.shape
    offset_x = (nx-width)//2
    offset_y = (ny-height)//2
    
    cropped = inp[offset_x:nx-offset_x, offset_y:ny-offset_y, ...]

    return slice_image(cropped, shape)

def transform(inp, k):
    """
    function for data augmentation
    TODO: zoom in, zoom out
    """
    if k == 0:
        return inp
    if k == 1:    
        return np.flipud(inp)
    if k == 2:
        return np.fliplr(inp)
    if k == 3:
        return np.rot90(inp, 1)
    if k == 4:
        return np.rot90(inp, 2)
    if k == 5:
        return np.rot90(inp, 3)
    if k == 6:
        return np.rot90(np.flipud(inp), 1)
    if k == 7:
        return np.rot90(np.flipud(inp), 2)
    if k == 8:
        return np.rot90(np.flipud(inp), 3)
    if k == 9:
        return np.rot90(np.fliplr(inp), 1)
    if k == 10:
        return np.rot90(np.fliplr(inp), 2)
    if k == 11:
        return np.rot90(np.fliplr(inp), 3)

def slice_volume(inp, shape):
    """
    slice 3D volume into blocks

    Args:
    inp: the original volume with shape (nx, ny, nz, chns)
    shape: the output slice shape(frames, sx, sy, chns)
    
    Returns:
    slices: (?, sx, sy, chns)
    """
    if len(shape) == 4:
        frames, sx, sy, _ = shape
    if len(shape) == 3:
        sx, sy, _ = shape
        frames = 1

    nx, ny, slices, _ = inp.shape
    inp = np.transpose(inp, [2, 0, 1, 3])

    # steps along x,y dimension
    steps_x = int(np.ceil(float(nx)/sx))
    steps_y = int(np.ceil(float(ny)/sy))
    pieces = steps_x*steps_y # number of pieces from the original slice
    
    # steps along slice dimension
    sliding_steps = slices - frames + 1

    # number of total blocks sliced into
    total = pieces*sliding_steps

    sliced_v = np.zeros([total] + shape)

    for i in range(sliding_steps):

        bf = i
        ef = i+frames
        
        for x in range(steps_x):

            if x == (steps_x-1):
                bx = nx-sx
                ex = nx
            else:
                bx = x*sx
                ex = bx + sx
            
            for y in range(steps_y):

                if y == (steps_y-1):
                    by = ny-sy
                    ey = ny
                else:
                    by = y*sy
                    ey = by + sy
                               
                sliced_v[x*steps_y+y+i*pieces, ...] = inp[bf:ef, bx:ex, by:ey, ...]

    return sliced_v

def stitch_volume(inp, shape):
    """
    combine pieces back to complete volumes
    Args:
    inp: (nx, ny, nz, chns)
    shape: the shape of original volume
    
    Returns:
    """

    nx, ny, slices, _ = shape
    inp = np.transpose(inp, [1,2,0,3])
    sx, sy, pieces, chns = inp.shape

    steps_x = int(np.ceil(float(nx)/sx))
    steps_y = int(np.ceil(float(ny)/sy))
    
    if steps_x*steps_y*slices != pieces:
        raise ValueError("Pieces input don't match the output shape")

    vol = np.zeros([nx,ny,slices, chns])

    for i in range(slices):
        for x in range(steps_x):

            if x == (steps_x-1):
                bx = nx-sx
                ex = nx
            else:
                bx = x*sx
                ex = bx + sx

            for y in range(steps_y):

                # the current location of pieces
                idx = i*steps_x*steps_y + x*steps_x + y

                if y == (steps_y-1):
                    by = ny-sy
                    ey = ny
                else:
                    by = y*sy
                    ey = by + sy

                vol[bx:ex,by:ey,i] = inp[:,:,idx, ...]

    return vol

def normalize_with_max(x, axis=(0,1), data_chns='CPLX'):
    """
    x is complex value
    x = x/(max(abs(x)))
    """
    scalor = np.max(abs(x), axis)

    if data_chns == 'CPLX':
        normalized_x = cplx2float(x/scalor)
        return normalized_x
    
    if data_chns == 'MAG':
        normalized_x = abs(x/scalor)
        return normalized_x
    

# TODO how the following normalization method will effect the prior
def normalize_with_std(x, info=False):
    """
    x is complex value
    global std normalization through read and imaginary channel
    then scaled to [-1,1]
    """

    x_float = cplx2float(x)
    x_float_mean = np.mean(x_float) 
    x_float_std = np.std(x_float)
    normalized_x = (x_float-x_float_mean)/(x_float_std+0.001)
    scalor = np.max(normalized_x)
    normalized_x = normalized_x/scalor
    if info:
        return normalized_x, 
    else:
        return normalized_x

def calculate_center_pos(nx, sx):
    idx = []
    offset = (nx-sx)//2
    for i in range(offset, sx+offset):
        for j in range(offset, sx+offset):
            idx.append(nx*i+j)
    return idx

#TODO simplify this function
def load_image(files, queue, sign, batch_size, input_shape, data_chns='CPLX', normalize_func=normalize_with_max, fileloader=mat_loader, key='rss', rseed=None, rand_s=1.0, labelizer=None):
    """
    load the batch of specified shape from the random 2D images
    """

    local_random = np.random.RandomState(rseed)
    nr_files = len(files)
    train_idx = local_random.permutation(nr_files)
    idx = 0
    
    is_next_file = True
    lefted = np.zeros((None))

    image_batch = np.zeros([int(batch_size)] + input_shape)

    if labelizer is not None:
        label_batch = np.zeros([batch_size], dtype='int32')
        lefted_label = np.zeros((None))
        center_pos = calculate_center_pos(nx=4, sx=2) # TODO: to be configurabale

    while True:
        while queue.full() == False:
            if is_next_file:
                if idx == nr_files:
                    sign.put(True)
                    idx = 0
                    train_idx = local_random.permutation(nr_files) # permute the order of file list
                    color_print("Files permuted", 'red')
                
                #print("Reading the file indexed with " + str(idx) + " " + files[train_idx[idx]])
                file_path = files[train_idx[idx]]
                if labelizer is not None:
                    vol, label = fileloader(file_path, key, True)
                else:
                    vol = fileloader(file_path, key)
                
                normalized_vol = normalize_func(vol, data_chns=data_chns)
                normalized_vol = normalized_vol*round(np.random.uniform(rand_s,1.0),4)
                #normalized_vol = transform(normalized_vol, local_random.randint(0,12))
                
                if np.isnan(normalized_vol).any():
                    raise Exception("Sorry, NAN occurred")
                
                pieces = slice_image(normalized_vol, input_shape)
                
                if labelizer is not None:
                    if 'resample' not in os.path.split(file_path)[-1]:
                        binary_labels = np.repeat(labelizer.transform([np.append(label, ['center'])]), pieces.shape[0], axis=0).astype('float32')
                    else:
                        binary_labels = np.array([], dtype='float32').reshape((0, len(labelizer.classes_)))
                        for i in range(pieces.shape[0]):
                            if i in center_pos:
                                binary_label = labelizer.transform([np.append(label, ['center'])]).astype('float32')
                            else:
                                binary_label = labelizer.transform([np.append(label, ['outer'])]).astype('float32')
                            binary_labels = np.concatenate([binary_labels, binary_label], axis=0)
                         
                if lefted.shape is ():
                    cur_remained = 0
                else:
                    cur_remained = lefted.shape[0]
                    
                is_next_file = False
                idx = idx + 1

            if batch_size <= cur_remained:
                
                image_batch = lefted[0:batch_size, ...]

                if labelizer is not None:
                    label_batch = lefted_label[0:batch_size, ...]

                if labelizer is not None:
                    queue.put((image_batch, label_batch))
                else:
                    queue.put(image_batch)

                cur_remained = cur_remained - batch_size
                lefted = lefted[batch_size:, ...]

                if labelizer is not None:
                    lefted_label = lefted_label[batch_size:, ...]

            else:
                is_next_file = True

                if lefted.shape is ():
                    lefted = pieces

                    if labelizer is not None:
                        lefted_label = binary_labels

                else:
                    lefted = np.concatenate([lefted, pieces], axis=0)

                    if labelizer is not None:
                        lefted_label = np.concatenate([lefted_label, binary_labels], axis=0)


def load_volume(files, queue, sign, batch_size, input_x_shape, file_loader=h5_loader, key='rss', rseed=None):
    """
    load the batch of specified shape from the random 3D volumes
    """

    local_random = np.random.RandomState(rseed)
    nr_files = len(files)
    train_idx = local_random.permutation(nr_files)
    idx = 0
    
    is_next_file = True
    lefted = np.zeros(None)

    while True:
        while queue.full() == False:
            single_batch = np.zeros([int(batch_size)] + input_x_shape)
            
            if is_next_file:
                
                if idx == nr_files:
                    sign.put(True)
                    idx = 0
                    train_idx = local_random.permutation(nr_files) # permute the order of file list
                print("Reading the file indexed with " + str(idx) + " " + files[train_idx[idx]])
                
                vol = file_loader(files[train_idx[idx]], key)
                scalar = np.max(abs(vol), axis=(1,2))[:, np.newaxis, np.newaxis]
                normalized_vol = vol/scalar
                if np.isnan(normalized_vol).any():
                    raise Exception("Sorry, NAN occurred")
                
                blocks = slice_volume(vol, input_x_shape)
                if lefted.shape is ():
                    cur_remained = 0
                else:
                    cur_remained = lefted.shape[0]

                is_next_file = False
                idx = idx + 1

            if batch_size <= cur_remained:

                single_batch = lefted[0:batch_size, ...]
                queue.put(single_batch)
                cur_remained = cur_remained - batch_size
                lefted = lefted[batch_size:, ...]

            else:
                is_next_file = True
                if lefted.shape is ():
                    lefted = blocks
                else:
                    lefted = np.concatenate([lefted, blocks], axis=0)


def create_dataloader_procs(datalist, 
                       train_queue,
                       sign,
                       n_thread,
                       train_procs,
                       batch_size,
                       input_shape,
                       data_chns='CPLX',
                       normalize_func=normalize_with_max,
                       fileloader=mat_loader,
                       read_func=load_image,
                       key='rss',
                       rand_scalor=1.0,
                       labelizer=None,
                       ):
    """
    create threads to read the images from hard drive and perturb them,
    the read_func must have args: datalist, train_queue, batch_size, seed
    """

    for _ in range(n_thread):
        seed = np.random.randint(1e8)
        train_proc = Process(target=read_func, 
                             args=(datalist,
                                   train_queue,
                                   sign,
                                   batch_size,
                                   input_shape,
                                   data_chns,
                                   normalize_func,
                                   fileloader,
                                   key,
                                   seed,
                                   rand_scalor,
                                   labelizer))
        train_proc.daemon = True
        train_proc.start()
        train_procs.append(train_proc)


def terminate_procs(train_procs):
    """
    terminate all the threads 
    """
    for procs in train_procs:
        procs.terminate()

def print_parameters():
    """
    print the trainable parameters info of tf graph
    """

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        print("========layer=======")
        shape = variable.get_shape()
        print("shape->",shape)
        print("shape->len",len(shape))
        variable_parameters = 1
        for dim in shape:
            print("dim->", dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)

def get_lr(step, lr, warmup_steps, hidden_size):
    """
    learning rate scheduler
    """
    lr_base = lr * 0.002 # for Adam correction
    ret = 5000. * hidden_size ** (-0.5) * \
          np.min([(step + 1) * warmup_steps ** (-1.5), (step + 1) ** (-0.5)])
    return ret * lr_base

#TODO delete this function
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

def write_video(path, vol, fps=60):
    """
    vol = [frames, width, height]
    """
    shape = vol.shape
    width = shape[1]
    height = shape[2]
    crf = 5
    vol = abs(vol)/np.max(abs(vol), axis=(1,2))[:, np.newaxis, np.newaxis]*255.0
    video = FFmpegWriter(path, 
            inputdict={'-r': str(fps), '-s':'{}x{}'.format(width,height)},
            outputdict={'-r': str(fps), '-c:v': 'libx264', '-crf': str(crf), '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'})
    video.writeFrame(vol[...,np.newaxis].astype(np.uint8))
    video.close()


def export_model(saver, sess, path, name, as_text=False, gpu_id=None):
    saver.save(sess, os.path.join(path, name))
    tf.train.write_graph(sess.graph, path, name+'.pb', as_text)
    if gpu_id is not None:
        with open(os.path.join(path, name+'_gpu_id'), 'w+') as fs:
            for i in range(len(gpu_id)):
                fs.write(gpu_id[i])
                fs.write('\t')


def save_config(x,path):
    with open(os.path.join(path, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(x, yaml_file, default_flow_style=False)


def create_train_procs(train_func, train_procs, sess, sign, step, epoch, t_queue, ops, saver, graph, config):
    """
    create one training thread
    """    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']
    
    proc = Process(target=train_func, 
                            args=(sess,
                                  sign,
                                  step,
                                  epoch,
                                  t_queue,
                                  ops,
                                  saver,
                                  graph,
                                  config))
    proc.daemon = True
    proc.start()
    train_procs.append(proc)

def check_events(status_path):
    data = load_config(status_path)
    return data['flag']

# TODO ugly but useful, hope the layout of nvidia-smi's output won't change in the future
def check_available_gpu(threshold=300):
    """
    check how many gpus are available
    """
    terminal_str = sp.check_output("nvidia-smi").decode()
    ds = terminal_str.split(' '*79)[0].split('\n')
    nr_gpus = (len(ds) - 8)//3
    begin_row = 8
    gs=[]
    for _ in range(nr_gpus):
        gs.append(int(ds[begin_row].split('|')[2].split('/')[0][:-4]) )
        begin_row = begin_row + 3
    
    ga = [i<threshold for i in gs]
    gpu_id = ''
    nr_available_gpu = sum(ga)
    for i in range(len(ga)):
        if ga[i]:
            gpu_id=gpu_id+str(i)+','

    return nr_available_gpu, gpu_id

def scale_down(x):
    """
    x is complex/float
    """
    scalor = np.max(abs(x))
    x = x/(scalor+1e-8)
    return x, scalor

def scale_up(x, scalor):
    """
    x is complex/float
    """
    return x*scalor

def pad_zeros(x, nx, ny):
    """
    pad first two dimensions with zeros
    """
    tmp = np.zeros([nx, ny]+list(x.shape[2:]), dtype='complex64')
    shape = x.shape
    offset_x = (nx - shape[0])//2
    offset_y = (ny - shape[1])//2
    tmp[offset_x:nx-offset_x, offset_y:ny-offset_y, ...] = x
    return tmp

def fill_zeros(x, nx, ny):
    """
    place zero between two continuous element of x for the first two dimensions
    """
    tmp = np.zeros([nx, ny]+list(x.shape[2:]), dtype='complex64')
    shape = x.shape
    step_x = nx//shape[0]
    step_y = ny//shape[1]
    tmp[0::step_x, 0::step_y, ...] = x
    return tmp

def split_logits(x, logits, nr_mix):
    """
    function to spilt distribution parameters
    TODO: compatible with images with the different number of channels
    """
    xs = list(x.shape) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = list(logits.shape) # predicted distribution, e.g. (B,32,32,60)
    nr_mix = int(ls[-1] / 6) # here and below: unpacking the params of the mixture of logistics
    probs = scipy.special.softmax(logits[:,:,:,:nr_mix], axis=-1)
    l = np.reshape(logits[:,:,:,nr_mix:], xs[:-1] + [5,nr_mix])
    means = l[:,:,:,:2,:]
    scales = np.exp(np.maximum(l[:,:,:,2:4,:], -7.))
    coeffs = np.tanh(l[:,:,:,4:,:])
    x = np.reshape(x, xs + [1]) + np.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = np.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
    means = np.concatenate([np.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2],axis=3)
    return probs, means, scales

def logistic_pdf(mean, scale, x_points):
    """
    generate plotting arrays for logistic probablistic density function
    Args:
        mean
        scale  
        x_min
        x_max
    Returns:
        discrete pdf curves
    """

    if not isinstance(mean, np.ndarray):
        mean = np.array(mean)
    if not isinstance(scale, np.ndarray):
        scale = np.array(scale)

    shape = mean.shape
        
    mean  = mean.reshape([np.prod(shape),1])
    scale = scale.reshape([np.prod(shape),1])

    tmp = np.exp((mean-x_points[np.newaxis,:])/scale)
    f = tmp/(scale*(1+tmp)**2)

    return f.reshape(list(shape)+[len(x_points)])

def my_imports(module_name):
    globals()[module_name] = __import__(module_name)

def init_config(path):
    """
    for the start of reconstruction
    """
    recon_config = load_config(path)
    model_config = load_config(os.path.join(recon_config['model_path'], 'config.yaml'))
    
    return recon_config, model_config

def bit_mask(dims):
    return np.sum(np.power(2, dims))

def get_label(model_config, recon_config):

    if model_config['conditional']:
        
        labelizer = get_labelizer(model_config['labels'])
        center_pos = calculate_center_pos(4,2)
        binary_labels = np.array([], dtype='float32').reshape((0, len(labelizer.classes_)))
        for i in range(recon_config['batch_size']):
            if i in center_pos:
                binary_label = labelizer.transform([['T2S','hku','center']]).astype('float32')
            else:
                binary_label = labelizer.transform([['T2S','hku','outer']]).astype('float32')
            binary_labels = np.concatenate([binary_labels, binary_label], axis=0)
    else:
        binary_labels=None

    return binary_labels

def color_print(strs, color='red', bold=True):
    if bold:
        print(colored(strs, color, attrs=['bold']))
    else:
        print(colored(strs, color))