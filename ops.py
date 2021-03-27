from scipy.fft import fftshift, ifft2, fft2
import numpy as np

def mfft2(x, dims, axis=(0,1)):
    nx, ny = dims
    return fftshift(fft2(fftshift(x), axes=axis))/np.sqrt(nx*ny)

def mifft2(x, dims, axis=(0,1)):
    nx, ny = dims
    return fftshift(ifft2(fftshift(x), axes=axis))*np.sqrt(nx*ny)

def A_cart(img, coilsen, mask, dims):
    """
    forward cartesian A 
    """
    coil_img = coilsen*img[..., np.newaxis]
    kspace = mfft2(coil_img, dims)
    kspace = np.multiply(kspace, mask[...,np.newaxis])
    return kspace

def AT_cart(kspace, coilsen, mask, dims):
    """
    adjoint cartesian AT
    """
    coil_img = mifft2(kspace*mask[...,np.newaxis], dims)
    coil_sum = np.sum(coil_img*np.conj(coilsen),axis=2)
    return coil_sum