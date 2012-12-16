__author__ = 'liuc'

import numpy, scipy.ndimage
from numpy.ma.core import exp
from scipy.constants.constants import pi

def ssim(img1,img2):

    img1=img1.astype(float)
    gaussian_kernel_size=11
    gaussian_kernel_sigma=1.5
    gaussian_kernel = numpy.zeros((gaussian_kernel_size,gaussian_kernel_size))
    for i in range(gaussian_kernel_size):
        for j in range(gaussian_kernel_size):
            gaussian_kernel[i,j]=( 1 / (2*pi*(gaussian_kernel_sigma**2)) )*\
                                 exp(-(((i-5)**2)+((j-5)**2))/(2*(gaussian_kernel_sigma**2)))

    mu1=scipy.ndimage.filters.convolve(img1,gaussian_kernel)
    mu2=scipy.ndimage.filters.convolve(img2,gaussian_kernel)
    var1=scipy.ndimage.filters.convolve((img1-mu1)**2,gaussian_kernel)#,mode='constant',cval=0)
    var2=scipy.ndimage.filters.convolve((img2-mu2)**2,gaussian_kernel)
    mu12=mu1*mu2
    var12=scipy.ndimage.filters.convolve((img1-mu1)*(img2-mu2),gaussian_kernel)

    L=255
    K1=0.01
    K2=0.03

    C1=(K1*L)**2
    C2=(K2*L)**2

    SSIM = (2*mu12+C1)*(2*var12+C2)/((mu1**2+mu2**2+C1)*(var1+var2+C2))

    return numpy.average(SSIM)
