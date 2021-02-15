#!/usr/bin/env python
# coding: utf-8

# # Medical Image Analysis
# ### Assignment 01: Backprojection
# 
# - Name:  Naren Akash R J
# - Roll Number: 2018-111-020

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, ifft, fftfreq

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import radon, rescale, warp
from scipy.interpolate import interp1d


# In[2]:


def radon_transform(image, theta=np.arange(180)):
    diagonal = np.sqrt(2) * max(image.shape)
    pad = [int(np.ceil(diagonal - s)) for s in image.shape]
    new_center = [(s + p) // 2 for s, p in zip(image.shape, pad)]
    old_center = [s // 2 for s in image.shape]
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    
    center = padded_image.shape[0] // 2
    radon_image = np.zeros((padded_image.shape[0], len(theta)), dtype=image.dtype)
    
    for i, angle in enumerate(np.deg2rad(theta)):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                      [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                      [0, 0, 1]])
        rotated = warp(padded_image, R, clip=False)
        radon_image[:, i] = rotated.sum(0)
    
    return radon_image


# In[3]:


def inverse_radon_transform(radon_image, theta=None, interpolation='cubic'):
    output_size = radon_image.shape[0]
    th = np.deg2rad(theta)

    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * radon_image.shape[0]))))
    pad_width = ((0, projection_size_padded - radon_image.shape[0]), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)
    
    reconstructed = np.zeros((output_size, output_size))
    
    mid_index = radon_image.shape[0] // 2
    [X, Y] = np.mgrid[0:output_size, 0:output_size]
    xpr = X - int(output_size) // 2
    ypr = Y - int(output_size) // 2
    
    for i in range(len(theta)):
        t = ypr * np.cos(th[i]) - xpr * np.sin(th[i])
        x = np.arange(radon_image.shape[0]) - mid_index
        if interpolation == 'linear':
            backprojected = np.interp(t, x, radon_image[:, i],
                                      left=0, right=0)
        else:
            interpolant = interp1d(x, radon_image[:, i], kind=interpolation,
                                   bounds_error=False, fill_value=0)
            backprojected = interpolant(t)
        reconstructed += backprojected

    return reconstructed * np.pi / (2 * len(th))


# In[4]:


image = io.imread('custom.png')
image = rgb2gray(image)
image = rescale(image, scale=0.4, mode='reflect', multichannel=False)

theta = np.arange(180)
sinogram = radon(image, theta, circle=False)
reconstruction_fbp = inverse_radon_transform(sinogram, theta=theta, interpolation='nearest')


# In[5]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

fig.tight_layout()
plt.show()


# In[6]:


plt.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)


# In[7]:


for i in range(1, 180):
    for interpolation in ['nearest', 'linear', 'cubic']:
        theta = np.linspace(0., 180., num=i, endpoint=False)
        reconstruction_fbp = inverse_radon_transform(sinogram, theta=theta, interpolation=interpolation)
        plt.title('Backprojection Output\nInterpolation: ' + str(interpolation) + ", Views: " + str(i))
        plt.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
        plt.savefig('results/bp/CustomImage/'+ str(i) + '_' + str(interpolation) +'.png')
        plt.show()

