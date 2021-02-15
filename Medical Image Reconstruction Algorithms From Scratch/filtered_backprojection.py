#!/usr/bin/env python
# coding: utf-8

# # Medical Image Analysis
# ### Assignment 01: Filtered Backprojection
# 
# - Name:  Naren Akash R J
# - Roll Number: 2018-111-020

# In[20]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, ifft, fftfreq

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, warp
from scipy.interpolate import interp1d
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import operator 
from skimage.util import random_noise


# In[21]:


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


# In[22]:


def inverse_radon_transform(radon_image, theta=None, interpolation='cubic', filt='ramp'):
    output_size = radon_image.shape[0]
    th = np.deg2rad(theta)

    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * radon_image.shape[0]))))
    pad_width = ((0, projection_size_padded - radon_image.shape[0]), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)
    f = fftfreq(projection_size_padded).reshape(-1, 1)   
    omega = 2 * np.pi * f
    
    if filt == 'ramp':
        fourier_filter = 2 * np.abs(f) 
    elif filt == 'hamming':
        fourier_filter = np.expand_dims(np.hamming(len(f)), axis=1)
    elif filt == 'hanning':
        fourier_filter = np.expand_dims(np.hanning(len(f)), axis=1)
    
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0))
    radon_filtered = radon_filtered[:radon_image.shape[0], :]
    
    reconstructed = np.zeros((output_size, output_size))
    
    mid_index = radon_image.shape[0] // 2
    [X, Y] = np.mgrid[0:output_size, 0:output_size]
    xpr = X - int(output_size) // 2
    ypr = Y - int(output_size) // 2
    
    for i in range(len(theta)):
        t = ypr * np.cos(th[i]) - xpr * np.sin(th[i])
        x = np.arange(radon_filtered.shape[0]) - mid_index
        if interpolation == 'linear':
            backprojected = np.interp(t, x, radon_filtered[:, i],
                                      left=0, right=0)
        else:
            interpolant = interp1d(x, radon_filtered[:, i], kind=interpolation,
                                   bounds_error=False, fill_value=0)
            backprojected = interpolant(t)
        reconstructed += backprojected

    return reconstructed * np.pi / (2 * len(th))


# In[23]:


def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


# In[28]:


image = shepp_logan_phantom()
# image = rescale(image, scale=0.4, mode='reflect', multichannel=False)
# image= random_noise (image)

theta = np.arange(180)
sinogram = radon(image, theta, circle=False)
reconstruction_fbp = inverse_radon_transform(sinogram, theta=theta, interpolation='nearest', filt='ramp')


# In[29]:


sinogram.shape


# In[30]:


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


# In[32]:


plt.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)


# In[9]:


value_nearest = []
value_linear = []
value_cubic = []


# In[10]:


# for i in range(1, 180):
#     for filt in ['ramp', 'hamming', 'hanning']:
#         for interpolation in ['nearest', 'linear', 'cubic']:
#             theta = np.linspace(0., 180., num=i, endpoint=False)
#             reconstruction_fbp = inverse_radon_transform(sinogram, theta=theta, interpolation=interpolation, filt=filt)
#             plt.title('Filtered Backprojection Output\nInterpolation: ' + str(interpolation) + ", Views: " + str(i) + ", Filter: " + str(filt))
#             plt.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
#             plt.savefig('results/fbp/Shepp-Logan/'+ str(i) + '_' + str(interpolation) + '_' + str(filt) + '.png')
#             plt.show()


# In[11]:


values_nearest_mse_ramp = []
values_nearest_mse_hamming = []
values_nearest_mse_hanning = []

values_linear_mse_ramp = []
values_linear_mse_hamming = []
values_linear_mse_hanning = []

values_cubic_mse_ramp = []
values_cubic_mse_hamming = []
values_cubic_mse_hanning = []


# In[12]:


for i in range(1, 180, 30):
    for filt in ['ramp', 'hamming', 'hanning']:
        for interpolation in ['nearest', 'linear', 'cubic']:
            theta = np.linspace(0., 180., num=i, endpoint=False)
            reconstruction_fbp = cropND(inverse_radon_transform(sinogram, theta=theta, interpolation=interpolation, filt=filt), (image.shape))
        
        if interpolation == 'nearest':
            if filt == 'ramp':
                values_nearest_mse_ramp.append(mean_squared_error(image, reconstruction_fbp))
            elif filt == 'hamming':
                values_nearest_mse_hamming.append(mean_squared_error(image, reconstruction_fbp))
            elif filt == 'hanning':
                values_nearest_mse_hanning.append(mean_squared_error(image, reconstruction_fbp))
        elif interpolation == 'linear':
            if filt == 'ramp':
                values_linear_mse_ramp.append(mean_squared_error(image, reconstruction_fbp))
            elif filt == 'hamming':
                values_linear_mse_hamming.append(mean_squared_error(image, reconstruction_fbp))
            elif filt == 'hanning':
                values_linear_mse_hanning.append(mean_squared_error(image, reconstruction_fbp))
        elif interpolation == 'cubic':
            if filt == 'ramp':
                values_cubic_mse_ramp.append(mean_squared_error(image, reconstruction_fbp))
            elif filt == 'hamming':
                values_cubic_mse_hamming.append(mean_squared_error(image, reconstruction_fbp))
            elif filt == 'hanning':
                values_cubic_mse_hanning.append(mean_squared_error(image, reconstruction_fbp))


# In[18]:


import plotly.graph_objects as go

x = np.arange(1, 180, 30)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=values_nearest_mse_ramp, name='MSE, Nearest', line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x=x, y=values_linear_mse_ramp, name='MSE, Linear', line=dict(color='royalblue', width=4)))
fig.add_trace(go.Scatter(x=x, y=values_cubic_mse_ramp, name='MSE, Cubic', line=dict(color='green', width=4)))

fig.update_layout(title='Reconstruction Error in Filtered Backprojection',
                   xaxis_title='Number of Views',
                   yaxis_title='MSE')

fig.show()


# In[15]:


x = np.arange(1, 180, 30)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=values_cubic_mse_ramp, name='MSE, Ramp', line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x=x, y=values_cubic_mse_hamming, name='MSE, Hamming', line=dict(color='royalblue', width=4)))
fig.add_trace(go.Scatter(x=x, y=values_cubic_mse_hanning, name='MSE, Hanning', line=dict(color='green', width=4)))

fig.update_layout(title='Reconstruction Error in Filtered Backprojection',
                   xaxis_title='Number of Views',
                   yaxis_title='MSE')

fig.show()


# In[ ]:




