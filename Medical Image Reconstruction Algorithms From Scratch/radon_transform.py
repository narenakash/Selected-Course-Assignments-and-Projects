#!/usr/bin/env python
# coding: utf-8

# # Medical Image Analysis
# ### Assignment 01: Radon Transform
# 
# - Name:  Naren Akash R J
# - Roll Number: 2018-111-020

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, warp


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


image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode='reflect', multichannel=False)


# In[4]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 4))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon_transform(image, theta=theta)
sinogram_builtin = radon(image, theta=theta, circle=False)
dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
           aspect='auto')

ax3.set_title("Radon transform builtin\n(Sinogram)")
ax3.set_xlabel("Projection angle (deg)")
ax3.set_ylabel("Projection position (pixels)")
ax3.imshow(sinogram_builtin, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, sinogram_builtin.shape[0] + dy),
           aspect='auto')

fig.tight_layout()
plt.show()


# In[5]:


for i in range(1, max(image.shape)):
    theta = np.linspace(0., 180., i, endpoint=False)
    sinogram = radon_transform(image, theta=theta)
    sinogram_builtin = radon(image, theta=theta, circle=False)
    
    dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
    plt.title("Radon transform\n(Sinogram)" + "\nNumber of views: " + str(i))
    plt.xlabel("Projection angle (deg)")
    plt.ylabel("Projection position (pixels)")
    plt.imshow(sinogram, cmap=plt.cm.Greys_r,
               extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
               aspect='auto')

    fig.tight_layout()
    plt.savefig('results/radon/' + str(i) + '.png')
    plt.show()

