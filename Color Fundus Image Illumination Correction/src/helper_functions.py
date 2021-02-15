import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.morphology as mp
import os

def normalize(image,method='max'):
	image = image.astype('float')
	if method=='255':
		return image/255
	else:
		return (image-image.min())/(image.max()-image.min())

def read_image(image_path, img_size=None, floatify=True, color='rgb'):
	img = cv2.imread(image_path, cv2.IMREAD_COLOR)
	if color=='rgb':
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	if color=='gray':	
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if img_size is not None:
		H,W = img.shape[:2]
		h,w = img_size
		if h==0:
			h = int(H*w/W)
		if w==0:
			w = int(W*h/H)
		img = cv2.resize(img, (w,h))
	if floatify:
		img = img.astype('float')/255
	return img

def show_image(image,show=True,axis=False):
	plt.imshow(image, cmap='gray')
	if not axis:
		plt.axis('off')
	if show:
		plt.show()

def image_grid(images, titles=None, subp=None, noaxis=False):
	if subp==None:
		subp = 10+len(images)
	if type(subp)==int:
		subp = [int(i) for i in str(subp)]
	rows, cols = subp
	for i, image in enumerate(images):
		plt.subplot(rows,cols,i+1)
		if titles is not None:
			plt.title(titles[i])
		show_image(image,False,noaxis)
	plt.show()

def get_coord_arrays(shape):
	H,W = shape[:2]
	x = np.tile(np.arange(W), (H,1)).astype('float')
	y = np.tile(np.arange(H).reshape(-1,1), (1,W)).astype('float')
	return x,y

def find_error(original, reconstructed, method='mse'):
	if method=='mse':
		return np.mean((original-reconstructed)**2)

def binary_filter(image, operation='open', kernel=None):
	image = (normalize(image)*255).astype('uint8')
	if kernel is None:
		kernel = mp.disk(3)
	out = None
	if operation=='open':
		out = cv2.dilate(cv2.erode(image, kernel), kernel)
	if operation=='close':
		out = cv2.erode(cv2.dilate(image, kernel), kernel)
	return normalize(out)

def pad_image(image, size, location='center'):
	new_image = np.zeros(size, image.dtype)
	H,W = new_image.shape
	h,w = image.shape
	if location=='center':
		x0,y0 = (W-w)//2,(H-h)//2
		new_image[y0:y0+h,x0:x0+w] = image
	return new_image

def center_crop(image, size):
	h,w = size
	H,W = image.shape[:2]
	x0,y0 = (W-w)//2,(H-h)//2
	return image[y0:y0+h,x0:x0+w]