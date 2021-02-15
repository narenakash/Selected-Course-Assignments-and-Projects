from helper_functions import *
import numpy as np
from scipy.signal import convolve

def retinex(image,h=None,output_type=0,retina_masking=True):
	if h is None:
		C = 1
		kernel_size = 10
		sigma = kernel_size//5
		x,y = get_coord_arrays((kernel_size,kernel_size))
		x-=kernel_size//2
		y-=kernel_size//2
		h = C*np.exp(-(x**2 + y**2)/(2*sigma**2))
		h /= np.sum(h)

	rgb_img = image.copy()
	I_in = rgb_img[:,:,1]
	I_in += 1e-7
	a = np.log(I_in)
	b = np.log(convolve(I_in,h,'same'))
	c = convolve(a,h,'same')

	I_out1 = a - b
	I_out2 = a - c
	I_out = np.log(I_in/convolve(I_in,h,'same'))

	I_out1 = normalize(I_out1)
	I_out2 = normalize(I_out2)
	I_out = normalize(I_out)

	eye_ball_mask = None
	if retina_masking:
		eye_ball_mask = binary_filter( (I_in>=(I_in.min()+0.01)), 'open' )

	outputs = []
	v = np.max(rgb_img,axis=-1)
	for g_corr in [I_out1,I_out2,I_out]:
		rgb_img_new = rgb_img * np.transpose([g_corr/(v+1e-7) for i in range(3)], (1,2,0))
		if retina_masking:
			rgb_img_new[eye_ball_mask==0] = 0
		outputs.append(rgb_img_new)
	return outputs[output_type]

def main():
	img = read_image('../images/retina.jpg', img_size=(512,0), color='rgb')
	output = retinex(img)
	image_grid([img,output],['Original','Enhanced'])

	img = read_image('../images/skin.jpg', color='rgb')
	output = retinex(img, retina_masking=False)
	image_grid([img,output],['Original','Enhanced'])

if __name__ == '__main__':
	main()