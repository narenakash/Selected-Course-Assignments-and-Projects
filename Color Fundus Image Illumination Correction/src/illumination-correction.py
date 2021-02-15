from helper_functions import *
import numpy as np
import scipy.interpolate
from Q3a import retinex

even_spaced_idx = lambda arr,n: np.round(np.linspace(0, len(arr) - 1, n)).astype(int)
none_value = -1

def Interpolate(data):
	mask = (data!=none_value)
	xx, yy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
	xym = np.vstack( (np.ravel(xx[mask]), np.ravel(yy[mask])) ).T
	data2 = np.ravel( data[mask] )
	interp2 = scipy.interpolate.NearestNDInterpolator(xym, data2)
	result2 = interp2(np.ravel(xx), np.ravel(yy)).reshape( xx.shape )
	return result2

def window_mu_sigma(image,points,w=100):
	shape = image.shape[:2]
	H,W = shape
	mu = np.full(shape,none_value,dtype='float')
	sigma = np.full(shape,none_value,dtype='float')
	for point in points:
		y,x = point
		y1,y2,x1,x2 = [
			np.clip(y-w//2,0,H),
			np.clip(y+w//2,0,H),
			np.clip(x-w//2,0,W),
			np.clip(x+w//2,0,W),
		]
		a = image[y1:y2,x1:x2]
		mu[y,x] = np.mean(a)
		sigma[y,x] = np.std(a)
	mu = Interpolate(mu)
	sigma = Interpolate(sigma)
	return mu,sigma

def colour_retinal_image_enhancement(image):
	rgb_img = image.copy()
	img = rgb_img[:,:,1]

	eye_ball_mask = binary_filter( (img>=(img.min()+0.01)), 'open' )
	H,W = img.shape[:2]
	middle_row = eye_ball_mask[H//2]
	eye_ball_radius = (W-(middle_row[::-1]!=0).argmax()-(middle_row!=0).argmax())//2
	img = pad_image(img, (eye_ball_radius*2,W))
	H,W = img.shape[:2]

	x,y = get_coord_arrays(img.shape)
	x-=W//2
	y-=H//2
	radial_gradient = (np.sqrt((x**2)+(y**2)))
	radial_gradient = np.round(radial_gradient)

	points = []
	r_max = min(H,W)//2
	C = 1/100
	D = 3
	roll_count = 0
	for r in range(r_max):
		a = np.vstack(np.where(radial_gradient==r)).T
		num_elements = int(C*(r**2))
		roll_count += int(r)*D
		points.append(np.roll(a,roll_count,axis=0)[even_spaced_idx(a,num_elements)])

	points = np.concatenate(points).astype('int')
	grid_mask = np.zeros_like(img)
	grid_mask[points.T[0],points.T[1]] = 1
	grid_mask = center_crop(grid_mask, (rgb_img.shape[:2]))
	img = center_crop(img, (rgb_img.shape[:2]))
	points = np.vstack(np.where(grid_mask>0)).T

	mu,sigma = window_mu_sigma(img,points)
	D = (img - mu)/(sigma+1e-7)
	background = D<D.mean()
	D[~background] = none_value
	D = Interpolate(D)
	D = normalize(D)
	g_corr = center_crop(D, rgb_img.shape[:2])
	v = np.max(rgb_img,axis=-1)
	rgb_img_new = rgb_img * np.transpose([g_corr/(v+1e-7) for i in range(3)], (1,2,0))
	rgb_img_new[eye_ball_mask==0] = 0
	return rgb_img_new

def main():
	img = read_image('../images/retina.jpg', img_size=(512,0), color='rgb')
	crie_out = colour_retinal_image_enhancement(img)
	retinex_out = retinex(img)
	image_grid([img,crie_out,retinex_out],['Original','Enhanced (CRIE Paper)','Enhanced (Retinex)'])

if __name__ == '__main__':
	main()