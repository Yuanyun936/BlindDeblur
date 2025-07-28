import numpy as np
from PIL import Image
from numpy.fft import fft2
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# functionName implies a torch version of the function
def fftn(x):
	x_fft = torch.fft.fftn(x,dim=[2,3])
	return x_fft

def ifftn(x):
	return torch.fft.ifftn(x,dim=[2,3])

def ifftshift(x):
	# Copied from user vmos1 in a forum - https://github.com/locuslab/pytorch_fft/issues/9
	for dim in range(len(x.size()) - 1, 0, -1):
		x = torch.roll(x, dims=dim, shifts=x.size(dim)//2)
	return x

def conv_fft(H, x):
	H = H.to(x.device)
	if x.ndim > 3: 
		# Batched version of convolution
		Y_fft = fftn(x)*H.repeat([x.size(0),1,1,1])
		y = ifftn(Y_fft)
	if x.ndim == 3:
		# Non-batched version of convolution
		Y_fft = torch.fft.fftn(x, dim=[1,2])*H
		y = torch.fft.ifftn(Y_fft, dim=[1,2])
	return torch.real(y)

def conv_fft_batch(H, x, mode='circular'):
	# Batched version of convolution
	if mode == 'circular':
		Y_fft = fftn(x)*H
		y = ifftn(Y_fft)
		return y.real
	else:
		_, _ , h, w = x.size()
		_, _, h1, w1 = H.size()
		h2, w2 = h//4, w//4
		m = nn.ReflectionPad2d( (h2,h2,w2,w2) )	
		x_pad = m(x)
		Y_fft = fftn(x_pad)*H
		y = ifftn(Y_fft)
		return y.real[:,:,h2:h+h2,w2:w+w2]


def scalar_to_tens(x):
	return torch.Tensor([x]).view(1,1,1,1)

def list_to_torch(x_list, device):
	xt_list = []
	for x in x_list:
		xt_list.append(img_to_tens(x).to(device))
	return xt_list

def torch_to_list(xt_list):
	x_list = []
	for xt in xt_list:
		x_list.append(tens_to_img(xt))

	return x_list

def conv_kernel(k, x, mode='cyclic'):
	k = k.to(x.device)
	if x.dim() == 3:
		_ , h, w = x.size()
		_, h1, w1 = k.size()
		k_pad, H = psf_to_otf(k.view(1,1,h1,w1), [1,1,h,w])
		H = H.view(1,h,w).repeat(x.size(0),1,1)
		Ax = conv_fft(H,x)

		return Ax, k_pad.view(1,h,w)

def conv_kernel_symm(k, x):
	_ , h, w = x.size()
	h2, w2 = np.shape(k)
	h3, w3 = h+2*h2, w+2*w2
	m = nn.ReflectionPad2d( (h2,h2,w2,w2) )
	
	x_pad = m(x.view(1,1,h,w)).view(1,h3,w3)
	k = torch.from_numpy(np.expand_dims(k,0))
	k_pad, H = psf_to_otf(k.view(1,1,h2,w2), [1,1,h3,w3])
	H = H.view(1,h3,w3)
	Ax_pad = conv_fft(H,x_pad)
	Ax = Ax_pad[:,h2:h+h2,w2:w+w2]
	
	return Ax, k_pad.view(1,h+2*h2,w+2*w2)

def psf_to_otf(ker, size):
    if ker.shape[2] % 2 == 0:
    	ker = F.pad(ker, (0,1,0,1), "constant", 0)
    psf = torch.zeros(size, device=ker.device)
    # circularly shift
    centre = ker.shape[2]//2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    # compute the otf
    # otf = torch.rfft(psf, 3, onesided=False)
    otf = torch.fft.fftn(psf, dim=[2,3])
    return psf, otf

def fft_kernel(ker, out_tens):
    if ker.shape[2] % 2 == 0:
    	ker = F.pad(ker, (0,1,0,1), "constant", 0)
    psf = torch.zeros_like(out_tens)
    # circularly shift
    centre = ker.shape[2]//2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    # compute the otf
    # otf = torch.rfft(psf, 3, onesided=False)
    otf = torch.fft.fftn(psf, dim=[2,3])
    return otf

def p4ip_wrapper(y, k, M, p4ip, mode ='circular'):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if mode == 'symmetric':
		H, W = np.shape(y)
		H1, W1 = H//2, W//2
		y = np.pad(y, ((H1,H1),(W1,W1)), mode='symmetric')
	
	Ht = img_to_tens(k).to(device).float()	
	yt = img_to_tens(y).to(device)
	Mt = scalar_to_tens(M).to(device)
	with torch.no_grad():
		x_rec_list = p4ip(yt, Ht, Mt)
	x_rec = x_rec_list[-1]
	x_rec = np.clip(x_rec.cpu().detach().numpy(),0,1)	
	x_out = x_rec[0,0,:,:]
	if mode == 'symmetric':
		x_out = x_out[H1:H+H1, W1:W+W1]

	return x_out

	

def unet_wrapper(y, unet):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	yt = img_to_tens(y).to(device)
	with torch.no_grad():
		x_rec = unet(yt)
	x_rec = np.clip(x_rec.cpu().detach().numpy(),0,1)
	x_out = x_rec[0,0,:,:]
	return x_out

def net_wrapper(y, srn):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	yt = img_to_tens(y).to(device)
	with torch.no_grad():
		x_rec = srn(yt)
	if isinstance(x_rec, list):
		x_out = x_rec[-1]
	else:
		x_out = x_rec
	x_out = np.clip(x_out.cpu().detach().numpy(),0,1)
	x_out = x_out[0,0,:,:]
	return x_out



def p4ip_denoiser(y, M, denoiser):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	yt = img_to_tens(y).to(device)
	Mt = scalar_to_tens(M).to(device)
	with torch.no_grad():
		x_rec = denoiser(yt, Mt)
	if isinstance(x_rec, list):
		x_rec = x_rec_list[-1]
	x_rec = np.clip(x_rec.cpu().detach().numpy(),0,1)
	x_out = x_rec[0,0,:,:]
	return x_out

def L2_Loss_Gradient(x, y):
	Dx_x, Dy_x = torch.gradient(x, dim=[2,3])
	Dx_y, Dy_y = torch.gradient(y, dim=[2,3])
	L2_LOSS = nn.MSELoss()
	return L2_LOSS(Dx_x, Dx_y) + L2_LOSS(Dy_x, Dy_y)


def sharpness(x):
	Dx, Dy = torch.gradient(x, dim=[2,3])
	return ((Dx.pow(2) + Dy.pow(2)).pow(0.5)).mean()

	return x0

def tens_to_img(xt, size=None):
	if size is None:
		x_np = np.squeeze(np.squeeze(xt.detach().cpu().numpy()))
		if x_np.ndim == 3:
			return np.transpose(x_np, (1,2,0))
		else:
			return x_np
	else:
		x_np = np.squeeze(np.squeeze(xt.detach().cpu().numpy()))
		return np.reshape(x_np, size)

def img_to_tens(x, size=None):
	if x.ndim == 2:
		xt = torch.from_numpy(np.expand_dims( np.expand_dims(x,0),0))
		if size is None:
			return xt
		else:
			return xt.view(size)
	if x.ndim == 3:
		xt = torch.from_numpy(np.expand_dims(np.transpose(x, (2,0,1)),0))
		return xt

# def Kernel_L1_Loss(k_out, k_target):
# 	# Make sure the k_target is of size k_out
# 	B, _, H, W = k_out.size()
# 	_, _, H1, W1 = k_target.size()
# 	H2, W2 = H//2, W//2
# 	H3, W3 = H1//2, W1//2
# 	if H < H1:
# 		# Cropping out the center of the k_target
# 		k_target = k_target[:,:,H3-H2:H3+H2, W3-W2:W3+W2]
# 		# Making sure the kernel is at the center
		

class NoRefSharpness(nn.Module):
	"""Sharpness Metric"""
	def __init__(self, window=16, alpha=1/3):
		super(NoRefSharpness, self).__init__()
		self.window_size = window

		self.avg = nn.AvgPool2d(kernel_size = self.window_size, stride = self.window_size)
		self.dx = torch.Tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]]).view(1,1,3,3)
		self.dy = torch.Tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]).view(1,1,3,3)
		self.alpha = alpha
		
	def forward(self, x):
		b, c, h, w = x.size()
		
		h1 = 0 if (h % self.window_size == 0) else h % self.window_size
		w1 = 0 if (w % self.window_size == 0) else w % self.window_size
		h2, w2 = h - h1, w-w1
		x = x[:,:,0:h2, 0:w2]
		if c > 1:
			x = torch.mean(x, 1, True)
		
		G_x = F.conv2d(x, self.dx.to(x.device))
		G_y = F.conv2d(x, self.dy.to(x.device))

		a, b, c = self.avg(G_x.pow(2)), self.avg(G_y.pow(2)), self.avg(G_x*G_y)
		m, p = 0.5*(a+b), a*b - c.pow(2)
		s1 = (m + torch.clip(m.pow(2)- p, 0).pow(0.5))
		s2 = (m - torch.clip(m.pow(2)- p, 0).pow(0.5))
		return (s1*(s1-s2)/(s1+s2)).mean().pow(-self.alpha)

	def coherence(self, x):
		b, c, h, w = x.size()
		
		h1 = 0 if (h % self.window_size == 0) else h % self.window_size
		w1 = 0 if (w % self.window_size == 0) else w % self.window_size
		h2, w2 = h - h1, w-w1
		x = x[:,:,0:h2, 0:w2]
		if c > 1:
			x = torch.mean(x, 1, True)
		
		G_x = F.conv2d(x, self.dx.to(x.device))
		G_y = F.conv2d(x, self.dy.to(x.device))

		a, b, c = self.avg(G_x.pow(2)), self.avg(G_y.pow(2)), self.avg(G_x*G_y)
		m, p = 0.5*(a+b), a*b - c.pow(2)
		s1 = (m + torch.clip(m.pow(2)- p, 0).pow(0.5))
		s2 = (m - torch.clip(m.pow(2)- p, 0).pow(0.5))
		return ((s1-s2)/(s1+s2+1e-6)).mean()


def get_pad_and_crop(k_size):
	pad = nn.ReflectionPad2d((k_size,k_size,k_size,k_size))
	crop = lambda xt: xt[:,:,k_size:-k_size,k_size:-k_size]

	return pad, crop

		



def get_first_moments(im):
	rows, cols = np.shape(im)
	seq1 = np.repeat(np.reshape(np.arange(rows), [rows,1]), cols, axis=1)
	seq2 = np.repeat(np.reshape(np.arange(cols), [1,cols]), rows, axis=0)
	mx, my = np.mean(seq1*im)/np.mean(im), np.mean(seq2*im)/np.mean(im)
	return mx, my

def center_kernel(kernel):
	N, _ = np.shape(kernel)
	# Center the image
	mx, my = get_first_moments(kernel)
	shift_x, shift_y =  N//2-np.int32(mx), N//2-np.int32(my)
	kernel = np.roll(kernel, (shift_x, shift_y), axis=[0,1])
	return kernel
	

	