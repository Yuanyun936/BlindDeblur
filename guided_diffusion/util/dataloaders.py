from os import listdir
from os.path import isfile, join
import sys
sys.path.append(".")


import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils

from motionblur.motionblur import Kernel
from util.utils_torch import conv_kernel, center_kernel

from util.img_utils import clear_color
from PIL import Image
from torchvision.transforms import v2

# constants
# helpers functions

def exists(x):
	return x is not None

def default(val, d):
	if exists(val):
		return val
	return d() if callable(d) else d

def identity(t, *args, **kwargs):
	return t

def cycle(dl):
	while True:
		for data in dl:
			yield data

def has_int_squareroot(num):
	return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
	groups = num // divisor
	remainder = num % divisor
	arr = [divisor] * groups
	if remainder > 0:
		arr.append(remainder)
	return arr

def convert_image_to_fn(img_type, image):
	if image.mode != img_type:
		return image.convert(img_type)
	return image



# dataset classes

class BlurDataset(Dataset):
	def __init__(
		self,
		folder_list,
		image_size = 256 ,
		kernel_size = 64,
		kernel_list = [],
		random_crop = True,
		augment_horizontal_flip = True,
		normalize = True,
		max_intensity = 1.0
	):
		super().__init__()
		self.folder_list = folder_list
		self.ks = kernel_size
		self.image_size = image_size
		self.paths = []

		self.pad = nn.ReflectionPad2d((self.ks,self.ks,self.ks,self.ks))
		self.crop = lambda x: x[:, self.ks:-self.ks, self.ks:-self.ks]
		for folder in self.folder_list:
			for f in listdir(folder):
				if isfile(join(folder,f)):
					self.paths.append(join(folder,f))
		self.paths.sort()
		if random_crop:
			self.transform = T.Compose([
				T.RandomResizedCrop(image_size),
				T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
				T.ToTensor()
			])
		else:
			# used for testing images
			self.transform = T.Compose([
				T.RandomCrop(256), 
				T.ToTensor()])

		self.kernel_list = kernel_list
		
	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		path = self.paths[index]
		
		# # Sample a random kernel first
		if len(self.kernel_list) > 0:
			kernel = self.kernel_list[np.random.randint(0,len(self.kernel_list))]
		else:
			kernel = Kernel(size=(self.ks,self.ks), intensity=np.random.uniform(0.1, 0.95)).kernelMatrix
		kernel = np.reshape(kernel, [64,64])
		# Center and normalize the kernel
		kernel = center_kernel(kernel)
		sum_k = np.sum(kernel)
		kernel_norm = kernel/sum_k
		k_torch = torch.from_numpy(np.expand_dims(kernel_norm,0))   
		
		# Then an image
		img = self.transform(Image.open(path).convert('RGB'))
		
		# Blur the image using symmetric boundary conditions
		blur_pad, _ = conv_kernel(k_torch, self.pad(img) )
		blur = self.crop(blur_pad)
		
		return img, blur, k_torch




if __name__ == "__main__":

	# dataset = BlurDataset(folder="../datasets/Flickr2K/train/", image_size = 256,
		# kernel_size = 64)

	dataset = NonBlindDataset(folder="../datasets/Flickr2K/train/", image_size = 256,
		kernel_size = 64)


	dataloader = DataLoader(dataset, batch_size = 1)

	for i, data in enumerate(dataloader):
		img, blur, k =  data
		print(torch.min(img), torch.max(img))
		print(torch.min(blur), torch.max(blur))
		print(torch.min(k), torch.max(k))
		plt.subplot(1,3,1); plt.imshow(clear_color(img)); plt.axis('off')
		plt.subplot(1,3,2); plt.imshow(clear_color(blur)); plt.axis('off')
		plt.subplot(1,3,3); plt.imshow(clear_color(k)); plt.axis('off')
		plt.show()

		if i > 100:
			break