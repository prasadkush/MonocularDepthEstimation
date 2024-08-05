from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch

import numpy as np  
import pandas as pd
from PIL import Image
from io import BytesIO
import random
from itertools import permutations
import os
import cv2
import pdb


def get_inverse_transforms(dataset='kitti'):
	"""
	Get inverse transforms to undo data normalization
	"""
	if dataset == 'kitti':
		inv_normalize_color = transforms.Normalize((-0.36878809/0.31531613, -0.36893588/0.31266074, -0.36062746/0.31447971),
	(1/0.31531613, 1/0.31266074, 1/0.314479710))
	#inv_normalize_depth = transforms.Normalize(
	#mean=[-0.480/0.295],
	#std=[1/0.295]
	#)

	return inv_normalize_color

def get_tensor_to_image_transforms(dataset='kitti'):
	"""
	Get transforms to go from Pytorch Tensors to PIL images that can be displayed
	"""
	tensor_to_image = transforms.ToPILImage()
	inv_normalize_color, inv_normalize_depth = get_inverse_transforms(dataset=dataset)
	return (transforms.Compose([inv_normalize_color,tensor_to_image]),
			transforms.Compose([inv_normalize_depth,tensor_to_image]))

#mean:  [0.38399986 0.39878138 0.3793309 ]
#std:  [0.32906724 0.31968708 0.31093021]

#std:  [0.30721843 0.31161108 0.3070735 ]
#mean:  [0.4326707  0.4251328  0.41189488]


def get_color_transform(dataset='kitti'):
	if dataset == 'kitti':
		color_transform = transforms.Compose([
		transforms.ConvertImageDtype(torch.float),
		transforms.Normalize((0.38399, 0.39878, 0.37933), (0.32906, 0.31968, 0.3109)),
		])
	elif dataset == 'CamVid':
		color_transform = transforms.Compose([
		transforms.ConvertImageDtype(torch.float),
		transforms.Normalize((0.4326707, 0.4251328, 0.41189488), (0.30721843, 0.31161108, 0.3070735)),
		])
	return color_transform

class getDataset(Dataset):
	"""
	The Dataset class 

	Arguments:
		data (int): list of tuples with data from the zip files
		is_mono (boolen): whether to return monocular or stereo data
		start_idx (int): start of index to use in data list  
		end_idx (int): end of i
	"""
	def __init__(self, rawdatapath=None, depthdatapath=None, max_depth=80, pct=1.0, train_val_split=1.0, dataset='kitti', data_augment_flip=0, data_augment_brightness_color=0, gt_present=True, mode='train'):
		self.start_idx = 0
		self.color_transform = transforms.Compose([
		transforms.ConvertImageDtype(torch.float),
		transforms.Normalize((0.38399, 0.39878, 0.37933), (0.32906, 0.31968, 0.3109)),
		])
		self.ConvertImageDtype = transforms.ConvertImageDtype(torch.float)
		if dataset == 'kitti':
			self.Normalize = transforms.Normalize((0.36878809, 0.36893588, 0.36062746),  (0.31531613, 0.31266074, 0.31447971))
	   # self.data = []
		self.data_orig = []
		self.images = []
		self.imgw = 480
		self.imgh = 360
		self.max_depth = max_depth
		self.dataset = dataset
		self.depth_data = []
		self.samples = []
		self.num_orig_samples = 0
		self.num_depth_data = []
		if gt_present == True:
			print('depth data')
			dircount = 0
			totalcount = 0
			for dirname in depthdatapath:
				dirnamerawdata = rawdatapath[dircount]
				print('dirname: ', dirname)
				print('dirnamerawdata: ', dirnamerawdata)
				count = 0
				for filename in os.listdir(dirname):
					filepath = os.path.join(dirname, filename)
					#print('filename: ', filename)
					img = np.array(Image.open(filepath), dtype=int)
					assert(np.max(img) > 255)
					img = img.astype(np.float) / 256.
					img = np.clip(img,0,self.max_depth)
					#img = cv2.imread(filepath)
					imgnew = cv2.resize(img, (480, 360), interpolation=cv2.INTER_NEAREST)
					t = torch.from_numpy(imgnew)
					self.depth_data.append(t)
					filepathrawdata = os.path.join(dirnamerawdata, filename)
					img = cv2.imread(filepathrawdata)
					imgnew = cv2.resize(img, (480, 360), interpolation=cv2.INTER_NEAREST)
					imgnew = imgnew.transpose(2,0,1)
					t = torch.from_numpy(imgnew)
					t = self.ConvertImageDtype(t)
					self.data_orig.append(t)
					t = self.Normalize(t)
					self.images.append(t)
					sample = {'image': self.images[totalcount], 'gt': self.depth_data[totalcount], 'original': self.data_orig[totalcount]}
					self.samples.append(sample)
					totalcount += 1
					count += 1
				print('num data: ', count)
				self.num_depth_data.append(count)
				dircount += 1
			#self.end_idx = int(pct * train_val_split * len(self.depth_data))
			#self.depth_data = self.depth_data[0:self.end_idx]
			print('totalcount: ', totalcount)
			#print('num depth_data: ', len(self.depth_data))
			print('len(self.samples): ', len(self.samples))
		if gt_present == False:
			print('raw images')
			dircount = 0
			totalcount = 0
			for dirname in rawdatapath:
				print('dirname: ', dirname)
				count = 0
				i = 0
				for filename in os.listdir(dirname):
					i = i + 1
					filepath = os.path.join(dirname, filename)
					#print('filename: ', filename)
					img = cv2.imread(filepath)
					imgnew = cv2.resize(img, (480, 360), interpolation=cv2.INTER_NEAREST)
					imgnew = imgnew.transpose(2,0,1)
					t = torch.from_numpy(imgnew)
					t = self.ConvertImageDtype(t)
					self.data_orig.append(t)
					t = self.Normalize(t)
					self.images.append(t)
					sample = {'image': self.images[totalcount], 'original': self.data_orig[totalcount]}
					self.samples.append(sample)
					totalcount += 1
					count += 1
				dircount += 1	
				print('num images: ', count)
			print('total images: ', totalcount)
		self.end_idx = int(pct * len(self.samples))
		self.samples = self.samples[0:self.end_idx]
		self.num_orig_samples = len(self.samples)
		print('num_orig_samples: ', self.num_orig_samples)

#			for filename in os.listdir(depthdatapath):
#				i = i + 1
#				filepath = os.path.join(depthdatapath, filename)
#				print('filename: ', filename)
#				img = np.array(Image.open(filepath), dtype=int)
#				assert(np.max(img) > 255)
#				img = img.astype(np.float) / 256.
#				img = np.clip(img,0,self.max_depth)
#				#img = cv2.imread(filepath)
#				imgnew = cv2.resize(img, (480, 360), interpolation=cv2.INTER_NEAREST)
#				t = torch.from_numpy(imgnew)
#				self.depth_data.append(t)
#				sample = {'image': self.images[i-1], 'gt': self.depth_data[i-1], 'original': self.data_orig[i-1]}
#				self.samples.append(sample)
#				if i == self.end_idx:
#					break
#		else:
#			for i in range(self.end_idx):
#				sample = {'image': self.images[i], 'original': self.data_orig[i]}
#				self.samples.append(sample)        

		if data_augment_brightness_color > 0:
			#auginds = np.random.randint(0,self.end_idx, size=int(min(data_augment_brightness_color,1)*(self.end_idx)))
			auginds = random.sample(range(self.num_orig_samples), int(min(data_augment_brightness_color,1)*self.num_orig_samples))
			inv_transform = get_inverse_transforms(dataset)
			for i in range(len(auginds)):
				sample = self.samples[auginds[i]]
				augimage = sample['image']
				#print('sample.keys: ', sample.keys())
				if gt_present == True:
					augdepth = sample['gt']
				augorig = sample['original']
				brighness_aug = 0.8 + 0.3*torch.rand(1)
				augimage = inv_transform(sample['image'])
				augimage = brighness_aug*augimage
				color_aug = 0.8 + 0.3*torch.rand((3,1,1))
				augimage = color_aug*augimage
				augimage = torch.clip(augimage, min=0, max=1)
				augimage = self.Normalize(augimage)
				augorig = brighness_aug*augorig
				augorig = color_aug*augorig
				augorig = torch.clip(augorig,min=0,max=1)
				if gt_present == True:
					sample = {'image': augimage, 'gt': augdepth, 'original': augorig}
				else:
					sample = {'image': augimage, 'original': augorig}
				#ind = np.random.randint(0,len(self.samples))
				#self.samples.insert(ind, sample)
				self.samples.append(sample)
		#self.end_idx = len(self.samples) - 1

		if data_augment_flip > 0:
			auginds = random.sample(range(self.num_orig_samples), int(min(data_augment,1)*self.num_orig_samples))
			#auginds = np.random.randint(0,self.end_idx, size=int(0.20*self.end_idx))
			for i in range(len(auginds)):
				sample = self.samples[auginds[i]]
				augimage = sample['image']
				if gt_present == True:
					augdepth = sample['gt']
				augorig = sample['original']
				augimage = torch.flip(augimage, [2])
				augdepth = torch.flip(augdepth, [1])
				augorig = torch.flip(augorig, [2])
				
				#print('image: ', sample['image'][0,0:5,:])
				#print('augimage: ', augimage[0,0:5,:])
				#print('semantic: ', sample['semantic'][0:5,:])
				#print('augsem: ', augsem[0:5,:])
				#print('original: ', sample['original'][0,0:5,:])
				#print('augorig: ', augorig[0,0:5,:])
				#cv2.imshow('image: ', sample['image'].numpy().transpose((1,2,0)))
				if gt_present == True:
					sample = {'image': augimage, 'gt': augdepth, 'original': augorig}
				else:
					sample = {'image': augimage, 'original': augorig}
				#ind = np.random.randint(0,len(self.samples))
				self.samples.append(sample)     
		#print('self.end_idx: ', self.end_idx)
		random.shuffle(self.samples)
		print('len(self.samples): ', len(self.samples))
		self.images = []
		self.depth_data = []
		self.data_orig = []
		#print('self.end_idx + int(data_augment_brightness_color*self.end_idx): ', self.end_idx + int(data_augment_brightness_color*self.end_idx))



	def __getitem__(self, idx):
		return self.samples[idx] # TODO 

	def __len__(self):
		return len(self.samples) # TODO 


def get_data_loaders(path,  
					batch_size=1, 
					train_val_split=1.0, 
					pct_dataset=1.0):
	"""
	The function to return the Pytorch Dataloader class to iterate through
	the dataset. 

	Arguments:
		is_mono (boolen): whether to return monocular or stereo data
		batch_size (int): batch size for both training and testing 
		train_test_split (float): ratio of data from training to testing
		pct_dataset (float): percent of dataset to use 
	"""
	training_dataset = SegnetDataset(path, pct, train_val_split) # TODO 
	#testing_dataset = DepthDatasetMemory(data, is_mono, test_start_idx, test_end_idx) # TODO 

	#return (DataLoader(training_dataset, batch_size, shuffle=True, pin_memory=True),
	#        DataLoader(testing_dataset, batch_size, shuffle=False, pin_memory=True))

	return DataLoader(training_dataset, batch_size, shuffle=True, pin_memory=True)


