import cv2
import torch
import torch.nn as nn
from data import getDataset, get_color_transform, get_inverse_transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from preprocess import get_mean_std
from losses import depth_loss
from depthmodel import NeWCRFDepth
from depth_metrics import RMSE, abs_rel_error, sq_rel_error
from Exceptions import ModelPathrequiredError
import os
import pdb
import time
import matplotlib.cm as cm
import matplotlib as mpl



def visualize_depth(depth_img, cmap='magma_r', vmin=0.01, vmax=85):
	cmapper = plt.get_cmap(cmap)
	vmin = (torch.min(depth_img)).numpy()
	vmax = (torch.max(depth_img)).numpy()
	#print('vmin: ', vmin)
	#print('vmax: ', vmax)
	value = (depth_img.numpy() - vmin)/(vmax - vmin)
	value[depth_img == 0] = 0
	value = cmapper(value, bytes=True)
	
	#print('type value: ', type(value))
	img = value[:,:,0:3]
	#print('np.min value: ', np.min(img))
	#print('np.max value: ', np.max(img))
	return img


def predict_and_visualize(dataloader, batch_size=4, dataset_name='kitti', imgdir=None, model=None, modelpath=None, modelname='NeWCRFDepth', gt_present=True, save_images=False, criterion=depth_loss, epoch=None):
	if model == None and modelpath == None:
		raise ModelPathrequiredError("Both model and modelpath are None")
	elif model == None:
		if modelname == 'NeWCRFDepth':
			model = NeWCRFDepth()
	if modelpath != None:
		checkpoint = torch.load(modelpath)
		model.load_state_dict(checkpoint['model_state_dict'])
		print('checkpoint[epoch]: ', checkpoint['epoch'])
		print('checkpoint[loss]: ', checkpoint['loss'])
	#mean, std = get_mean_std(dataset_name)
	#batch_size = 8
	#loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
	model.eval()
	print('model parameters: ', list(model.parameters()))
	for name, param in model.named_parameters():
		print('name: ', name)
		print('param: ', param)
	numimgs = 0
	total_time = 0
	total_loss = 0
	timeimg = 0
	randi = np.random.randint(0, int(len(dataloader.dataset)/batch_size) - 2)
	#randi = int(randi/batch_size) 
	print('randi: ', randi)
	plasma = plt.get_cmap('plasma')
	greys = plt.get_cmap('Greys')
	normalizer = mpl.colors.Normalize(vmin=0, vmax=model.max_depth)
	mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
	Normalize = get_inverse_transforms('kitti')
	print('inside predict, len(dataloader): ', len(dataloader.dataset))
	with torch.no_grad():
		for i, data in enumerate(dataloader):
			img = data['image']
			dimg = data['original']
			start = time.time()
			out = model.forward(img)
			elapsed_time = time.time() - start
			total_time += elapsed_time
			print('avg img time: ', elapsed_time/img.shape[0])
			#outimg = outimg.astype('uint8')
			if gt_present == True:
				gt = data['gt']
				gtn = ((gt.numpy()/model.max_depth)*255).astype(np.uint8)
				#gtn = visualize_depth(gt, vmin=0.01, vmax=model.max_depth)
				rmse = RMSE(out, gt)
				absrel_error = abs_rel_error(out, gt)
				sqrel_err = sq_rel_error(out, gt)
				loss = criterion(out, gt)
				print('rmse: ', rmse)
				print('abs_rel_error: ', absrel_error)
				print('sqrel_error: ', sqrel_err)
				print('loss: ', loss)
				out2 = model.forward(img)
				rmse2 = RMSE(out2, gt)
				absrel_error2 = abs_rel_error(out2, gt)
				sqrel_err2 = sq_rel_error(out2, gt)
				loss2 = criterion(out2, gt)
				print('rmse2: ', rmse2)
				print('abs_rel_error2: ', absrel_error2)
				print('sqrel_error2: ', sqrel_err2)
				print('loss2: ', loss2)

			imgorig = Normalize(img)
			imgorig = torch.permute(imgorig, (0,2,3,1))
			imgorig = (255*imgorig).numpy().astype('uint8')
			dn = ((out.numpy()/model.max_depth)*255).astype(np.uint8)
			#dn = out.numpy()
			#dn = visualize_depth(out, vmin=0.01, vmax=model.max_depth)
			print('gt min: ', torch.min(gt))
			print('gt max: ', torch.max(gt))
			if save_images:
				for j in range(img.shape[0]):
					#coloredDepth = (mapper.to_rgba(dn[j,:,:])[:, :, :3] * 255).astype(np.uint8)
					#dn[j,:,:] = ((dn[j,:,:]/model.max_depth)*255).astype('uint8')
					#cv2.imwrite(imgdir + '/depth/' + str(numimgs + j) + '_depth1_' + '.jpg', coloredDepth)
					dn = visualize_depth(out[j,:,:], vmin=0.01, vmax=model.max_depth)
					gtn = visualize_depth(gt[j,:,:], vmin=0.01, vmax=model.max_depth)
					cv2.imwrite(imgdir + '/depth/' + str(numimgs + j) + '_depth2_' + '.jpg', dn)
					cv2.imwrite(imgdir + '/orig/' + str(numimgs + j) + '_imgorig_' + '.jpg', imgorig[j,:,:,:])
					if gt_present:
						cv2.imwrite(imgdir + '/origdepth' + str(numimgs + j) + '_depthorig_' + '.jpg', gtn)
					#if gt_present == True:
					#	cv2.imwrite(imgdir + '/' + str(numimgs) + '_img_' + '.jpg', outimg2[i,:,:,:])
				 	#pixelacc += np.sum(np.equal(inds[i,:,:].numpy(), ds[i,:,:].numpy()))/(inds.shape[1]*inds.shape[2])
					#print('np.sum(np.equal(inds[i,:,:], imgs[i,:,:]))/(inds.shape[1]*inds.shape[2]): ', np.sum(np.equal(inds[i,:,:].numpy(), imgs[i,:,:].numpy()))/(inds.shape[1]*inds.shape[2]))
			else:
				for j in range(img.shape[0]):
					#coloredDepth = (mapper.to_rgba(dn[j,:,:])[:, :, :3] * 255).astype(np.uint8)
					#dn[j,:,:] = ((dn[j,:,:]/model.max_depth)*255).astype('uint8')
					#cv2.imwrite(imgdir + '/e' + str(epoch) + '_depth1_' + str(j) + '.jpg', coloredDepth)
					#cv2.imwrite(imgdir + '/e' + str(epoch) + '_depth2_'  + str(j) + '.jpg', dn[j,:,:])
					#cv2.imwrite(imgdir + '/e' + str(epoch) + '_imgorig_' + str(j)  + '.jpg', imgorig[j,:,:,:])
					vmin = np.min(out[j,:,:].numpy())
					vmax = np.max(out[j,:,:].numpy())
					print('vmin: ', vmin)
					print('vmax: ', vmax)
					#dn = visualize_depth(out[j,:,:], vmin=0.01, vmax=model.max_depth)
					#value = (out[j,:,:].numpy() - vmin)/(vmax - vmin)
					#value[out[j,:,:].numpy() == 0] = 0
					#value = (value*255).astype('uint8')
					#gtn = visualize_depth(gt[j,:,:], vmin=0.01, vmax=model.max_depth)
					cv2.imshow('predicted depth: ', dn[j,:,:])
					cv2.imshow('original image: ', imgorig[j,:,:,:])
					if gt_present:
						cv2.imshow('ground truth depth: ', gtn[j,:,:])		
					cv2.waitKey(0)		
			numimgs += img.shape[0]

	print('average time: ', total_time/numimgs)
	print('total imgs: ', numimgs)

	

def compute_metrics(dataloader, batch_size=4, dataset_name='kitti', imgdir=None, model=None, modelpath=None, modelname='NeWCRFDepth', gt_present=True, save_images=False, criterion=depth_loss, epoch=None):
	if model == None and modelpath == None:
		raise ModelPathrequiredError("Both model and modelpath are None")
	elif model == None:
		if modelname == 'NeWCRFDepth':
			model = NeWCRFDepth()
	if modelpath != None:
		checkpoint = torch.load(modelpath)
		model.load_state_dict(checkpoint['model_state_dict'])
		print('checkpoint[epoch]: ', checkpoint['epoch'])
		print('checkpoint[loss]: ', checkpoint['loss'])
	#mean, std = get_mean_std(dataset_name)
	#batch_size = 8
	#loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
	model.eval()
	numimgs = 0
	total_time = 0
	total_loss = 0
	timeimg = 0
	randi = np.random.randint(0, int(len(dataloader.dataset)/batch_size) - 2)
	#randi = int(randi/batch_size) 
	print('randi: ', randi)
	plasma = plt.get_cmap('plasma')
	greys = plt.get_cmap('Greys')
	normalizer = mpl.colors.Normalize(vmin=0, vmax=model.max_depth)
	mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
	totalrmse = 0
	totalrelerror = 0
	totalsqrelerr = 0
	Normalize = get_inverse_transforms('kitti')
	print('inside predict, len(dataloader): ', len(dataloader.dataset))
	with torch.no_grad():
		for i, data in enumerate(dataloader):
			img = data['image']
			dimg = data['original']
			start = time.time()
			out = model.forward(img)
			elapsed_time = time.time() - start
			total_time += elapsed_time
			print('avg img time: ', elapsed_time/img.shape[0])
			#outimg = outimg.astype('uint8')
			if gt_present == True:
				gt = data['gt']
				rmse = RMSE(out, gt)
				absrel_error = abs_rel_error(out, gt)
				sqrel_err = sq_rel_error(out, gt)
				loss = criterion(out, gt)
				totalrmse += rmse*img.shape[0]
				totalrelerror += absrel_error*img.shape[0]
				totalsqrelerr += sqrel_err*img.shape[0]
				total_loss += loss*img.shape[0]
				gtn = ((gt.numpy()/model.max_depth)*255).astype(np.uint8)
			imgorig = Normalize(img)
			imgorig = torch.permute(imgorig, (0,2,3,1))
			imgorig = (255*imgorig).numpy().astype('uint8')
			dn = ((out.numpy()/model.max_depth)*255).astype(np.uint8)
			#dn = out.numpy()
			if save_images:
				for j in range(img.shape[0]):
					coloredDepth = (mapper.to_rgba(dn[j,:,:])[:, :, :3] * 255).astype(np.uint8)
					#dn[j,:,:] = ((dn[j,:,:]/model.max_depth)*255).astype('uint8')
					cv2.imwrite(imgdir + '/depth/' + str(numimgs + j) + '_depth1_' + '.jpg', coloredDepth)
					cv2.imwrite(imgdir + '/depth/' + str(numimgs + j) + '_depth2_' + '.jpg', dn[j,:,:])
					cv2.imwrite(imgdir + '/orig/' + str(numimgs + j) + '_imgorig_' + '.jpg', imgorig[j,:,:,:])
					if gt_present:
						cv2.imwrite(imgdir + '/origdepth' + str(numimgs + j) + '_depthorig_' + '.jpg', gtn[j,:,:] )
					#if gt_present == True:
					#	cv2.imwrite(imgdir + '/' + str(numimgs) + '_img_' + '.jpg', outimg2[i,:,:,:])
				 	#pixelacc += np.sum(np.equal(inds[i,:,:].numpy(), ds[i,:,:].numpy()))/(inds.shape[1]*inds.shape[2])
					#print('np.sum(np.equal(inds[i,:,:], imgs[i,:,:]))/(inds.shape[1]*inds.shape[2]): ', np.sum(np.equal(inds[i,:,:].numpy(), imgs[i,:,:].numpy()))/(inds.shape[1]*inds.shape[2]))
			elif i == randi and epoch != None:
				for j in range(img.shape[0]):
					coloredDepth = (mapper.to_rgba(dn[j,:,:])[:, :, :3] * 255).astype(np.uint8)
					#dn[j,:,:] = ((dn[j,:,:]/model.max_depth)*255).astype('uint8')
					cv2.imwrite(imgdir + '/e' + str(epoch) + '_depth1_' + str(j) + '.jpg', coloredDepth)
					cv2.imwrite(imgdir + '/e' + str(epoch) + '_depth2_'  + str(j) + '.jpg', dn[j,:,:])
					cv2.imwrite(imgdir + '/e' + str(epoch) + '_imgorig_' + str(j)  + '.jpg', imgorig[j,:,:,:])
					if gt_present:
						cv2.imwrite(imgdir + '/e' + str(epoch) + '_depthorig_' + str(j) + '.jpg', gtn[j,:,:])				
			numimgs += img.shape[0]

			#print('intersect: ', intersect)
			#print('union: ', union)
	#print('total_time: ', total_time)
	print('average time: ', total_time/numimgs)
	print('total imgs: ', numimgs)
	if gt_present == True:
		rmse_ = totalrmse/numimgs
		absrel_error_ = totalrelerror/numimgs
		sqrel_error_ = totalsqrelerr/numimgs
		loss_ = total_loss/numimgs
		print('rmse: ', rmse_)
		print('abs rel error: ', absrel_error_)
		print('sq_rel_error: ', sqrel_error_)
		print('loss: ', loss_)
		return rmse_, absrel_error_, sqrel_error_, loss_
		return rmse_, absrel_error_, sqrel_error_, loss_




'''
i = 0
for filename in os.listdir(datapath):
	if i > 0:
		break
	filepath = os.path.join(datapath, filename)
	semrgbfilepath = os.path.join(semrgbdatapath, filename)
	semfilepath = os.path.join(semdatapath, filename)
	img = cv2.imread(filepath)
	imgs = cv2.imread(semrgbfilepath)
	imginds = cv2.imread(semfilepath)
	imginds = cv2.resize(imginds, (480, 360), interpolation=cv2.INTER_NEAREST)
	imginds = imginds[:,:,0]
	print('imginds before: ', imginds)
	myids = getmyids()
	arr = np.arange(34)
	d = np.nonzero(imginds[:,:,np.newaxis] == arr)
	myids = getmyids()
	imginds[d[0],d[1]] = myids[d[2]]
	print('imginds after: ', imginds)
	#print('img shape: ', img.shape)
	#cv2.imshow('img: ', img)
	#cv2.waitKey(0)
	predict(img, imgs, imginds, modelpath)
	i = i+1
'''