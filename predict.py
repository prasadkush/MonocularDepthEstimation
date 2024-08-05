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
import os
import pdb
import time
import matplotlib.cm as cm
import matplotlib as mpl



def compute_metrics(dataset, dataset_name='kitti', imgdir=None, model=None, modelpath=None, modelname='NeWCRFDepth', gt_present=True, save_images=False, criterion=depth_loss, epoch=None):
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
	imgh = dataset.imgh
	imgw = dataset.imgw
	batch_size = 8
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
	model.eval()
	numimgs = 0
	total_time = 0
	total_loss = 0
	timeimg = 0
	randi = np.random.randint(0, len(dataset) - batch_size)
	randi = int(randi/batch_size) 
	plasma = plt.get_cmap('plasma')
	greys = plt.get_cmap('Greys')
	normalizer = mpl.colors.Normalize(vmin=0, vmax=model.max_depth)
	mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
	totalrmse = 0
	totalrelerror = 0
	totalsqrelerr = 0
	Normalize = get_inverse_transforms('kitti')
	with torch.no_grad():
		for i, data in enumerate(loader):
			img = data['image']
			dimg = data['original']
			start = time.time()
			out = model.forward(img)
			total_time += time.time() - start
			#print('timeimg: ', timeimg)
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
	if gt_present == True:
		rmse = totalrmse/numimgs
		absrel_error = totalrelerror/numimgs
		sqrel_error = totalsqrelerr/numimgs
		loss = total_loss/numimgs
		print('rmse: ', rmse)
		print('abs rel error: ', absrel_error)
		print('sq_rel_error: ', sqrel_error)
		print('loss: ', loss)
		return rmse, absrel_error, sqrel_error, loss




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