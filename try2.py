from preprocess import get_std
import cv2
import numpy as np
import torch.nn as nn
from data import getDataset, get_inverse_transforms
from torch.utils.data import DataLoader, Dataset
import torch
from modelv2 import Segnet
from modelv3 import Segnet as SegnetSkip3
#from predict import compute_accuracy
import os
from test_segmentation_camvid import label_colours
from model_dilated import SegmentationDilated as SegmentationDil
from model_dilated2 import SegmentationDilated as SegmentationDil2
from model_dilated3 import SegmentationDilated as SegmentationDil3
from model_dilated4 import SegmentationDilated as SegmentationDil4
from model_dilated5 import SegmentationDilated as SegmentationDil5
from model_dilated_attention import SegmentationDilated as SegmentationDilattn
from depthmodel import SwinTransformer, NeWCRFDepth
from data_config import rawdatalist, depthdatalist, valrawdatalist, valdepthdatalist
from depth_metrics import RMSE, abs_rel_error, sq_rel_error
from predict import compute_metrics, predict_and_visualize
from time import time
from torchsummary import summary
from losses import depth_loss
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

print('hi')

#std, mean = get_std(datapath=None, mean=None, dataset='kitti')
#print('std: ', std)
#print('mean: ', mean)


#dataset = getDataset(rawdatapath=rawdatapath, depthdatapath=depthdatapath, pct=1.0, train_val_split=1.0, dataset='kitti', data_augment=False, gt_present=True, mode='train')
#dataset = getDataset(rawdatapath=rawdatalist, depthdatapath=depthdatalist, max_depth=85, pct=1.0, train_val_split=1.0, dataset='kitti', data_augment_flip=0, data_augment_brightness_color=0.20, gt_present=True, mode='train')

valdataset = getDataset(rawdatapath=rawdatalist, depthdatapath=depthdatalist, max_depth=85, pct=0.1, train_val_split=1.0, dataset='kitti', data_augment_flip=0, data_augment_brightness_color=0, gt_present=True, mode='train')
loader = DataLoader(valdataset, batch_size=4, shuffle=False, pin_memory=True)

#model = SegnetSkip3(kernel_size=7, padding=3, out_channels=12)
#model = SegmentationDil(kernel1_size=7, kernel2_size=3, kernel3_size=5, padding=3)
#model = SegmentationDil2(kernel1_size=7, kernel2_size=3, kernel3_size=5, padding=3)

modelpath = 'results/trial2/latestdepthmodelnew.pt'
model = NeWCRFDepth()

predict_and_visualize(loader, batch_size=4, dataset_name='kitti', imgdir=None, model=model, modelpath=modelpath, modelname='NeWCRFDepth', gt_present=True, save_images=False)
#segmimgdir = imgdir + '/segm'
#origimgdir = imgdir + '/orig'
#overlayimgdir = imgdir + '/overlay'

#checkpoint = torch.load(modelpath)
#print('checkpoint[epoch]: ', checkpoint['epoch'])
#print('checkpoint[loss]: ', checkpoint['loss'])
#print('checkpoint[mean_iou]: ', checkpoint['mean_iou'])

'''
#loader = DataLoader(dataset, batch_size=4, shuffle=False, pin_memory=True)
pixelacc, meaniou, loss, intersect, union = compute_accuracy(datasetval, dataset_name='CamVid', imgdir=imgdir, model=model, modelpath=modelpath, modelname='SegmentationDil4', gt_present=True, save_images=True, criterion=criterion)
print('pixelacc: ', pixelacc)
print('loss: ', loss)
print('mean_iou: ', meaniou)
print('intersect: ', intersect)
print('union: ', union)
'''

'''
i = 0
alpha = 0.50
newimg = 255*np.ones((360,480,3))
#overimg = 255*np.ones((360,480,3))
blankcol = 255*np.ones((360,3,3))
for i in range(101):
	imgpath = segmimgdir + '/' + str(i) + '_outimg_.jpg'
	segm = cv2.imread(imgpath)
	origimgpath = origimgdir +'/' + str(i) + '_imgorig_.jpg'
	origimg = cv2.imread(origimgpath)
	overpath = overlayimgdir + '/' + str(i) + '_overlayimg_.jpg'
	overimg = cv2.addWeighted(segm, alpha, origimg, 1 - alpha,
		0)
	#cv2.imwrite(imgdir + '/' + str(i) + '_overlayimg_.jpg', overimg)
	#overlayimg = cv2.imread(overpath)
	newimg = np.concatenate((segm, blankcol, origimg, blankcol, overimg),axis=1)
	cv2.imwrite(overlayimgdir + '/' + str(i) + '_overlayimg_.jpg', newimg)
'''

summary(model, input_size = (3, 360, 480), batch_size=4)

#overlay = cv2.imread(imgdir + '/1_outimg_.jpg')
#output = cv2.imread(imgdir + '/1_imgorig_.jpg')
#alpha = 0.5

#cv2.addWeighted(overlay, alpha, output, 1 - alpha,
#		0, output)

#cv2.imwrite(imgdir + '/1_overlayed.jpg', output)

#loader = DataLoader(dataset, batch_size=4, shuffle=False, pin_memory=True)
#pixelacc, meaniou, loss = compute_accuracy(dataset, dataset_name='kitti', imgdir=None, model=None, modelpath=modelpath, modelname='SegnetSkip', gt_present=True, save_images=False, criterion=criterion)
#print('pixelacc: ', pixelacc)
#print('loss: ', loss)
#print('mean_iou: ', meaniou)

#loader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)

#loaderiter = iter(loader)

#data = next(loaderiter)

#color_arr = label_colours
#unarr = np.empty((0,))
#Normalize = get_inverse_transforms('kitti')

#print('data:', data)
'''
for i, data in enumerate(loader):
	d = data['image']
	ds = data['semantic']
	#print('d.shape: ', d.shape )
	#print('np.unique(np.flatten(ds)): ', np.unique(ds[2,:,:,].flatten()))
	#print('np.unique(np.flatten(ds)): ', np.unique(ds[3,:,:,].flatten()))
	indices = np.indices((d.shape[0], 360,480))
	outimg2 = np.ones((d.shape[0], 360,480,3))
	#outimg2[indices[0,:,:],indices[1,:,:],:] = color_arr[imgs]
	outimg2[indices[0,:,:,:], indices[1,:,:,:],indices[2,:,:,:],:] = color_arr[ds]
	outimg2 = outimg2.astype('uint8')	#
	imgorig = Normalize(d)
	imgorig = torch.permute(imgorig, (0,2,3,1))
	imgorig = 255*imgorig
	imgorig = imgorig.numpy().astype('ui#nt8')
	print('d.shape: ', d.shape#)
	print('ds.shape: ', ds.sha#pe)
	for j in range(d.shape[0])#:
		arr = np.unique(ds[j,:#,:,].flatten())
		unarr = np.union1d(unarr, arr)
		print('arr: ', arr)
		print('unarr: ', unarr)
		#cv2.imshow('seg: ', outimg2[i,:,:,:])
		#cv2.imshow('img: ', imgorig[i,:,:,:])
		#cv2.waitKey(0)
'''


#loss_criterion = depth_loss()
#rmse, absrel_error, sqrel_error, loss = compute_metrics(valdataset, dataset_name='kitti', imgdir='results', model=model, modelpath=None, modelname='NeWCRFDepth', gt_present=True, save_images=False, criterion=loss_criterion, epoch=1)

#print('rmse: ', rmse)
#print('absrel_error: ', absrel_error)
#print('sqrel_error: ', sqrel_error)
#print('loss: ', loss)

'''
blankcol = 255*np.ones((360,3,3)).astype(np.uint8)
plasma = plt.get_cmap('plasma')
greys = plt.get_cmap('Greys')
eps = 1e-7
normalizer = mpl.colors.Normalize(vmin=0, vmax=85)
mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
for i, data in enumerate(loader):
	img = data['image']
	imgorig1 = data['original']
	#x = model.forward(img)
	#print('x shape: ', x.shape)
	d = data['gt']
	#rmse = RMSE(x, d)
	#absrel_error = abs_rel_error(x,d)
	#sqrel_err = sq_rel_error(x,d)
	#loss = loss_criterion(x, d)
	#print('loss: ', loss)
	#print('RMSE: ', rmse)
	#print('absrel_error: ', absrel_error)
	#print('sq_rel_error: ', sqrel_err)
	imgorig = Normalize(img)
	imgorig = torch.permute(imgorig, (0,2,3,1))
	imgorig1 = torch.permute(imgorig1, (0,2,3,1))
	imgorign = imgorig.numpy()
	imgorign = (255*imgorign).astype('uint8')
	#imgorig = imgorig.numpy().astype('uint8')
	dn = d.numpy()[:,:,:]
	#dn = np.clip(dn,0,80)
	#depth = depth_png.astype(np.float) / 256.
    #depth[depth_png == 0] = -1.
	#print('dn shape: ', dn.shape)
	#print('d[0,:,:,0] == d[0,:,:,1]: ', np.sum(d[0,:,:,0] == d[0,:,:,1]))
	for j in range(img.shape[0]):
		#print('np.min(dn[j,:,:]): ', np.min(dn[j,:,:]))
		#print('np.max(dn[j,:,:]): ', np.max(dn[j,:,:]))
		#print('dn[j,:,:]*255/80: ', dn[j,:,:]*255/80)
		#print('dn[j,:,:]: ', dn[j,:,:])
		coloredDepth = (mapper.to_rgba(dn[j,:,:])[:, :, :3] * 255).astype(np.uint8)
		print('min mapper.to_rgba(dn[j,:,:])[:, :, :3]: ', np.min(mapper.to_rgba(dn[j,:,:])[:, :, :3]))
		print('max mapper.to_rgba(dn[j,:,:])[:, :, :3]: ', np.max(mapper.to_rgba(dn[j,:,:])[:, :, :3]))
		#print('mapper.to_rgba(dn[j,:,:])[:, :, :3]: ', mapper.to_rgba(dn[j,:,:])[:, :, :3])
		#coloredDepth = (greys(np.log10(dn[j,:,:]))[:, :, :3] * 255).astype('uint8')
		print('coloredDepth shape: ', coloredDepth.shape)
		print('coloredDepth type: ', type(coloredDepth))
		#print('imgorig[j,:,:,0]: ', imgorig[j,:,:,0])
		newimg = np.concatenate((imgorig[j,:,:,:], blankcol, coloredDepth, blankcol, imgorig1[j,:,:,:]), axis=1)
		dn[j,:,:] = ((dn[j,:,:]/85)*255).astype('uint8')
		#print('newimg: ', newimg[:,:,0])
		#cv2.imshow('img: ', imgorig[j,:,:,:])
		#cv2.imshow('coloredDepth: ', coloredDepth)
		cv2.imwrite('results/imgorig' + str(j) + '.png', imgorign[j,:,:,:])
		cv2.imwrite('results/depth' + str(j) + '.png', coloredDepth)
		cv2.imwrite('results/depth2' + str(j) + '.png', dn[j,:,:])
		cv2.imshow('newimg: ', newimg)
		cv2.waitKey(0)
		#print('min depth value: ', torch.min(d[j,:,:,:]))
		#print('max depth value: ', torch.max(d[j,:,:,:]))

'''

'''

loss_criterion = depth_loss()
d = data['image']
gt = data['gt']
#d = torch.unsqueeze(d,0)
#net = Segnet(7,3)
print('d shape: ', d.shape)
start_time = time()
x = model.forward(d)
end_time = time()
x = torch.squeeze(x,1)
print('x.shape: ', x.shape)
print('gt shape: ', gt.shape)
print('avg_time: ', (end_time - start_time)/x.shape[0])




loss = loss_criterion(x, gt)

print('loss: ', loss)

#print('std: ', std)
#print('mean: ', mean)

#cv2.imshow('img: ', img)
#cv2.waitKey(0)

'''