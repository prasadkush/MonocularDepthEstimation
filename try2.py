from preprocess import get_std
import cv2
import numpy as np
import torch.nn as nn
from data import getDataset, get_inverse_transforms
from torch.utils.data import DataLoader, Dataset
import torch
from modelv2 import Segnet
from modelv3 import Segnet as SegnetSkip3
from predict import compute_accuracy
import os
from test_segmentation_camvid import label_colours
from model_dilated import SegmentationDilated as SegmentationDil
from model_dilated2 import SegmentationDilated as SegmentationDil2
from model_dilated3 import SegmentationDilated as SegmentationDil3
from model_dilated4 import SegmentationDilated as SegmentationDil4
from model_dilated5 import SegmentationDilated as SegmentationDil5
from model_dilated_attention import SegmentationDilated as SegmentationDilattn
from depthmodel import SwinTransformer
from time import time
from torchsummary import summary

print('hi')

#datapath = 'C:/Users/Kush/OneDrive/Desktop/CV-Ml/datasets/data_semantics/training'

#datapathtest = 'C:/Users/Kush/OneDrive/Desktop/CV-Ml/datasets/data_semantics/testing'

#filepath = 'C:/Users/Kush/OneDrive/Desktop/CV-ML/datasets/SegNet-Tutorial-master/SegNet-Tutorial-master/CamVid/trainannot/0001TP_006690.png'

datapathcam = 'C:/Users/Kush/OneDrive/Desktop/CV-ML/datasets/SegNet-Tutorial-master/SegNet-Tutorial-master/CamVid'

criterion = nn.CrossEntropyLoss()

dataset = getDataset(datapathcam, dataset='CamVid', data_augment=False, gt_present=True, mode='train')

#datasetval = getDataset(datapathcam, dataset='CamVid', data_augment=False, gt_present=True, mode='val')

#print('length dataset: ', len(dataset))

imgdir = 'results/trial12_CamVid_Dil4/val'

resultsdir = 'results/trial12_CamVid_Dil4'

#modelpath = resultsdir + '/bestlosssegnetmodelnew.pt'

modelpath = resultsdir + '/bestmeaniousegmentmodelnew.pt'

#checkpoint = torch.load(modelpath)

#print('epochs: ', checkpoint['epoch'])

#model = SegnetSkip3(kernel_size=7, padding=3, out_channels=12)
#model = SegmentationDil(kernel1_size=7, kernel2_size=3, kernel3_size=5, padding=3)
#model = SegmentationDil2(kernel1_size=7, kernel2_size=3, kernel3_size=5, padding=3)
model = SwinTransformer()


segmimgdir = imgdir + '/segm'
origimgdir = imgdir + '/orig'
overlayimgdir = imgdir + '/overlay'

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

#summary(model, input_size = (3, 360, 480), batch_size=4)

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

loader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)

loaderiter = iter(loader)

data = next(loaderiter)

#color_arr = label_colours
#unarr = np.empty((0,))
#Normalize = get_inverse_transforms('CamVid')

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

d = data['image']
#d = torch.unsqueeze(d,0)
#net = Segnet(7,3)
print('d shape: ', d.shape)
start_time = time()
x = model.forward(d)
end_time = time()
print('x.shape: ', x.shape)
print('avg_time: ', (end_time - start_time)/x.shape[0])
#std, mean = get_std(datapath=datapathcam, dataset='CamVid')

#print('std: ', std)
#print('mean: ', mean)


#cv2.imshow('img: ', img)
#cv2.waitKey(0)