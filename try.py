import cv2
import numpy as np
from data import getDataset
from torch.utils.data import DataLoader, Dataset
from modelv2 import Encoder, Segnet
from modelv3 import Segnet as SegnetSkip3
import torch
from train import train
from model_dilated import SegmentationDilated as SegmentationDil
from model_dilated2 import SegmentationDilated as SegmentationDil2
from model_dilated3 import SegmentationDilated as SegmentationDil3
from model_dilated4 import SegmentationDilated as SegmentationDil4
from model_dilated5 import SegmentationDilated as SegmentationDil5
from labels import mylabels, Label, id2myid, id2label
from depthmodel import SwinTransformer, NeWCRFDepth
from depthmodel2 import NeWCRFDepth as NeWCRFDepth2
from losses import depth_loss
from data_config import rawdatalist, depthdatalist, valrawdatalist, valdepthdatalist




#model = SegnetSkip3(kernel_size=7, padding=3, out_channels=12)
#model = SegmentationDil(kernel1_size=7, kernel2_size=3, kernel3_size=5, padding=3)
#model = SegmentationDil2(kernel1_size=7, kernel2_size=3, kernel3_size=5, padding=3)
#model = SegmentationDil5(kernel1_size=7, kernel2_size=3, kernel3_size=5, padding=3)

model = NeWCRFDepth()

criterion = depth_loss(lambda_=0.55, alpha=10)

dataset = getDataset(rawdatapath=rawdatalist, depthdatapath=depthdatalist, max_depth=85, pct=1.0, train_val_split=1.0, dataset='kitti', data_augment_flip=0, data_augment_brightness_color=0.20, gt_present=True, mode='train')

valdataset = getDataset(rawdatapath=valrawdatalist, depthdatapath=valdepthdatalist, max_depth=85, pct=0.30, train_val_split=1.0, dataset='kitti', data_augment_flip=0, data_augment_brightness_color=0.0, gt_present=True, mode='train')

resultsdir = 'results/trial5'

#resultsdiropen = 'results/trial5'

modelpath='results/trial5/bestlossdepthmodelnew.pt'

#bestmodelpath = 'results/trial0//bestvallossdepthmodelnew.pt'

#modelpath = 'results/trial0//bestvallossdepthmodelnew.pt'

batch_size = 4

batch_size_val = 4

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

val_data_loader = DataLoader(valdataset, batch_size=batch_size_val, shuffle=True, pin_memory=True)


print('len(dataset): ', len(dataset))
print('len(data_loader.dataset): ', len(data_loader.dataset))

train(data_loader, val_data_loader, model, criterion, epochs=60, batch_size=batch_size, batch_size_val=batch_size_val, dataset_name='kitti', shuffle=True, resume_training=True, resultsdir=resultsdir, resultsdiropen=resultsdir, modelpath=modelpath, initialize_from_model=False)
#loader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)

#loaderiter = iter(loader)

#data = next(loaderiter)

#print('data:', data)
#d = data['image'].numpy()
#ds = data['semantic'].numpy()
#print('d shape: ', d.shape)
#print('ds shape: ', ds.shape)
#for i in range(4):
#	print('np.unique(ds.flatten()) ', i, ': ', (np.unique(ds[i,:,:].flatten())))
#print('ds: ', ds)
#print('ds[0,0,0,:]: ', ds[0,0,0,:])
#print('ds[0,0,1,:]: ', ds[0,0,1,:])
#print('ds[0,0,2,:]: ', ds[0,0,2,:])
#print('ds[0,1,0,:]: ', ds[0,21,0,:])
#print('ds[0,0,0,:]: ', ds[1,200,120,:])
#print('ds[0,0,1,:]: ', ds[1,100,100,:])
#print('ds[0,0,2,:]: ', ds[0,200,200,:])
#print('ds[0,1,0,:]: ', ds[0,100,150,:])
#print('ds[0,:,:,0] == ds[0,:,:,1]: ', ds[0,:,:,0] == ds[0,:,:,1])

#print('np.unique(np.flatten(ds)): ', np.unique(ds[1,:,:,].flatten()))
#print('np.unique(np.flatten(ds)): ', np.unique(ds[2,:,:,:], axis=2))
#print('np.unique(np.flatten(ds)): ', np.unique(ds[3,:,:,:], axis=2))

#arr = np.unique(ds[1,:,:,].flatten())
#print('arr: ', arr)
#print('tpye(arr): ', type(arr))

#res = map(id2myid, arr)
#print('res: ', list(res))
#print('type(res): ', type(res))
#resarr = map(id2myid, ds[1,:,:])
#print('resarr: ', list(resarr))

#print('ds[0,0:50,0:50,0]): ', ds[0,0:50,0:50,0])
#print('ds[0,0:50,0:50,1]): ', ds[0,0:50,0:50,1])
#print('ds[0,0:50,0:50,2]): ', ds[0,0:50,0:50,2])

#print('ds0new[0,:,:,0] == ds0new[0,:,:,1]: ', ds[0,:,:,0] == ds[0,:,:,1])
#print('ds0new[0,:,:,2] == ds0new[0,:,:,1]: ', ds[0,:,:,2] == ds[0,:,:,1])



#imgsem = ds[0,:,:]


#print('imgsem[0:200, 0:200]: ', imgsem[0:200,0:200])
#print('np.unique(imgsem.flatten()): ', np.unique(imgsem.flatten()))
#d_ = np.nonzero(imgsem[:,:,np.newaxis] == arr)
#print('type(d_[0]): ', type(d_[0]))
#imgsem[d_[0],d_[1]] = myids[d_[2]]
#print('np.unique(imgsem.flatten()): ', np.unique(imgsem.flatten()))

#net = Segnet()
#x = net.forward(torch.from_numpy(d.transpose((0,3,1,2))))

#print('x: ', x)
#print('x shape: ', x.shape)
#print('net: ', net)
#cv2.imshow('ds0: ', ds[0,:,:,:])
#cv2.waitKey(0)