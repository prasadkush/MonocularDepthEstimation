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

from labels import mylabels, Label, id2myid, id2label, names2mynames

def getmyids():
    myids = np.zeros((34,))
    for i in range(34):
        name = id2label[i].name
        if name in list(mylabels.keys()):
            myids[i] = mylabels[name]
        elif name in list(names2mynames.keys()):
            namekey = names2mynames[name]
            myids[i] = mylabels[namekey] 
        else:
            myids[i] = 14
    return myids

def load_zip_to_mem(zip_file, is_mono=True):
    """
    Function to load CLEVR-D data from the zip file.
    """
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    file_dict = {name.split('/')[1]: input_zip.read(name) for 
            name in input_zip.namelist() if '.png' in name}
    data = []
    for file_name in file_dict:
      #Only deal with right rgb images, all else via dict lookup
      if 'right' in file_name and 'CLEVR-D' not in file_name:
        rgb_right = file_dict[file_name]
        right_depth_name = file_name.replace('CLEVR','CLEVR-D')
        depth_right = file_dict[right_depth_name]
        if is_mono:
          data.append( (rgb_right, depth_right))
        else:
          rgb_left = file_dict[file_name.replace('right','left')]
          depth_left = file_dict[right_depth_name.replace('right','left')]
          data.append( (rgb_right,rgb_left, depth_right,depth_left))
    return data

def get_inverse_transforms(dataset='kitti'):
    """
    Get inverse transforms to undo data normalization
    """
    if dataset == 'kitti':
        inv_normalize_color = transforms.Normalize((-0.38399/0.32906, -0.39878/0.31968, -0.37933/0.3109),
    (1/0.32906, 1/0.31968, 1/0.3109))
    elif dataset == 'CamVid':
        inv_normalize_color = transforms.Normalize((-0.4326707/0.30721843, -0.4251328/0.31161108, -0.41189488/0.3070735),
    (1/0.30721843, 1/0.31161108, 1/0.3070735))    
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
    def __init__(self, datapath, pct=1.0, train_val_split=1.0, dataset='kitti', data_augment=False, gt_present=True, mode='train'):
        self.start_idx = 0
        self.color_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.38399, 0.39878, 0.37933), (0.32906, 0.31968, 0.3109)),
        ])
        self.ConvertImageDtype = transforms.ConvertImageDtype(torch.float)
        if dataset == 'kitti':
            self.Normalize = transforms.Normalize((0.38399, 0.39878, 0.37933), (0.32906, 0.31968, 0.3109))
        elif dataset == 'CamVid':
            self.Normalize = transforms.Normalize((0.4326707, 0.4251328, 0.41189488), (0.30721843, 0.31161108, 0.3070735))
        self.data = []
        self.data_orig = []
        self.images = []
        self.imgw = 480
        self.imgh = 360
        self.dataset = dataset
        myids = getmyids()
        if self.dataset == 'kitti':
            imgdir = os.path.join(datapath, 'image_2')
        elif self.dataset == 'CamVid':
            if mode == 'train':
                imgdir = os.path.join(datapath, 'train')
            elif mode == 'val':
                imgdir = os.path.join(datapath, 'val')
            elif mode == 'test':
                imgdir = os.path.join(datapath, 'test')
        for filename in os.listdir(imgdir):
            filepath = os.path.join(imgdir, filename)
            img = cv2.imread(filepath)
            imgnew = cv2.resize(img, (480, 360), interpolation=cv2.INTER_NEAREST)
            imgnew = imgnew.transpose(2,0,1)
            t = torch.from_numpy(imgnew)
            t = self.ConvertImageDtype(t)
            self.data_orig.append(t)
            t = self.Normalize(t)
            self.data.append(t)
        self.end_idx = int(pct * train_val_split * len(self.data))
        self.images = self.data[0:self.end_idx]
        print('num images: ', len(self.images))
        self.semantic_images = []
        i = 0
        self.semantic_data = []
        self.samples = []
        if self.dataset == 'kitti':
            imgdir = os.path.join(datapath, 'semantic')
            self.num_classes = 16
        elif self.dataset == 'CamVid':
            if mode == 'train':
                imgdir = os.path.join(datapath, 'trainannot')
            elif mode == 'val':
                imgdir = os.path.join(datapath, 'valannot')
            elif mode == 'test':
                imgdir = os.path.join(datapath, 'testannot')
            self.num_classes = 12
        myids = 0
        self.myfreqs = np.zeros((self.num_classes,1)).astype('int32') # num of pixels of a particular class/total num of pixels in images in which the class occurs
        self.numimages = np.zeros((self.num_classes,1)).astype('int32') # no of images in which the class occurs
        if dataset == 'kitti':
            myids = getmyids()
        print('myids: ', myids)
        if gt_present == True:
            arr2 = np.arange(self.num_classes)
            for filename in os.listdir(imgdir):
                i = i + 1
                filepath = os.path.join(imgdir, filename)
                img = cv2.imread(filepath)
                imgnew = cv2.resize(img, (480, 360), interpolation=cv2.INTER_NEAREST)
                imgsem = imgnew[:,:,0]
                if dataset == 'kitti':
                    arr = np.arange(34)
                    d = np.nonzero(imgsem[:,:,np.newaxis] == arr)
                    imgsem[d[0],d[1]] = myids[d[2]]
                #print('imgsem[np.newaxis,:,:] == arr2: ', imgsem[:,:,np.newaxis] == arr2)
                #print('np.equal(imgsem[np.newaxis,:,:],arr2): ', np.equal(imgsem[:,:,np.newaxis],arr2))
                self.myfreqs += np.sum(imgsem[:,:,np.newaxis] == arr2, axis=(0,1)).reshape((self.num_classes,1))
                self.numimages += (self.myfreqs > 0).astype('int16')
                t = torch.from_numpy(imgsem)
                t = t.long()
                self.semantic_data.append(t)
                sample = {'image': self.images[i-1], 'semantic': self.semantic_data[i-1], 'original': self.data_orig[i-1]}
                self.samples.append(sample)
                if i == self.end_idx:
                    break
        else:
            for i in range(self.end_idx):
                sample = {'image': self.images[i], 'original': self.data_orig[i]}
                self.samples.append(sample)
        if data_augment == True:

            auginds = np.random.randint(0,self.end_idx, size=int(0.20*self.end_idx))
            for i in range(auginds.shape[0]):
                sample = self.samples[auginds[i]]
                augimage = sample['image']
                augsem = sample['semantic']
                augorig = sample['original']
                augimage = torch.flip(augimage, [2])
                augsem = torch.flip(augsem, [1])
                augorig = torch.flip(augorig, [2])
                #print('image: ', sample['image'][0,0:5,:])
                #print('augimage: ', augimage[0,0:5,:])
                #print('semantic: ', sample['semantic'][0:5,:])
                #print('augsem: ', augsem[0:5,:])
                #print('original: ', sample['original'][0,0:5,:])
                #print('augorig: ', augorig[0,0:5,:])
                #cv2.imshow('image: ', sample['image'].numpy().transpose((1,2,0)))
                sample = {'image': augimage, 'semantic': augsem, 'original': augorig}
                ind = np.random.randint(0,len(self.samples))
                self.samples.insert(ind, sample)        
        if gt_present == True:
            self.weights = self.myfreqs/(self.numimages*self.imgw*self.imgh)
            print('weights: ', self.weights)
            median = np.median(self.weights)
            self.weights = median/self.weights
            self.weights = torch.from_numpy(self.weights)
            print('weights: ', self.weights)
        print('self.end_idx: ', self.end_idx)
        print('len(self.samples): ', len(self.samples))
        print('self.end_idx + int(0.20*self.end_idx): ', self.end_idx + int(0.20*self.end_idx))



    def __getitem__(self, idx):
        return self.samples[idx] # TODO 

    def __len__(self):
        return len(self.samples) # TODO 

    def getWeights(self):
        return self.weights


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


