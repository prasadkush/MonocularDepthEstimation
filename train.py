import cv2
import torch
import torch.nn as nn
from torch import optim
from data import getDataset, get_inverse_transforms
from torch.utils.data import DataLoader, Dataset
from model import Encoder, Segnet
import matplotlib.pyplot as plt
import numpy as np
from config import datapath, datapathcam
import pickle
import pdb
from predict import compute_metrics
from depth_metrics import RMSE, sq_rel_error, abs_rel_error
from Exceptions import ModelPathrequiredError
#from torchvision.models import vgg16_bn

weight_dict = {}

def save_model(loss, path, epoch, model, optimizer, lr_schedule, lr_milestones, metrics, metrics_val):
    EPOCH = epoch
    PATH_ = path
    LOSS_ = loss
    METRICS_ = metrics
    METRICS_VAL_ = metrics_val

    torch.save({
        'epoch': EPOCH, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS_, 'metrics': METRICS_, 'metrics_val': METRICS_VAL_, 'lr_schedule': lr_schedule, 'lr_milestones': lr_milestones}, PATH_)


def weights_init(m):
    #print('m: ', m)
    print('m.__class__: ', m.__class__)
    classname = m.__class__.__name__
    print('initializing weights, classname: ', classname)
    if isinstance(m, nn.Conv2d):
        #torch.nn.init.normal_(m.weight, mean=0.2, std=1)
        #torch.nn.init.uniform_(m.weight)  
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        #torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        #torch.nn.init.normal_(m.weight, mean=0.2, std=1)
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        #torch.nn.init.uniform_(m.weight)
        #torch.nn.init.zeros_(m.bias)
    


def train(data_loader, val_data_loader, model, epochs=60, batch_size=4, batch_size_val=4, shuffle=True, val_dataset=None, modelpath=None, bestmodelpath=None, resume_training=False, useWeights=False, resultsdir=None,
    layer_wise_training=False):

    criterion = depth_loss()


    lr_initial = 0.05
    lr_new = 0.05
    if resume_training == False:
        model.apply(weights_init)
    print('after model.apply')
    optimizer = optim.Adam(model.parameters(), lr=lr_initial, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #optimizer = optim.SGD(model.parameters(), lr=lr_initial,  momentum=0.9, nesterov=True)
    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

 #   batch_size_val = 8
 #   loader_val = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True)
    #loaderiter_val = iter(loader_val)
    num_val = 0
    compute_val_over_whole_data = True

    start_epoch = 0
    loss = 0
    best_loss = 100000
    best_val_loss = 10000
    best_rmse = 0
    best_absrel_val = 0
    total_loss = 0
    training_loss_list = []
    #mean_list = []
    rmselist = []
    absrellist = []
    rmse_val_list = []
    absrel_val_list = []
    loss_val_list = []
    epochs = 60
    lr_schedule = [lr_initial, 0.01, 0.005, 0.001]
    lr_milestones = [30, 40, 50, epochs]

    if resume_training == True and modelpath != None:
        checkpoint = torch.load(modelpath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        if bestmodelpath != None:
            checkpoint = torch.load(bestmodelpath)
            best_loss = checkpoint['loss']
        else:
            best_loss = checkpoint['loss']
        start_epoch = epoch + 1
        ind_sch = np.searchsorted(np.array(lr_milestones), start_epoch, side='right')
        lr_new = lr_schedule[ind_sch]

        bestvalmodelpath = resultsdir + '/bestvallossdepthmodelnew.pt'
        valcheckpoint = torch.load(bestvalmodelpath)
        valepoch = valcheckpoint['epoch']
        best_val_loss = valcheckpoint['loss']

        bestvalabsrelmodelpath = resultsdir + '/bestvalabsreldepthmodelnew.pt'
        valabsrelcheckpoint = torch.load(bestvalabsrelodelpath)
        valabsrelepoch = valabsrelcheckpoint['epoch']
        best_absrel_val = valabsrelcheckpoint['metrics']['absrel']

        with open(resultsdir + '/training_loss_list.pkl', 'rb') as f:
            training_loss_list = pickle.load(f)
            training_loss_list = training_loss_list[0:min(start_epoch,len(training_loss_list))]
            print('len(training_loss_list): ', len(training_loss_list))
        #with open(resultsdir + '/mean_list.pkl', 'rb') as f:
        #    mean_list = pickle.load(f)
        with open(resultsdir + '/absrellist.pkl', 'rb') as f:
            absrellist = pickle.load(f)
            absrellist = absrellist[0:min(start_epoch,len(absrellist))]
            print('len(absrellist): ', len(absrellist))
        with open(resultsdir + '/rmselist.pkl', 'rb') as f:
            rmselist = pickle.load(f)
            rmselist = rmselist[0:min(start_epoch,len(rmselist))]
            print('len(rmselist): ', len(rmselist))
        with open(resultsdir + '/absrel_val_list.pkl', 'rb') as f:
            absrel_val_list = pickle.load(f)
            absrel_val_list = absrel_val_list[0:min(start_epoch,len(absrel_val_list))]
            best_absrel_val2 = min(absrel_val_list)
            print('len(absrel_val_list): ', len(absrel_val_list))
        with open(resultsdir + '/rmse_val_list.pkl', 'rb') as f:
            rmse_val_list = pickle.load(f)
            rmse_val_list = rmse_val_list[0:min(start_epoch,len(rmse_val_list))]
            print('len(pixelacc_val_list): ', len(pixelacc_val_list))
        with open(resultsdir + '/loss_val_list.pkl', 'rb') as f:
            loss_val_list = pickle.load(f)
            loss_val_list = loss_val_list[0:min(start_epoch,len(loss_val_list))]
            print('len(loss_val_list): ', len(loss_val_list))
            best_val_loss2 = min(loss_val_list)
    elif resume_training == True and modelpath == None:
        raise ModelPathrequiredError("Provide Model path if resume_training is set to True")


   # layers = ['Decoder.layer2', 'Decoder.layer3', 'Decoder.layer4', 'Encoder.layer15', 'Encoder.layer16', 'Encoder.layer17']
    if layer_wise_training == True:
        for name, param in model.named_parameters():
            ind = name.index('.')
            ind2 = name.find('.', ind + 1)
            layername = name[0:ind2]
            if layername not in layers and param.requires_grad == True:
                param.requires_grad = False
            print('name: ', name, ' layername: ', layername, ' requires_grad: ', param.requires_grad)


    print('resume_training: ', resume_training)
    print('start_epoch: ', start_epoch)
    print('best_loss: ', best_loss)
    print('best_val_loss: ', best_val_loss)
    print('best_val_absrel: ', best_absrel_val)
    if resume_training == True:
        #print('best_mean_iou2: ', best_mean_iou2)
        #print('best_val_loss2: ', best_val_loss2)
        print('valepoch: ', valepoch)
        print('valabsrelepoch: ', valabsrelepoch)
        print('ind_sch: ', ind_sch)

    #epochs = 46
    #lr_schedule = [lr_initial, lr_initial/2, lr_initial/5, lr_initial/10]
    #lr_milestones = [10, 20, 30, epochs]
    #lr_schedule = [lr_initial, 0.01, 0.005, 0.001]
    #lr_milestones = [12, 24, 34, epochs]
    #for g in optimizer.param_groups:
    #            g['lr'] = lr_new
    print('lr_schedule: ', lr_schedule)
    print('lr_milestones: ', lr_milestones)
    #print('lr_new: ', lr_new)
    print('resultsdir: ', resultsdir)
    #loaderiter = iter(loader)
    if resume_training == False:
        ind_sch = 0
    for e in range(start_epoch, epochs):
        print('epoch: ', e)
        count = 0
        for g in optimizer.param_groups:
            print('g[lr]: ', g['lr'])
            if count == 0:
                break
        if e in lr_milestones:
            ind_sch += 1
            lr_new = lr_schedule[ind_sch]
            for g in optimizer.param_groups:
                g['lr'] = lr_new

        total_loss = 0
        model.train()
        numimgs = 0
        totalrmse = 0
        totalabsrel = 0
        totalsqrel = 0
        for i, data in enumerate(data_loader):

            print('epoch: ', e, ' i: ', i)
            d = data['image']
            dd = data['gt']
            dimg = data['original']
            optimizer.zero_grad()
            x = model.forward(d)
            print('x shape: ', x.shape)
            numimgs += x.shape[0]            
            print('min(dd): ', torch.min(dd))
            print('max(dd): ', torch.max(dd))
            output = criterion(x,dd)
            rmse = RMSE(x,dd)
            absrel = abs_rel_error(x,dd)
            sqrel = sq_rel_error(x,dd)
            print('output: ', output)
            #breakpoint()
            output.backward(retain_graph=False)
            optimizer.step()
            loss = output.item()
            total_loss = total_loss + batch_size*loss
            totalrmse = totalrmse + batch_size*rmse
            totalabsrel = totalabsrel + batch_size*absrel
            totalsqrel = totalsqrel + batch_size*sqrel
            if i == 0:
                break


        training_loss = total_loss/numimgs
        training_loss_list.append(training_loss)
        epoch_list = range(len(training_loss_list))
        rmse = totalrmse/numimgs
        absrel = totalabsrel/numimgs
        sqrel = totalsqrel/numimgs
        print('training rmse: ', rmse)
        print('training absrel: ', absrel)
        print('training sqrel: ', sqrel)
        print('training loss: ', training_loss)
        rmselist.append(rmse)
        absrellist.append(absrel)

        with open(resultsdir + '/training_loss_list.pkl', 'wb') as f:
            pickle.dump(training_loss_list, f)
        with open(resultsdir + '/absrellist.pkl', 'wb') as f:
            pickle.dump(absrellist, f)
        with open(resultsdir + '/rmselist.pkl', 'wb') as f:
            pickle.dump(rmselist, f)


        if compute_val_over_whole_data:
            rmse_val, absrel_val, sqrel_val, val_loss = compute_metrics(val_data_loader, batch_size_val, dataset_name='kitti', imgdir=resultsdir, model=model, modelpath=None, modelname='NeWCRFDepth', gt_present=True, save_images=False, criterion=criterion, epoch=e) 
            print('rmse_val: ', rmse_val)
            print('absrel_val: ', absrel_val)
            print('val_loss: ', val_loss)
            print('sqrel_val: ', sqrel_val)

        rmse_val_list.append(rmse_val)
        absrel_val_list.append(absrel_val)
        loss_val_list.append(val_loss)

        with open(resultsdir + '/rmse_val_list.pkl', 'wb') as f:
            pickle.dump(rmse_val_list, f)
        with open(resultsdir + '/absrel_val_list.pkl', 'wb') as f:
            pickle.dump(absrel_val_list, f)
        with open(resultsdir + '/loss_val_list.pkl', 'wb') as f:
            pickle.dump(loss_val_list, f)

        metrics = dict{'rmse':rmse, 'absrel':absrel, 'sqrel': sqrel}
        metrics_val = dict{'rmse':rmse_val, 'absrel':absrel_val, 'sqrel_val': sqrel_val}
        
        if training_loss <= best_loss:
            path = resultsdir + '/bestlossdepthmodelnew.pt'
            save_model(training_loss, path, e, model, optimizer, lr_schedule, lr_milestones, metrics, metrics_val)
            best_loss = training_loss

        if best_loss != training_loss:
            path = resultsdir + '/latestdepthmodelnew.pt'
            save_model(training_loss, path, e, model, optimizer, lr_schedule, lr_milestones, metrics, metrics_val)

        metrics = dict{'rmse':rmse_val, 'absrel':absrel_val, 'sqrel_val': sqrel_val}

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = resultsdir + '/bestvallossdepthmodelnew.pt'
            save_model(val_loss, path, e, model, optimizer, lr_schedule, lr_milestones, metrics, metrics_val)

        if absrel_val < best_absrel_val:
            best_absrel_val = absrel_val
            path = resultsdir + '/bestabsreldepthmodelnew.pt'
            save_model(val_loss, path, e, model, optimizer, lr_schedule, lr_milestones, metrics, metrics_val)

