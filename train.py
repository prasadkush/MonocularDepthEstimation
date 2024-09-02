import cv2
import torch
import torch.nn as nn
from torch import optim
from data import getDataset, get_color_transform, get_inverse_transforms
from torch.utils.data import DataLoader, Dataset
from model import Encoder, Segnet
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pdb
from predict import compute_metrics
from depth_metrics import RMSE, sq_rel_error, abs_rel_error, error_less_than, norm_error
from Exceptions import ModelPathrequiredError
from losses import depth_loss
from utils import find_nth_occurence
import random
from time import time
import matplotlib.cm as cm
import matplotlib as mpl
#from torchvision.models import vgg16_bn

weight_dict = {}

def save_model(loss, val_loss, path, epoch, model, optimizer, lr_schedule, lr_milestones, metrics, metrics_val):
    EPOCH = epoch
    PATH_ = path
    LOSS_ = loss
    VAL_LOSS_ = val_loss
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
        #torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(m.weight, a=0.25, mode='fan_in', nonlinearity='leaky_relu')
        #torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        #torch.nn.init.normal_(m.weight, mean=0.2, std=1)
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        #torch.nn.init.uniform_(m.weight)
        #torch.nn.init.zeros_(m.bias)
    


def train(data_loader, val_data_loader, model, criterion, epochs=60, batch_size=4, batch_size_val=4, dataset_name='kitti', shuffle=True, val_dataset=None, modelpath=None, bestmodelpath=None, resume_training=False, useWeights=False, resultsdir=None, resultsdiropen=None,
    layer_wise_training=False, initialize_from_model=False):

    #criterion = depth_loss()


    lr_initial = 0.004
    lr_new = 0.004
    if resume_training == False:
        if initialize_from_model and modelpath != None:
            checkpoint = torch.load(modelpath)
            loaded_dict = checkpoint['model_state_dict']
            new_dict = {k:v for k, v in loaded_dict.items() if 'convup2' not in k}
            model.load_state_dict(new_dict)
        elif not initialize_from_model:
            model.apply(weights_init)
    print('after model.apply')
    paramlist = list(model.parameters())
    #print('paramlist: ')
    for name, param in list(model.named_parameters()):
        print('name: ', name)
        print('param requires_grad: ', param.requires_grad)
        if param.requires_grad == False:
            param.requires_grad = True
    #print('paramlist: ', paramlist)
    paramlist1 = []
    paramlist2 = []
    paramlist3 = []
    paramlist4 = []
    for name, param in model.named_parameters():
        if 'stageblock3.transformerblocks' in name or 'stageblock4.transformerblocks' in name or 'crf1.layers' in name or 'crf2.layers' in name:
            paramlist2.append(param)
        elif 'stageblock1.transformerblocks' in name or 'stageblock2.transformerblocks' in name or 'crf3.layers' in name or 'crf4.layers' in name:
            paramlist3.append(param)
        elif 'PPMhead' in name:
            paramlist4.append(param)
        else:
            paramlist1.append(param)
        print('name: ', name)
        print('param shape: ', param.shape)
            #if name in paramnames_to_print:
            #    param_to_print.append({'name':name, 'param':param})
    #optimizer = optim.Adam(paramlist, lr=lr_initial, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8)
    optimizer = optim.Adam([{'params': paramlist1, 'lr':lr_initial, 'betas':(0.9, 0.999), 'eps':1e-08},
                        {'params':paramlist2, 'lr':lr_initial, 'betas':(0.0,0.999), 'eps':1e-8},
                        {'params':paramlist3, 'lr':lr_initial, 'betas':(0.9,0.999), 'eps':1e-8},
                        {'params':paramlist4, 'lr':lr_initial, 'betas':(0.9,0.999), 'eps':1e-8}],
     weight_decay=1e-8)
    
    #paramlist = [(name, p) for name, p in model.named_parameters() if ('stageblock3.transformerblocks' not in name and 'stageblock4.transformerblocks' not in name and 'crf1.layers' not in name and 'crf2.layers' not in name)]
    #print('paramlist: ')
    #for name, param in paramlist:
    #    print('name: ', name)
    #    print('param shape: ', param.shape)
    #print('paramlist: ', paramlist)
    #paramlist = []
    #for name, param in model.named_parameters():
    #    if 'stageblock3.transformerblocks' not in name and 'stageblock4.transformerblocks' not in name and 'crf1.layers' not in name and 'crf2.layers' not in name:
    #        paramlist.append(param)
    #        print('name: ', name)
    #        print('param shape: ', param.shape)
            #if name in paramnames_to_print:
            #    param_to_print.append({'name':name, 'param':param})
    #    else:
    #        param.requires_grad = False
            #if name in paramnames_to_print:
            #    param_to_print.append({'name':name, 'param':param})


    #optimizer = optim.Adam(paramlist, lr=lr_initial, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    #optimizer = optim.SGD(model.parameters(), lr=lr_initial,  momentum=0.9, nesterov=True)
    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    #for name, param in list(model.named_parameters()):
    #    print('name: ', name)
    #    print('param shape: ', param.shape)
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
    best_absrel_val = 1000
    total_loss = 0
    training_loss_list = []
    #mean_list = []
    rmselist = []
    absrellist = []
    rmse_val_list = []
    absrel_val_list = []
    loss_val_list = []
    epochs = 16
    #lr_schedule = [lr_initial, 0.01, 0.005, 0.001]
    #lr_schedule = [lr_initial, 0.002, 0.001, 0.002, 0.001]
    lr_schedule = [0.004, 0.002, 0.001, 0.0005]
    #lr_milestones = [30, 40, 50, epochs]
    lr_milestones = [0, 1, 3, 10, 16]
    #lr_milestones = [1, 2, 16]
    lr_initial = lr_schedule[0]
    lr_new = lr_schedule[0]

    if resume_training == True and modelpath != None:
        checkpoint = torch.load(modelpath)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        if bestmodelpath != None:
            checkpoint = torch.load(bestmodelpath)
            best_loss = checkpoint['loss']
        else:
            best_loss = checkpoint['loss']
        start_epoch = epoch + 1
        ind_sch = np.searchsorted(np.array(lr_milestones), start_epoch, side='right') - 1
        lr_new = lr_schedule[ind_sch]
        lr_initial = lr_schedule[ind_sch]
        print('start_epoch: ', start_epoch)
        print('ind_sch: ', ind_sch)
        print('lr_new: ', lr_new)
        count = 0
        for g in optimizer.param_groups:
            g['lr'] = lr_initial
            print('gr: ', count, '  lr: ', g['lr'])
            count += 1  
        #bestvalmodelpath = resultsdiropen + '/bestvallossdepthmodelnew.pt'
        #valcheckpoint = torch.load(bestvalmodelpath)
        #valepoch = valcheckpoint['epoch']
        #best_val_loss = valcheckpoint['loss']

        #bestvalabsrelmodelpath = resultsdiropen + '/bestvalabsreldepthmodelnew.pt'
        #valabsrelcheckpoint = torch.load(bestvalabsrelodelpath)
        #valabsrelepoch = valabsrelcheckpoint['epoch']
        #best_absrel_val = valabsrelcheckpoint['metrics']['absrel']

        with open(resultsdiropen + '/training_loss_list.pkl', 'rb') as f:
            training_loss_list = pickle.load(f)
            training_loss_list = training_loss_list[0:min(start_epoch,len(training_loss_list))]
            print('len(training_loss_list): ', len(training_loss_list))
            print('training_loss_list: ', training_loss_list)
            best_loss = min(training_loss_list)
        #with open(resultsdir + '/mean_list.pkl', 'rb') as f:
        #    mean_list = pickle.load(f)
        with open(resultsdiropen + '/absrellist.pkl', 'rb') as f:
            absrellist = pickle.load(f)
            absrellist = absrellist[0:min(start_epoch,len(absrellist))]
            print('len(absrellist): ', len(absrellist))
            print('absrellist: ', absrellist)
        with open(resultsdiropen + '/rmselist.pkl', 'rb') as f:
            rmselist = pickle.load(f)
            rmselist = rmselist[0:min(start_epoch,len(rmselist))]
            print('rmselist: ', rmselist)
        with open(resultsdiropen + '/absrel_val_list.pkl', 'rb') as f:
            absrel_val_list = pickle.load(f)
            absrel_val_list = absrel_val_list[0:min(start_epoch,len(absrel_val_list))]
            best_absrel_val = min(absrel_val_list)
            print('len(absrel_val_list): ', len(absrel_val_list))
            print('absrel_val_list: ', absrel_val_list)
        with open(resultsdiropen + '/rmse_val_list.pkl', 'rb') as f:
            rmse_val_list = pickle.load(f)
            rmse_val_list = rmse_val_list[0:min(start_epoch,len(rmse_val_list))]
            print('len(rmse_val_list): ', len(rmse_val_list))
            print('rmse_val_list: ', rmse_val_list)
        with open(resultsdiropen + '/loss_val_list.pkl', 'rb') as f:
            loss_val_list = pickle.load(f)
            loss_val_list = loss_val_list[0:min(start_epoch,len(loss_val_list))]
            print('len(loss_val_list): ', len(loss_val_list))
            print('loss_val_list: ', loss_val_list)
            best_val_loss = min(loss_val_list)
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
        #print('valepoch: ', valepoch)
        #print('valabsrelepoch: ', valabsrelepoch)
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
    print('lr_new: ', lr_new)
    print('resultsdir: ', resultsdir)
    #loaderiter = iter(loader)

    plasma = plt.get_cmap('plasma')
    greys = plt.get_cmap('Greys')
    normalizer = mpl.colors.Normalize(vmin=0, vmax=model.max_depth)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    Normalize = get_inverse_transforms('kitti')
    #params_to_print = [model.swintransformer.stageblock1.transformerblocks[0].msablock.linear1.weight, model.swintransformer.stageblock1.transformerblocks[1].mlpblock.mlplayer[3].weight,
    #model.crf1.layers[0].mlpblock.mlplayer[0].weight, model.crf2.layers[0].mlpblock.mlplayer[3].weight]
    params_to_print1 = ['swintransformer.stageblock1.transformerblocks.0.msablock.linear1.weight', 'swintransformer.stageblock2.transformerblocks.1.mlpblock.mlplayer.3.weight', 'swintransformer.stageblock3.transformerblocks.2.msablock.linear1.weight',
    'swintransformer.stageblock4.transformerblocks.1.mlpblock.mlplayer.3.weight', 'PPMhead.ppm_modules.1.1.weight', 'PPMhead.bottleneck.0.weight', 'crf1.layers.0.mlpblock.mlplayer.0.weight', 'crf2.layers.0.mlpblock.mlplayer.3.weight',
    'crf3.layers.0.mlpblock.mlplayer.0.weight', 'crf4.layers.0.mlpblock.mlplayer.3.weight']
    paramnames = [p[0:min(find_nth_occurence(p,'.',n=6),len(p))] for p in params_to_print1]
    paramshortlist = []
    wt_values = []
    prev_wt_values = []
    diff_wts = []
    for p in params_to_print1:
        paramshortlist.append(model.get_parameter(p))
        if len(paramshortlist[-1].data.shape) <= 2:
            wt_values.append(paramshortlist[-1].data)
        elif len(paramshortlist[-1].data.shape) == 4:
            wt_values.append(paramshortlist[-1].data[:,:,0,0])
        #elif len(paramshortlist[-1].data.shape) == 3:
        #    wt_values.append(paramshortlist[-1].data[:,:,0,])
        prev_wt_values.append(torch.zeros(wt_values[-1].shape))
        diff_wts.append(0)
    #for param in params_to_print:
    #    print('param.is_leaf: ', param.is_leaf)
    #    if  not param.is_leaf:
    #        param.requires_grad_(requires_grad=True)
    #        param.retain_grad()
    if resume_training == False:
        ind_sch = 0
    for e in range(start_epoch, epochs):
        print('epoch: ', e)
        if e > 1:
            if e == 7:
                for l in range(3):
                    for p in optimizer.param_groups[l]['params']:
                        p.requires_grad = True
            #if e % 2 == 1:
            #    pind1 = 1
            #    pind2 = 2
            #elif e % 2 == 0:
            #    pind1 = 2
            #    pind2 = 1
            #for p in optimizer.param_groups[pind1]['params']:
            #    p.requires_grad = True
            #for p in optimizer.param_groups[pind2]['params']:
            #    p.requires_grad = False
            #    p.grad = None
            if e == 6:
                for l in range(3):
                    for p in optimizer.param_groups[l]['params']:
                        p.requires_grad = False
                        p.grad = None

        count = 0
        
        if e in lr_milestones and e != start_epoch:
            ind_sch += 1
            lr_new = lr_schedule[ind_sch]
            for g in optimizer.param_groups:
                g['lr'] = lr_new
        for g in optimizer.param_groups:
            print('gr: ', count, '  lr: ', g['lr'])
            count += 1
        total_loss = 0
        model.train()
        numimgs = 0
        totalrmse = 0
        totalabsrel = 0
        totalsqrel = 0
        for i, data in enumerate(data_loader):
            start_time1 = time()
            print('epoch: ', e, ' i: ', i)
            print('lr: ', optimizer.param_groups[0]['lr'])
            d = data['image']
            dd = data['gt']
            dimg = data['original']
            optimizer.zero_grad()
            start_time = time()
            x = model.forward(d)
            elapsed_time = time() - start_time
            print('x shape: ', x.detach().shape)
            print('avg img time: ', elapsed_time/d.shape[0])          
            numimgs += d.shape[0]            
            #print('min(dd): ', torch.min(dd))
            #print('max(dd): ', torch.max(dd))
            output = criterion(x,dd)
            rmse = RMSE(x.detach(),dd)
            absrel = abs_rel_error(x.detach(),dd)
            sqrel = sq_rel_error(x.detach(),dd)
            err_less_than_125 = error_less_than(x.detach(), dd)
            inds = np.argwhere(dd[0,:,:].numpy() != 0)
            ind = random.randint(0,inds.shape[0] - 6)
            inds = inds[ind:ind+5,:]
            #ix = random.randint(0,d.shape[2] - 7)
            #iy = random.randint(0,d.shape[3] - 2)
            #breakpoint()
            output.backward(retain_graph=False)
            optimizer.step()
            loss = output.detach().item()
            total_loss = total_loss + d.shape[0]*loss
            totalrmse = totalrmse + d.shape[0]*rmse
            totalabsrel = totalabsrel + d.shape[0]*absrel
            totalsqrel = totalsqrel + d.shape[0]*sqrel
            l1error = norm_error(x.detach(), dd, norm=1)
            l2error = norm_error(x.detach(), dd, norm=2)
            print('loss: ', loss)
            print('rmse: ', rmse)
            print('absrel: ', absrel)
            print('epoch rmse: ', totalrmse/(d.shape[0]*(i+1)))
            print('epoch absrel: ', totalabsrel/(d.shape[0]*(i+1)))
            #print('x[0:3, ix, iy]: ', x.detach()[0, ix:ix+5, iy])
            print('err_less_than 1.25', err_less_than_125, '\n')
            print('l1 error: ', l1error)
            print('l2 error: ', l2error)
            #print('dd[0:3, ix, iy]: ', dd[0, ix:ix+5, iy])
            print('x[0:3, ix, iy]: ', x.detach()[0, inds[0:5,0], inds[0:5,1]])
            print('dd[0:3, ix, iy]: ', dd[0, inds[0:5,0], inds[0:5,1]])
            for l, param in enumerate(paramshortlist):
                #print('wt_values[l] norm: ', torch.norm(wt_values[l], p=2)/torch.numel(wt_values[l]))
                #wt_values[l] = param.data
                #print('wt_values[l] norm: ', torch.norm(wt_values[l], p=2)/torch.numel(wt_values[l]))
                diff_wts[l] = wt_values[l] - prev_wt_values[l]
                prev_wt_values[l].copy_(wt_values[l])
                if param.grad != None:
                    ix =  random.randint(0,param.grad.shape[0] - 6)
                    if len(param.grad.shape) == 2:
                        iy = random.randint(0,param.grad.shape[1] - 1)
                        print('grad ', l, ': ', param.grad[ix:ix+5,iy])
                        print('grad norm ', l, ': ', torch.norm(param.grad, p=1).item()/torch.numel(param.grad), '              name: ', paramnames[l])
                    elif len(param.grad.shape) == 4:
                        iy = random.randint(0,param.grad.shape[1] - 1)
                        print('grad ', l, ': ', param.grad[ix:ix+5,iy,0,0])
                        print('grad norm ', l, ': ', torch.norm(param.grad[:,:,0,0], p=1).item()/torch.numel(param.grad[:,:,0,0]), '              name: ', paramnames[l])
                    elif len(param.grad.shape) == 1:
                        print('grad ', l, ': ', param.grad[ix:ix+5])
                        print('grad norm ', l, ': ', torch.norm(param.grad, p=1).item()/torch.numel(param.grad), '              name: ', paramnames[l])                                        
                else:
                    print('grad ', l, ': ', param.grad)
                print('diff_wts[l] norm: ', torch.norm(diff_wts[l], p=1).item()/torch.numel(diff_wts[l]), '     wt norm: ', torch.norm(wt_values[l], p=1).item()/torch.numel(wt_values[l]))
            if i % 50 == 0:
                print('epoch training_loss: ', total_loss/(d.shape[0]*(i+1)))
                print('epoch rmse: ', totalrmse/(d.shape[0]*(i+1)))
                print('epoch absrel: ', totalabsrel/(d.shape[0]*(i+1)))
            if i % int(len(data_loader.dataset)/(4*d.shape[0])) == 0:
                print('epoch training_loss: ', total_loss/(d.shape[0]*(i+1)))
                print('epoch rmse: ', totalrmse/(d.shape[0]*(i+1)))
                print('epoch absrel: ', totalabsrel/(d.shape[0]*(i+1)))
                imgorig = Normalize(d)
                imgorig = torch.permute(imgorig, (0,2,3,1))
                imgorig = (255*imgorig).numpy().astype('uint8')
                dn = ((x.detach().numpy()/model.max_depth)*255).astype(np.uint8)
                gtn = ((dd.numpy()/model.max_depth)*255).astype(np.uint8)
                for j in range(d.shape[0]):
                    coloredDepth = (mapper.to_rgba(dn[j,:,:])[:, :, :3] * 255).astype(np.uint8)
                    cv2.imwrite(resultsdir + '/training/e_' + str(e) + '_i_' + str(i) + '_depth1_' + str(j) + '.jpg', coloredDepth)
                    cv2.imwrite(resultsdir + '/training/e_' + str(e) + '_i_' + str(i) + '_depth2_'  + str(j) + '.jpg', dn[j,:,:])
                    cv2.imwrite(resultsdir + '/training/e_' + str(e) + '_i_' + str(i) + '_imgorig_' + str(j)  + '.jpg', imgorig[j,:,:,:])
                    cv2.imwrite(resultsdir + '/training/e_' + str(e) + '_i_' + str(i) + '_depthorig_' + str(j) + '.jpg', gtn[j,:,:])
            elapsed_time1 = time() - start_time1
            print('elapsed_time1: ', elapsed_time1)

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
            rmse_val, absrel_val, sqrel_val, val_loss = compute_metrics(val_data_loader, batch_size_val, dataset_name='kitti', imgdir=resultsdir + '/imgs', model=model, modelpath=None, modelname='NeWCRFDepth', gt_present=True, save_images=False, criterion=criterion, epoch=e) 
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

        metrics = {'rmse':rmse, 'absrel':absrel, 'sqrel': sqrel}
        metrics_val = {'rmse':rmse_val, 'absrel':absrel_val, 'sqrel_val': sqrel_val}
        
        if training_loss <= best_loss:
            path = resultsdir + '/bestlossdepthmodelnew.pt'
            save_model(training_loss, val_loss, path, e, model, optimizer, lr_schedule, lr_milestones, metrics, metrics_val)
            best_loss = training_loss

        if best_loss != training_loss:
            path = resultsdir + '/latestdepthmodelnew.pt'
            save_model(training_loss, val_loss, path, e, model, optimizer, lr_schedule, lr_milestones, metrics, metrics_val)

        #metrics = {'rmse':rmse_val, 'absrel':absrel_val, 'sqrel_val': sqrel_val}

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = resultsdir + '/bestvallossdepthmodelnew.pt'
            save_model(training_loss, val_loss, path, e, model, optimizer, lr_schedule, lr_milestones, metrics, metrics_val)

        if absrel_val < best_absrel_val:
            best_absrel_val = absrel_val
            path = resultsdir + '/bestabsreldepthmodelnew.pt'
            save_model(training_loss, val_loss, path, e, model, optimizer, lr_schedule, lr_milestones, metrics, metrics_val)

