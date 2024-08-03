import torch  
import torch.nn as nn  
import torch.nn.functional as F 
import torchvision.models as models  
import numpy as np
from Exceptions import OutofIndexError
from torchvision.models import vgg16_bn
from torchvision.models import VGG16_BN_Weights
from time import time
from einops import rearrange
from torch.nn.functional import pad
from torchvision.transforms import Resize
from torch.nn.functional import interpolate

class PatchEmbeddingLayer(nn.Module):
    def __init__(self, patch_size, in_channels, embedding_dims):
        super(PatchEmbeddingLayer, self).__init__()
        self.ConvLayer = nn.Conv2d(in_channels, embedding_dims, kernel_size=patch_size, stride=patch_size)
        self.flattenLayer = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        out = self.ConvLayer(x)
        out = self.flattenLayer(out)
        out = out.permute((0,2,1))
        return out

class MLPblock(nn.Module):
    def __init__(self, embedding_dims, hidden_dims):
        super(MLPblock, self).__init__()
        self.embedding_dims = embedding_dims
        self.hidden_dims = hidden_dims
        self.layerNorm = nn.LayerNorm(embedding_dims)
        self.mlplayer = nn.Sequential(nn.Linear(in_features = embedding_dims, out_features = hidden_dims), nn.GELU(),
            nn.Dropout(p=0.30), nn.Linear(in_features = hidden_dims, out_features = embedding_dims), nn.Dropout(p=0.30))

    def forward(self, x):
        out = self.layerNorm(x)
        out = self.mlplayer(out) + x
        return out

class ConvLayer(nn.Module):
    def __init__(self, inputfeatures, outputinter, kernel_size=7, stride=1, padding=3, dilation=1, output=64, layertype=1, droupout=False):
        super(ConvLayer, self).__init__()
        if droupout == False:
            self.layer1 = nn.Sequential(
            nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer2 = nn.Sequential(
            nn.Conv2d(outputinter, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer3 = nn.Sequential(
            nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(output),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        else: 
            self.layer1 = nn.Sequential(
            nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer2 = nn.Sequential(
            nn.Conv2d(outputinter, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer3 = nn.Sequential(
            nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(output), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))

        self.layer4 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.layer5 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=False)
        self.layertype = layertype

    def forward(self, x):
        #print('ConvLayer: ')
        #print('x shape: ', x.shape)
        out1 = self.layer1(x)
        #print('out1 shape: ', out1.shape)
        if self.layertype == 1:
            out1 = self.layer3(out1)
            #print('out2 shape: ', out2.shape)
            out1, inds = self.layer4(out1)
            #print('out3 shape: ', out3.shape)
            return out1, inds
        elif self.layertype == 2:
            out1 = self.layer2(out1)
            #print('out2 shape: ', out2.shape)
            out1 = self.layer3(out1)
            #print('out3 shape: ', out3.shape)
            out1, inds = self.layer4(out1)
            #print('out4 shape: ', out4.shape)
            return out1, inds
        elif self.layertype == 3:
            out1 = self.layer3(out1)
            return out1
        elif self.layertype == 4:
            out1 = self.layer3(out1)
            #print('out2 shape: ', out2.shape)
            out1 = self.layer5(out1)
            #print('out3 shape: ', out3.shape)
            return out1


class MultiheadSelfAttentionblock(nn.Module):
    def __init__(self, embedding_dims, num_heads):
        super(MultiheadSelfAttentionblock, self).__init__()
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        self.layernorm = nn.LayerNorm(embedding_dims)
        self.multiheadattention = nn.MultiheadAttention(embedding_dims, num_heads, batch_first=True)

    def forward(self, x):
        out = self.layernorm(x)
        out, _ = self.multiheadattention(out, out, out, need_weights=False)
        print('out shape after multiheadattention: ', out.shape)
        out = out + x
        return out

class Transformerblock(nn.Module):
    def __init__(self, embedding_dims, hidden_dims, num_heads):
        super(Transformerblock, self).__init__()
        self.msablock = MultiheadSelfAttentionblock(embedding_dims, num_heads)
        self.mlpblock = MLPblock(embedding_dims, hidden_dims)

    def forward(self, x):
        out = self.msablock(x)
        out = self.mlpblock(out)
        return out


class ShiftedWindowMSA(nn.Module):
    def __init__(self, emb_size, num_heads, window_sizex=7, window_sizey=7, shifted=True, relative_pos_embeddings=True):
        super(ShiftedWindowMSA, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.window_sizex = window_sizex
        self.window_sizey = window_sizey
        self.shifted = shifted
        self.linear1 = nn.Linear(emb_size, 3*emb_size)
        self.pos_embeddings = nn.Parameter(torch.randn(window_sizex*2- 1, window_sizey*2 - 1))
        self.indices = torch.from_numpy(np.indices((window_sizex,window_sizey))).permute((1,2,0)).reshape((window_sizex*window_sizey,2))
        self.rel_indices = self.indices[None,:,:] - self.indices[:,None,:] + torch.tensor([window_sizex-1, window_sizey-1])
        self.rel_pos_embeddings = relative_pos_embeddings
        

    def forward(self, x):
        h_dim = self.emb_size / self.num_heads
        height = x.shape[1]
        width = x.shape[2]
        x = self.linear1(x)

        x = rearrange(x, 'b h w (c k) -> b h w c k', h=height, w=width, k=3, c=self.emb_size)

        #x = rearrange(x, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k', w1 = self.window_size, w2 = self.window_size, H = self.num_heads)
        
        #print('x shape: ', x.shape)

        if self.shifted:
            x = torch.roll(x, (-self.window_sizex//2, -self.window_sizey//2), dims=(1,2))
        
        x = rearrange(x, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k', w1 = self.window_sizex, w2 = self.window_sizey, H = self.num_heads)            
        
                
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        wei = (Q @ K.transpose(4,5)) / np.sqrt(h_dim)

        if self.rel_pos_embeddings:

            rel_pos_embeddings = self.pos_embeddings[self.rel_indices[:,:,0],self.rel_indices[:,:,1]]
            #print('rel_pos_embeddings shape: ', rel_pos_embeddings.shape)
            wei += rel_pos_embeddings

        #print('wei shape: ', wei.shape)
        
        if self.shifted:
            row_mask = torch.zeros((self.window_sizex*self.window_sizey, self.window_sizex*self.window_sizey))
            row_mask[-self.window_sizex * (self.window_sizey//2):, 0:-self.window_sizex * (self.window_sizey//2)] = float('-inf')
            row_mask[0:-self.window_sizex * (self.window_sizey//2), -self.window_sizex * (self.window_sizey//2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_sizex, w2=self.window_sizey)
            wei[:, :, -1, :] += row_mask
            wei[:, :, :, -1] += column_mask

        #print('wei shape: ', wei.shape)



        wei = F.softmax(wei, dim=-1) @ V

        #print('wei shape: ', wei.shape)

        #print('wei[0, 0, 0, 0, 0:25, 0:5]: ', wei[0, 0, 0, 0, 0:25, 0:5])

        wei = rearrange(wei, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (e H)', w1 = self.window_sizex, w2 = self.window_sizey)

        if self.shifted:
            x = torch.roll(x, (self.window_sizex//2, self.window_sizey//2), dims=(1,2))

        #print ('wei shape: ', wei.shape)

        #print('wei[0, 0:5, 0:5, 0:5]: ', wei[0, 0:5, 0:5, 0:5])

        return wei



class SwinTransformerblock(nn.Module):
    def __init__(self, embedding_dims, hidden_dims, num_heads, window_sizex=5, window_sizey=5, shifted_window=True, relative_pos_embeddings=True):
        super(SwinTransformerblock, self).__init__()
        self.layernorm = nn.LayerNorm(embedding_dims)
        self.msablock = ShiftedWindowMSA(embedding_dims, num_heads, window_sizex=window_sizex, window_sizey=window_sizey, shifted=shifted_window, relative_pos_embeddings=True)
        self.mlpblock = MLPblock(embedding_dims, hidden_dims)
        #self.window_size = window_size
        self.window_sizex = window_sizex
        self.window_sizey = window_sizey

    def forward(self, x):
        out = self.layernorm(x)
        #x = rearrange(x, 'b (Wh w1) (Ww w2) (e H) -> b H Wh Ww (w1 w2) e', w1 = self.window_size, w2 = self.window_size, H = self.num_heads)
        out = self.msablock(out) + x
        out = self.mlpblock(out) + out
        return out

#class TransformerblockWMSA(nn.Module):

class patchify(nn.Module):
    def __init__(self, patch_size=4):
        super(patchify, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        out = rearrange(x, 'b c (h h1) (w w1) -> b h w (h1 w1 c)', h1=self.patch_size, w1=self.patch_size)
        return out

class patchmerging(nn.Module):
    def __init__(self, patch_size=2, embedded_dims=256, output_dims=512):
        super(patchmerging, self).__init__()
        self.patch_size = patch_size
        #self.Linearlayer = nn.Linear(patch_size*patch_size*embedded_dims,resolution*embedded_dims)
        self.Linearlayer = nn.Linear(patch_size*patch_size*embedded_dims,output_dims)


    def forward(self, x):
        out = rearrange(x, 'b (h h1) (w w1) c -> b h w (h1 w1 c)', h1=self.patch_size, w1=self.patch_size)
        #print('out shape: ', out.shape)
        out = self.Linearlayer(out)
        return out


class stageblock(nn.Module):
    def __init__(self, stage=1, numblocks=2, resolution=2, embedding_dims=256, output_dims=512, patch_size=2, num_heads=8, window_sizex=5, window_sizey=5, relative_pos_embeddings=True):
        super(stageblock, self).__init__()
        #self.patchmerging = patchmerging(patch_size=patch_size,embedded_dims=embedding_dims,resolution=2)
        self.patchmerging = patchmerging(patch_size=patch_size,embedded_dims=embedding_dims, output_dims=output_dims)
        #self.newembedded_dims = resolution*embedded_dims
        self.newembedded_dims = output_dims
        self.newhidden_dims = 2*self.newembedded_dims
        self.numblocks = numblocks
        self.transformerblocks = [SwinTransformerblock(self.newembedded_dims, self.newhidden_dims, num_heads, window_sizex=window_sizex, window_sizey=window_sizey, shifted_window=i%2, relative_pos_embeddings=True) for i in range(numblocks)]
        #self.TransformerblockSWMSA2 = SwinTransformerblock(embedding_dims, hidden_dims, num_heads, window_size=window_size, shifted_window=True)
        #self.TransformerblockWMSA2 = SwinTransformerblock(embedding_dims, hidden_dims, num_heads, window_size=window_size, shifted_window=False)        

    def forward(self, x):
        out = self.patchmerging(x)
        print('after patch merging out shape: ', out.shape)
        for i in range(self.numblocks):
            out = self.transformerblocks[i](out)
            #print('out shape: ', out.shape)
        return out

class SwinTransformer(nn.Module):
    def __init__(self, embedding_dims=256, hidden_dims=512, num_heads=8, window_size=5):
        super(SwinTransformer, self).__init__()
        #self.downsamplelayer(scale_factor=2,mode='bilinear')
        self.layer1 = ConvLayer(3, 64, kernel_size=4, padding=3, dilation=2, output=64, layertype=4)
        self.layer2 = ConvLayer(64, 64, kernel_size=7, padding=3, output=128, layertype=1)
        self.layer3 = ConvLayer(128, 128, kernel_size=7, output=256, layertype=1)
        self.TransformerblockSWMSA = SwinTransformerblock(embedding_dims, hidden_dims, num_heads, window_sizex=5, window_sizey=5, shifted_window=True, relative_pos_embeddings=True)
        self.TransformerblockWMSA = SwinTransformerblock(embedding_dims, hidden_dims, num_heads, window_sizex=5, window_sizey=5, shifted_window=False, relative_pos_embeddings=True)
        self.TransformerblockSWMSA2 = SwinTransformerblock(embedding_dims, hidden_dims, num_heads, window_sizex=5, window_sizey=5, shifted_window=True, relative_pos_embeddings=True)
        self.TransformerblockWMSA2 = SwinTransformerblock(embedding_dims, hidden_dims, num_heads, window_sizex=5, window_sizey=5, shifted_window=False, relative_pos_embeddings=True)
        self.window_size = window_size
        self.Linearlayer = nn.Linear(4*4*3,embedding_dims)
        #self.patchmerging1 = patchmerging(patch_size=2,embedded_dims=embedding_dims,resolution=2)
        #self.TransformerblockSWMSA2 = SwinTransformerblock(embedding_dims, hidden_dims, num_heads, window_size=window_size, shifted_window=True)
        #self.TransformerblockWMSA2 = SwinTransformerblock(embedding_dims, hidden_dims, num_heads, window_size=window_size, shifted_window=False)
        self.stageblock1 = stageblock(stage=2, numblocks=2, resolution=2, embedding_dims=256, output_dims=512, patch_size=2, num_heads=8, window_sizex=5, window_sizey=5, relative_pos_embeddings=True)
        self.stageblock2 = stageblock(stage=3, numblocks=2, resolution=2, embedding_dims=512, output_dims=1024, patch_size=2, num_heads=8, window_sizex=6, window_sizey=6, relative_pos_embeddings=True)
        self.stageblock3 = stageblock(stage=4, numblocks=2, resolution=2, embedding_dims=1024, output_dims=1024, patch_size=2, num_heads=8, window_sizex=6, window_sizey=5, relative_pos_embeddings=True)

    def forward(self, x):
        #out = rearrange(x, 'b (h h1) (w w1) c -> b h w (h1 w1 c)', h1=self.window_size, w1=self.window_size)
        print('x shape: ', x.shape)
        out = rearrange(x, 'b c (h h1) (w w1) -> b h w (h1 w1 c)', h1=4, w1=4)
        #print('x[0,:,0:4,0:4]: ', x[0,:,0:4,0:4])
        #print('out[0,0,0,:]: ', out[0,0,0,:])
        #print('out shape: ', out.shape)
        out = self.Linearlayer(out)
        #print('out shape: ', out.shape)
        out = self.TransformerblockWMSA(out)
        #print('out shape: ', out.shape)
        out = self.TransformerblockSWMSA(out)
        #print('out shape: ', out.shape)
        out = self.TransformerblockWMSA2(out)
        #print('out shape: ', out.shape)
        out = self.TransformerblockSWMSA2(out)
        #print('out shape: ', out.shape)
        #out = self.patchmerging1(out)
        out1 = self.stageblock1(out)
        #print('after stageblock1 out shape: ', out.shape)
        out2 = pad(out1, (0,0,0,0,1,2), 'replicate')
        #print('out shape: ', out.shape)
        out2 = self.stageblock2(out2)
        #print('after stageblock2 out shape: ', out.shape)
        out3 = self.stageblock3(out2)
        #print('after stageblock3 out shape: ', out.shape)
        return out, out1, out2, out3

class NeWCRFlayer(nn.Module):
    def __init__(self, )

class PSPhead(nn.Module):
    def __init__(self, input_dim, output_dims, pool_scales=[1,2,3,6]):
        super(PSPhead, self).__init__()
        self.ppm_modules = [nn.Sequential(nn.AdaptiveAvgPool2d(pool), nn.Conv2d(input_dim, output_dims, kernel_size=1),
            nn.BatchNorm2d(output_dims),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None)) for pool in pool_scales]

        self.bottleneck = nn.Sequential(nn.Conv2d(input_dim + output_dims*len(pool_scales), output_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dims),
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))


    def forward(self, x):
        x = x.permute((0,3,1,2))
        ppm_outs = []
        ppm_outs.append(x)
        for ppm in self.ppm_modules:
            #ppm_out = Resize((x.shape[2], x.shape[3]), interpolation=InterpolationMode.BILINEAR)
            ppm_out = interpolate(ppm(x), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=None)
            ppm_outs.append(ppm_out)
        ppm_outs = torch.cat(ppm_outs, dim=1)
        print('ppm_outs shape: ', ppm_outs.shape)
        ppm_head_out = self.bottleneck(ppm_outs)
        ppm_head_out = ppm_head_out.permute((0,2,3,1))
        #print('ppm_head_out shape: ', ppm_head_out.shape)
        return ppm_head_out

class NeWCRFDepth(nn.Module):
    def __init__(self):
        super(NeWCRFDepth, self).__init__()
        self.transformer = SwinTransformer()
        self.PPMhead = PSPhead(1024, 256)

    def forward(self, x):
        trans_outs = self.transformer(x)
        print('trans_outs[0].shape: ', trans_outs[0].shape)
        print('trans_outs[1].shape: ', trans_outs[1].shape)
        print('trans_outs[2].shape: ', trans_outs[2].shape)
        print('trans_outs[3].shape: ', trans_outs[3].shape)
        ppm_head_out = self.PPMhead(trans_outs[3])
        print('ppm_head_out shape: ', ppm_head_out.shape)
        return ppm_head_out


