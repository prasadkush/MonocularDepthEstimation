import torch  
import torch.nn as nn  
import torch.nn.functional as F 
import torchvision.models as models  
import numpy as np

eps = 1e-7
#lambda_ = 0.85
#alpha = 10


class depth_loss(nn.Module):
	def __init__(self, lambda_=0.85, alpha=10):
		super(depth_loss, self).__init__()
		self.lambda_ = lambda_
		self.alpha = alpha

	def forward(self, output, gt):
		deltad = torch.log((output + eps)/(gt + eps))
		num_pixels = torch.sum(gt != 0, dim=(1,2))
		#print('num_pixels: ', num_pixels)
		#breakpoint()
		#print('deltad/num_pixels: ', deltad/num_pixels)
		deltad[gt == 0] = 0
		#loss = slef.alpha*torch.mean(torch.sqrt(torch.mean(torch.square(deltad), dim=(1,2)) - self.lambda_*torch.square(torch.mean(deltad, dim=(1,2)))))
		#temp = self.alpha*torch.sqrt(torch.sum(torch.square(deltad), dim=(1,2))/num_pixels - self.lambda_*torch.square(torch.sum(deltad, dim=(1,2))/num_pixels))
		#print('temp: ', temp)
		#breakpoint()
		loss = self.alpha*torch.mean(torch.sqrt(torch.sum(torch.square(deltad), dim=(1,2))/num_pixels - self.lambda_*torch.square(torch.sum(deltad, dim=(1,2))/num_pixels)))
		return loss

