# Data processing 
import pandas as pd
from skimage import io, transform
# Math
import numpy as np
import math
# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
# Custom
from utils import vMF
'''
	input: (240, 320, 3) image
	output: two heads--> 1. distrubution (1D 240 vector), 2. parameters (1D 4 vector)
'''
class IlluminationModule(nn.Module):
	def __init__(self):
		super().__init__()
		self.cv_block1 = conv_bn_elu(3, 64, kernel_size=7, stride=2) # 160 x 120
		self.cv_block2 = conv_bn_elu(64, 128, kernel_size=5, stride=2) # 80 x 60
		self.cv_block3 = conv_bn_elu(128, 256, stride=2) # 40 x 30
		self.cv_block4 = conv_bn_elu(256, 256)
		self.cv_block5 = conv_bn_elu(256, 256, stride=2)  # 20 x 15
		self.cv_block6= conv_bn_elu(256, 256)
		self.cv_block7 = conv_bn_elu(256, 256, stride=2) # 10 x 8

		self.fc = nn.Linear(256*10*8, 2048)
		self.fc_bn = nn.BatchNorm1d(2048)
		''' two heads regression'''
		self.ds_fc = nn.Linear(2048, 256) # sky distribution
		self.ds_bn = nn.BatchNorm1d(256)
		self.pr_fc = nn.Linear(2048, 4) # sky and camera parameters
		self.pr_bn = nn.BatchNorm1d(4)
	
	def forward(self, x):
		x = self.cv_block1(x)
		x = self.cv_block2(x)
		x = self.cv_block3(x)
		x = self.cv_block4(x)
		x = self.cv_block5(x)
		x = self.cv_block6(x)
		x = self.cv_block7(x)
		x = x.view(-1, 256*10*8)
		x = F.elu(self.fc_bn(self.fc(x)))
		return F.log_softmax(self.ds_bn(self.ds_fc(x)), dim=1), self.pr_bn(self.pr_fc(x))
		

def conv_bn_elu(in_, out_, kernel_size=3, stride=1, padding=True):
	## conv layer with BN and ELU function 
	pad = int(kernel_size/2)
	if padding is False:
		pad = 0
	return nn.Sequential(
		nn.Conv2d(in_, out_, kernel_size, stride=stride, padding=pad),
		nn.BatchNorm2d(out_),
		nn.ELU(),
	)

'''
	Dataset loader 
	dataset standardization ==> 
		mean: [0.48548178 0.48455666 0.46329196] std: [0.21904471 0.21578524 0.23359051]
'''
class Train_Dataset(Dataset):
	def __init__(self, csv_path):
		self.to_tensor = transforms.ToTensor()
		self.normalize = transforms.Normalize(mean=[0.48548178, 0.48455666, 0.46329196],
												std= [0.21904471, 0.21578524, 0.23359051])
		self.data = pd.read_csv(csv_path)
		self.path_arr = np.asarray(self.data['filepath'])

		''' sun position: theta, phi '''
		self.theta_arr = np.asarray(self.data['theta'])
		self.phi_arr = np.asarray(self.data['phi'])

		''' parameters: turbidity, omega(scaling factor, exposure), camera evelation (degrees) and FoV (radians) '''
		self.tub_arr = np.asarray(self.data['turbidity'])
		self.omg_arr = np.asarray(self.data['exposure'])
		self.elv_arr = np.asarray(self.data['elevation'])
		self.fov_arr = np.asarray(self.data['fov'])

		self.data_len = len(self.data.index)

	def __getitem__(self, index):
		source_img_name = self.path_arr[index]
		source_img = io.imread(source_img_name)[:, :, :3]/255.0
		tensor_img = self.to_tensor(source_img)
		sun_pos = np.asarray([self.theta_arr[index], self.phi_arr[index]])
		sp_pdf = vMF(sun_pos) # target probability distribution of sun position
		pr_vec = np.asarray([self.tub_arr[index], self.omg_arr[index], float(self.elv_arr[index]), math.radians(float(self.fov_arr[index]))]) # target parameters
		label = {'img': self.normalize(tensor_img), 'dis': sp_pdf, 'prrs': pr_vec}
		return label

	def __len__(self):
		return self.data_len

class Eval_Dataset(Dataset):
	def __init__(self, csv_path):
		self.to_tensor = transforms.ToTensor()
		self.normalize = transforms.Normalize(mean=[0.48548178, 0.48455666, 0.46329196],
												std= [0.21904471, 0.21578524, 0.23359051])
		self.data = pd.read_csv(csv_path)
		self.path_arr = np.asarray(self.data['filepath'])

		''' sun position: theta, phi '''
		self.theta_arr = np.asarray(self.data['theta'])
		self.phi_arr = np.asarray(self.data['phi'])

		''' parameters: turbidity, omega(scaling factor, exposure), camera evelation and FoV '''
		self.tub_arr = np.asarray(self.data['turbidity'])
		self.omg_arr = np.asarray(self.data['exposure'])
		self.elv_arr = np.asarray(self.data['elevation'])
		self.fov_arr = np.asarray(self.data['fov'])

		self.data_len = len(self.data.index)

	def __getitem__(self, index):
		source_img_name = self.path_arr[index]
		source_img = io.imread(source_img_name)[:, :, :3]/255.0
		tensor_img = self.to_tensor(source_img)
		sun_pos = np.asarray([self.theta_arr[index], self.phi_arr[index]])
		sp_pdf = vMF(sun_pos) # target probability distribution of sun position
		pr_vec = np.asarray([self.tub_arr[index], self.omg_arr[index], float(self.elv_arr[index]), math.radians(float(self.fov_arr[index]))]) # target parameters
		label = {'img': self.normalize(tensor_img), 'dis': sp_pdf, 'prrs': pr_vec, 'sp': sun_pos}
		return label

	def __len__(self):
		return self.data_len

class Inference_Data(Dataset):
	def __init__(self, img_path):
		self.input_img = io.imread(img_path)
		self.to_tensor = transforms.ToTensor()
		self.normalize = transforms.Normalize(mean=[0.48548178, 0.48455666, 0.46329196],
												std= [0.21904471, 0.21578524, 0.23359051])
		self.data_len = 1

	def __getitem__(self, index):
		tensor_img = self.to_tensor(self.input_img)
		return self.normalize(tensor_img)

	def __len__(self):
		return self.data_len
