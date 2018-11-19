# System
import argparse
# Math
import numpy as np
import math
# Visualize
import progressbar
# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Custom
from loader import IlluminationModule, Eval_Dataset
from libs.projections import bin2Sphere
from utils import getAngle

def evaluate(module):
	# loss function
	sun_crit = nn.KLDivLoss()
	prr_crit = nn.MSELoss()
	
	# data loader
	eval_dataset = Eval_Dataset(csv_path='data/test_list.csv')
	eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=4)
	
	module.eval()
	'''
	Totally five predictions: 
		1. sun position (angular error)
		2. tubidity ( abs difference)
		3. exposure ( abs difference)
		4. elevation (difference in degrees)
		5. FoV (difference in degrees)
	'''
	sum_err = np.asarray([0, 0, 0, 0, 0]).astype('float64')
	sum_loss = 0.0
	data_length = len(eval_dataloader)
	with progressbar.ProgressBar(max_value=data_length) as bar:
		for i, sample in enumerate(eval_dataloader):

			# processing sampled data
			input_img = sample['img'].cuda().float()
			label_dis, label_prrs = sample['dis'].cuda().float(), sample['prrs'].cuda().float()
			sunpos = sample['sp'][0].numpy()

			with torch.no_grad():
				pred_dis, pred_prrs = module(input_img)
				sun_loss, prr_loss = sun_crit(pred_dis, label_dis), prr_crit(pred_prrs, label_prrs)
				beta = 64
				loss = sun_loss + beta * prr_loss
				sum_loss += loss.item()
				
				# calculate all prediction error
				pred_sunpos = bin2Sphere(np.argmax(pred_dis.cpu().numpy()[0])) # predicted sun position is the bin with highest probability
				sunpos_err = getAngle(sunpos, pred_sunpos)
				tur_err = abs(pred_prrs.cpu().numpy()[0][0] - label_prrs.cpu().numpy()[0][0])
				omg_err = abs(pred_prrs.cpu().numpy()[0][1] - label_prrs.cpu().numpy()[0][1])
				elv_err = abs(pred_prrs.cpu().numpy()[0][2] - label_prrs.cpu().numpy()[0][2])
				fov_err = math.degrees(abs(pred_prrs.cpu().numpy()[0][3] - label_prrs.cpu().numpy()[0][3]))
				sum_err += np.asarray([sunpos_err, tur_err, omg_err, elv_err, fov_err])
			bar.update(i)
	# print average prediction errors across all testing dataset
	print('Testing avg loss:', sum_loss/data_length)
	final_err = sum_err/data_length
	print('Average predictions error:')
	print('sun position (angular error):', final_err[0])
	print('tubidity:', final_err[1])
	print('exposure:', final_err[2])
	print('elevation (degrees):', final_err[3])
	print('FoV (degrees):', final_err[4])

def main(args):
	print('loading weights ...')
	# device configuration
	torch.cuda.set_device(0)
	# get network module 
	module = IlluminationModule().cuda()
	#load pre-trained weight
	module.load_state_dict(torch.load(args.pre_trained))

	print('start evaluating ...')
	evaluate(module)
	print('evaluation done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
  
    parser.add_argument('--pre_trained', default='pre-trained/weights.pth', help='pre-trained weight path')

    args = parser.parse_args()
    main(args)