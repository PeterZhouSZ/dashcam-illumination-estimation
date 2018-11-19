# Math
import numpy as np
# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Custom
from loader import IlluminationModule, Train_Dataset

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def trainloop(module, lr=0.01, train_epochs=50):
	# set training dataset
	train_dataset = Train_Dataset(csv_path='data/train_list.csv')
	train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)

	# using KL divergence loss for sun distribution and MSE loss  
	sun_crit = nn.KLDivLoss()
	prr_crit = nn.MSELoss()
	# set optimizer
	optimizer = torch.optim.Adam(module.parameters(), lr=lr)

	# train the model
	cur_lr = lr
	for i in range(train_epochs):
		for i_batch, sample in enumerate(train_dataloader):
			# training input and targets
			img = sample['img'].cuda().float()
			label_dis, label_prrs = sample['dis'].cuda().float(), sample['prrs'].cuda().float()

			# forward pass 
			pred_dis, pred_prrs = module(img)
			beta = 0.1 # to compensate for the number od bins in output distribution
			sun_loss, prr_loss = sun_crit(pred_dis, label_dis), prr_crit(pred_prrs, label_prrs)
			loss = sun_loss + beta * prr_loss
			
			# optimization
			print('epoch:', i+1, 'steps:', i_batch+1, 'loss:', loss.item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		# Decay learning rate (0.5/15 epochs)
		if i % 15 == 0:
			cur_lr *= 0.5
			update_lr(optimizer, cur_lr)
		

	# Save the model checkpoint
	torch.save(module.state_dict(), 'weights.pth')

def main():

	# device configuration
	torch.cuda.set_device(0)
	# get network module 
	illuminationModule = IlluminationModule().cuda()

	trainloop(illuminationModule)

if __name__ == '__main__':
	main()