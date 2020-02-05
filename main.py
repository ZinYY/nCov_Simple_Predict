import torch
import torch.nn as nn
from torch.autograd import Variable
from easyModel import SNet
import torch.optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from dataReader import reader
import os

if __name__ == '__main__':
	read = reader()
	read.get()
	
	x = read.days
	y = read.num
	datanum = len(x)
	x = torch.tensor(x, dtype=torch.float32)
	x = x.reshape((datanum, 1))
	y = torch.tensor(y, dtype=torch.float32)
	y = y.reshape((datanum, 1))
	
	# y = y - 3000.0
	y = y / 1000.0
	
	print(y)
	# Dataloader
	dum_dataset = TensorDataset(x, y)
	dum_dataloader = DataLoader(dataset=dum_dataset,
								batch_size=datanum,
								shuffle=False,
								num_workers=0)
	
	snet = SNet()
	# 产生优化器
	optimizer = torch.optim.SGD(snet.parameters(), lr=0.0001, momentum=0.9, weight_decay=0)
	
	epoch_num = 100000
	print_freq = 1000
	
	# 开始训练
	best_loss = 100000000
	best_state = None
	# 如果存在存档，读取
	if os.path.exists('snet.pkl'):
		snet.load_state_dict(torch.load('snet.pkl'))
	# snet.topnum.data = snet.topnum.data * 10.0
	
	for epoch in range(epoch_num):
		for i, step_data in enumerate(dum_dataloader):
			# 从dataloader中读取该batch的数据
			inputs, labels = step_data
			inputs, labels = Variable(inputs), Variable(labels)
			
			# 运算网络输出out,清零梯度，单步训练
			optimizer.zero_grad()
			out = snet(inputs)
			
			criterion = nn.MSELoss()
			loss = criterion(out, labels)
			
			loss.backward()
			optimizer.step()
		# print(i)
		if epoch % print_freq == 0:
			loss = loss.data.float().item()
			print('epoch: {}\t\t\t\tloss: {:.6f}'.format(epoch, loss))
			
			if best_loss > loss:
				best_loss = loss
				best_state = snet.state_dict()
		
		if epoch % (print_freq * 20) == 0:
			print('----------------------------------')
			print('best loss till now:', best_loss)
			print(best_state)
			print('----------------------------------')
			torch.save(snet.state_dict(), 'snet.pkl')
	
	print('----------------------------------')
	print('best loss till now:', best_loss)
	print(best_state)
	print('----------------------------------')
	torch.save(snet.state_dict(), 'snet.pkl')
