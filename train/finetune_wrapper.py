import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models
import matplotlib.pyplot as plt
import time
import os
import sys
import threading
import resnet
import transforms

# for interactive purpose
plt.ion()

model_sel = {
	'inception': torchvision.models.inception.inception_v3(pretrained=True),
	'resnet152': resnet.resnet152(pretrained=True),
	'resnet101': torchvision.models.resnet.resnet101(pretrained=True),
}

args = {
	# arch params
	'data_dir': '/home/ubuntu/jd_ai/data3',
	'test_data_dir': '/home/ubuntu/jd_ai/data/test',
	'model_name': 'resnet152',
	'input_size': 400,
	'output_size': 30,
	'batch_size': 16,
	'requires_grad': True,
	'use_gpu': torch.cuda.is_available(),

	# train params
	'epoch': 100,

	'start_lr': 0.0025,
	'momentum': 0.9,

	'gamma': 0.1,
	'lr_decay_step_size': 30,

	'loss_fun': nn.CrossEntropyLoss(),
}

data_trans = {
    'train': transforms.Compose([
    	transforms.RandomRotation(degrees=20),
    	transforms.RandomResizedCrop(args['input_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
    	transforms.Resize(args['input_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def data_load(data_dir=args['data_dir']):
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
		data_trans[x]) for x in ['train', 'val']}

	dataloader_1s_1 = {x: torch.utils.data.DataLoader(
		image_datasets[x], args['batch_size'], shuffle=True, num_workers=4) for x in ['train']}
	
	dataset_sizes_1 = {x: len(image_datasets[x]) for x in ['train']}

	dataloader_1s_2 = {x: torch.utils.data.DataLoader(
		image_datasets[x], 1, shuffle=False, num_workers=4) for x in ['val']}
	
	dataset_sizes_2 = {x: len(image_datasets[x]) for x in ['val']}

	return dataloader_1s_1, dataset_sizes_1, dataloader_1s_2, dataset_sizes_2

def model_tools(args=args):
	model_conv = model_sel[args['model_name']]
	for param in model_conv.parameters():
	    param.requires_grad = args['requires_grad']

	num_ftrs = model_conv.fc.in_features

	model_conv.fc = nn.Linear(num_ftrs, args['output_size'])

	if (args['use_gpu']):
		model_conv = model_conv.cuda()

	optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=args['start_lr'], momentum=args['momentum'])

	lr_lambda = lambda epoch: (1-epoch/args['epoch'])**0.9
        
	poly_lr_scheduler = lr_scheduler.LambdaLR(optimizer_conv, lr_lambda = lr_lambda)

        
	return (model_conv, optimizer_conv, poly_lr_scheduler)

def poly_lr(epoch):
        return (1-(epoch/args['epoch']))^0.9

def train(model, optimizer, scheduler, dataloader_1, data_sizes_1, dataloader_2, data_sizes_2):
	since = time.time()

	loss_func = args['loss_fun']

	best_model = model.state_dict()
	best_acc = 0.0

	for epoch in range(args['epoch']):
		print('Epoch {}/{}'.format(epoch, args['epoch'] - 1))
		print('-' * 10)
		time_epoch = time.time()

		scheduler.step()
                
		model.train(True)

		running_loss = 0.0
		running_acc = 0

		for data in dataloader_1['train']:

			inputs, labels = data

			if args['use_gpu']:
				inputs = Variable(inputs.cuda())
				labels = Variable(labels.cuda())
			else:
				inputs = Variable(inputs)
				labels = Variable(labels)

			optimizer.zero_grad()
			outputs = model(inputs)



			if(args['model_name'] == 'inception'): # adjust inception auxiliary output
				_, preds = torch.max(outputs[0].data, 1)
				loss = sum((loss_func(out,labels) for out in outputs))
			else:
				_, preds = torch.max(outputs.data, 1)
				loss = loss_func(outputs, labels)

			loss.backward()
			optimizer.step()

			running_loss += loss.data[0]
			running_acc += torch.sum(preds == labels.data)

		epoch_loss = running_loss / data_sizes_1['train']
		epoch_acc = running_acc / data_sizes_1['train']

		print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

		print('finished this epoch in ',time.time() - time_epoch,'seconds')
		print()

		this_acc = model_eval(model, dataloader_2, data_sizes_2)

		if (this_acc > best_acc):
			best_acc = this_acc
			best_model = model.state_dict()

		print('Validation Acc: ',this_acc)

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	model.load_state_dict(best_model)
	return model

def model_eval(model, dataloader_2, data_sizes_2):
	model.train(False)

	running_acc = 0.0

	for data in dataloader_2['val']:
		inputs, labels = data

		if (args['use_gpu']):
			inputs = Variable(inputs.cuda())
			labels = Variable(labels.cuda())
		else:
			inputs = Variable(inputs)
			labels = Variable(labels)

		outputs = model(inputs)

		_, preds = torch.max(outputs.data, 1)

		running_acc += torch.sum(preds == labels.data)

	acc = running_acc / data_sizes_2['val']

	return acc

def model_test(model, data_dir=args['test_data_dir']):
	model.train(False)

	image_datasets = {x: datasets.ImageFolder(os.path.join(args['test_data_dir'], x), 
		data_trans[x]) for x in ['val']}
	
	dataloader_1s_1 = {x: torch.utils.data.dataloader_1(
		image_datasets[x], args['batch_size'], shuffle=True, num_workers=4) for x in ['val']}

	for data in dataloader_1['val']:
		inputs, labels = data

		if (args['use_gpu']):
			inputs = Variable(inputs.cuda())
			labels = Variable(labels.cuda())
		else:
			inputs = Variable(inputs)
			labels = Variable(labels)

		outputs = model(inputs)
		_, preds = torch.max(outputs.data, 1)

	return preds

if __name__ == '__main__':
	dataloader_1, data_sizes_1, dataloader_2, data_sizes_2 = data_load()
	model, optimizer, scheduler = model_tools()
	model = train(model, optimizer, scheduler, dataloader_1, data_sizes_1, dataloader_2, data_sizes_2)
	torch.save(model, 'best_152_v3.pt')
