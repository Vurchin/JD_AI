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
import io
from PIL import Image
from subprocess import call


plt.ion()

model = torch.load('/home/ubuntu/jd_ai/models/best_50.pt')

data_dir = '/home/ubuntu/jd_ai/data/test'

preprocess_1 = transforms.Compose([
    			transforms.Resize(400),
        		transforms.ToTensor(),
        		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

preprocess_2 = transforms.Compose([
    			transforms.Resize(500),
        		transforms.ToTensor(),
        		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

counter = 0

for file in os.listdir(data_dir + '/val/1'):
	print(counter)
	counter = counter + 1
	im = Image.open(data_dir + '/val/1/' + file)
	
	im_tensor_1 = preprocess_1(im)
	im_input_1 = Variable(im_tensor_1.cuda())
	output_1 = model(im_input_1.unsqueeze(0))

	im_tensor_2 = preprocess_2(im)
	im_input_2 = Variable(im_tensor_2.cuda())
	output_2 = model(im_input_2.unsqueeze(0))

	softmax = nn.Softmax()

	out1 = softmax(output_1)
	out2 = softmax(output_2)

	output = (out1 + out2) / 2

	value, idx = output[0].max(0)

	output = softmax(output)

	for i in range(30):
		output.data[0][i] = 0.5/29

	output.data[0][idx.data] = 0.5

	for i in range(30):
		result = open('result_v3.csv', 'a')
		name = str(file)[:-4]
		res = name + ',' + str(i+1) + ',' + str(format(output.data[0][i], '.10f')) + '\n'
		result.write(res)
		result.close()

	im.close()

















	
