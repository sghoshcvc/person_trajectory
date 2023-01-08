import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from model import VRNN
from dataloader import Trajectory
from util import to_var

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


def train(epoch):
	train_loss = 0
	for batch_idx, (data) in enumerate(train_loader):
		
		#transforming data
		#data = Variable(data)
		#to remove eventually
		# data = Variable(data.squeeze().transpose(0, 1))
		data = data.float()
		data = to_var(data.transpose(0, 1))
		data = (data - data.min().data) / (data.max().data - data.min().data)
		
		#forward + backward + optimize
		optimizer.zero_grad()
		kld_loss, nll_loss, _, _ = model(data)
		loss = kld_loss + nll_loss
		loss.backward()
		optimizer.step()

		#grad norm clipping, only in pytorch version >= 1.10
		nn.utils.clip_grad_norm(model.parameters(), clip)

		#printing
		if batch_idx % print_every == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
				epoch, batch_idx * data.shape[1], len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				kld_loss.item() / batch_size,
				nll_loss.item() / batch_size))

			sample = model.sample(1024)
			plt.imshow(sample.numpy())
			plt.pause(1e-6)

		train_loss += loss.item()


	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
	"""uses test data to evaluate 
	likelihood of the model"""
	
	mean_kld_loss, mean_nll_loss = 0, 0
	for i, (data) in enumerate(val_loader):                                            
		
		#data = Variable(data)
		data = data.float()
		data = to_var(data.squeeze().transpose(0, 1))
		data = (data - data.min().data) / (data.max().data - data.min().data)

		kld_loss, nll_loss, _, _ = model(data)
		mean_kld_loss += kld_loss.item()
		mean_nll_loss += nll_loss.item()

	mean_kld_loss /= len(val_loader.dataset)
	mean_nll_loss /= len(val_loader.dataset)

	print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
		mean_kld_loss, mean_nll_loss))


#hyperparameters
x_dim = 12
h_dim = 32
z_dim = 12
n_layers =  1
n_epochs = 100
clip = 10
learning_rate = 1e-3
batch_size = 16
ranseed = 128
print_every = 10
save_every = 10

#manual seed
torch.manual_seed(ranseed)
# plt.ion()

trajectory_Obj = Trajectory("/home/suman/person_trajectory/varRNN/VariationalRNN/VariationalRecurrentNeuralNetwork-master/data/thor")

#init model + optimizer + datasets
train_loader = torch.utils.data.DataLoader(trajectory_Obj, batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(trajectory_Obj, batch_size=batch_size, shuffle=False)
model = VRNN(x_dim, h_dim, z_dim, n_layers)
if torch.cuda.is_available():
       model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, n_epochs + 1):
	
	#training + testing
	train(epoch)
	test(epoch)

	#saving model
	if epoch % save_every == 1:
		fn = '/home/suman/person_trajectory/varRNN/VariationalRNN/VariationalRecurrentNeuralNetwork-master/saves/vrnn_state_dict_'+str(epoch)+'.pth'
		torch.save(model.state_dict(), fn)
		print('Saved model to '+fn)