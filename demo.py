import os
import numpy as np
from types import SimpleNamespace
from utils import *
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from networks import *
import torch.optim as optim
import time


args = SimpleNamespace(batch_size=1000,
                       epochs=1500,

                       device="cuda" if torch.cuda.is_available() else "cpu",

                       ## network
                       IW=False,
                       n_stohastic_layers=2,
                       k=6,

                       ## adam optimizer hyperparameters
                       adam_beta_1=0.9,
                       adam_beta_2=0.99,
                       base_lr=1e-3,

                       ## MNIST dataset
                       dataset_path="C:/Users/johny/Desktop/AML_project_21/mnist_dataset/MNIST/processed/",
                       )


session_folder = ("IW{}_nlayers{}_k{}/").format(args.IW, args.n_stohastic_layers, args.k)
if not os.path.exists(session_folder):
    os.makedirs(session_folder)

# setting seeds
set_seeds(25)

## train dataloader
train_dataset = MNIST(dataset_path=args.dataset_path+"training.pt", device=args.device)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

## test dataloader
test_dataset = MNIST(dataset_path=args.dataset_path+"test.pt", device=args.device)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)# no point in shufflying during evaluation

## test labels dataloader
test_dataset_labels = MNIST_wlabels(dataset_path=args.dataset_path+"test.pt", device=args.device)
test_loader_labels = DataLoader(test_dataset_labels, batch_size=1, shuffle=False)# no point in shufflying during evaluation


if args.n_stohastic_layers == 1:
    model = VAE_1(latent_dim=50, hidden_dim=200, input_dim=784, IW=args.IW,  k=args.k, device=args.device)

elif args.n_stohastic_layers == 2:
    model = VAE_2(latent_dim_1=100, latent_dim_2=50, hidden_dim_1=200, hidden_dim_2=100, input_dim=784, IW=args.IW,  k=args.k, device=args.device)

else:
    print("Provide one of the available n_stohastic_layers [1 or 2]")

optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(args.adam_beta_1, args.adam_beta_2))

model.epochs = args.epochs
counter_passes = 0


for current_epoch in range(0, args.epochs):

    model.train_epoch(train_loader, optimizer)

    if (current_epoch+1) % 1000 == 0 or (current_epoch+1) == model.epochs:
        torch.save(model.state_dict(), session_folder+str(current_epoch+1)+".pth")

model.generate(10, "C:/Users/johny/Desktop/AML_project_21/"+session_folder+"generated/")
model.reconstruct(test_loader, 10, "C:/Users/johny/Desktop/AML_project_21/"+session_folder+"reconstruct/")

model.plot_latent(test_loader_labels, "C:/Users/johny/Desktop/AML_project_21/"+session_folder+"plot_latent/")

total_nll = model.test_nll(test_loader)

print("NNL", total_nll)
