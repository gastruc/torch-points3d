# Needed for remote rendering 
import os
os.environ["DISPLAY"] = ":99.0"
os.environ["PYVISTA_OFF_SCREEN"]="true"
os.environ["PYVISTA_PLOT_THEME"]="true"
os.environ["PYVISTA_USE_PANEL"]="true"
os.environ["PYVISTA_AUTO_CLOSE"]="false"
os.system("Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &")

import os
import sys
from omegaconf import OmegaConf
import pyvista as pv
import torch
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple, deque
from itertools import count
import math


np.random.seed(22)

from torch_points3d.applications.RL import PointNet2

DIR = ""

#@title Choose ModelNet parameters {run: "auto"}
MODELNET_VERSION="40" #@param ["10", "40"]
USE_NORMAL = True
device="cuda"

from torch_points3d.datasets.classification.modelnet import SampledModelNet
import torch_points3d.core.data_transform as T3D
import torch_geometric.transforms as T

from torch_points3d.datasets.batch import SimpleBatch

from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq


class PointNet2CLassifier(torch.nn.Module):
    def __init__(self):
        super().__init__() 
        self.encoder = PointNet2("encoder", input_nc= 3 * USE_NORMAL,output_nc = int(MODELNET_VERSION), num_layers=3,kwargs="multiscale")
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
    
    @property
    def conv_type(self):
        """ This is needed by the dataset to infer which batch collate should be used"""
        return self.encoder.conv_type
    
    def get_output(self):
        """ This is needed by the tracker to get access to the ouputs of the network"""
        return self.output
    
    def get_labels(self):
        """ Needed by the tracker in order to access ground truth labels"""
        return self.labels
    
    def get_current_losses(self):
        """ Entry point for the tracker to grab the loss """
        return {"loss_class": float(self.loss_class)}
    
    def forward(self, data):
        # Set labels for the tracker
        self.labels = data.y.squeeze()

        # Forward through the network
        #print(data.x.shape)
        data_out = self.encoder(data)
        self.output = self.log_softmax(data_out.x.squeeze())

        # Set loss for the backward pass
        self.loss_class = torch.nn.functional.nll_loss(self.output, self.labels)
        
    def veri(self, inp,indice):
        x = inp.to(device)
        data_out = self.encoder(x)
        inp.x = inp.x.transpose(1, 2)
        #return(torch.argmax(self.log_softmax(data_out.x.squeeze())[indice])==inp.y.squeeze()[indice])
        result=self.log_softmax(data_out.x.squeeze())[indice]
        return(torch.argmax(result)==inp.y.squeeze()[indice],result[inp.y.squeeze()[indice]])
        
        
    def extract(self, data):
        return(self.encoder(data,True))
    
    def sortie(self, data):
        inp=self.encoder(data)
        data.x = data.x.transpose(1, 2)
        return(inp)
    
    def backward(self):
         self.loss_class.backward()


            
class DQN(nn.Module):

    def __init__(self, h):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(1,1,kernel_size=32,stride=3)
        #self.bn1 = nn.BatchNorm1d(161)

        self.head1 = nn.Linear(164, 256)
        self.head2 = nn.Linear(256, 2)
        

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, inp,indice,y):
        inp = inp.to(device)
        tr=model_128.extract(inp)
        inp.x = inp.x.transpose(1, 2)
        print(tr)
        print(tr['x'].shape,indice)
        print(tr['x'][[indice]].shape)
        tr=torch.unsqueeze(tr['x'][[indice]],1)
        tr=torch.squeeze(tr,3)
        tr=self.conv1(tr)
        tr=torch.squeeze(tr,1)
        #x=torch.unsqueeze(x,1)
        tr = F.relu(tr)
        tr=torch.squeeze(tr)
        tr=torch.cat((tr,torch.squeeze(y)),0)
        tr=F.relu(self.head1(tr))
        return self.head2(tr)
    
def get_list_random(k,l):
    return (list(np.random.randint(l, size=k)),list(np.random.randint(l, size=k)),list(np.random.randint(l, size=k)))

model_128 = PointNet2CLassifier()
model_128.load_state_dict(torch.load("2021-04-26 10:28:01.360039/modele_"+str(128)+".pth"))
model_128.to(device)
model_128.eval()   
    
NUM_WORKERS = 4
BATCH_SIZE = 3
policy_net = DQN(128).to(device)
policy_net.load_state_dict(torch.load("policy_net2.pth"))
policy_net.to(device)
policy_net.eval() 

optimizer = torch.optim.RMSprop(policy_net.parameters())

yaml_config = """
            task: classification
            class: modelnet.ModelNetDataset
            name: modelnet
            dataroot: %s
            number: %s
            pre_transforms:
                - transform: NormalizeScale
                - transform: GridSampling3D
                  lparams: [0.02]
            train_transforms:
                - transform: FixedPoints
                  lparams: [%d]
                - transform: RandomNoise
                - transform: RandomRotate
                  params:
                    degrees: 180
                    axis: 2
                - transform: AddFeatsByKeys
                  params:
                    feat_names: [norm]
                    list_add_to_x: [%r]
                    delete_feats: [True]
            test_transforms:
                - transform: FixedPoints
                  lparams: [%d]
                - transform: AddFeatsByKeys
                  params:
                    feat_names: [norm]
                    list_add_to_x: [%r]
                    delete_feats: [True]
            """ % (os.path.join(DIR, "data"),MODELNET_VERSION, 512,USE_NORMAL, 512,USE_NORMAL)

from omegaconf import OmegaConf
params = OmegaConf.create(yaml_config)

                # Instantiate dataset
from torch_points3d.datasets.classification.modelnet import ModelNetDataset
dataset = ModelNetDataset(params)

                # Setup the data loaders
dataset.create_dataloaders(
                model_128, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                num_workers=NUM_WORKERS, 
                precompute_multi_scale=False)

test_loader = dataset.test_dataloaders[0]
tracker = dataset.get_tracker(False, True)

def batch_to_batch(data,random,j):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects. 
        """

    keys = ['x','y','pos','grid_size']

    batch = SimpleBatch()
    batch.__data_class__ = data.__class__
    
    l1,l2,l3=get_list_random(random,len(data['x'][0]))

    for key in data.keys:
        if key in ['y','grid_size']:
            item = data[key]
            batch[key]=item[[0,1]]
        else:
            item = data[key]
            batch[key]=torch.cat((torch.unsqueeze(item[0,l1,:],0),torch.unsqueeze(item[j,l1,:],0)),axis=0)
            #batch[key]=item[:,:128,:]
    return batch.contiguous(),l1


def test_epoch_128(device,random):
    model_128.to(device)
    model_128.eval()
    tracker.reset("test")
    test_loader = dataset.test_dataloaders[0]
    iter_data_time = time.time()
    booles,confs=[],[]
    
    for i, data in enumerate(test_loader):
        for j in range (len(data['x'])):
            t_data = time.time() - iter_data_time
            data.to(device)
            state,points=batch_to_batch(data,random,j)
            state.to(device)
            boole,conf=parcours(data,state,points,1)
            print(boole,conf)
            booles.append(boole)
            confs.append(conf)
            
    print("Accuracy:",sum(booles)/len(booles))
    print("Loss:",sum(conf)/len(conf))


def parcours(data,state,points,j):
    n_actions=64
    action=0
    while action==0 and len(points)<128:
        l=[]
        for i in range (n_actions):
            with torch.no_grad():
                samp=torch.tensor([[random.random(),random.random(),random.random()]], device=device)
                result=policy_net(state,j,samp)
                print(torch.max(result),torch.argmax(result))
                l.append((max(result),torch.argmax(result),i,samp))
        try:
            _,action,_,samp=max(l)
            print("action",action)
        except:
            print("error",l)
        if action==0:
            state,points=find_neighbor(data,state,samp,points,1)
        
    return(model_128.veri(state,1))

def get_min(general,samp,indice):
    bidule=np.array(general.x.cpu()[indice])
    samp2=np.array(samp.cpu())
    l=[np.linalg.norm(bidule[j,:]-samp2) for j in range(len(bidule))]
    #l=[(tensor[pos,i,0]-tensor[pos,j,0])**2+(tensor[pos,i,1]-tensor[pos,j,1])**2+(tensor[pos,i,2]-tensor[pos,j,2])**2 for j in l]
    return(np.argmin(l))       
        
def find_neighbor(general,state,samp,points,indice):
    u=get_min(general,samp,indice)
    points.append(u)
    return(batch_to_batch3(general,points,indice),points)
    
def batch_to_batch3(data,l,j):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects. 
        """

    keys = ['x','y','pos','grid_size']

    batch = SimpleBatch()
    batch.__data_class__ = data.__class__

    for key in data.keys:
        if key in ['y','grid_size']:
            item = data[key]
            batch[key]=item[[0,1]]
        else:
            item = data[key]
            batch[key]=torch.cat((torch.unsqueeze(item[0,l,:],0),torch.unsqueeze(item[j,l,:],0)),axis=0)
            #batch[key]=item[:,:128,:]
    return batch.contiguous()
        
test_epoch_128('cuda',64)










