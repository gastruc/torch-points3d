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

np.random.seed(22)

from torch_points3d.applications.pointnet import PointNet2

DIR = ""

#@title Choose ModelNet parameters {run: "auto"}
MODELNET_VERSION="40" #@param ["10", "40"]
USE_NORMAL = True

from torch_points3d.datasets.classification.modelnet import SampledModelNet
import torch_points3d.core.data_transform as T3D
import torch_geometric.transforms as T

from torch_points3d.datasets.batch import SimpleBatch

from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq


def get_list(tensor,k):
    l1,l2,l3=[],[],[]
    norme0 = [(tensor[0,i,0]**2+tensor[0,i,1]**2+tensor[0,i,2]**2,i) for i in range (len(tensor[0]))]
    norme0.sort()
    norme1 = [(tensor[1,i,0]**2+tensor[1,i,1]**2+tensor[1,i,2]**2,i) for i in range (len(tensor[0]))]
    norme1.sort()
    norme2 = [(tensor[2,i,0]**2+tensor[2,i,1]**2+tensor[2,i,2]**2,i) for i in range (len(tensor[2]))]
    norme2.sort()
    i=-1
    for j in range (k):
        u,v=norme0[i]
        l1.append(v)
        u,v=norme1[i]
        l2.append(v)
        u,v=norme2[i]
        l3.append(v)
        i-=1
    return (l1,l2,l3)

def get_normes(tensor,i,pos,l):
    l=[np.linalg.norm(tensor[pos,i,:]-tensor[pos,j,:]) for j in l]
    #l=[(tensor[pos,i,0]-tensor[pos,j,0])**2+(tensor[pos,i,1]-tensor[pos,j,1])**2+(tensor[pos,i,2]-tensor[pos,j,2])**2 for j in l]
    return(sum(l))

def get_list_upgraded(tensor,k,l1,l2,l3):
    iter_data_time = time.time()
    for i in range (k):
        norme0 = [(get_normes(tensor,i,0,l1),i) for i in range (len(tensor[0]))]
        norme1 = [(get_normes(tensor,i,1,l2),i) for i in range (len(tensor[0]))]
        norme2 = [(get_normes(tensor,i,2,l3),i) for i in range (len(tensor[0]))]
        u,v=max(norme0)
        l1.append(v)
        u,v=max(norme1)
        l2.append(v)
        u,v=max(norme2)
        l3.append(v)
        print("done",time.time() - iter_data_time)
    print("finish",time.time() - iter_data_time)
    return (l1,l2,l3)

def get_list_random(k,l):
    return (list(np.random.randint(l, size=k)),list(np.random.randint(l, size=k)),list(np.random.randint(l, size=k)))
    


def batch_to_batch(data,random,furthest,furthest_upgraded):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects. 
        """

    keys = ['x','y','pos','grid_size']

    batch = SimpleBatch()
    batch.__data_class__ = data.__class__
    
    l1,l2,l3=get_list(data['x'],furthest)
    l11,l22,l33=get_list_random(random,len(data['x'][0]))
    l1,l2,l3=l1+l11,l2+l22,l3+l33
    l1,l2,l3=get_list_upgraded(data['x'],furthest_upgraded,l1,l2,l3)

    for key in data.keys:
        if key in ['y','grid_size']:
            item = data[key]
            batch[key]=item
        else:
            item = data[key]
            batch[key]=torch.cat((torch.unsqueeze(item[0,l1,:],0),torch.unsqueeze(item[1,l2,:],0), torch.unsqueeze(item[2,l3,:],0)),axis=0)
            #batch[key]=item[:,:128,:]
    return batch.contiguous()




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
        data_out = self.encoder(data)
        self.output = self.log_softmax(data_out.x.squeeze())

        # Set loss for the backward pass
        self.loss_class = torch.nn.functional.nll_loss(self.output, self.labels)
    
    def backward(self):
         self.loss_class.backward()


def test_epoch1_128(device,random,furthest,furthest_upgraded):
    model_128.to(device)
    model_128.eval()
    tracker.reset("test")
    test_loader = dataset.test_dataloaders[0]
    iter_data_time = time.time()
    
    for i, data in enumerate(test_loader):
        if len(data['x'])==3:
            print(i,time.time() - iter_data_time)
            t_data = time.time() - iter_data_time
            data=batch_to_batch(data,random,furthest,furthest_upgraded)
            data.to(device)
            model_128.forward(data)
            tracker.track(model_128)
        
def test_epoch_128(device):
    model_128.to(device)
    model_128.eval()
    tracker.reset("test")
    test_loader = dataset.test_dataloaders[0]
    iter_data_time = time.time()
    
    for i, data in enumerate(test_loader):
        if len(data['x'])==3:
            t_data = time.time() - iter_data_time
            iter_start_time = time.time()
            data.to(device)
            model_128.forward(data)
            tracker.track(model_128)

def test_epoch1_256(device):
    model_256.to(device)
    model_256.eval()
    tracker.reset("test")
    test_loader = dataset.test_dataloaders[0]
    iter_data_time = time.time()
    
    for i, data in enumerate(test_loader):
        if len(data['x'])==3:
            t_data = time.time() - iter_data_time
            iter_start_time = time.time()
            data=batch_to_batch(data)
            data.to(device)
            model_256.forward(data)
            tracker.track(model_256)
        
def test_epoch_256(device):
    model_256.to(device)
    model_256.eval()
    tracker.reset("test")
    test_loader = dataset.test_dataloaders[0]
    iter_data_time = time.time()
    
    for i, data in enumerate(test_loader):
        t_data = time.time() - iter_data_time
        iter_start_time = time.time()
        data.to(device)
        model_256.forward(data)
        tracker.track(model_256)

model_128 = PointNet2CLassifier()
model_128.load_state_dict(torch.load("2021-04-26 10:28:01.360039/modele_"+str(128)+".pth"))

model_256 = PointNet2CLassifier()
model_256.load_state_dict(torch.load("2021-04-26 10:28:01.360039/2021-04-26 17:25:16.894236/modele_"+str(256)+".pth"))
for u in [128]:
    NUM_WORKERS = 4
    BATCH_SIZE = 3


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
            """ % (os.path.join(DIR, "data"),MODELNET_VERSION, 128,USE_NORMAL, u,USE_NORMAL)

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
                precompute_multi_scale=False
            )

    tracker = dataset.get_tracker(False, True)
    
    print("Modèle 128:")
    test_epoch_128('cuda')
    print(tracker.publish(0)['current_metrics']['acc'])
    print(tracker.publish(0)['current_metrics']['loss_class'])
    
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
                precompute_multi_scale=False
            )

    tracker = dataset.get_tracker(False, True)
    print("Modèle 128 + 128 aléatoires:")
    #test_epoch1_128('cuda',256,0,0)
    #print(tracker.publish(0)['current_metrics']['acc'])
    #print(tracker.publish(0)['current_metrics']['loss_class'])
    
    tracker = dataset.get_tracker(False, True)
    print("Modèle 128 + 128 plus loins:")
    #test_epoch1_128('cuda',128,128,0)
    #print(tracker.publish(0)['current_metrics']['acc'])
    #print(tracker.publish(0)['current_metrics']['loss_class'])
    
    tracker = dataset.get_tracker(False, True)
    print("Modèle 128 + 128 plus loins des autres un par un:")
    test_epoch1_128('cuda',128,0,128)
    print(tracker.publish(0)['current_metrics']['acc'])
    print(tracker.publish(0)['current_metrics']['loss_class'])
    
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
            """ % (os.path.join(DIR, "data"),MODELNET_VERSION, 256,USE_NORMAL, 256,USE_NORMAL)

    from omegaconf import OmegaConf
    params = OmegaConf.create(yaml_config)

                # Instantiate dataset
    from torch_points3d.datasets.classification.modelnet import ModelNetDataset
    dataset = ModelNetDataset(params)

                # Setup the data loaders
    dataset.create_dataloaders(
                model_256, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                num_workers=NUM_WORKERS, 
                precompute_multi_scale=False
            )

    tracker = dataset.get_tracker(False, True)
    print("Modèle 256:")
    test_epoch_256('cuda')
    print(tracker.publish(0)['current_metrics']['acc'])
    print(tracker.publish(0)['current_metrics']['loss_class'])
    
    
    
    
    
    

    
    
    
