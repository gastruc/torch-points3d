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


def batch_to_batch(data):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects. 
        """

    keys = ['x','y','pos','grid_size']

    batch = SimpleBatch()
    batch.__data_class__ = data.__class__

    for key in keys:
        batch[key] = []

    for key in data.keys:
        if key in ['y','gris_size']:
            item = data[key]
            batch[key].append(item)
        else:
            item = data[key]
            item=[item[i][:128] for i in range (len(item))]
            batch[key].append(torch.FloatTensor(item))

    for key in batch.keys:
        item = batch[key][0]
        if (
                torch.is_tensor(item)
                or isinstance(item, int)
                or isinstance(item, float)
            ):
            batch[key] = torch.stack(batch[key])
        else:
            raise ValueError("Unsupported attribute type")

    return batch.contiguous()

def train_epoch(device):
    model.to(device)
    model.train()
    tracker.reset("train")
    train_loader = dataset.train_dataloader
    iter_data_time = time.time()
        
    for i, data in enumerate(train_loader):
        t_data = time.time() - iter_data_time
        iter_start_time = time.time()
        print(data['x'].shape)
        print(type(data['x']))
        data2=batch_to_batch(data)
        
        optimizer.zero_grad()
        data2.to(device)
        model.forward(data2)
        model.backward()
        optimizer.step()
        if i % 10 == 0:
            tracker.track(model)

def test_epoch(device):
    model.to(device)
    model.eval()
    tracker.reset("test")
    test_loader = dataset.test_dataloaders[0]
    iter_data_time = time.time()
    
    for i, data in enumerate(test_loader):
        t_data = time.time() - iter_data_time
        iter_start_time = time.time()
        data.to(device)
        model.forward(data)
        tracker.track(model)


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
           

        
        
l=[]

for u in [512,1024,2048]:
    l1=[]
    model = PointNet2CLassifier()
    
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
            """ % (os.path.join(DIR, "data"),MODELNET_VERSION, u,USE_NORMAL, u,USE_NORMAL)

    from omegaconf import OmegaConf
    params = OmegaConf.create(yaml_config)

                # Instantiate dataset
    from torch_points3d.datasets.classification.modelnet import ModelNetDataset
    dataset = ModelNetDataset(params)

                # Setup the data loaders
    dataset.create_dataloaders(
                model, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                num_workers=NUM_WORKERS, 
                precompute_multi_scale=False
            )

            # Setup the tracker and actiavte tensorboard loging
    logdir = "" # Replace with your own path
    logdir = os.path.join(logdir, str(datetime.datetime.now()))
    os.mkdir(logdir)
    os.chdir(logdir)
    tracker = dataset.get_tracker(False, True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    EPOCHS = 100
    for i in range(EPOCHS):
        print("=========== EPOCH %i ===========" % i)
        time.sleep(0.5)
        train_epoch('cuda')

    for v in [128,256,512]:
        if not((u,v) in []):
            print(u,v)

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
            """ % (os.path.join(DIR, "data"),MODELNET_VERSION, u,USE_NORMAL, v,USE_NORMAL)

            from omegaconf import OmegaConf
            params = OmegaConf.create(yaml_config)

                # Instantiate dataset
            from torch_points3d.datasets.classification.modelnet import ModelNetDataset
            dataset = ModelNetDataset(params)

                # Setup the data loaders
            dataset.create_dataloaders(
                model, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                num_workers=NUM_WORKERS, 
                precompute_multi_scale=False
            )

            # Setup the tracker and actiavte tensorboard loging
            logdir = "" # Replace with your own path
            logdir = os.path.join(logdir, str(datetime.datetime.now()))
            os.mkdir(logdir)
            os.chdir(logdir)
            tracker = dataset.get_tracker(False, True)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            test_epoch('cuda')
            print(u,v)
            print(tracker.publish(0)['current_metrics']['acc'])
            l1.append(tracker.publish(0)['current_metrics']['acc'])
    l.append(l1)

    
print(l)
sys.stdout.flush()
