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

np.random.seed(22)

from torch_points3d.applications.pointnet2 import PointNet2

DIR = ""

#@title Choose ModelNet parameters {run: "auto"}
MODELNET_VERSION="40" #@param ["10", "40"]
USE_NORMAL = True

from torch_points3d.datasets.classification.modelnet import SampledModelNet
import torch_points3d.core.data_transform as T3D
import torch_geometric.transforms as T

dataroot = os.path.join(DIR, "data/modelnet")
pre_transform = T.Compose([T.NormalizeScale(), T3D.GridSampling3D(0.02)])
dataset = SampledModelNet(dataroot, name=MODELNET_VERSION, train=True, transform=None,
                 pre_transform=pre_transform, pre_filter=None)
dataset[0]

from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq

def train_epoch(device):
    model.to(device)
    model.train()
    tracker.reset("train")
    train_loader = dataset.train_dataloader
    iter_data_time = time.time()
    for i, data in enumerate(train_loader):
        t_data = time.time() - iter_data_time
        iter_start_time = time.time()
        optimizer.zero_grad()
        data.to(device)
        model.forward(data)
        model.backward()
        optimizer.step()
        if i % 10 == 0:
            tracker.track(model)
        
    """
    with Ctq(train_loader) as tq_train_loader:
        for i, data in enumerate(tq_train_loader):
            t_data = time.time() - iter_data_time
            iter_start_time = time.time()
            optimizer.zero_grad()
            data.to(device)
            model.forward(data)
            model.backward()
            optimizer.step()
            if i % 10 == 0:
                tracker.track(model)

            tq_train_loader.set_postfix(
                **tracker.get_metrics(),
                data_loading=float(t_data),
                iteration=float(time.time() - iter_start_time),
            )
            iter_data_time = time.time()
     """

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
    
    """
    with Ctq(test_loader) as tq_test_loader:
        for i, data in enumerate(tq_test_loader):
            t_data = time.time() - iter_data_time
            iter_start_time = time.time()
            data.to(device)
            model.forward(data)           
            tracker.track(model)

            tq_test_loader.set_postfix(
                **tracker.get_metrics(),
                data_loading=float(t_data),
                iteration=float(time.time() - iter_start_time),
            )
            iter_data_time = time.time()
"""

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
    for v in [512,1024,2048]:

        model = PointNet2CLassifier()

        NUM_WORKERS = 4
        BATCH_SIZE = 4


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

        EPOCHS = 150
        somme=0
        for i in range(EPOCHS):
            print("=========== EPOCH %i ===========" % i)
            time.sleep(0.5)
            train_epoch('cuda')
            test_epoch('cuda')
            if i>=100:
                somme+=tracker.publish(i)['current_metrics']['acc']
        print((tracker.publish(i)['current_metrics']['acc'],somme/50))
        l1.append((tracker.publish(i)['current_metrics']['acc'],somme/50))
    l.append(l1)

    
print(l)


