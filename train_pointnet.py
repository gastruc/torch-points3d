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

DIR = "" 

#@title Choose ModelNet parameters {run: "auto"}
MODELNET_VERSION="40" #@param ["10", "40"]
USE_NORMAL = True #@param {type:"boolean"}


from torch_points3d.datasets.classification.modelnet import SampledModelNet
import torch_points3d.core.data_transform as T3D
import torch_geometric.transforms as T

dataroot1 = os.path.join(DIR, "data/modelnet1")
dataroot2 = os.path.join(DIR, "data/modelnet2")
pre_transform = T.Compose([T.NormalizeScale(), T3D.GridSampling3D(0.02)])

NUM_WORKERS = 4
BATCH_SIZE = 16

from torch_points3d.applications.pointnet2 import PointNet2

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
            
def train_epoch(device,train_loader):
    model.to(device)
    model.train()
    tracker.reset("train")
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
            
def test_epoch(device,test_loader):
    model.to(device)
    model.eval()
    tracker.reset("test")
    iter_data_time = time.time()
    
    for i, data in enumerate(test_loader):
        t_data = time.time() - iter_data_time
        iter_start_time = time.time()
        data.to(device)
        model.forward(data)
        tracker.track(model)

l=[]

for u in [512,1024,2048]:
    l1=[]
    for v in [512,1024,2048]:
        model = PointNet2CLassifier()


        transform = T.FixedPoints(u)
        dataset = SampledModelNet(dataroot1, name=MODELNET_VERSION, train=True, transform=transform,
                         pre_transform=pre_transform, pre_filter=None)

        collate_function = lambda datalist: SimpleBatch.from_data_list(datalist)
        train_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS,  
            collate_fn=collate_function
        )
        for i, data in enumerate(train_loader):
            print(data)
            break

        transform = T.FixedPoints(v)
        dataset = SampledModelNet(dataroot2, name=MODELNET_VERSION, train=True, transform=transform,
                         pre_transform=pre_transform, pre_filter=None)

        collate_function = lambda datalist: SimpleBatch.from_data_list(datalist)
        test_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS,  
            collate_fn=collate_function
        )
        
        for i, data in enumerate(test_loader):
            print(data)
            break


        logdir = "" # Replace with your own path
        logdir = os.path.join(logdir, str(datetime.datetime.now()))
        os.mkdir(logdir)
        os.chdir(logdir)
        
        from torch_points3d.datasets.classification.modelnet import ModelNetDataset
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


        
        dataset = ModelNetDataset(params)
        tracker = dataset.get_tracker(False, True)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        EPOCHS = 100
        somme=0
        for i in range(EPOCHS):
            print("=========== EPOCH %i ===========" % i)
            time.sleep(0.5)
            train_epoch('cuda',train_loader)
            test_epoch('cuda',test_loader)
            if i>=80:
                somme+=tracker.publish(i)['current_metrics']['acc']
        print((tracker.publish(i)['current_metrics']['acc'],somme/20))
        l1.append((tracker.publish(i)['current_metrics']['acc'],somme/20))
    l.append(l1)

print(l)



