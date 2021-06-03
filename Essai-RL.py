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
        
    def veri(self, data,indice):
        data = data.to(device)
        print("data.y",data['y'].squeeze()[indice])
        data_out = self.encoder(data)
        print(self.log_softmax(data_out.x.squeeze()))
        print(torch.argmax(self.log_softmax(data_out.x.squeeze())[indice]))
        print(self.log_softmax(self.encoder(data).x.squeeze())==data.y.squeeze())
        return(self.log_softmax(self.encoder(data).x.squeeze())==data.y.squeeze())
        
        
    def extract(self, data):
        return(self.encoder(data,True))
    
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
    def forward(self, x,indice,y):
        x = x.to(device)
        data=model_128.extract(x)
        x=torch.unsqueeze(data['x'][[indice]],1)
        x=torch.squeeze(x,3)
        x=self.conv1(x)
        x=torch.squeeze(x,1)
        #x=torch.unsqueeze(x,1)
        x = F.relu(x)
        x=torch.squeeze(x)
        x=torch.cat((x,torch.squeeze(y)),0)
        x=F.relu(self.head1(x))
        return self.head2(x)
    
    
Transition = namedtuple('Transition',
                        ('state', 'action','samp', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
BATCH_SIZE = 4
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

n_actions = 2

policy_net = DQN(128).to(device)

optimizer = torch.optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(200)


steps_done = 0


def select_action(state,indice):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
    #if True:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            samp=torch.tensor([[random.random(),random.random(),random.random()]], device=device, dtype=torch.long)
            #return policy_net(state,indice,samp).max(1)[1].view(1, 1),samp
            return torch.argmax(policy_net(state,indice,samp)),samp
    else:
        return torch.tensor([[random.randrange(int(n_actions))]], device=device, dtype=torch.long),torch.tensor([[random.random(),random.random(),random.random()]], device=device, dtype=torch.long)


episode_durations = []
    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    samp_batch = torch.cat(batch.samp)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch,samp_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

NUM_WORKERS = 4     
model_128 = PointNet2CLassifier()
model_128.load_state_dict(torch.load("2021-04-26 10:28:01.360039/modele_"+str(128)+".pth"))
model_128.to(device)
model_128.eval()

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

def step(general,state,samp,action,points,indice):
    #if action==0:
    if False:
        state,points=find_neighbor(general,state,samp,points,indice)
        return(state,points,-0.01,False)
    #elif action==1:
    elif True:
        if model_128.veri(state,indice):
            return(state,points,2,True)
        else:
            return(state,points,-1,True)
    else:
        print("Problème",action)

def get_min(general,samp,indice):
    general2=general.to("cpu")
    l=[np.linalg.norm(general2.x[indice,j,:]-samp) for j in range(len(general2.x[indice]))]
    #l=[(tensor[pos,i,0]-tensor[pos,j,0])**2+(tensor[pos,i,1]-tensor[pos,j,1])**2+(tensor[pos,i,2]-tensor[pos,j,2])**2 for j in l]
    return(np.argmin(l))       
        
def find_neighbor(general,state,samp,points,indice):
    u=get_min(general,samp,indice)
    points.append(u)
    return(batch_to_batch3(general),points)
    
def batch_to_batch3(data,l):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects. 
        """

    keys = ['x','y','pos','grid_size']

    batch = SimpleBatch()
    batch.__data_class__ = data.__class__

    for key in data.keys:
        if key in ['y','grid_size']:
            item = data[key]
            batch[key]=item
        else:
            item = data[key]
            batch[key]=torch.cat((torch.unsqueeze(item[0,l,:],0),torch.unsqueeze(item[1,l,:],0)),axis=0)
            #batch[key]=item[:,:128,:]
    return batch.contiguous(),l1


    
def batch_to_batch2(data,random):
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
            batch[key]=item
        else:
            item = data[key]
            batch[key]=torch.cat((torch.unsqueeze(item[0,l1,:],0),torch.unsqueeze(item[1,l1,:],0)),axis=0)
            #batch[key]=item[:,:128,:]
    return batch.contiguous(),l1




train_loader = dataset.train_dataloader
DEPART=64
num_episodes = 20
for i_episode in range(num_episodes):
    # Initialize the environment and state
    if i_episode%1==0:
        print(i_episode)
    for i, data in enumerate(train_loader):
        indice=random.randint(0,1)
        data.to(device)
        state,points=batch_to_batch2(data,DEPART)
        for t in count():
            # Select and perform an action
            action,samp = select_action(state,indice)
            print(t,state.x.shape)
            next_state,points, reward,done= step(data,state,samp,action,points,indice)
            reward = torch.tensor([reward], device=device)

            # Store the transition in memory
            memory.push(state, action, samp,next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break


print('Complete')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
""" 
    
    

def test_epoch1_128(device,random,furthest,furthest_upgraded):
    model_128.to(device)
    model_128.eval()
    tracker.reset("test")
    test_loader = dataset.test_dataloaders[0]
    iter_data_time = time.time()
    
    for i, data in enumerate(test_loader):
        if len(data['x'])==3:
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


    
"""
    
    
    

    
    
    
