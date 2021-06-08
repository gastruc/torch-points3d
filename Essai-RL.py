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
import copy

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
        #print(data.x.shape)
        data_out = self.encoder(data)
        self.output = self.log_softmax(data_out.x.squeeze())

        # Set loss for the backward pass
        self.loss_class = torch.nn.functional.nll_loss(self.output, self.labels)
        
    def veri(self, inp,indice):
        x = inp.to(device)
        data_out = self.encoder(x)
        inp.x = inp.x.transpose(1, 2)
        return(torch.argmax(self.log_softmax(data_out.x.squeeze())[indice])==data.y.squeeze()[indice])
        
        
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
    
    
Transition = namedtuple('Transition',
                        ('general','state', 'action','samp','points' ,'indice','next_state', 'reward'))


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
    
    
BATCH_SIZE = 3
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

n_actions = 2

policy_net = DQN(128).to(device)
#policy_net.load_state_dict(torch.load("policy_net.pth"))
target_net = DQN(128).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(200)


steps_done = 0


def select_action(state,indice,p):
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
            samp=torch.tensor([[random.random(),random.random(),random.random()]], device=device)
            #return policy_net(state,indice,samp).max(1)[1].view(1, 1),samp
            return torch.tensor([[torch.argmax(policy_net(state,indice,samp))]], device=device, dtype=torch.long),samp
    else:
        if random.random()>p:
            return torch.tensor([[1]], device=device, dtype=torch.long),torch.tensor([[random.random(),random.random(),random.random()]], device=device)
        else:
            return torch.tensor([[0]], device=device, dtype=torch.long),torch.tensor([[random.random(),random.random(),random.random()]], device=device)


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
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = [s for s in batch.next_state if s is not None]
    non_final=[i for i in range(len(batch.next_state)) if batch.next_state[i] is not None]
    #state_batch = list_to_batch(batch.state)
    action_batch = torch.cat(batch.action)
    samp_batch = torch.cat(batch.samp)
    reward_batch = torch.cat(batch.reward)
    indice_batch=list(batch.indice)
    #print(len(batch.points),batch.points[0][-5:],batch.points[1][-5:],batch.points[2][-5:])
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    #print(policy_net(batch.state[0],batch.indice[0],samp_batch[0]))
    #print(torch.cat([policy_net(batch.state[i],batch.indice[i],samp_batch[i]) for i in range (len(batch.state))]).shape,action_batch.shape)
    state_action_values = (torch.cat([policy_net(batch.state[i],batch.indice[i],samp_batch[i]) for i in range (len(batch.state))])).gather(0, torch.squeeze(action_batch))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #print(torch.cat([model_128(non_final_next_states[i])[batch.indice[non_final[i]]] for i in range (len(non_final_next_states))]))
    inter=torch.tensor([parcours(batch.general[non_final[i]],non_final_next_states[i],copy.deepcopy(batch.points[non_final[i]]),indice_batch[non_final[i]]) for i in range (len(non_final_next_states))], device=device)
    #inter=torch.cat([model_128.sortie(non_final_next_states[i]).x[indice_batch[non_final[i]]] for i in range (len(non_final_next_states))])
    next_state_values[non_final_mask]=inter
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def parcours(data,state,points,j):
    n_actions=64
    action=0
    etapes=0
    while action==0 and len(points)<128:
        action_max=0
        maxi=-100000
        samp_max=torch.tensor([[random.random(),random.random(),random.random()]], device=device)
        for i in range (n_actions):
            with torch.no_grad():
                samp=torch.tensor([[random.random(),random.random(),random.random()]], device=device)
                result=target_net(state,j,samp)
                if max(result)>maxi:
                    action_max=torch.argmax(result)
                    samp_max=samp
                    maxi=max(result)
        action,samp=action_max,samp_max
        etapes+=1
        if action==0:
            state,points=find_neighbor(data,state,samp,points,j)
    if model_128.veri(state,j):
        return(2*(GAMMA**etapes)-0.01*(1-GAMMA**etapes)/(1-GAMMA))
    else:
        return(-1*(GAMMA**etapes)-0.01*(1-GAMMA**etapes)/(1-GAMMA))
    
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
    if action==0:
    #if False:
        next_state,points=find_neighbor(general,state,samp,points,indice)
        return(next_state,points,-0.01,False)
    elif action==1:
    #elif True:
        if model_128.veri(state,indice):
            return(state,points,2,True)
        else:
            return(state,points,-1,True)
    else:
        print("Problème",action)

def get_min(general,samp,indice,points):
    ind=[i for i in range(len(general.x[0])) if not(i in points)]
    bidule=np.array(general.x.cpu()[indice,ind,:])
    samp2=np.array(samp.cpu())
    l=[np.linalg.norm(bidule[j,:]-samp2) for j in range(len(bidule))]
    #l=[(tensor[pos,i,0]-tensor[pos,j,0])**2+(tensor[pos,i,1]-tensor[pos,j,1])**2+(tensor[pos,i,2]-tensor[pos,j,2])**2 for j in l]
    return(ind[np.argmin(l)])       
        
def find_neighbor(general,state,samp,points,indice):
    u=get_min(general,samp,indice,points)
    points.append(u)
    return(batch_to_batch3(general,points),points)
    
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
            batch[key]=item[[0,1]]
        else:
            item = data[key]
            batch[key]=torch.cat((torch.unsqueeze(item[0,l,:],0),torch.unsqueeze(item[1,l,:],0)),axis=0)
            #batch[key]=item[:,:128,:]
    return batch.contiguous()


    
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
            batch[key]=item[[0,1]]
        else:
            item = data[key]
            batch[key]=torch.cat((torch.unsqueeze(item[0,l1,:],0),torch.unsqueeze(item[1,l1,:],0)),axis=0)
            #batch[key]=item[:,:128,:]
    return batch.contiguous(),l1

def list_to_batch(data):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects. 
        """

    keys = ['x','y','pos','grid_size']

    batch = SimpleBatch()
    batch.__data_class__ = data.__class__
    

    for key in data[0].keys:
        if key in ['y','grid_size']:
            item=torch.cat([data[i][key] for i in range(len(data))])
            batch[key]=item
        else:
            item=torch.cat([data[i][key] for i in range(len(data))])
            batch[key]=item
    return batch.contiguous()

proba=0.95
train_loader = dataset.train_dataloader
DEPART=64
num_episodes = 2
TARGET_UPDATE = 1
timer=time.time()
for i_episode in range(num_episodes):
    # Initialize the environment and state
    if i_episode%10==0:
        print("épisode",i_episode)
    for i, data in enumerate(train_loader):
        indice=random.randint(0,1)
        data.to(device)
        state,points=batch_to_batch2(data,DEPART)
        print("chg")
        for t in count():
            # Select and perform an action
            action,samp = select_action(state,indice,proba)
            next_state,points, reward,done= step(data,state,samp,action,points,indice)
            reward = torch.tensor([reward], device=device)
            # Store the transition in memory
            if done:
                memory.push(data, state, action, samp, copy.deepcopy(points), indice, None, reward)
            else:
                memory.push(data, state, action, samp, copy.deepcopy(points), indice, next_state, reward)                


            # Move to the next state
            state = next_state
            # Perform one step of the optimization (on the policy network)
            optimize_model()
            if done or next_state.x.shape[1]>128:
                episode_durations.append(t + 1)
                break
                
    target_net.load_state_dict(policy_net.state_dict())

torch.save(policy_net.state_dict(), "policy_net.pth")
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
    
    
    

    
    
    
