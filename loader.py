from os import listdir
from os.path import join
import cv2
import os
import h5py
from sklearn.cluster import KMeans as KM


import numpy as np
import torch.nn as nn
import torch as torch
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
from skimage.segmentation import find_boundaries

def default_loader(filepath):
    return Image.open(filepath).convert('RGB')

def gray_loader(filepath):
    return Image.open(filepath).convert('L')

#https://jaidevd.com/posts/weighted-loss-functions-for-instance-segmentation/
#UNET paper 
def make_weight_map(masks):
    w0 = 10
    sigma = 10
    """
    Generate the weight maps as specified in the UNet paper
    for a set of binary masks.
    
    Parameters
    ----------
    masks: array-like
        A 3D array of shape (n_masks, image_height, image_width),
        where each slice of the matrix along the 0th axis represents
	one binary mask.

    Returns
    -------
    array-like
        A 2D array of shape (image_height, image_width)
    
    """
    nrows, ncols = masks.shape[1:]
    masks = (masks > 0).astype(int)
    distMap = np.zeros((nrows * ncols, masks.shape[0]))
    X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
    X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
    for i, mask in enumerate(masks):
        # find the boundary of each mask,
        # compute the distance of each pixel from this boundary
        bounds = find_boundaries(mask, mode='inner')
        X2, Y2 = np.nonzero(bounds)
        xSum = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
        ySum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
        distMap[:, i] = np.sqrt(xSum + ySum).min(axis=0)
    ix = np.arange(distMap.shape[0])
    if distMap.shape[1] == 1:
        d1 = distMap.ravel()
        border_loss_map = w0 * np.exp((-1 * (d1) ** 2) / (2 * (sigma ** 2)))
    else:
        if distMap.shape[1] == 2:
            d1_ix, d2_ix = np.argpartition(distMap, 1, axis=1)[:, :2].T
        else:
            d1_ix, d2_ix = np.argpartition(distMap, 2, axis=1)[:, :2].T
        d1 = distMap[ix, d1_ix]
        d2 = distMap[ix, d2_ix]
        border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))
    xBLoss = np.zeros((nrows, ncols))
    xBLoss[X1, Y1] = border_loss_map
    # class weight map
    loss = np.zeros((nrows, ncols))
    w_1 = 1 - masks.sum() / loss.size
    w_0 = 1 - w_1
    loss[masks.sum(0) == 1] = w_1
    loss[masks.sum(0) == 0] = w_0
    ZZ = xBLoss + loss
    return ZZ


    
class Reader(data.Dataset):
    def __init__(self, image_list, labels_list=[], edges_list=[], transform=None, target_transform=None, use_cache=True, loader=default_loader,targ_loader = None):
        
        self.images = image_list
        self.loader = loader
        self.targ_loader = targ_loader
        
        
        if len(labels_list) is not 0:
            assert len(image_list) == len(labels_list)
            self.labels = labels_list
            self.edges = edges_list
        else:
            self.labels = False
            self.edges = False

        self.transform = transform
        self.target_transform = target_transform

        self.cache = {}
        self.use_cache = use_cache

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx not in self.cache:           
            img = self.loader(self.images[idx])
            if self.labels:
                if (self.targ_loader == None):
                    target = Image.open(self.labels[idx])
                    edge = Image.open(self.edges[idx])
                else:
                    target = self.loader(self.labels[idx])
                    edge = self.loader(self.edges[idx])
            else:
                target = None
                edge = None
        else:
            img,target,edge = self.cache[idx]
            
        if self.use_cache:
            self.cache[idx] = (img, target, edge)

        seed = np.random.randint(2147483647)
        np.random.seed(seed)
        displacement_val = np.random.randn(2, 5, 5) * 10
        disp = torch.tensor(displacement_val)
        
        #NOT WORKING random.seed(seed)
        torch.manual_seed(seed)
        if self.transform is not None:
            img = self.transform(img)
            #img = etorch.deform_grid(img,disp,order=1)
            
        #NOT WORKING random.seed(seed)
        torch.manual_seed(seed)
        if self.labels:
            if self.target_transform is not None:
                target = self.target_transform(target)
                #target = etorch.deform_grid(target,disp,order=1)
                
                
        #NOT WORKING random.seed(seed)
        torch.manual_seed(seed)
        if self.edges:
            if self.target_transform is not None:
                edge = self.target_transform(edge)
                #edge = etorch.deform_grid(edge,disp,order=1)
            
        return np.array(img), np.array(target), np.array(edge)
    
#### Training dataset import 
class hela:
    def __init__(self, file_dir,test=False):
        self.basepath = file_dir
        self.rgb = sorted([join(self.basepath, f) for f in 
                           listdir(self.basepath) if f.startswith('t')])
        self.labels = sorted([join(self.basepath, f) for f in 
                              listdir(self.basepath) if f.startswith('man')])
                 
        self.edges = self.add_borders()
        
        if test == False:
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.RandomResizedCrop(448,scale=(0.7,1.)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485],
                                      std=[0.229])])

            self.transform_target = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.RandomResizedCrop(448,scale=(0.7,1.)),
                 ToLogits()])
        else:
            self.transform = transforms.Compose(
                [transforms.RandomResizedCrop(448,scale=(1.,1.)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485],
                                      std=[0.229])])

            self.transform_target = transforms.Compose(
                [transforms.RandomResizedCrop(448,scale=(1.,1.)),
                 ToLogits()])
        
        # Check the names are paired correctly
        assert np.array([img[-8:] == lbl[-8:] for img, lbl in zip(self.rgb, self.labels)]).all() == True
        
        if 0 == len(self.rgb):
            print("No HeLa dataset found in:" + self.basepath)
            exit(-1)
        
    def add_edges(self):
        for l in self.labels:
            # check if edges emissing 
            if not os.path.exists(self.basepath+ '/' +l[-8:-5]+"_edge.png"):
                num_array = np.array(Image.open(l)).astype(np.float32)
                nap = np.zeros_like(num_array)
                dIdx = abs(num_array[:-1,:]-num_array[1: ,:])>0
                dIdy = abs(num_array[:,:-1]-num_array[: ,1:])>0
                nap[1:]+=dIdx
                nap[:,1:]+=dIdy
                nap = np.clip(nap,0,1)
                nap = cv2.dilate(nap,np.ones((10,10))).astype('uint8')
                pil = Image.fromarray(nap)
                pil.save(self.basepath+ '/' +l[-8:-5]+"_edge.png","PNG")

        return sorted([join(self.basepath, f) for f in 
                       listdir(self.basepath) if f.endswith('_edge.png')])
    
    def add_borders(self):
        for l in self.labels:
            # check if edges emissing 
            if not os.path.exists(self.basepath+ '/' +l[-8:-5]+"_edge.png"):
                num_array = np.array(Image.open(l)).astype(np.float32)
                
                n_mask = np.zeros([np.unique(num_array).shape[0]-1,num_array.shape[0],num_array.shape[1]])
                for obj in range(1,len(np.unique(num_array))):
                    img_ones = np.ones_like(num_array)*obj
                    n_mask[obj-1,:,:] = np.equal(num_array,img_ones).astype('int')
                
                nap = make_weight_map(n_mask).astype('uint8')

                pil = Image.fromarray(nap).convert('L')
                pil.save(self.basepath+ '/' +l[-8:-5]+"_edge.png","PNG")

        return sorted([join(self.basepath, f) for f in 
                       listdir(self.basepath) if f.endswith('_edge.png')])
    
class cvppp:
    def __init__(self, file_dir, test = False):
        self.basepath = file_dir
        self.rgb = sorted([join(self.basepath, f) for f in 
                           listdir(self.basepath) if f.endswith('_rgb.png')])
        self.labels = sorted([join(self.basepath, f) for f in 
                              listdir(self.basepath) if f.endswith('_label.png')])
                 
        self.edges = self.add_edges()
        
        if test:
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.RandomResizedCrop(448,scale=(0.7,1.)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

            self.transform_target = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.RandomResizedCrop(448,scale=(0.7,1.)),
                 ToLogits()])
        else:
            self.transform = transforms.Compose(
                [transforms.RandomResizedCrop(448,scale=(1.,1.)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

            self.transform_target = transforms.Compose(
                [transforms.RandomResizedCrop(448,scale=(1.,1.)),
                 ToLogits()])
        
        # Check the names are paired correctly
        assert np.array([img[:-7] == lbl[:-9] for img, lbl in 
                         zip(self.rgb, self.labels)]).all() == True
        
        if 0 == len(self.rgb):
            print("No cvppp dataset found in:" + self.basepath)
            exit(-1)
        
    def add_edges(self):
        for l in self.labels:
            # check if edges emissing 
            if not os.path.exists(l[:-9]+"edge.png"):
                num_array = np.array(Image.open(l)).astype(np.float32)
                nap = np.zeros_like(num_array)
                dIdx = abs(num_array[:-1,:]-num_array[1: ,:])>0
                dIdy = abs(num_array[:,:-1]-num_array[: ,1:])>0
                nap[1:]+=dIdx
                nap[:,1:]+=dIdy
                nap = np.clip(nap,0,1)
                nap = cv2.dilate(nap,np.ones((5,5))).astype('uint8')
                pil = Image.fromarray(nap)
                pil.save(l[:-9]+"edge.png","PNG")

        return sorted([join(self.basepath, f) for f in 
                       listdir(self.basepath) if f.endswith('_edge.png')])

class ToLogits(object):
    def __init__(self,expand_dim=None):
        self.expand_dim = expand_dim

    def __call__(self, pic):
        if pic.mode == 'I':
            img = torch.from_numpy(numpy.array(pic, numpy.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(numpy.array(pic, numpy.int32, copy=True))
        elif pic.mode == 'F':
            img = torch.from_numpy(numpy.array(pic, numpy.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(numpy.array(pic, numpy.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if self.expand_dim is not None:
            return img.unsqueeze(self.expand_dim)
        return img

class H5Reader(data.Dataset):
    def __init__(self, image_list, labels_list=[], transform=None, target_transform=None, use_cache=True, loader=default_loader):
        
        self.images = image_list
        self.loader = loader
        
        if len(labels_list) is not 0:
            assert len(image_list) == len(labels_list)
            self.labels = labels_list
        else:
            self.labels = False

        self.transform = transform
        self.target_transform = target_transform

        self.cache = {}
        self.use_cache = use_cache

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx not in self.cache:           
            img = self.loader(self.images[idx])
            if self.labels:
                target = Image.open(self.labels[idx])
            else:
                target = None
        else:
            img,target = self.cache[idx]
            
        if self.use_cache:
            self.cache[idx] = (img, target)

        seed = np.random.randint(2147483647)
        
        #NOT WORKING random.seed(seed)
        torch.manual_seed(seed)
        if self.transform is not None:
            img = self.transform(img)

        #NOT WORKING random.seed(seed)
        torch.manual_seed(seed)
        if self.labels:
            if self.target_transform is not None:
                target = self.target_transform(target)
            
        return np.array(img), np.array(target)
    
def guide_function(alpha,beta,phase,x_dim,y_dim):
    alpha = torch.FloatTensor(alpha).unsqueeze(-1).unsqueeze(-1)
    beta = torch.FloatTensor(beta).unsqueeze(-1).unsqueeze(-1)
    phase= torch.FloatTensor(phase).unsqueeze(-1).unsqueeze(-1)

    xx_channel = torch.linspace(0.,1.,x_dim).repeat(1, y_dim, 1).float()
    yy_channel = torch.linspace(0.,1.,y_dim).repeat(1, x_dim, 1).transpose(1, 2).float()

    xx_channel = xx_channel * alpha
    yy_channel = yy_channel * beta
    return torch.sin(xx_channel+ yy_channel+phase)

def accuracy(inLabel,gtLabel,method):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: score: Dice score
# From CVPPP challenge

    score = 0 # initialize output
    
    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return score
    
    maxInLabel = np.max(inLabel) # maximum label value in inLabel
    minInLabel = np.min(inLabel) # minimum label value in inLabel
    maxGtLabel = np.max(gtLabel) # maximum label value in gtLabel
    minGtLabel = np.min(gtLabel) # minimum label value in gtLabel
    
    if(maxInLabel==minInLabel): # trivial solution
        return score
    
    for i in range(minInLabel+1,maxInLabel+1): # loop all labels of inLabel, but background
        sMax = 0; # maximum Dice value found for label i so far
        for j in range(minGtLabel+1,maxGtLabel+1): # loop all labels of gtLabel, but background
            s = method(inLabel, gtLabel, i, j) # compare labelled regions
            # keep max Dice value for label i
            if(sMax < s):
                sMax = s
        score = score + sMax; # sum up best found values
    score = score/(maxInLabel-minInLabel)
    return score

def Dice(inLabel, gtLabel, i, j):
# calculate Dice score for the given labels i and j
    
    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return 0

    one = np.ones(inLabel.shape)
    inMask = (inLabel==i*one) # find region of label i in inLabel
    gtMask = (gtLabel==j*one) # find region of label j in gtLabel
    inSize = np.sum(inMask*one) # cardinality of set i in inLabel
    gtSize = np.sum(gtMask*one) # cardinality of set j in gtLabel
    overlap= np.sum(inMask*gtMask*one) # cardinality of overlap of the two regions
    if ((inSize + gtSize)>1e-8):
        out = 2*overlap/(inSize + gtSize) # Dice score
    else:
        out = 0

    return out    

def IOU(inLabel, gtLabel, i, j):
    
    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return 0

    one = np.ones(inLabel.shape)
    inMask = (inLabel==i*one) # find region of label i in inLabel
    gtMask = (gtLabel==j*one) # find region of label j in gtLabel
    inSize = np.sum(inMask*one) # cardinality of set i in inLabel
    gtSize = np.sum(gtMask*one) # cardinality of set j in gtLabel
    overlap= np.sum(inMask*gtMask*one) # cardinality of overlap of the two regions
    if ((inSize + gtSize)>1e-8):
        out = overlap/(inSize + gtSize - overlap) # iou
    else:
        out = 0

    return out    

def DiC(inp,targ):
    return abs(np.unique(inp).shape[0] - np.unique(targ).shape[0])

def km_cluster(x,fg, n_clusters, omega):
    
    x_dim = 448
    y_dim = 448
    
    xx_channel = torch.linspace(-1,1.,x_dim).repeat(
                        1, y_dim, 1).float()*fg
    xx = xx_channel.reshape(xx_channel.shape[0],-1).T
    
    yy_channel = torch.linspace(-1,1.,y_dim).repeat(
                        1, x_dim, 1).transpose(1, 2).float()*fg
    yy = yy_channel.reshape(yy_channel.shape[0],-1).T
    
    
    feats = (x*fg).reshape(3,-1).T
    
    xt = np.hstack([omega*feats, xx.numpy(),yy.numpy()])
    
    return KM(n_clusters=n_clusters, random_state=0).fit_predict(xt).reshape(448,448)