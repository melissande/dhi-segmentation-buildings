import numpy as np
from numpy import newaxis
import torch
from random import randint
from numpy.random import rand
import numpy as np
from skimage.transform import rescale

def crop_center(img,cropx,cropy):
    """
       Crops the image img to a size of cropx x cropy but from the center of the image
    """

    y = img.shape[0]
    x = img.shape[1]
    
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    

    return img[starty:starty+cropy, startx:startx+cropx]

class Transform(object):
    """
        Potentially transforms the batch with T1=rot +90 degrees then T2= Flip up then T3= Flip right then T4=upscale
    """
    def  __call__(self,sample):
        proba=randint(0,1)
        X, Y = sample['input'], sample['groundtruth']

        #Rot +90
        if proba:
            X=np.rot90(X,axes=(0,1))
            Y=np.rot90(Y,axes=(0,1))
         
        proba=randint(0,1)
        #Flip up/down
        if proba:
            X=np.flip(X,0)
            Y=np.flip(Y,0)
            
        proba=randint(0,1)
        #Flip right/left
        if proba:
            X=np.flip(X,1)
            Y=np.flip(Y,1)
            
        proba=randint(0,1)
        #Scale up
        ratio=1.0+rand()*0.2
  
        if proba:
     
            w=X.shape[1]
            h=X.shape[0]
            X=rescale(X,ratio)   
            Y=rescale(Y,ratio)
            X=crop_center(X,w,h)
            Y=crop_center(Y,w,h)
            
            
        X,Y=(X-np.amin(X))/(np.amax(X)-np.amin(X)),(Y-np.amin(Y))/(np.amax(Y)-np.amin(Y))    
        Y=Y.astype(int)

 
        return {'input': X, 'groundtruth': Y}
        
    

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        X, Y = sample['input'], sample['groundtruth']
        return {'input': torch.from_numpy(X),
                'groundtruth': torch.from_numpy(Y)}