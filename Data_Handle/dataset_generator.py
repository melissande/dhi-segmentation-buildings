import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py


PATH_INPUT='INPUT/'
PATH_OUTPUT='OUTPUT/'
NB_CLASSES=2

def _parse_image(path_input,path_output,nb_classes):
    '''
    Reads the paths of input and output and returns corresponding numpy arrays X,Y. The output is processed to be one hot encoded for background (channel 0) and building footprints (channel 1). Paths should contain .h5 files
    :paths_input path of the input tensor that has to be read
    :paths_output path of the groundtruth mask of building footprints over background
    returns input and output image as array
    '''
    
    with h5py.File(path_input, 'r') as hf:
            X =np.array(hf.get('data'))
    with h5py.File(path_output, 'r') as hf:
            Y_build=np.array(hf.get('data'))
            Y_build=(Y_build>0).astype(int)
            Y_other= (1-Y_build).astype(int)
            Y=np.stack((Y_other,Y_build),axis=2)
            
    return X,Y


class Dataset_sat(Dataset):
    """Satellite images dataset with rastered footprints in groundtruth."""

    def __init__(self,paths_input: np.ndarray,paths_output: np.ndarray,nb_classes: int,transform=None):
        """
        Args:
            paths_input: paths of the patch images  in input
            paths_output: paths of the patch groundtruth  in output
            nb_classes: number of classes in the rasterized version of groundtruth
            transform: for data augmentation
            #paths_input and paths_output should a be lists ordered in the same way to get correspondence between patches
            #each path contained in the list is a .h5 file
            
        """
        self.paths_input = paths_input
        self.paths_output = paths_output
        self.nb_classes=nb_classes
        self.transform = transform

    @classmethod
    def from_root_folder(cls, root_folder: str, nb_classes: int,*,transform=None, max_data_size:  int = None):
        paths_input = []
        paths_output=[]
        
        
        for filename in sorted(os.listdir(root_folder+PATH_INPUT))[:max_data_size]:
            paths_input.append(os.path.join(root_folder+PATH_INPUT, filename))

        for filename in sorted(os.listdir(root_folder+PATH_OUTPUT))[:max_data_size]:

            paths_output.append(os.path.join(root_folder+PATH_OUTPUT, filename))
        

        return Dataset_sat(np.asarray(paths_input), np.asarray(paths_output),nb_classes,transform)

    def __len__(self):
        return len(self.paths_input)
    


    def __getitem__(self, idx):
        
        X,Y=_parse_image(self.paths_input[idx],self.paths_output[idx],self.nb_classes)
        sample = {'input': X, 'groundtruth': Y}

 
        if self.transform:
            sample = self.transform(sample)
  

        return sample


