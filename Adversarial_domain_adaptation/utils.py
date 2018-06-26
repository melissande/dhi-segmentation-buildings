import os
import sys
sys.path.append(os.path.abspath('../'))
from Data_Handle.image_utils import standardize
from Data_Handle.dataset_generator import Dataset_sat
from Data_Handle.data_augmentation import *
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import params

def make_variable_lab_da(tensor):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)


def get_data_loader(root_path,dataset_type,*,shuffle_eval=False):
    path=os.path.join(root_path, dataset_type)
    if not os.path.exists(path):
        return print('%s dataset path not valid. This folder should contain INTPUT/ and OUTPUT/'%path)
    if dataset_type=='TRAINING/':
        transform=params.data_aug
        max_data_size=params.max_train_size
        batch_size=params.batch_size
        shuffle=True
    elif dataset_type=='VALIDATION/':
        transform=None
        max_data_size=params.max_val_size
        batch_size=params.val_batch_size
        shuffle=shuffle_eval
    elif dataset_type=='TEST/':
        transform=None
        max_data_size=params.max_val_size
        batch_size=params.val_batch_size
        shuffle=shuffle_eval
    else:
        return print('%s not valid dataset type'%dataset_type)
        
        
    generator=Dataset_sat.from_root_folder(path,params.input_channels,max_data_size=max_data_size,transform=transform)
    return DataLoader(generator, batch_size=batch_size,shuffle=shuffle, num_workers=1) 
def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    

    # restore model weights
    if restore=='':
        print('Model trained from scratch')
    elif restore!='' and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        print("Restore model from: {}".format(os.path.abspath(restore)))
    else:
        return print('%s path for restoring model is not valid'%restore)
        

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net
def save_model(net, filename):
    
    """Save trained model."""

    torch.save(net.state_dict(),os.path.join(params.global_path, filename))
    print("save pretrained model to: {}".format(os.path.join(params.global_path, filename+'CP{}.pth'.format(epoch))))

    
class Store_learning_adda(object):
    """
    Text files creations and update to store learning tracking: avg loss train at each epoch, verificaiton loss at each epoch for generative and discriminative network
    error rate verification set at each epoch for discriminative
    """
    def __init__(self, global_pred_path):

        self.prediction_path=global_pred_path+'STORE_LEARNING_ADDA/'
        if not os.path.exists(self.prediction_path):
            os.makedirs(self.prediction_path)

       
        
    def initialize(self,mode):
        #STORE loss Discriminative for ANALYSIS
        
        self.avg_loss_train_d=[]
        self.file_train_d = open(self.prediction_path+'avg_loss_train_d.txt',mode) 
        
        self.file_verif_d = open(self.prediction_path+'loss_verif_d.txt',mode) 
        #STORE loss Generative for ANALYSIS
        
        self.avg_loss_train_g=[]
        self.file_train_g = open(self.prediction_path+'avg_loss_train_g.txt',mode) 
        
        self.file_verif_g = open(self.prediction_path+'loss_verif_g.txt',mode) 
        
        #STORE ERROR RATE Discriminateive 

        self.error_rate_file_verif_d = open(self.prediction_path+'error_verif_d.txt',mode)
        

        
    def write_file(self,file,value):
        
        
        file.write(str(value)+'\n')
        file.flush()