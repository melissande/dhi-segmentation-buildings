import sys
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import logging


import torch
import torch.nn as nn
from torch import optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader

import time
from random import randint



from IOU_computations import *
from Data_Handle.dataset_generator import Dataset_sat
from predict_and_evaluate import *
from Data_Handle.data_augmentation import *


###### PATH TO STORE MODEL ############
GLOBAL_PATH='MODEL_TEST_GHANA/'


if not os.path.exists(GLOBAL_PATH):
            os.makedirs(GLOBAL_PATH)
######################################



###################################

INPUT_CHANNELS=9 #9 channels for panchromatic + 8 pansharpened. If not set to 9, plotting of patches will mess up.
                # so only works for INPUT_CHANNELS=9 anyway.

NB_CLASSES=2 #Building and Background. Only works for NB_CLASSES=2 anyway, otherwise this network doesn't work.
SIZE_PATCH=128# patches of size 128x128. Needs to be equal to the size of the patches of the dataset.
############## 
MODEL_PATH_SAVE=GLOBAL_PATH+'RESUNET_test_'
MODEL_PATH_RESTORE='' #Path of Model to restore ex: 'MODELS/RUBV3D2_final_model_ghana.pth'
TEST_SAVE=GLOBAL_PATH+'TEST_SAVE/' #to store some patches initial and final epoch of validation set + models + performance curves
if not os.path.exists(TEST_SAVE):
            os.makedirs(TEST_SAVE)
        
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

##############

DROPOUT=0.35
DEFAULT_BATCH_SIZE =8 #has to be set to 8 for Ghana dataset and can be set up to 32 for Spacenet dataset
DEFAULT_EPOCHS =2
DEFAULT_VALID=32  # Batch size for validation set. 
                #Knowing that around 1200 elements in ghana validation and 15000 in spacenet validation

DISPLAY_STEP=100 #how often (in terms of iterations) is displayed measures during an epoch
IOU_STEP=15 # how often is computed IOU measures over validations et

###############
DEFAULT_LAYERS=3 #number of layers of the UNET (not considering bottom layer) = number of downsmapling stages
DEFAULT_FEATURES_ROOT=32 # number of filters in the first layer of the Unet
DEFAULT_BN=True # Batch normalization layers included

#####

DEFAULT_FILTER_WIDTH=3 #convolution kernel size. ex, here: 3x3
DEFAULT_LR=1e-3#1e-3for spacenet and ghana
DEFAULT_N_RESBLOCKS=1 #can add residual blocks inside each stage. Make the network heavier. Not advised.

###Tune Learning rate
REDUCE_LR_STEPS = [1,5, 50, 100,200] #reduce everytime one of these epochs is reached

################


DISTANCE_NET='v2' #can be set to none if no distance module wants to be used
BINS=10
THRESHOLD=20

if DISTANCE_NET is None:
    DISTANCE_NET_UNET=False # has to be set to False if no distance module is used, otherwise error.
else:
    DISTANCE_NET_UNET=True


##### Data augmentation set for training ###
DATA_AUG=None
# DATA_AUG=transforms.Compose([Transform(),ToTensor()])

####### TMP folder for IOU ###
## not to worry about, compulsory for vectorizing masks ""



#######  Data: where the dataset is stored ###
# root_folder ='../SPACENET_DATA/SPACENET_DATA_PROCESSED/DATASET/128_x_128_8_bands_pansh/'
root_folder = '../2_DATA_GHANA/DATASET/128_x_128_8_pansh/'

#type of loss used 
LOSS_FN='cross-entropy'# or 'jaccard_approx'


class Trainer(object):
    """
    Trains a unet instance
    
    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param lr: learning rate
    :nb_classes: always set to 2 ->background and building
    :type of loss: 'cross-entropy' or 'jaccard_approx-approx'
    """
    def __init__(self, net, batch_size=32, lr=0.001,nb_classes=2,loss_fn=LOSS_FN):
        self.net = net
        self.batch_size = batch_size
        self.lr = lr
        self.nb_classes=nb_classes
        self.loss_fn=loss_fn
        
        
        
            
    def _initialize(self, prediction_path,store_learning,iou_step,dist_net,threshold,bins):
        
        self.optimizer = optim.Adam(self.net.parameters(),lr=self.lr)
        self.prediction_path = prediction_path
        self.store_learning=store_learning
        self.IOU_STEP=iou_step
        self.threshold=threshold
        self.bins=bins
        self.dist_net=dist_net
        
    def train(self, data_provider_path,store_learning, save_path='', restore_path='',  epochs=3, dropout=0.2, display_step=100, validation_batch_size=30, prediction_path = '',dist_net=None,threshold=20,bins=15,iou_step=1,reduce_lr_steps=[1,10,100,200],data_aug=None):
        """
        Lauches the training process
        
        :param data_provider_path: where the DATASET folder is
        :param store_learning: to store the metrics during the training as .txt file
        :param save_path: path where to store checkpoints
        :param restore_path: path where is the model to restore is stored
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param validation_batch_size: batch size of the validation set
        :param prediction_path: where to store output of training (patches, losses .txt file, models)
        :param dist_net: distance module or not 
        :param threshold: threshold of distance module
        :param bins: number of bins for distance module
        :iou_step: how often is computed Iou measures over the validation set
        :reduce_lr_steps: epoch at which the learning rate is halved
        :data_aug: 'yes' or 'no' if the training set is augmented
        """
        
        ##SET UP PATHS FOR TRAINING ##
        #check they exist?
        PATH_TRAINING=data_provider_path+'TRAINING/'
        if not os.path.exists(PATH_TRAINING):
            print('Training dataset path not valid. Should be path_to_dataset/TRAINING/ and this folder should contain INTPUT/ and OUTPUT/')
            raise
        PATH_VALIDATION=data_provider_path+'VALIDATION/'
        if not os.path.exists(PATH_VALIDATION):
            print('Validation dataset path not valid. Should be path_to_dataset/VALIDATION/ and this folder should contain INTPUT/ and OUTPUT/')
            raise
        PATH_TEST=data_provider_path+'TEST/'
        if not os.path.exists(PATH_TEST):
            print('Test dataset path not valid. Should be path_to_dataset/TEST/ and this folder should contain INTPUT/ and OUTPUT/')
            raise
        
        TMP_IOU=prediction_path+'TMP_IOU/'
        if not os.path.exists(TMP_IOU):
                    os.makedirs(TMP_IOU)
        
 
        loss_train=[]

        if epochs == 0:
            print('Epoch set 0, model won\'t be trained')
            raise 
        if save_path=='':
            print('Specify a path where to store the Model')
            raise
        
        if prediction_path=='':
            print('Specify where to stored visualization of training')
            raise
            
        if restore_path=='':
            store_learning.initialize('w')
            store_learning
            print('Model trained from scratch')
        else:
            store_learning.initialize('a')
            self.net.load_state_dict(torch.load(restore_path))
            print('Model loaded from {}'.format(restore_path))
            
        self._initialize(prediction_path,store_learning,iou_step,dist_net,threshold,bins)
        
    
            
        ###Validation loader

        val_generator=Dataset_sat.from_root_folder(PATH_VALIDATION,self.nb_classes)
        val_loader = DataLoader(val_generator, batch_size=validation_batch_size,shuffle=False, num_workers=1)
        RBD=randint(0,int(val_loader.__len__())-1)
        self.info_validation(val_loader,-1,RBD,"_init",TMP_IOU)

        ###Training loader

        train_generator=Dataset_sat.from_root_folder(PATH_TRAINING,self.nb_classes,transform=data_aug)#max_data_size=4958 
        
        
        logging.info("Start optimization")

        counter=0
        
        for epoch in range(epochs):
            
            ##tune learning reate
            if epoch in reduce_lr_steps:
                self.lr = self.lr * 0.5
                self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            
            total_loss = 0
            error_tot=0   
            train_loader = DataLoader(train_generator, batch_size=self.batch_size,shuffle=True, num_workers=1)
            for i_batch,sample_batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                predict_net=Train_or_Predict(sample_batch,self.dist_net,self.loss_fn,self.threshold,self.bins,self.net)
                loss,_,probs_seg=predict_net.forward_pass()
 
                loss,self.optimizer,self.net=predict_net.backward_prog(loss,self.optimizer)
                
                total_loss+=loss.data[0]
                loss_train.append(loss.data[0])

                
                counter+=1
                
                if i_batch % display_step == 0:
                    self.output_training_stats(i_batch,loss,predict_net.batch_y,probs_seg)
                    
                
            
            avg_loss_train_value=total_loss/train_loader.__len__()
            (self.store_learning).avg_loss_train.append(avg_loss_train_value)
            (self.store_learning).write_file((self.store_learning).file_train,avg_loss_train_value)
            logging.info(" Training {:}, Minibatch Loss= {:.4f}".format("epoch_%s"%epoch,avg_loss_train_value))
            self.info_validation(val_loader,epoch,RBD,"epoch_%s"%epoch,TMP_IOU)
            torch.save(self.net.state_dict(),save_path + 'CP{}.pth'.format(epoch))
            print('Checkpoint {} saved !'.format(epoch))
    
        self.info_validation(val_loader,-2,RBD,'_last_',TMP_IOU)
#         time.sleep(4)
#         plt.close(fig)
        return save_path + 'CP{}.pth'.format(epoch)
        

    def output_training_stats(self, step, loss,batch_y,probs_seg):
    # Calculate batch loss and accuracy
        loss_v=loss.data[0]
        groundtruth_seg_v=np.asarray(batch_y)
        prediction_seg_v=probs_seg.data.cpu().numpy()
  
    
        logging.info("Iter {:}, Minibatch Loss= {:.4f}, Minibatch error= {:.4f}%".format(step,loss_v,error_rate(prediction_seg_v, groundtruth_seg_v)))
    
    def info_validation(self,val_loader,epoch,RBD,name,TMP_IOU):

        loss_v=0
        error_rate_v=0
        iou_acc_v=0
        f1_v=0
        if name=="_init":
            display_patches=True
            save_patches=True
            save_IOU_metrics=False
        elif name=='_last_':
            display_patches=True
            save_patches=True
            save_IOU_metrics=False
        else:
            display_patches=True
            save_patches=False
            save_IOU_metrics=True
            
        
       
        for i_batch,sample in enumerate(val_loader):
            predict_net=Train_or_Predict(sample,self.dist_net,self.loss_fn,self.threshold,self.bins,self.net)
            loss,probs_dist,probs_seg=predict_net.forward_pass()
             
            prediction_seg_v=probs_seg.data.cpu().numpy()
            groundtruth_seg_v=np.asarray(predict_net.batch_y)
            prediction_dist_v=probs_dist.data.cpu().numpy()
            groundtruth_dist=np.asarray(predict_net.batch_y_dist)
            
            loss_v+=loss.data[0]
            error_rate_v+=error_rate(prediction_seg_v,groundtruth_seg_v)
            
     
            if (save_IOU_metrics and (epoch+1)%self.IOU_STEP==0):
                iou_acc,f1,_=predict_score_batch(TMP_IOU,np.argmax(groundtruth_seg_v,3),np.argmax(prediction_seg_v,3))
                iou_acc_v+=iou_acc
                f1_v+=f1
        
        loss_v/=val_loader.__len__()   
        error_rate_v/=val_loader.__len__()  
        logging.info("Verification  loss= {:.4f},error= {:.4f}%".format(loss_v,error_rate_v))
        
        if (name!="_init" and name!='_last_'):
            (self.store_learning).write_file((self.store_learning).file_verif,loss_v)
            (self.store_learning).write_file((self.store_learning).error_rate_file_verif,error_rate_v)

        
        if (save_IOU_metrics and (epoch+1)%self.IOU_STEP==0):
            iou_acc_v/=val_loader.__len__()  
            f1_v/=val_loader.__len__()  
            logging.info("Verification   IOU Precision = {:.4f}%, F1 IOU= {:.4f}%".format(iou_acc_v,f1_v))
            (self.store_learning).write_file((self.store_learning).IOU_acc_file_verif,iou_acc_v)
            (self.store_learning).write_file((self.store_learning).f1_IOU_file_verif,f1_v)
    
            

if __name__ == '__main__':
    
#     python train_model.py ../2_DATA_GHANA/DATASET/128_x_128_8_pansh/ MODEL_TEST_GHANA/ RESUNET_test_ '' --epochs=6 --iou_step=2
    
    root_folder=sys.argv[1]
     ##########
    GLOBAL_PATH=sys.argv[2]
    

    if not os.path.exists(GLOBAL_PATH):
            os.makedirs(GLOBAL_PATH)
    TEST_SAVE=GLOBAL_PATH+'TEST_SAVE/'
    if not os.path.exists(TEST_SAVE):
            os.makedirs(TEST_SAVE)
    ##########
    
    
    MODEL_PATH_SAVE=GLOBAL_PATH+sys.argv[3]
    MODEL_PATH_RESTORE=sys.argv[4]
    
    for i in range(5, len(sys.argv)):
        arg = sys.argv[i]
        if arg.startswith('--input_channels'):
            INPUT_CHANNELS=int(arg[len('--input_channels='):])
        elif arg.startswith('--nb_classes'):
            NB_CLASSES=int(arg[len('--nb_classes='):])
#         elif arg.startswith('--unet_version'):
#             UNET_V=int(arg[len('--unet_version='):])
        elif arg.startswith('--nb_layers'):
            DEFAULT_LAYERS=int(arg[len('--nb_layers='):])
        elif arg.startswith('--filter_width'):
            DEFAULT_FILTER_WIDTH=int(arg[len('--filter_width='):]) 
        elif arg.startswith('--nb_features_root'):
            DEFAULT_FEATURES_ROOT=int(arg[len('--nb_features_root='):])
        elif arg.startswith('--learning_rate'):
            DEFAULT_LR=float(arg[len('--learning_rate='):])
        elif arg.startswith('--batch_size'):
            DEFAULT_BATCH_SIZE = int(arg[len('--batch_size='):])
        elif arg.startswith('--epochs'):
            DEFAULT_EPOCHS = int(arg[len('--epochs='):])
        elif arg.startswith('--dropout'):
            DROPOUT = float(arg[len('--dropout='):])
        elif arg.startswith('--display_step'):
            DISPLAY_STEP = int(arg[len('--display_step='):])
        elif arg.startswith('--validation_size_batch'):
            DEFAULT_VALID = int(arg[len('--validation_size_batch='):])  
        elif arg.startswith('--distance_net'):
            DISTANCE_NET = arg[len('--distance_net='):]
            if DISTANCE_NET=='v2':
                BINS=10
                THRESHOLD=20
                DISTANCE_NET_UNET=True
            elif DISTANCE_NET=='None':
                DISTANCE_NET=None
                DISTANCE_NET_UNET=False
            else:
                raise ValueError('Unknown argument %s' % str(arg))
        elif arg.startswith('--batch_norm'):
            DEFAULT_BN = eval(arg[len('--batch_norm='):])
        elif arg.startswith('--iou_step'):
            IOU_STEP = int(arg[len('--iou_step='):])
        elif arg.startswith('--lr_reduce_steps'):
            REDUCE_LR_STEPS = np.asarray(arg[len('--lr_reduce_steps='):].split(',')).astype(int)
        elif arg.startswith('--data_aug'):
            if (arg[len('--data_aug='):].lower()=='yes'):
                DATA_AUG=transforms.Compose([Transform(),ToTensor()])
            elif (arg[len('--data_aug='):].lower()=='no'):
                DATA_AUG=None
            else:
                raise ValueError('Unknown argument %s' % str(arg))
        elif arg.startswith('--loss_func'):
            LOSS_FN=arg[len('--loss_func='):]
            
        else:
            raise ValueError('Unknown argument %s' % str(arg))
    
    from RUBV3D2 import UNet 
    
    model=UNet(INPUT_CHANNELS,NB_CLASSES,depth =DEFAULT_LAYERS,n_features_zero =DEFAULT_FEATURES_ROOT,width_kernel=DEFAULT_FILTER_WIDTH,dropout=DROPOUT,distance_net=DISTANCE_NET_UNET,bins=BINS,batch_norm=DEFAULT_BN)
    
    model.cuda()
    
    cudnn.benchmark = True
    
    print('### Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    trainer=Trainer(model,DEFAULT_BATCH_SIZE,DEFAULT_LR,NB_CLASSES,LOSS_FN)
    store_learning=Store_learning(GLOBAL_PATH)
    save_path=trainer.train( root_folder,store_learning, MODEL_PATH_SAVE, MODEL_PATH_RESTORE,DEFAULT_EPOCHS,DROPOUT, DISPLAY_STEP, DEFAULT_VALID, TEST_SAVE,DISTANCE_NET,THRESHOLD,BINS,IOU_STEP,REDUCE_LR_STEPS,DATA_AUG)
    print('Last model saved is %s: '%save_path)