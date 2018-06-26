import os
from utils import make_variable_lab_da
import torch



class Train_or_Predict_adda():
    
    def __init__(self,discri_net,criterion):


        self.discri_net=discri_net
        self.criterion=criterion


    def forward_pass_d(self,feature_maps_src,feature_maps_tgt):
        
         # predict on discriminator
        feat_concat=[]
        for feature_map_src,feature_map_tgt in zip(feature_maps_src,feature_maps_tgt):
            feat_concat.append(torch.cat((feature_map_src.detach(), feature_map_tgt.detach()), 0))    
          
        
        pred_concat = self.discri_net(feat_concat)
        

        # prepare real and fake label
        label_src = make_variable_lab_da(torch.ones(len(feature_maps_src[0])).long())
        label_tgt = make_variable_lab_da(torch.zeros(len(feature_maps_tgt[0])).long())
        label_concat = torch.cat((label_src, label_tgt), 0)
        
        
            
        # compute loss for critic
        loss_discri = self.criterion(pred_concat, label_concat)
        
    
        return loss_discri,feat_concat,pred_concat,label_concat
    
   
    
    def forward_pass_g(self,feature_maps_tgt):


        # predict on discriminator
        pred_tgt = self.discri_net(feature_maps_tgt)

        # prepare fake labels
        label_tgt = make_variable_lab_da(torch.ones(len(feature_maps_tgt[0])).long())

        # compute loss for target encoder
        loss_tgt = self.criterion(pred_tgt, label_tgt)
        return loss_tgt,pred_tgt,label_tgt


    def backward_prog(self,loss,optimizer):
        loss.backward()
        optimizer.step()
        
        
   
    
    
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
        
        self.file_train_d = open(self.prediction_path+'avg_loss_train_d.txt',mode) 
        
        self.file_verif_d = open(self.prediction_path+'loss_verif_d.txt',mode) 
        #STORE loss Generative for ANALYSIS
        
        self.file_train_g = open(self.prediction_path+'avg_loss_train_g.txt',mode) 
        
        self.file_verif_g = open(self.prediction_path+'loss_verif_g.txt',mode) 
        
        #STORE accuracy Discriminateive 

        self.acc_rate_file_verif_d = open(self.prediction_path+'acc_verif_d.txt',mode)
        

        
    def write_file(self,file,value):
        
        
        file.write(str(value)+'\n')
        file.flush()