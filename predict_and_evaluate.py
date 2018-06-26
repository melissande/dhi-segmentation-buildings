import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from Data_Handle.image_utils import standardize,distance_map_batch_v2
import matplotlib.pyplot as plt
import time
import os

def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """

    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))


class Train_or_Predict():
    
    def __init__(self,sample,dist_net,loss_fn,threshold,bins,net):
        
        

        self.dist_net=dist_net
        self.loss_fn=loss_fn
        self.threshold=threshold
        self.bins=bins
        self.sample=sample
        self.net=net
        self.feature_maps=None
    def get_feature_maps(self):
        
        return self.feature_maps
       
    
    def forward_pass(self):
        
        ##Variables output transformed for cuda
        
        X=self.initialize_input()
        self.batch_y=self.sample['groundtruth']
        Y = Variable(self.batch_y.float())
        Y=Y.cuda()
                
        ## fwd
        if self.dist_net=='v2':
            self.batch_y_dist=distance_map_batch_v2(self.batch_y,self.threshold,self.bins)
            Y_dist = Variable(self.batch_y_dist.float())
            Y_dist=Y_dist.cuda()
            probs_dist,probs_seg=self.predict(X)
            loss_seg=self.criterion(Y,probs_seg,self.loss_fn)
            loss_dist=self.criterion(Y_dist,probs_dist,'cross-entropy')
            loss=loss_seg+loss_dist

            
        else:
            self.batch_y_dist=None
            probs_seg=self.predict(X)
            probs_dist=None
            loss=self.criterion(Y,probs_seg,self.loss_fn)

    
        return loss,probs_dist,probs_seg
    

    def initialize_input(self):
         ##Variables input transformed for cuda
            
        self.batch_x=standardize(self.sample['input'])
        X = Variable(self.batch_x.float())
        X=X.permute(0,3,1,2).cuda()  #for the model to fwd/back in pytorch always batch size x channels x height x width

        return X
    
    def jaccard_approx(self,y_true,y_est):
        '''
            Loss advised by Vladimir on Spacenet Challenge: http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/
        '''
        sigmo=nn.Sigmoid()
        y_est=sigmo(y_est)

        jaccard_approx=1/2*(torch.sum(y_true[:,0]*y_est[:,0])/(torch.sum(y_true[:,0])+torch.sum(y_est[:,0])-torch.sum(y_true[:,0]*y_est[:,0]))
                            +torch.sum(y_true[:,1]*y_est[:,1])/(torch.sum(y_true[:,1])+torch.sum(y_est[:,1])-torch.sum(y_true[:,1]*y_est[:,1])))


        loss_func=-1/len(y_est)*(torch.sum(y_true[:,0]*torch.log(y_est[:,0]))+torch.sum(y_true[:,1]*torch.log(y_est[:,1])))

        loss=loss_func-torch.log(jaccard_approx)

        return loss

    def criterion(self,y_true,y_est,loss_fn):
        '''
        Loss used --> cross entropy or jaccard approx
        '''
        y_true = y_true.contiguous().view(-1,y_true.size()[-1])
        y_est = y_est.contiguous().view(-1,y_true.size()[-1])
        y_true_flat = y_true.max(-1)[1]
        if loss_fn=='cross-entropy':
            loss_func=nn.CrossEntropyLoss()
            loss = loss_func(y_est,y_true_flat)#be careful inverse order of arguments
        elif loss_fn=='jaccard_approx':
            loss=self.jaccard_approx(y_true,y_est) 
        return loss


    def predict(self,X):
        '''
            Predict with the model, X has to be a CUDA Variable
        '''
        if self.dist_net=='v2':
            logits_dist,logits_seg,self.feature_maps=self.net(X)
            return logits_dist.permute(0,2,3,1),logits_seg.permute(0,2,3,1)  
        else:
            logits,self.feature_maps=self.net(X)
            return logits.permute(0,2,3,1)
        
    def backward_prog(self,loss,optimizer):
        loss.backward()
        optimizer.step()
        
        return loss,optimizer,self.net
        
class Plot_patches():
    """
        congifured to be used in a notebook (draw for plot)
    """
    
    def __init__(self,prediction_seg,groundtruth_seg=None,prediction_dist=None,groundtruth_dist=None):
        '''
        all parameters ares numpy array and dense labels for groundtruth and predictions
        '''
        
        
        self.prediction_seg=prediction_seg
        self.groundtruth_seg=groundtruth_seg
        self.prediction_dist=prediction_dist
        self.groundtruth_dist=groundtruth_dist
        

     
    def produce_pansharp(self,batch_x):
        
        pansharp=np.stack((batch_x[:,:,:,5],batch_x[:,:,:,3],batch_x[:,:,:,2]),axis=3)

        return pansharp
        
    def plot_patches_with_gt(self,batch_x,epoch,prediction_path,save_patches):
        pansharp=self.produce_pansharp(batch_x)
        
        if self.prediction_dist is None and self.groundtruth_dist is None:
            fig,axs=plt.subplots(3, len(pansharp),figsize=(3*len(pansharp),9))

            labels=np.argmax(self.groundtruth_seg, 3) 
            logits=np.argmax(self.prediction_seg, 3)

            for i in range(len(pansharp)):

                axs[0,i].imshow(pansharp[i])
                axs[0,i].axis('off')
                axs[1,i].imshow(labels[i]) 
                axs[1,i].axis('off')
                axs[2,i].imshow(logits[i])
                axs[2,i].axis('off')


                if save_patches:
                    plt.imsave(prediction_path+epoch+'_Panchro_'+str(i)+'.jpg',pansharp[i])
                    plt.imsave(prediction_path+epoch+'_Groundtruth_'+str(i)+'.jpg',labels[i])
                    plt.imsave(prediction_path+epoch+'_Predictions_'+str(i)+'.jpg',logits[i])
        else:

            fig,axs=plt.subplots(5, len(pansharp),figsize=(5*len(pansharp),15))

            labels_seg=np.argmax(self.groundtruth_seg, 3) 
            logits_seg=np.argmax(self.prediction_seg, 3)
            labels_dist=np.argmax(self.groundtruth_dist, 3) 
            logits_dist=np.argmax(self.prediction_dist, 3)

            for i in range(len(pansharp)):

                axs[0,i].imshow(pansharp[i])
                axs[0,i].axis('off')
                axs[1,i].imshow(labels_seg[i])
                axs[1,i].axis('off')
                axs[2,i].imshow(logits_seg[i])
                axs[2,i].axis('off')
                axs[3,i].imshow(labels_dist[i],cmap="jet")
                axs[3,i].axis('off')
                axs[4,i].imshow(logits_dist[i],cmap="jet")
                axs[4,i].axis('off')


                if save_patches:
                    plt.imsave(prediction_path+epoch+'_Panchro_'+str(i)+'.jpg',pansharp[i])
                    plt.imsave(prediction_path+epoch+'_Groundtruth_'+str(i)+'.jpg',labels_seg[i])
                    plt.imsave(prediction_path+epoch+'_Predictions_'+str(i)+'.jpg',logits_seg[i])
                    plt.imsave(prediction_path+epoch+'_Groundtruth_dist_'+str(i)+'.jpg',labels_dist[i],cmap="jet")
                    plt.imsave(prediction_path+epoch+'_Predictions_dist_'+str(i)+'.jpg',logits_dist[i],cmap="jet")

        fig.canvas.draw()
        time.sleep(10)
        plt.close(fig)
        
    def plot_patches_only_pred(self,batch_x,prediction_path,save_patches): 
        pansharp=self.produce_pansharp(batch_x)
        
        if self.prediction_dist is None:
            fig,axs=plt.subplots(2, len(pansharp),figsize=(3*len(pansharp),6))

            logits=np.argmax(self.prediction_seg, 3)

            for i in range(len(pansharp)):

                axs[0,i].imshow(pansharp[i]) 
                axs[1,i].imshow(logits[i])


                if save_patches:
                    plt.imsave(prediction_path+'_Panchro_'+str(i)+'.jpg',pansharp[i])
                    plt.imsave(prediction_path+'_Predictions_'+str(i)+'.jpg',logits[i])
        else:

            fig,axs=plt.subplots(3, len(pansharp),figsize=(3*len(pansharp),9))


            logits_seg=np.argmax(self.prediction_seg, 3)
            logits_dist=np.argmax(self.prediction_dist, 3)

            for i in range(len(pansharp)):

                axs[0,i].imshow(pansharp[i])
                axs[1,i].imshow(logits_seg[i])
                axs[2,i].imshow(logits_dist[i],cmap="jet")


                if save_patches:
                    plt.imsave(prediction_path+'_Panchro_'+str(i)+'.jpg',pansharp[i])
                    plt.imsave(prediction_path+'_Predictions_'+str(i)+'.jpg',logits_seg[i])
                    plt.imsave(prediction_path+'_Predictions_dist_'+str(i)+'.jpg',logits_dist[i],cmap="jet")

        fig.canvas.draw()
        time.sleep(10)
        plt.close(fig)
        
class Store_learning(object):
    """
    Text files creations and update to store learning tracking: avg loss train at each epoch, verificaiton loss at each epoch,
    error rate verification set at each epoch,Spacenet IoU accuracy every certain steps of epoch, Spacenet F1 score every certain steps of epoch 
    """
    def __init__(self, global_pred_path):

        self.prediction_path=global_pred_path+'STORE_LEARNING/'
        if not os.path.exists(self.prediction_path):
            os.makedirs(self.prediction_path)

       
        
    def initialize(self,mode):
        #STORE loss for ANALYSIS
        
        self.avg_loss_train=[]
        self.file_train = open(self.prediction_path+'avg_loss_train.txt',mode) 
        
        self.file_verif = open(self.prediction_path+'loss_verif.txt',mode) 
        #STORE ERROR RATE

        self.error_rate_file_verif = open(self.prediction_path+'error_verif.txt',mode)
        #STORE IOU_ACC for ANALYSIS

        self.IOU_acc_file_verif = open(self.prediction_path+'iou_acc_verif.txt',mode)
        #STORE f1_IOU for ANALYSIS

        self.f1_IOU_file_verif = open(self.prediction_path+'f1_iou_verif.txt',mode) 

        
    def write_file(self,file,value):

        
        file.write(str(value)+'\n')
        file.flush()