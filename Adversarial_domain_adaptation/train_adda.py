import sys
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import logging


import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms

import time
from random import randint

sys.path.append(os.path.abspath('../'))
from IOU_computations import *
from predict_and_evaluate import *

import params
from utils import get_data_loader, init_model,make_variable_lab_da,Store_learning_adda
from predict_and_evaluate_adda import *
from discriminator import *


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class Trainer_adda():
    
    def __init__(self,src_encoder,tgt_encoder,discri_net,store_learning,store_learning_adda):
        self.src_encoder=src_encoder
        self.tgt_encoder=tgt_encoder
        self.discri_net=discri_net
        self.store_learning=store_learning
        self.store_learning_adda=store_learning_adda
        
    def _initialize(self):
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                           lr=params.c_learning_rate)
        self.optimizer_discri = optim.Adam(discri_net.parameters(),
                              lr=params.d_learning_rate)
        
        
        self.loss_tr_d=[]
        self.loss_val_d=[]
        self.loss_tr_g=[]
        self.loss_val_g=[]
        self.loss_tr_cl_tg=[]
        self.loss_val_cl_tg=[]
        
        
        
    def train(self):
        ####################
        # 1. setup network #
        ####################
        
        # setup dataset
        self._initialize()
        
        ##Source Dataset
        src_data_loader_tr = get_data_loader(params.src_dataset,'TRAINING/')
        src_data_loader_val = get_data_loader(params.src_dataset,'VALIDATION/')
        ##Target Dataset
        tgt_data_loader_tr = get_data_loader(params.tgt_dataset,'TRAINING/')
        tgt_data_loader_val = get_data_loader(params.tgt_dataset,'VALIDATION/')

        

#         # set train state for Dropout and BN layers
#         (self.tgt_encoder).train()
#         (self.discri_net).train()

        # setup criterion and optimizer

        len_data_loader = min(len(src_data_loader_tr), len(tgt_data_loader_tr))
        val_data_zip = enumerate(zip(src_data_loader_val, tgt_data_loader_val))
        val_len_data_loader = min(len(src_data_loader_val), len(tgt_data_loader_val))
        RBD=randint(0,int(val_len_data_loader-1))

        self.info_validation(val_data_zip,val_len_data_loader,-1,RBD,"_init")
#         c_learning_rate=params.c_learning_rate
#         d_learning_rate=params.d_learning_rate
        for epoch in range(params.num_epochs):

            # zip source and target data pair
            tr_data_zip = enumerate(zip(src_data_loader_tr, tgt_data_loader_tr))
            total_loss_cl_tgt = 0            
            total_loss_discri = 0
            total_loss_tgt_da = 0


            for step, (sample_src, sample_tgt) in tr_data_zip:
#                 if step in params.c_lr_reduc:
#                     c_learning_rate=c_learning_rate/2 
#                     self.optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
#                            lr=c_learning_rate)
#                 if step in params.d_lr_reduc:
#                     d_learning_rate=d_learning_rate/2
#                     self.optimizer_discri = optim.Adam(discri_net.parameters(),
#                            lr=d_learning_rate)        
                if (step % params.train_dis_step == 0):

                    ###########################
                    # 2.1 train discriminator #
                    ###########################

                    # zero gradients for optimizer
                    (self.optimizer_discri).zero_grad()


                    predict_net_src=Train_or_Predict(sample_src,params.distance_net,params.loss_fn,params.threshold,params.bins,self.src_encoder)
                    predict_net_tgt=Train_or_Predict(sample_tgt,params.distance_net,params.loss_fn,params.threshold,params.bins,self.tgt_encoder)



                    images_src=predict_net_src.initialize_input()
                    images_tgt=predict_net_tgt.initialize_input()

                    # extract and concat features

                    _,_ = predict_net_src.predict(images_src)
                    _,_ = predict_net_tgt.predict(images_tgt)
                    feature_maps_src=predict_net_src.get_feature_maps()
                    feature_maps_tgt=predict_net_tgt.get_feature_maps()


                    predict_discri_net=Train_or_Predict_adda(self.discri_net,self.criterion)
                    loss_discri,_,pred_concat,label_concat=predict_discri_net.forward_pass_d(feature_maps_src,feature_maps_tgt)
                    predict_discri_net.backward_prog(loss_discri,self.optimizer_discri)

                    pred_cls = torch.squeeze(pred_concat.max(1)[1])
                    acc = (pred_cls == label_concat).float().mean()


                ############################
                # 2.2 train target encoder #
                ############################

                # zero gradients for optimizer
                (self.optimizer_discri).zero_grad()
                (self.optimizer_tgt).zero_grad()

                # extract and target features
                predict_net_tgt=Train_or_Predict(sample_tgt,params.distance_net,params.loss_fn,params.threshold,params.bins,self.tgt_encoder)
                loss_cl,_,feat_tgt=predict_net_tgt.forward_pass()
                feature_maps_tgt=predict_net_tgt.get_feature_maps()
                
                predict_discri_net=Train_or_Predict_adda(self.discri_net,self.criterion)
                loss_tgt,_,_=predict_discri_net.forward_pass_g(feature_maps_tgt)
                predict_discri_net.backward_prog(loss_tgt,self.optimizer_tgt)            

                ## saves losses and acc rates

                total_loss_cl_tgt+=loss_cl.data[0]
                total_loss_discri+=loss_discri.data[0]
                total_loss_tgt_da+=loss_tgt.data[0]

                #######################
                # 2.3 print step info #
                #######################
                if ((step + 1) % params.display_step == 0):

                    groundtruth_seg_v=np.asarray(predict_net_tgt.batch_y)
                    prediction_seg_v=feat_tgt.data.cpu().numpy()

                    logging.info("Epoch [{}/{}] Step [{}/{}]\n"
                                 "Adda Information: d_loss={:.4f} g_loss={:.4f} d_f_acc={:.4f}%\n"
                            "Target RUBV3D2: Minibatch Loss= {:.4f}, Minibatch error= {:.4f}%\n"
                          .format(epoch + 1,
                                  params.num_epochs,
                                  step + 1,
                                  len_data_loader,
                                  loss_discri.data[0],
                                  loss_tgt.data[0],
                                  acc.data[0]*100,
                                  loss_cl.data[0],
                                  error_rate(prediction_seg_v, groundtruth_seg_v)))



            (self.loss_tr_cl_tg).append(total_loss_cl_tgt/len_data_loader)
            (self.store_learning).write_file((store_learning).file_train,total_loss_cl_tgt/len_data_loader)  

            (self.loss_tr_d).append(total_loss_discri/len_data_loader)
            (self.store_learning_adda).write_file((self.store_learning_adda).file_train_d,total_loss_discri/len_data_loader)  
            (self.loss_tr_g).append(total_loss_tgt_da/len_data_loader)
            (self.store_learning_adda).write_file((self.store_learning_adda).file_train_g,total_loss_tgt_da/len_data_loader) 
            
            val_data_zip = enumerate(zip(src_data_loader_val, tgt_data_loader_val))
            self.info_validation(val_data_zip,val_len_data_loader,epoch+1,RBD,"epoch_%s"%(epoch+1))
        val_data_zip = enumerate(zip(src_data_loader_val, tgt_data_loader_val))
        self.info_validation(val_data_zip,val_len_data_loader,epoch+1,RBD,'_last_')
            
            
            
    def info_validation(self,val_data_zip,val_len_data_loader,epoch,RBD,name):



        loss_v=0
        error_rate_v=0
        iou_acc_v=0
        f1_v=0


        total_loss_discri = 0
        total_loss_tgt_da = 0
        total_acc_discri=0



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


        for i_batch,(sample_src, sample_tgt) in val_data_zip:
            
       
            predict_net_src=Train_or_Predict(sample_src,params.distance_net,params.loss_fn,params.threshold,params.bins,self.src_encoder)
            predict_net_tgt=Train_or_Predict(sample_tgt,params.distance_net,params.loss_fn,params.threshold,params.bins,self.tgt_encoder)

            images_src=predict_net_src.initialize_input()
            images_tgt=predict_net_tgt.initialize_input()

            # extract and concat features

            _,_=predict_net_src.predict(images_src)
            _,_=predict_net_tgt.predict(images_tgt)
            

            feature_maps_src=predict_net_src.get_feature_maps()
            feature_maps_tgt=predict_net_tgt.get_feature_maps()

            predict_discri_net=Train_or_Predict_adda(self.discri_net,self.criterion)
            loss_discri,feat_concat,pred_concat,label_concat=predict_discri_net.forward_pass_d(feature_maps_src,feature_maps_tgt)

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()


            ############################
            # 2.2 train target encoder #
            ############################

            # extract and target features
            predict_net_tgt=Train_or_Predict(sample_tgt,params.distance_net,params.loss_fn,params.threshold,params.bins,self.tgt_encoder)
            loss_cl,feat_tgt_dist,feat_tgt=predict_net_tgt.forward_pass()
            feature_maps_tgt=predict_net_tgt.get_feature_maps()
            
            loss_tgt,pred_tgt,label_tgt=predict_discri_net.forward_pass_g(feature_maps_tgt)


            prediction_seg_v=feat_tgt.data.cpu().numpy()
            groundtruth_seg_v=np.asarray(predict_net_tgt.batch_y)
            prediction_dist_v=feat_tgt_dist.data.cpu().numpy()
            groundtruth_dist=np.asarray(predict_net_tgt.batch_y_dist)
#             plot_patches=Plot_patches(prediction_seg_v,groundtruth_seg_v,prediction_dist_v,groundtruth_dist)

            loss_v+=loss_cl.data[0]
            error_rate_v+=error_rate(prediction_seg_v,groundtruth_seg_v)


            total_loss_discri+=loss_discri.data[0]
            total_loss_tgt_da+=loss_tgt.data[0]
            total_acc_discri+=acc.data[0]

            if i_batch==RBD:
                
                batch_x=np.asarray(predict_net_tgt.batch_x)
#                 plot_patches.plot_patches_with_gt(batch_x,name,params.test_save,save_patches)

            if (save_IOU_metrics and (epoch+1)%params.iou_step==0):
                iou_acc,f1,_=predict_score_batch(params.tmp_iou,np.argmax(groundtruth_seg_v,3),np.argmax(prediction_seg_v,3))
                iou_acc_v+=iou_acc
                f1_v+=f1

        loss_v/=val_len_data_loader
        error_rate_v/=val_len_data_loader
        total_loss_discri/=val_len_data_loader
        total_loss_tgt_da/=val_len_data_loader
        total_acc_discri/=val_len_data_loader

        logging.info(" {:} \n"
                     "Adda Information: d_loss={:.4f} g_loss={:.4f} d_acc={:.4f}%\n"
                    " Target RUBV3D2: Verification  loss= {:.4f},error= {:.4f}%"
                     .format(name,
                             total_loss_discri,
                             total_loss_tgt_da,
                             total_acc_discri*100,
                             loss_v,
                             error_rate_v))

        if (name!="_init" and name!='_last_'):
            (self.store_learning).write_file((self.store_learning).file_verif,loss_v)
            (self.store_learning).write_file((self.store_learning).error_rate_file_verif,error_rate_v)
            (self.store_learning_adda).write_file((self.store_learning_adda).file_verif_d,total_loss_discri)
            (self.store_learning_adda).write_file((self.store_learning_adda).file_verif_g,total_loss_tgt_da)
            (self.store_learning_adda).write_file((self.store_learning_adda).acc_rate_file_verif_d,total_acc_discri)
            
            (self.loss_val_cl_tg).append(loss_v)
            (self.loss_val_d).append(total_loss_discri)
            (self.loss_val_g).append(total_loss_tgt_da)


        if (save_IOU_metrics and (epoch+1)%params.iou_step==0):
            iou_acc_v/=val_len_data_loader 
            f1_v/=val_len_data_loader
            logging.info("Target RUBV3D2: Verification   IOU Precision = {:.4f}%, F1 IOU= {:.4f}%".format(iou_acc_v,f1_v))
            (self.store_learning).write_file((self.store_learning).IOU_acc_file_verif,iou_acc_v)
            (self.store_learning).write_file((self.store_learning).f1_IOU_file_verif,f1_v)
        

if __name__ == '__main__':        

    ##Final Model Classification
    from RUBV3D2 import UNet 

    src_encoder = init_model(net=UNet(params.input_channels,params.nb_classes,params.default_layers,params.default_features_root,params.default_filter_width,params.distance_unet,params.bins,params.default_bn),
                                 restore=params.src_encoder_restore)

    tgt_encoder = init_model(net=UNet(params.input_channels,params.nb_classes,params.default_layers,params.default_features_root,params.default_filter_width,params.distance_unet,params.bins,params.default_bn),
                                 restore=params.tgt_encoder_restore)
    discri_net = init_model(net=Discriminator(input_dims=params.d_input_dims,
                                      output_dims=params.d_output_dims,len_feature_map=params.d_len_feature_maps,size_in=params.d_size_in),
                        restore=params.d_model_restore)




    store_learning=Store_learning(params.global_path)
    if params.d_model_restore=='':#if there is a model for discriminator
        store_learning.initialize('w')
    else:
        store_learning.initialize('a')


    store_learning_adda=Store_learning_adda(params.global_path)
    if params.d_model_restore=='':
        store_learning_adda.initialize('w')
    else:
        store_learning_adda.initialize('a')


    trainer=Trainer_adda(src_encoder,tgt_encoder,discri_net,store_learning,store_learning_adda)
    trainer.train()
