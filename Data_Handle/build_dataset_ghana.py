import h5py
import numpy as np
from numpy import newaxis
import cv2
import sys
import os
from image_utils import read_data_h5,write_data_h5
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    if len(sys.argv)<3:
        print('Specify the path of the folders for Raw patches and dataset ')
        exit()
        
    
    path_patches=sys.argv[1]
    #path_patches='../2_DATA_GHANA/RAW_PATCHES/128_x_128/'
    Path_dataset=sys.argv[2]
    #path_dataset='../2_DATA_GHANA/DATASET/'
    if not os.path.exists(Path_dataset):
            os.makedirs(Path_dataset)
    
        
    training_ratio=0.8 #so    test_ratio=0.2
    validation_ratio=0.2
    
    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg.startswith('--training_ratio'):
            training_ratio=float(arg[len('--training_ratio='):])
        if arg.startswith('--validation_ratio'):
            validation_ratio=float(arg[len('--validation_ratio='):])
    
    
   
    training_ratio=0.8 #so    test_ratio=0.2
    validation_ratio=0.2
    
    
    list_input_panchro=[]
    list_input_ms=[]
    list_input_pansharp=[]
    for filename in sorted(os.listdir(path_patches)):
        if filename.startswith('panchro'):
            print('Reading %s'%filename)
            list_input_panchro.append(read_data_h5(path_patches+filename))
        if filename.startswith('ms'):
            print('Reading %s'%filename)
            list_input_ms.append(read_data_h5(path_patches+filename))
        if filename.startswith('pansharp'):
            print('Reading %s'%filename)
            list_input_pansharp.append(read_data_h5(path_patches+filename))
        if filename.startswith('groundtruth'):
            print('Reading %s'%filename)
            list_output=read_data_h5(path_patches+filename)

            
    ## READ ALL
    
    list_input_panchro=np.squeeze(np.asarray(list_input_panchro))[newaxis,:,:,:]
    list_input_pansharp=np.squeeze(np.asarray(list_input_pansharp))
    list_input_ms=np.squeeze(np.asarray(list_input_ms))
    list_input=np.concatenate((list_input_panchro,list_input_pansharp,list_input_ms),axis=0)
    
    list_input=np.transpose(list_input,(1,2,3,0))
    list_output=np.squeeze(list_output)
    print('Size patches list input [%d,%d,%d,%d]'%list_input.shape)
    print('Size patches list output [%d,%d,%d]'%list_output.shape)
    
    print('Dataset read')
    idx_shuffle = np.arange(len(list_input))
    np.random.shuffle(idx_shuffle)
    print('Dataset shuffled')    
    list_input=list_input[idx_shuffle]
    list_output=list_output[idx_shuffle]
    
    print('Save indices of shuffle at %s' %'indices_dataset.txt')
    np.savetxt('indices_dataset.txt',idx_shuffle)
    #Do the split
    training_size=int(round(training_ratio*list_input.shape[0]))
    test_size=list_input.shape[0]-training_size
    validation_size=int(round(validation_ratio*training_size))
    training_size=training_size-validation_size
    
    
    print('Split (TRAINING - VALIDATION:%f) - TEST:%f  done'%(1-validation_ratio,training_ratio))
    print('Training size:%d, Validation size:%d, Test size: %d'%(training_size,validation_size,test_size))
    
    
    path_dataset_all=[]
    path_dataset=Path_dataset+'128_x_128_8_pansh/'
    if not os.path.exists(path_dataset):
            os.makedirs(path_dataset)
    index=np.arange(9)
    path_dataset_all.append([path_dataset,index])
#     path_dataset=Path_dataset+'120_x_120_4_pansh/'
#     if not os.path.exists(path_dataset):
#             os.makedirs(path_dataset)
#     index=[0,1,2,4,6]
#     path_dataset_all.append([path_dataset,index])
#     path_dataset=Path_dataset+'120_x_120_8_ms/'
#     if not os.path.exists(path_dataset):
#             os.makedirs(path_dataset)
#     index=[0,9,10,11,12,13,14,15,16]
#     path_dataset_all.append([path_dataset,index])
            
#     path_dataset=Path_dataset+'120_x_120_8_pansh_8_ms/'
#     if not os.path.exists(path_dataset):
#             os.makedirs(path_dataset)
#     index=np.arange(17)
#     path_dataset_all.append([path_dataset,index])
           
    #     #Save the dataset
    for path in path_dataset_all:
        path_dataset=path[0]
        if not os.path.exists(path_dataset+'TRAINING'):
                os.makedirs(path_dataset+'TRAINING')
                if not os.path.exists(path_dataset+'TRAINING/INPUT'):
                    os.makedirs(path_dataset+'TRAINING/INPUT')
                if not os.path.exists(path_dataset+'TRAINING/OUTPUT'):
                    os.makedirs(path_dataset+'TRAINING/OUTPUT')
        if not os.path.exists(path_dataset+'VALIDATION'):
                os.makedirs(path_dataset+'VALIDATION')
                if not os.path.exists(path_dataset+'VALIDATION/INPUT'):
                    os.makedirs(path_dataset+'VALIDATION/INPUT')
                if not os.path.exists(path_dataset+'VALIDATION/OUTPUT'):
                    os.makedirs(path_dataset+'VALIDATION/OUTPUT')
        if not os.path.exists(path_dataset+'TEST'):
                os.makedirs(path_dataset+'TEST')
                if not os.path.exists(path_dataset+'TEST/INPUT'):
                    os.makedirs(path_dataset+'TEST/INPUT')
                if not os.path.exists(path_dataset+'TEST/OUTPUT'):
                    os.makedirs(path_dataset+'TEST/OUTPUT')
       
    

    
    print('BUILD TRAINING SET')
    for i in range(training_size):
        print('Patch %d/%d Tot size: %d'%(i,training_size,list_input.shape[0]))
        for path in path_dataset_all:
            write_data_h5(path[0]+'TRAINING/INPUT/input_'+str(i)+'.h5',np.transpose(list_input[i,:,:,path[1]],(1,2,0)))
            write_data_h5(path[0]+'TRAINING/OUTPUT/output_'+str(i)+'.h5',list_output[i])
        
    print('BUILD VALIDATION SET')
    for i in range(training_size,training_size+validation_size):
        print('Patch %d/%d Tot size: %d'%(i,training_size+validation_size,list_input.shape[0]))
        for path in path_dataset_all:
            write_data_h5(path[0]+'VALIDATION/INPUT/input_'+str(i)+'.h5',np.transpose(list_input[i,:,:,path[1]],(1,2,0)))
            write_data_h5(path[0]+'VALIDATION/OUTPUT/output_'+str(i)+'.h5',list_output[i])
        
    print('BUILD TEST SET')
    for i in range(training_size+validation_size,list_input.shape[0]):
        print('Patch %d/%d Tot size: %d'%(i,list_input.shape[0],list_input.shape[0]))
        for path in path_dataset_all:
            write_data_h5(path[0]+'TEST/INPUT/input_'+str(i)+'.h5',np.transpose(list_input[i,:,:,path[1]],(1,2,0)))
            write_data_h5(path[0]+'TEST/OUTPUT/output_'+str(i)+'.h5',list_output[i])
