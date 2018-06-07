import tensorflow as tf
import h5py
import numpy as np
from numpy import newaxis
import cv2
import sys
import os
from image_utils import read_images,write_data_h5,read_labels
#import matplotlib.pyplot as plt


'''
patche_exctraction.py is used to generate patches from the satellite image of Accra provided by DHI Gras. This is image is around 12000x12000 pixels and the MS bands are also upsampled in this script. The patches, stored in .h5 format are stored as number_of_patches x width x heights x number of bands for panchro, pansharp, ms and groundtruth
Arguments:
python patche_extraction.py ../DATA_GHANA/RAW_DATA/ ../DATA_GHANA/RAW_PATCHES/120_x_120/ width_patch=120

The two first arguments are obligatory. '../DATA_GHANA/RAW_DATA/' contains the input folder where the files corresponding to the bands of the image should be stored under these names: 'panchro.tif' for the Panchromatic Image, 'pansharp.tif' for the Pansharpened image, 'ms.tif' for the Multi Spectral Bands Image and 'groundtruth.png' for the image containing the mask with building footprints (0 if not and 1 if there is a building). The second argument '../DATA_GHANA/RAW_PATCHES/120_x_120/' is the path of the folder where to store the patches. width_patch is the size of the patches to extract.

'''

NAME_PANCHRO='panchro.tif'
NAME_PANSHARP='pansharp.tif'
NAME_MS='ms.tif'
NAME_LABELS='groundtruth.png'

WIDTH=128
STRIDE=128#needs to be equal to the width as we don't want overlapping patches
'''
All the lines with display have been commented at the scripts were launched on command line remote server so the display was not available
'''

def prepare_ms_hr(ms_lr,size_hr):
    '''
    Prepares the upsampled MS image 
    :ms_lr input image to upsample
    :size_hr to upsample the Low Resolution MS image to the dimension of the High Resolution panchromatic image
    '''
    tf.reset_default_graph()
    ms_ph=tf.placeholder(tf.float64, [ms_lr.shape[0],ms_lr.shape[1],ms_lr.shape[2],ms_lr.shape[3]], name='ms_placeholder')
    ms_hr=tf.image.resize_images(ms_ph, [size_hr[0], size_hr[1]])
    ms_hr=tf.cast(ms_hr,tf.float64,name='cast_ms_hr')

    with tf.Session() as sess: 
        ms_hr= sess.run(ms_hr,feed_dict={ms_ph: ms_lr})
        return ms_hr
        
def extract_patches(data,width,stride):
    '''
    Extract patches from images 
    :data input image 
    :width dimensiton of the patch
    :stride stride of patch selection on the image
    '''
    tf.reset_default_graph()
    print('Patch extraction with stride=%d and width=%d begins'%(stride,width) )
    data_pl=tf.placeholder(tf.float64, [data.shape[0],data.shape[1],data.shape[2],data.shape[3]], name='data_placeholder')
    data_o=tf.extract_image_patches(images=data_pl,ksizes=[1,width,width,1],strides=[1,stride,stride,1],rates=[1,1,1,1],padding='VALID')
    print('Patch extraction done')
    size_tot=data_o.get_shape().as_list()
    data_o=tf.reshape(data_o,[size_tot[1]*size_tot[2],width,width,data.shape[3]])
    with tf.Session() as sess:
        Data_o= sess.run(data_o,feed_dict={data_pl: data})
        print('%d patches of size %d x %d created as list'%(Data_o.shape[0],Data_o.shape[1],Data_o.shape[2]))
        return Data_o
    
def save_patches(data,path_out):
    '''
    Write the patches list to .h5 file format
    :data patches list
    :path_out where to save the patches
    '''
    write_data_h5(path_out,data)
    
if __name__ == '__main__':
    
    if len(sys.argv)<3:
        print('Specify all the paths of folder input and output')
        exit()
    
    
    #path='../DATA_GHANA/RAW_DATA/'
    path=sys.argv[1]

    #path_patches='../DATA_GHANA/RAW_PATCHES/120_x_120/'
    path_patches=sys.argv[2]
    if not os.path.exists(path_patches):
            os.makedirs(path_patches)
    
    
    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg.startswith('--width_patch'):
            WIDTH=int(arg[len('--width_patch='):])
            STRIDE=WIDTH
    
    #patch_test_number=300
     ## Panchromatic
    panchromatic_file=path+NAME_PANCHRO
    panchromatic=read_images(panchromatic_file)
    hr_size=panchromatic.shape
    panchromatic=panchromatic[newaxis,:,:,newaxis]
    print('\n PANCHROMATIC \n\n')
    panchromatic=extract_patches(panchromatic,WIDTH,STRIDE)
    
    ## Find patches to discard
    keep=np.arange(len(panchromatic))
    discard=[]
    print('Original size of the dataset is %d'%len(panchromatic))
    for i in range(len(panchromatic)):
        if (np.sum(panchromatic[i,:,:,:])).astype(int)==0:
            discard.append(i)

    discard=np.asarray(discard)
    keep=np.delete(keep,discard)
    print('Final size of the dataset is %d'%len(keep))
    ## Save Panchromatic
    panchromatic=panchromatic[keep]
    save_patches(panchromatic,path_patches+'panchro.h5') 
#     plt.imshow(panchromatic[patch_test_number,:,:,0])
#     plt.show()
            

    ##MS bands
    
    ms_file=path+NAME_MS 
    ms=read_images(ms_file)
    ms=np.transpose(ms,(1,2,0))
    print('\n MS BANDS\n\n')
    for i in range(ms.shape[2]):
        print('\n Band %d \n'%i)
        ms_i=ms[:,:,i]
        print('Upscale')
        ms_hr=prepare_ms_hr(ms_i[newaxis,:,:,newaxis],hr_size)
        ms_hr=extract_patches(ms_hr,WIDTH,STRIDE)
        ms_hr=ms_hr[keep]
        save_patches(ms_hr,path_patches+'ms_hr_'+str(i)+'.h5')
#         plt.imshow(ms_hr[patch_test_number,:,:,0])
#         plt.show()

    
    ##Pansharpened bands
    
    pansharpened_file=path+NAME_PANSHARP 
    pansharpened=read_images(pansharpened_file)
    pansharpened=np.transpose(pansharpened,(1,2,0))
    print('\n BANDS\n\n')
    for i in range(pansharpened.shape[2]):
        print('\n Band %d \n'%i)
        pansharpened_i=pansharpened[:,:,i]
        pansharpened_i=extract_patches(pansharpened_i[newaxis,:,:,newaxis],WIDTH,STRIDE)
        pansharpened_i=pansharpened_i[keep]
        save_patches(pansharpened_i,path_patches+'pansharpened_'+str(i)+'.h5')
#         plt.imshow(pansharpened_i[patch_test_number,:,:,0])
#         plt.show()
        
    ## Label patches
    
    labels_file=path+NAME_LABELS
    labels=read_labels(labels_file)
    labels=labels[newaxis,:,:,newaxis]
    
    print('\n LABELS \n\n')
    labels=extract_patches(labels,WIDTH,STRIDE)
    labels=labels[keep]
    save_patches(labels,path_patches+'groundtruth.h5')
#     plt.imshow(labels[patch_test_number,:,:,0])
#     plt.show()
    
    
    
    
    
    

    