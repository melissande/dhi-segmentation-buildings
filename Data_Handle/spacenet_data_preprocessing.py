from image_utils import *
from skimage.transform import rescale,resize
from scipy import misc
from numpy import newaxis
import numpy as np

#Dataset from SPACENET for Vegas, Paris, Khartum and Shanghai have resolution of 30 cm per pixel for pansharpen whereas
# GHANA has 50 cm but for Rio it is also 50 cm

ratio_ghana=3.0/5.0


def align_resolution_with_ghana(image):
    '''
    Aligns resolution of images from SPACENET (Vegas, Paris, Khartum and Shanghai) which are not the same than Ghana dataset
    :image ndarray with dimension (number of channels x width x height)
    return ndarray with dimension ( width x height x number of channels)
    '''
    image=np.transpose(image,(1,2,0))
    return rescale(image,ratio_ghana)[newaxis,:,:,:]
def prepare_ms_hr(ms_lr,size_hr):
    '''
    Prepares the upsampled MS image 
    :ms_lr input image to upsample
    :size_hr to upsample the Low Resolution MS image to the dimension of the High Resolution panchromatic image
    '''
    ms_lr=np.transpose(ms_lr,(1,2,0))
    return resize(ms_lr, size_hr)[newaxis,:,:,:]
def prepare_panchro(pansharp_rgb_path):
    return misc.imread(pansharp_rgb_path,'L')[newaxis,:,:,newaxis]
if __name__ == '__main__':
    
    path=sys.argv[0]#'/scratch/SPACENET_DATA/BUILDING_DATASET/'
    
    path_out=sys.argv[1]#'/scratch/SPACENET_DATA_PROCESSED/'
    path_raw=path_out+'RAW_IMAGES/'
    if not os.path.exists(path_raw):
                os.makedirs(path_raw)
    
    ## build RAW patches folders
    for filedir in os.listdir(path):
        if not os.path.exists(path_raw+filedir+'/PANCHRO/'):
                os.makedirs(path_raw+filedir+'/PANCHRO/')
        if not os.path.exists(path_raw+filedir+'/PANSHARP/'):
                os.makedirs(path_raw+filedir+'/PANSHARP/')
        if not os.path.exists(path_raw+filedir+'/GROUNDTRUTH/'):
                os.makedirs(path_raw+filedir+'/GROUNDTRUTH/')
        
    
    
    for filedir in os.listdir(path):

        if filedir.startswith('AOI_1_RIO'):
            _3_bands_folder=path+'AOI_1_RIO/processedBuildingLabels/3band/'
            _8_bands_folder=path+'AOI_1_RIO/processedBuildingLabels/8band/'
            geojson_folder=path+'AOI_1_RIO/processedBuildingLabels/vectordata/geojson/'
            file=filedir.split('AOI_')[1]
            print('Preparing %s'%file)
            
            panchro_size=[]
            panchro_files=[]
            for filename in  sorted(os.listdir(_3_bands_folder)):
                file=filename.split('RIO_')[1]
                panchro_files.append(_3_bands_folder+filename)

#                 print('Preparing Panchro %s'%file)
                panchro=prepare_panchro(_3_bands_folder+filename)
                panchro_size.append(panchro.shape[1:3])
#                 print('Panchro Shape [%d,%d,%d,%d]'%panchro.shape)

                write_data_h5(path_raw+'AOI_1_RIO/PANCHRO/panchro_1_RIO_'+file.split('.tif')[0]+'.h5',panchro)
                
            panchro_size=np.asarray(panchro_size)  
            panchro_files=np.asarray(panchro_files)
            count=0
            for filename in  sorted(os.listdir(_8_bands_folder)):
                file=filename.split('RIO_')[1]
#                 print('Preparing Pansharp %s'%file)
                pansharp=read_images(_8_bands_folder+filename)
                pansharp=prepare_ms_hr(pansharp,panchro_size[count])
#                 print('Pansharp Shape [%d,%d,%d,%d]'%pansharp.shape)
                write_data_h5(path_raw+'AOI_1_RIO/PANSHARP/pansharp_1_RIO_'+file.split('.tif')[0]+'.h5',pansharp)
                count+=1     
            count=0
            for filename in  sorted(os.listdir(geojson_folder)):
                file=filename.split('RIO_')[1]
                createRasterFromGeoJson(geojson_folder+filename, panchro_files[count],path_raw+'AOI_1_RIO/GROUNDTRUTH/groundtruth_1_RIO_'+file.split('.geojson')[0]+'.png')
                labels=read_labels(path_raw+'AOI_1_RIO/GROUNDTRUTH/groundtruth_1_RIO_'+file.split('.geojson')[0]+'.png')
                os.remove(path_raw+'AOI_1_RIO/GROUNDTRUTH/groundtruth_1_RIO_'+file.split('.geojson')[0]+'.png') 
#                 print('Groundtruth shape [%d,%d,%d,%d]'%labels[newaxis,:,:,newaxis].shape)
                write_data_h5(path_raw+'AOI_1_RIO/GROUNDTRUTH/groundtruth_1_RIO_'+file.split('.geojson')[0]+'.h5',labels[newaxis,:,:,newaxis])
                count+=1
            
        else:

            panchro_folder=path+filedir+'/'+filedir+'_Train/PAN/'
            pansharp_folder=path+filedir+'/'+filedir+'_Train/MUL-PanSharpen/'
            geojson_folder=path+filedir+'/'+filedir+'_Train/geojson/buildings/'
            file=filedir.split('AOI_')[1]
            print('Preparing %s'%file)
            panchro_files=[]
            for filename in  sorted(os.listdir(panchro_folder)):
                file=filename.split('AOI_')[1]
                panchro_files.append(panchro_folder+filename)
#                 print('Preparing Panchro %s'%file)
                panchro=read_images(panchro_folder+filename)
                panchro=align_resolution_with_ghana(panchro[newaxis,:,:])
#                 print('Panchro Shape [%d,%d,%d,%d]'%panchro.shape)
                write_data_h5(path_raw+filedir+'/PANCHRO/'+'panchro_'+file.split('.tif')[0]+'.h5',panchro)
           
            panchro_files=np.asarray(panchro_files)
            count=0          
            for filename in  sorted(os.listdir(pansharp_folder)):
                file=filename.split('AOI_')[1]
#                 print('Preparing Pansharp %s'%file)
                pansharp=read_images(pansharp_folder+filename)
                pansharp=align_resolution_with_ghana(pansharp)
#                 print('Pansharp Shape [%d,%d,%d,%d]'%pansharp.shape)
                write_data_h5(path_raw+filedir+'/PANSHARP/'+'pansharp_'+file.split('.tif')[0]+'.h5',pansharp)
            for filename in  sorted(os.listdir(geojson_folder)):
                file=filename.split('AOI_')[1]
                createRasterFromGeoJson(geojson_folder+filename, panchro_files[count],path_raw+filedir+'/GROUNDTRUTH/'+'groundtruth_'+file.split('.geojson')[0]+'.png')
                labels=read_labels(path_raw+filedir+'/GROUNDTRUTH/'+'groundtruth_'+file.split('.geojson')[0]+'.png')
                os.remove(path_raw+filedir+'/GROUNDTRUTH/'+'groundtruth_'+file.split('.geojson')[0]+'.png')
                labels=align_resolution_with_ghana(labels[newaxis,:,:])
#                 print('Groundtruth shape [%d,%d,%d,%d]'%labels.shape)
                write_data_h5(path_raw+filedir+'/GROUNDTRUTH/'+'groundtruth_'+file.split('.geojson')[0]+'.h5',labels)
                count+=1
            
        