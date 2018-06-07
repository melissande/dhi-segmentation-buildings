import h5py
import numpy as np
from numpy import newaxis
import cv2
import sys
import os
from osgeo import gdal,osr,ogr
import matplotlib.pyplot as plt
import torch
from scipy.ndimage.morphology import distance_transform_bf


NAME_PANCHRO='panchro.tif'
NAME_PANSHARP='pansharp.tif'
NAME_MS='ms.tif'
geojson_file='buildings_accra.geojson'

'''
This script gathers all the functions useful to open, read and write images in .png or .h5.
The main is used to visualize the image with all bands from Accra by DHI Gras and to create the groundtruth mask in .png
python image_utils.py ../DATA_GHANA/RAW_DATA/
where the first argument is the path are where are the raw images of Accra and the geojson corresponding files.
Files should be named: 'panchro.tif' for the panchromatic band, 'pansharp.tif' for the pansharpened bands, 'ms.tif' for the multi spectral bands, 'buildings_accra.geojson' for the geojson file of Accra.
'''
def read_images(path):
    '''
    Reads tif images (panchromatic, MS, pansharpened)
    :path where is the image
    '''
    raster = gdal.Open(path)
    data = np.array(raster.ReadAsArray())
    return data

def read_labels(path):
    '''
    Reads png images (label mask groundtruth)
    :path where is the image
    '''
    data=cv2.imread(path,cv2.IMREAD_UNCHANGED)
    return data

def read_data_h5(path):
    '''
    Reads h5 file
    :path path of the file to read
    returns data as an array
    '''
    with h5py.File(path, 'r') as hf:
        data=np.array(hf.get('data'))
    return data

def write_data_h5(path,data_):
    '''
    Writes h5 file
    :path path of the file to write
    :data_ data to write into the file
    '''
    with h5py.File(path,'w') as hf:
           hf.create_dataset('data',data=data_)
    print('File'+path+' created')


def plot_images(image,figsize=(8,8), plot_name='',add_title=False,save_path='',save_images=False):
    '''
    Plot each band of an image
    :image numpy array, should be width x height x number_channels
    '''
    
    number_channels=image.shape[2]
    
#     fig, axs = plt.subplots(1, number_channels, figsize=(number_channels*figsize[0], figsize[1]))

#     if number_channels==1:
#         axs=[axs]
    
#     if add_title:
#         suptitle = fig.suptitle(plot_name.split('/')[-1], fontsize='large')
    for i in range(number_channels):
#         axs[i].imshow(image[:,:,i])
#         axs[i].set_title('Band '+str(i))
        if save_images:
            print('Save image %s band %d'%(plot_name,i))
            plt.imsave(save_path+plot_name+'_'+str(i)+'.png',image[:,:,i])
        
#     plt.tight_layout()
#     if add_title:
#         suptitle.set_y(0.95)
#         fig.subplots_adjust(top=0.96)



def createRasterFromGeoJson(srcGeoJson, srcRasterFileName, outRasterFileName):
    '''
    Creates a raster image from a geojson file. Create the raster mask of buildings footprints and saves it as .png
    :srcGeoJson geojson file of building polygons
    :srcRasterFileName raster image corresponding, can be the panchromatic band
    :outRasterFileName name under which the mask has to be saved
    '''
    NoData_value = 0
    source_ds = ogr.Open(srcGeoJson)
    source_layer = source_ds.GetLayer()

    srcRaster = gdal.Open(srcRasterFileName)


    # Create the destination data source
    target_ds = gdal.GetDriverByName('GTiff').Create(outRasterFileName, srcRaster.RasterXSize, srcRaster.RasterYSize, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(srcRaster.GetGeoTransform())
    target_ds.SetProjection(srcRaster.GetProjection())
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])
    band.FlushCache()


def standardize(data):
    '''
    Standardize the input data of the network
    :param data to be standardized (size nb_batches x WIDTH x HEIGHT x number of channels) 
    
    returns data standardized size nb_batches x WIDTH x HEIGHT x number of channels 
    
    '''

    WIDTH=data.shape[1]
    HEIGHT=data.shape[2]
    channels=data.shape[3]
    
    
    mean_t=torch.mean(data.view(len(data)*WIDTH*HEIGHT,channels),0)
    std_t=torch.std(data.view(len(data)*WIDTH*HEIGHT,channels), 0)
    data=(data-mean_t)/std_t

    #For normalization 
    min_t=torch.min(data.view(len(data)*WIDTH*HEIGHT,channels), 0)
    max_t=torch.max(data.view(len(data)*WIDTH*HEIGHT,channels), 0)
    data=(data-min_t[0])/((max_t[0]-min_t[0]))

    return data



def distance_map_batch_v2(Y_batch,threshold=20,bins=15):
    '''
    Computes distance map of https://arxiv.org/abs/1709.05932 .
    Threshold and number of bins are parameters explained in the paper.
    Y_batch is the one hot encoded mask of the map with buildings and background. Mask for background is stored on channel 0 and Mask for buildings is stored on channel 1. Y_batch is a torch tensor
    returns: torch tensor distance map
    
    '''
    Y_batch_dist=[]
    for i in range(len(Y_batch)):
        distance_build=distance_transform_bf(np.asarray(Y_batch)[i,:,:,1],sampling=2)
        distance_background=distance_transform_bf(np.asarray(Y_batch)[i,:,:,0],sampling=2)
        distance_build=np.minimum(distance_build,threshold*(distance_build>0))
        distance_background=np.minimum(distance_background,threshold*(distance_background>0))
        distance=(distance_build-distance_background)
        distance=(distance-np.amin(distance))/(np.amax(distance)-np.amin(distance)+1e-50)*(bins-1)
        inp=torch.LongTensor(distance)
        inp_ = torch.unsqueeze(inp, len(distance.shape))
        one_hot = torch.FloatTensor(distance.shape[0], distance.shape[1], bins).zero_()
        one_hot.scatter_(len(distance.shape), inp_, 1)
        one_hot=np.asarray(one_hot)
        Y_batch_dist.append(one_hot)

    return torch.FloatTensor(np.asarray(Y_batch_dist))
   

if __name__ == '__main__':

    
    #path='../DATA_GHANA/RAW_DATA/'
    path=sys.argv[1]
    
    save_folder=path+'test_display/'
    if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
    name_labels=path+'groundtruth.png'
    
    ## Display Pansharpened
    print('Display Pansharpened')
    pansharpened_file=path+NAME_PANSHARP
    pansharpened=read_images(pansharpened_file)
    pansharpened=np.transpose(pansharpened,(1,2,0))
    plot_images(pansharpened,figsize=(8,8), plot_name='Pansharpened_Bands',add_title=False,save_path=save_folder,save_images=True)
    
    ## Display Panchromatic
    print('Display Panchromatic')
    panchromatic_file=path+NAME_PANCHRO
    panchromatic=read_images(panchromatic_file)
    panchromatic=panchromatic[:,:,newaxis]
    plot_images(panchromatic,figsize=(8,8), plot_name='Panchromatic_Band',add_title=True,save_path=save_folder,save_images=True)
    
    ## Display MS
    print('Display MS')
    ms_file=path+NAME_MS
    ms=read_images(ms_file)
    ms=np.transpose(ms,(1,2,0))
    plot_images(ms,figsize=(8,8), plot_name='Multi_Spectral_Bands',add_title=True,save_path=save_folder,save_images=True)
    
#     ## Display  Labels
    print('Display Labels')
    createRasterFromGeoJson(path+geojson_file, panchromatic_file,name_labels)
    labels=read_images(name_labels)
    labels=labels[:,:,newaxis]
    plot_images(labels,figsize=(8,8), plot_name='groundtruth',add_title=True,save_path=save_folder,save_images=True)
    
    
    
    