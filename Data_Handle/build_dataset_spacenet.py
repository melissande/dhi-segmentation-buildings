from image_utils import *
import os
import tensorflow as tf
from numpy import newaxis

WIDTH=128
STRIDE=128

def extract_patches(sess,data,width,stride):
    '''
    Extract patches from images 
    :data input image 
    :width dimensiton of the patch
    :stride stride of patch selection on the image
    '''
    print('Patch extraction with stride=%d and width=%d begins'%(stride,width) )
    data_pl=tf.placeholder(tf.float64, [data.shape[0],data.shape[1],data.shape[2],data.shape[3]], name='data_placeholder')
    data_o=tf.extract_image_patches(images=data_pl,ksizes=[1,width,width,1],strides=[1,stride,stride,1],rates=[1,1,1,1],padding='VALID')
    print('Patch extraction done')
    size_tot=data_o.get_shape().as_list()
    data_o=tf.reshape(data_o,[size_tot[1]*size_tot[2],width,width,data.shape[3]])
    
    Data_o= sess.run(data_o,feed_dict={data_pl: data})
    print('%d patches of size %d x %d created as list'%(Data_o.shape[0],Data_o.shape[1],Data_o.shape[2]))
    return Data_o
    
path_raw='../SPACENET_DATA/SPACENET_DATA_PROCESSED/RAW_IMAGES/'

path_dataset='../SPACENET_DATA/SPACENET_DATA_PROCESSED/DATASET/128_x_128_8_bands_pansh/'
if not os.path.exists(path_dataset):
        os.makedirs(path_dataset)

training_ratio=0.8 #so    test_ratio=0.2
validation_ratio=0.2

path_panchro=[]
path_pansharp=[]
path_groundtruth=[]

for citydir in  sorted(os.listdir(path_raw)):
    if citydir.startswith('AOI_1_RIO'):
        continue
    else:
        for bandsdir in  sorted(os.listdir(os.path.join(path_raw,citydir))):
            if bandsdir.startswith('PANCHRO'):
                for filename in sorted(os.listdir(os.path.join(path_raw,citydir,bandsdir))):
                    path_panchro.append(os.path.join(path_raw,citydir,bandsdir,filename))
            if bandsdir.startswith('PANSHARP'):
                for filename in sorted(os.listdir(os.path.join(path_raw,citydir,bandsdir))):
                    path_pansharp.append(os.path.join(path_raw,citydir,bandsdir,filename))
            if bandsdir.startswith('GROUNDTRUTH'):
                for filename in sorted(os.listdir(os.path.join(path_raw,citydir,bandsdir))):
                    path_groundtruth.append(os.path.join(path_raw,citydir,bandsdir,filename))
    
    
print('Do the splitting for ORIGINAL SIZE of patches\n')    
path_panchro=np.asarray(path_panchro)
print('Length List panchro %d'%path_panchro.shape)
path_pansharp=np.asarray(path_pansharp)
print('Length List pansharp %d'%path_panchro.shape)
path_groundtruth=np.asarray(path_groundtruth)
print('Length List groundtruth %d'%path_panchro.shape)


idx_shuffle = np.arange(len(path_panchro))
np.random.shuffle(idx_shuffle)


path_panchro=path_panchro[idx_shuffle]
path_pansharp=path_pansharp[idx_shuffle]
path_groundtruth=path_groundtruth[idx_shuffle]


#Do the split
training_size=int(round(training_ratio*path_panchro.shape[0]))
test_size=path_panchro.shape[0]-training_size
validation_size=int(round(validation_ratio*training_size))
training_size=training_size-validation_size

print('Split (TRAINING - VALIDATION:%f) - TEST:%f  done'%(1-validation_ratio,training_ratio))
print('Training size:%d, Validation size:%d, Test size: %d'%(training_size,validation_size,test_size))


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
with tf.Session() as sess:
    count_tr=0        
    print('BUILD TRAINING SET')
    for i in range(training_size):
        filename=path_pansharp[i].split('pansharp_')[1]
        filename=filename.split('.h5')[0]

        panchro=read_data_h5(path_panchro[i])
        pansharp=read_data_h5(path_pansharp[i])
        groundtruth=read_data_h5(path_groundtruth[i])
        input_=np.concatenate((panchro,pansharp),axis=3)
        output_=groundtruth

        input_=extract_patches(sess,input_,WIDTH,STRIDE)
        output_=extract_patches(sess,output_,WIDTH,STRIDE)

        for j in range(input_.shape[0]):
            write_data_h5(path_dataset+'TRAINING/INPUT/input_'+filename+'_'+str(j)+'.h5',input_[j,:,:,:])
            write_data_h5(path_dataset+'TRAINING/OUTPUT/output_'+filename+'_'+str(j)+'.h5',output_[j,:,:,0])
            count_tr+=1


    print('BUILD VALIDATION SET')
    count_val=0
    for i in range(training_size,training_size+validation_size):
        filename=path_pansharp[i].split('pansharp_')[1]
        filename=filename.split('.h5')[0]

        panchro=read_data_h5(path_panchro[i])
        pansharp=read_data_h5(path_pansharp[i])
        groundtruth=read_data_h5(path_groundtruth[i])
        input_=np.concatenate((panchro,pansharp),axis=3)
        output_=groundtruth

        input_=extract_patches(sess,input_,WIDTH,STRIDE)
        output_=extract_patches(sess,output_,WIDTH,STRIDE)

        for j in range(input_.shape[0]):
            write_data_h5(path_dataset+'VALIDATION/INPUT/input_'+filename+'_'+str(j)+'.h5',input_[j,:,:,:])
            write_data_h5(path_dataset+'VALIDATION/OUTPUT/output_'+filename+'_'+str(j)+'.h5',output_[j,:,:,0])
            count_val+=1

    count_test=0

    print('BUILD TEST SET')
    for i in range(training_size+validation_size,path_panchro.shape[0]):
        filename=path_pansharp[i].split('pansharp_')[1]
        filename=filename.split('.h5')[0]

        panchro=read_data_h5(path_panchro[i])
        pansharp=read_data_h5(path_pansharp[i])
        groundtruth=read_data_h5(path_groundtruth[i])
        input_=np.concatenate((panchro,pansharp),axis=3)
        output_=groundtruth

        input_=extract_patches(sess,input_,WIDTH,STRIDE)
        output_=extract_patches(sess,output_,WIDTH,STRIDE)
        for j in range(input_.shape[0]):
            write_data_h5(path_dataset+'TEST/INPUT/input_'+filename+'_'+str(j)+'.h5',input_[j,:,:,:])
            write_data_h5(path_dataset+'TEST/OUTPUT/output_'+filename+'_'+str(j)+'.h5',output_[j,:,:,0])
            count_test+=1
    
print('Elements in Training set %d'%count_tr) 
print('Elements in Validation set %d'%count_val)   
print('Elements in Test set %d'%count_test) 