# Virtual environments on cluster

 

## Basic libraries
Run
```sh
$ conda create -n env_thales python=3.6 numpy pip
$ source activate env_dhi
$ pip install scipy
$ pip install matplotlib
$ pip install h5py
$ pip install tensorflow-gpu 
$ conda install -c menpo opencv
$ pip install pandas
$ conda install gdal
$ conda install -c ioos rtree 
$ pip install centerline
$ pip install osmnx
$ pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
$ pip install torchvision
```
pay attention to the cuda version installed, you need to know what version of tensorflow-gpu and cuda/cdnn is corresponding to 								add it to the bashrc 

### Add to bashrc for TensorFlow 1.5 is expecting Cuda 9.0 ( NOT 9.1 ), as well as cuDNN 7
```sh
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-9.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-9.0
export LD_LIBRARY_PATH=/usr/local/cuDNNv7.0-8/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
```




## Create jupyter notebook
### For jupyter notebook
```sh
$pip install ipykernel
$python -m ipykernel install --user --name=env_dhi
```

### On the cluster
```sh
$CUDA_VISIBLE_DEVICES=0 jupyter notebook --no-browser --port=8888
```
### From local machine
```sh
$ssh -N -f -L localhost:8881:localhost:8888 s161362@mnemosyne.compute.dtu.dk
```
## AWS to download Spacenet Dataset

AWS create account
Get the credentials keys on the desktop of AWS online
create a bucket and make it "requester payer" (see: https://docs.aws.amazon.com/fr_fr/AmazonS3/latest/dev/configure-requester-pays-console.html )
install aws console:
```sh
$ pip install awscli
```
put credentials connection info  (only put key and secret key, the rest do enter) 
```sh
$ aws configure
```
check what is in the bucket spaceNet
```sh
$ aws s3 ls spacenet-dataset --request-payer requester
```
get the list of what is in the bucket 
```sh
$ aws s3api list-objects --bucket spacenet-dataset --request-payer requester
```

Download Building Dataset Spacenet
### Rio
```sh
$ aws s3api get-object --bucket spacenet-dataset \
    --key AOI_1_Rio/processedData/processedBuildingLabels.tar.gz \
    --request-payer requester /scratch/SPACENET_DATA/BUILDING_DATASET/AOI_1_RIO/processedBuildingLabels.tar.gz
```
### Vegas
#### Train
```sh
$ aws s3api get-object --bucket spacenet-dataset \
    --key AOI_2_Vegas/AOI_2_Vegas_Train.tar.gz \
    --request-payer requester /scratch/SPACENET_DATA/BUILDING_DATASET/AOI_2_Vegas/AOI_2_Vegas_Train.tar.gz
```
#### Test
```sh
$ aws s3api get-object --bucket spacenet-dataset \
    --key AOI_2_Vegas/AOI_2_Vegas_Test_public.tar.gz \
    --request-payer requester /scratch/SPACENET_DATA/BUILDING_DATASET/AOI_2_Vegas/AOI_2_Vegas_Test_public.tar.gz
```
### Paris
#### Train
```sj
$ aws s3api get-object --bucket spacenet-dataset \
    --key AOI_3_Paris/AOI_3_Paris_Train.tar.gz \
    --request-payer requester /scratch/SPACENET_DATA/BUILDING_DATASET/AOI_3_Paris/AOI_3_Paris_Train.tar.gz

```
#### Test
```sh
$ aws s3api get-object --bucket spacenet-dataset \
    --key AOI_3_Paris/AOI_3_Paris_Test_public.tar.gz \
    --request-payer requester /scratch/SPACENET_DATA/BUILDING_DATASET/AOI_3_Paris/AOI_3_Paris_Test_public.tar.gz
```
### Shanghai
#### Train
```sh
$ aws s3api get-object --bucket spacenet-dataset \
    --key AOI_4_Shanghai/AOI_4_Shanghai_Train.tar.gz \
    --request-payer requester /scratch/SPACENET_DATA/BUILDING_DATASET/AOI_4_Shanghai/AOI_4_Shanghai_Train.tar.gz
```
#### Test
```sh
$ aws s3api get-object --bucket spacenet-dataset \
    --key AOI_4_Shanghai/AOI_4_Shanghai_Test_public.tar.gz \
    --request-payer requester /scratch/SPACENET_DATA/BUILDING_DATASET/AOI_4_Shanghai/AOI_4_Shanghai_Test_public.tar.gz
```
### Karthoum
#### Train
```sh
$ aws s3api get-object --bucket spacenet-dataset \
    --key AOI_5_Khartoum/AOI_5_Khartoum_Train.tar.gz \
    --request-payer requester /scratch/SPACENET_DATA/BUILDING_DATASET/AOI_5_Khartoum/AOI_5_Khartoum_Train.tar.gz

```
#### Test
```sh
$ aws s3api get-object --bucket spacenet-dataset \
    --key AOI_5_Khartoum/AOI_5_Khartoum_Test_public.tar.gz \
    --request-payer requester /scratch/SPACENET_DATA/BUILDING_DATASET/AOI_5_Khartoum/AOI_5_Khartoum_Test_public.tar.gz
```

