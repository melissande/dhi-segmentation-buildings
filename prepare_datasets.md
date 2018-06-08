# Preparation of the datasets: Ghana dataset and Spacenet dataset

## 1) Ghana dataset

### Images available and rasterization of geojson

Pansharpened, multi spectral bands and panchromatic are raster bands availables under .tif files. A geojson file can be extraced from shape files available of the buildings using QGIS.

Create a folder where to store the raw data, let's say `DATA_GHANA/RAW_DATA/`. In this folder, stored:
-panchro.tif: panchromatic image
-pansharp.tif: pansharpened image
-ms.tif: multispectral bands image
-buildings.geojson: geojson file with building footprints stored as polygons

To produce a groundtruth map, run, on a cuda device:
```sh
$python Data_Handle/image_utils.py DATA_GHANA/RAW_DATA/
```
 which will display some bands of the images and create the groundtruth map as `RAW_DATA/groundtruth.png`
### Patches extraction

As the image provided is really big (around 13000x10000), patches have to be extracted to feed the neural network. Indeed, if the entire image was used, the computations would be huge and the program would end up crashing. I did a script that makes the size of patches to be chosen by the operator. Patches are stored for each band under \textbf{.h5} format, and MS bands are upsampled. One could ask why I didnt' stack all the bands but again, it is too memory demanding. Some parts of the big input image are actually not images but only nan values as the image is squared but the mapping of the area in Ghana is not. I discarded patches extracted from these areas considering that inputing these patches won't disturb the network (it will easily learn that 0 in input is 0 in output) but it will impact the evalution as the loss will be decreased as the network learns "super well" that 0 gives 0. 

To produce patches, run, on a cuda device:
```sh
$python Data_Handle/patches_extraction_ghana.py DATA_GHANA/RAW_DATA/ DATA_GHANA/RAW_PATCHES/128_x_128/ --width_patch=128
```

It will by default create patches of 128x128 but it the size of the patches can be modified by tunning `--width_patch`. Square patches are selected all around the image and don't overlap.

### Building sets

These patches have to be separated into 3 datasets: Training, Validation and Test. To do so, run:

```sh
$python Data_Handle/build_dataset_ghana.py DATA_GHANA/RAW_PATCHES/128_x_128/ DATA_GHANA/DATASET/ --training_ratio=0.8 --validation_ratio=0.2
```

Careful, it will create `DATA_GHANA/DATASET/128_x_128_8_pansh/` containing the dataset ready to use. So the name is hard coded and has to be changed inside the script.`--training_ratio=0.8 --validation_ratio=0.2` can be changed.
All patches,from all bands, are stacked patch per patch and located in an 'INPUT' folder whereas the groundtruth corresponding patches will be located in an 'OUTPUT' folder, both located into a 'DATASET' folder with a prepared split into 3 folders: 'TRAINING', 'VALIDATION' and 'TEST'. These patches are shuffled before being split.

## 2) Spacenet dataset


SpaceNet Dataset for buildings is composed of data avaiable for Round 1 and 2.

 - Rio de Janeiro: spatial resolution of 50 cm, 6940 patches of 438-9 x 406-7 and RGB pansharpened bands + 8 Multi Spectral bands.
 - Las Vegas: spatial resolution of 30 cm, 3851 patches of 676 x 730 and Panchromatic band + 8 Pansharpened bands.
 - Paris: spatial resolution of 30 cm, 1148 patches of 676 x 730 and Panchromatic band + 8 Pansharpened bands.
 - Shannghai: spatial resolution of 30 cm, 4582 patches of 676 x 730 and Panchromatic band + 8 Pansharpened bands.
 - Khartoum: spatial resolution of 30 cm, 1012 patches of 676 x 730 and Panchromatic band + 8 Pansharpened bands. 


### Preprocessing of Spacenet dataset
The dataset from Ghana has a resolution per pixel of 50 cm and we decided to use Panchromatic band + 8 Pansharpened bands based on previous experiences. We therefore, for Rio de Janeiro, had to upsample the 8 Multi Spectral bands to the size of the pansharpened and to combine the 3 bands RGB into gray scale. For the other cities, the images had to be downsample to 3/5 to achieve the 50 cm resolution. Groundtruth raster images had to be generated from geojson files available too. I stored this data in a folder called `SPACENET_DATA_PROCESSED/RAW_IMAGES/` with subfolder of cities. In each city subfolder, 3 folders can be found: `PANCHROMATIC/`, `PANSHARPENED/`, `GROUNDTRUTH/` containing respectively, the panchromatic images, pansharpened images and groundtruth images which the rasterized versions of the geojson original files.

These actions are done by running, on CUDA device:
```sh
$ python Data_Handle/spacenet_data_preprocessing.py SPACENET_DATA/BUILDING_DATASET/  SPACENET_DATA_PROCESSED/
```
`SPACENET_DATA/BUILDING_DATASET/ ` is the folder where have been extracted Spacenet Dataset from AWS (see README.md)


### Patch extraction of Spacenet dataset

As we investigated that using patches of 128x128 for the dataset of Ghana was the best, I generated a split of the available data in `RAW_IMAGES/` into `TRAINING/`, `VALIDATION/` and `TEST/` set with the split (0.8 TRAINING -0.2 VALIDATION)-0.2 TEST with patches of 128x128 extracted. Inside these data folders, `INPUT/` and `OUTPUT/` folders can be found with respectively panchromatic+pansharpened combined and groundtruth inside. I took care of keeping the original name of the cities in the name of the files to be able to deduce conclusions in data visualization.

The paths are hardcoded but running the following:
```sh
$ python Data_Handle/build_dataset_spacenet.py
```
produces the final dataset at the path `SPACENET_DATA_PROCESSEDDATASET/128_x_128_8_bands_pansh/` taking input images from `sSPACENET_DATA_PROCESSED/RAW_IMAGES/` 
