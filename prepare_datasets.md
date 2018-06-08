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
`sh
$python Data_Handle/image_utils.py DATA_GHANA/RAW_DATA/
`
 which will display some bands of the images and create the groundtruth map as `RAW_DATA/groundtruth.png`
### Patches extraction

As the image provided is really big (around 13000x10000), patches have to be extracted to feed the neural network. Indeed, if the entire image was used, the computations would be huge and the program would end up crashing. I did a script that makes the size of patches to be chosen by the operator. Patches are stored for each band under \textbf{.h5} format, and MS bands are upsampled. One could ask why I didnt' stack all the bands but again, it is too memory demanding. Some parts of the big input image are actually not images but only nan values as the image is squared but the mapping of the area in Ghana is not. I discarded patches extracted from these areas considering that inputing these patches won't disturb the network (it will easily learn that 0 in input is 0 in output) but it will impact the evalution as the loss will be decreased as the network learns "super well" that 0 gives 0. 

To produce patches, run, on a cuda device:
`sh
$python Data_Handle/patches_extraction_ghana.py DATA_GHANA/RAW_DATA/ DATA_GHANA/RAW_PATCHES/128_x_128/ --width_patch=128
`

It will by default create patches of 128x128 but it the size of the patches can be modified by tunning `--width_patch`. Square patches are selected all around the image and don't overlap.

### Building sets

These patches have to be separated into 3 datasets: Training, Validation and Test. To do so, run:

`sh
$python Data_Handle/build_dataset_ghana.py DATA_GHANA/RAW_PATCHES/128_x_128/ DATA_GHANA/DATASET/ --training_ratio=0.8 --validation_ratio=0.2
`

Careful, it will create `DATA_GHANA/DATASET/128_x_128_8_pansh/` containing the dataset ready to use. So the name is hard coded and has to be changed inside the script.`--training_ratio=0.8 --validation_ratio=0.2` can be changed.


