import os
"""Params for ADDA."""

# params for dataset and data loader
data_root_ghana = "../../2_DATA_GHANA_BALANCED/DATASET/128_x_128_8_pansh/"
data_root_spacenet = "../../SPACENET_DATA/SPACENET_DATA_PROCESSED/DATASET/128_x_128_8_bands_pansh/"
image_size = 128
nb_classes=2
input_channels=9
batch_size = 8
val_batch_size=32
test_batch_size=32

max_train_size=None#256
max_val_size=None#64
max_test_size=None#32

# params for source dataset
src_dataset = data_root_spacenet
src_encoder_restore = "../TRAINED_MODELS/RUBV3D2_final_model_spacenet.pth"
src_model_trained = True

# params for target dataset
tgt_dataset = data_root_ghana
tgt_encoder_restore = "../TRAINED_MODELS/RUBV3D2_final_model_spacenet.pth"
tgt_model_trained = True

# params for setting up classifying model
dropout=0.1#0.35
default_layers=3
default_features_root=32
default_bn=True
default_filter_width=3
default_n_resblocks=1
distance_unet=True
distance_net='v2'
bins=10
threshold=20
data_aug=None #transforms.Compose([Transform(),ToTensor()])


# params for setting up discriminative model
model_root = "../TRAINED_MODELS/"
d_size_in=16 #size patch bottom layer Unet
d_len_feature_maps=5 #number of stages Unet + dist modeul
d_input_dims=bins+240 #size concat number filters all layers expansion path Unet
d_output_dims = 2
d_model_restore = ""

# params for training network
num_epochs=2000#6
display_step=20#10
iou_step=5#2
train_dis_step=1

#saving_paths

global_path='MODEL_ADDA_2/'
if not os.path.exists(global_path):
            os.makedirs(global_path)

model_save_tgt_fold=os.path.join(global_path,'ADDA_GHANA/')
if not os.path.exists(model_save_tgt_fold):
            os.makedirs(model_save_tgt_fold)
model_save_tgt=os.path.join(model_save_tgt_fold,'RUBV3D2_model_adda_ghana')

model_save_da_fold=os.path.join(global_path,'DISCRIMINATIVE/')
if not os.path.exists(model_save_da_fold):
            os.makedirs(model_save_da_fold)
model_save_da=os.path.join(model_save_da_fold,'RUBV3D2_model_discri')


test_save=os.path.join(global_path,'TEST_SAVE/')
if not os.path.exists(test_save):
            os.makedirs(test_save)
        

tmp_iou=os.path.join(test_save,'TMP_IOU/')
if not os.path.exists(tmp_iou):
            os.makedirs(tmp_iou)    

        
loss_fn='cross-entropy'

# params for optimizing models
d_learning_rate = 1e-4
d_lr_reduc=[10,50,100,1000]
c_learning_rate = 1e-4
c_lr_reduc=[10,50,100,1000]
