import os
import glob
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras
from keras.models import Model
from keras.layers import Dense, Input, concatenate, Conv2D, GaussianNoise
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import keras.backend as K
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.models import load_model
from keras.utils import multi_gpu_model
from PIL import Image
import matplotlib.pyplot as plt
from random import randint
import imageio
from io import StringIO, BytesIO

# Configure file paths
TRAIN="dataset/train_data"
VALIDATION="dataset/val_data"
TRAIN_NUM=len(glob.glob(TRAIN+"/*/*"))
VAL_NUM=len(glob.glob(VALIDATION+"/*/*"))
TEST_DATA="test/testdata.npy"

CHECKPOINT="checkpoints/steg_model-{epoch:02d}-{val_loss:.2f}.hdf5"
PRETRAINED='checkpoints/steg_model-04-0.03.hdf5'

# Configure batch size
BATCH_SIZE=12


# Load test data as numpu arrays
test_images=np.load(TEST_DATA)

# Sample test data
tsecret = test_images[0].reshape((1,224,224,3))
tcover = test_images[1].reshape((1,224,224,3))

# Preprocessing functions
def normalize_batch(imgs):
    '''Performs channel-wise z-score normalization'''

    return (imgs -  np.array([0.485, 0.456, 0.406])) /np.array([0.229, 0.224, 0.225])
                                                        
def denormalize_batch(imgs,should_clip=True):
    '''Denormalize the images for prediction'''

    imgs= (imgs * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    
    if should_clip:
        imgs= np.clip(imgs,0,1)
    return imgs

# Image data generator
input_imgen = ImageDataGenerator(rescale = 1./255)
test_imgen = ImageDataGenerator(rescale = 1./255)

# Custom generator for loading training images from directory
def generate_generator_multiple(generator, direc):
    genX1 = generator.flow_from_directory(direc, target_size=(224,224), batch_size = BATCH_SIZE, shuffle=True, seed=3, class_mode=None)
    genX2 = generator.flow_from_directory( direc, target_size=(224, 224), batch_size = BATCH_SIZE, shuffle=True, seed=8, class_mode=None)
    
    while True:
            X1i = normalize_batch(genX1.next()) 
            X2i = normalize_batch(genX2.next())

            yield( {'secret': X1i, 'cover': X2i}, {'hide_conv_f': X2i, 'revl_conv_f': X1i})  #Yield both images and their mutual label
            
# Train data generator            
inputgenerator=generate_generator_multiple(generator=input_imgen, direc=TRAIN)
       
# Validation data generator     
testgenerator=generate_generator_multiple(test_imgen, direc=VALIDATION)

# Loss functions
def custom_loss_1(secret,secret_pred):
    
    # Compute L2 loss(MSE) for secret image
    secret_mse = keras.losses.mean_squared_error(secret,secret_pred)
    return secret_mse

def custom_loss_2(cover,cover_pred):

    # Compute L2 loss(MSE) for cover image
    cover_mse = keras.losses.mean_squared_error(cover,cover_pred)
    return cover_mse 

# Custom loss dictionary
losses = {
	"hide_conv_f": custom_loss_2,
	"revl_conv_f": custom_loss_1,
}

# Loss weights
lossWeights = {"hide_conv_f": 1.0, "revl_conv_f": 00.75}    

def steg_model(pretrain=False):

    if(pretrain):
        model=load_model(PRETRAINED,custom_objects={'custom_loss_1': custom_loss_1, 'custom_loss_2': custom_loss_2})
        return model
    
    # Inputs
    secret = Input(shape=(224,224,3), name='secret')
    cover = Input(shape=(224,224,3), name='cover')

    # Prepare Network 
    x = Conv2D(32, kernel_size=3, padding='same')(secret)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = SpatialDropout2D(0.1)(x)
    
    x = Conv2D(32, kernel_size=4, padding='same')(x)  
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = SpatialDropout2D(0.1)(x)

    x = Conv2D(32, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x) 
    prepared = SpatialDropout2D(0.1)(x)
    
    # Hiding Network
    x = concatenate([cover, prepared])
    
    x = Conv2D(32, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = SpatialDropout2D(0.1)(x)
    
    x = Conv2D(32, kernel_size=4, padding='same')(x)
    x = BatchNormalization()(x)  
    x = LeakyReLU()(x)
    x = SpatialDropout2D(0.1)(x)

    x = Conv2D(32, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = SpatialDropout2D(0.1)(x)
    
    cover_pred = Conv2D(3, kernel_size=1, padding='same', activation='sigmoid', name='hide_conv_f')(x)
    
    # Noise Layer
    noised_cover = GaussianNoise(0.5)(cover_pred)
    
    # Reveal Network 
    x = concatenate([noised_cover, prepared])
    
    x = Conv2D(32, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = SpatialDropout2D(0.1)(x)
    
    x = Conv2D(32, kernel_size=4, padding='same')(x)  
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = SpatialDropout2D(0.1)(x)

    x = Conv2D(32, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    secret_pred = Conv2D(3, kernel_size=1, padding='same', activation='sigmoid', name='revl_conv_f')(x)
    
    model = Model(inputs=[secret, cover], outputs=[cover_pred, secret_pred])
    
    model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights)
    
    return model


# Model object
model = steg_model(pretrain=False)

# Summarize layers
print(model.summary())

# Plot graph
plot_model(model, to_file='steg_model.png')

# Tensorboard 
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True)
   
# TF file writer for images 
im_writer = tf.summary.FileWriter('./logs/im')
 
# TF image logger 
def log_images(tag, images, step):
     """Logs a list of images."""

     im_summaries = []
     for nr, img in enumerate(images):
            
            # Write the image to a string
            s = BytesIO()
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

     # Create and write Summary
     summary = tf.Summary(value=im_summaries)
     im_writer.add_summary(summary, step)

# Custom keras image callback
class TensorBoardImage(Callback):
    def __init__(self, tag):
        super().__init__() 
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):

        # Load random test images        
        secretin = test_images[np.random.choice(len(test_images), size=4, replace=False)]
        coverin = test_images[np.random.choice(len(test_images), size=4, replace=False)]
        
        # Predict on batch
        coverout, secretout=model.predict([normalize_batch(secretin),normalize_batch(coverin)])

        # Postprocess output cover image
        coverout = denormalize_batch(coverout)
        coverout=np.squeeze(coverout)*255.0
        coverout=np.uint8(coverout)
        
        # Postprocess output secret image
        secretout=denormalize_batch(secretout)
        secretout=np.squeeze(secretout)*255.0
        secretout=np.uint8(secretout)
        
        # Convert images to UINT8 format (0-255)
        coverin=np.uint8(np.squeeze(coverin*255.0))
        secretin=np.uint8(np.squeeze(secretin*255.0))
        
        # Log image summary
        log_images("cover_in", coverin, epoch)
        log_images("secret_in", secretin, epoch)
        log_images("cover_out", coverout, epoch)
        log_images("secret_out", secretout, epoch)
        
        return

    
# Custom image logger
image_summary = TensorBoardImage('Image Example')

#Checkpoint path
filepath=CHECKPOINT

# Callback functions
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=False , save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.000001, verbose=1)
callbacks_list = [checkpoint, tensorboard, image_summary, reduce_lr]


# Train the model
model.fit_generator(inputgenerator, epochs=100, 
                          steps_per_epoch = TRAIN_NUM//BATCH_SIZE,
                          validation_data=testgenerator, 
                          validation_steps=VAL_NUM//BATCH_SIZE,
                          use_multiprocessing=True, 
                          callbacks=callbacks_list)


'''
Sample run: python train.py
[Ensure the file and directory paths are properly configured]
'''
