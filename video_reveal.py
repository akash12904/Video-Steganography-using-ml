import numpy as np
import keras
import sys
import cv2
from keras.models import Model
from keras.utils import plot_model
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from random import randint
import imageio
import argparse
from tqdm import tqdm
import os

# Construct argument parser
parser = argparse.ArgumentParser(description='Decode steganographic video')        
parser.add_argument("--model", required=True, help="path to trained model")
parser.add_argument("--stego", required=True, help="path to steganographic video")
parser.add_argument("--key", required=True, type=int, help="integer key to determine frames")
args= vars(parser.parse_args())

# Load the model
model_reveal=load_model(args['model'],compile=False)

# Normalize inputs
def normalize_batch(imgs):
    '''Performs channel-wise z-score normalization'''

    return (imgs -  np.array([0.485, 0.456, 0.406])) /np.array([0.229, 0.224, 0.225])

# Denormalize outputs
def denormalize_batch(imgs,should_clip=True):
    imgs= (imgs * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    
    if should_clip:
        imgs= np.clip(imgs,0,1)
    return imgs

# Break the video into frames
cap = cv2.VideoCapture(args['stego'])
frames = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    frames.append(frame)
cap.release()

# Define the frames to be altered based on the key
np.random.seed(args['key'])
key_frames_indices = np.sort(np.random.choice(len(frames), 12, replace=False))

# Process each frame
secret_parts = []
for idx in tqdm(range(len(frames)), desc="Decoding frames"):
    frame = frames[idx]
    if idx in key_frames_indices:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        stegoin = np.array(frame).reshape(1,224,224,3)/255.0

        # Predict the output       
        secretout=model_reveal.predict([normalize_batch(stegoin)])

        # Postprocess the output
        secretout = denormalize_batch(secretout)
        secretout=np.squeeze(secretout)*255.0
        secretout=np.uint8(secretout)

        # Save the secret part
        secret_parts.append(secretout)

secret_image = np.vstack(secret_parts)
secret_image = cv2.cvtColor(secret_image, cv2.COLOR_GRAY2BGR) if len(secret_image.shape) == 2 else secret_image

# Resize to 224x224
resized_image = cv2.resize(secret_image, (224, 224))

# Convert to BGR and save the resized image
resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('secret_image_224x224.png', resized_image)