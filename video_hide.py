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

# Check if file exists
def check_file(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        sys.exit()

# Construct argument parser
parser = argparse.ArgumentParser(description='Use block shuffle')         
parser.add_argument("--model", required=True, help="path to trained model")
parser.add_argument("--secret", required=True, help="path to secret image")
parser.add_argument("--cover", required=True, help="path to cover video")
parser.add_argument("--key", required=True, type=int, help="integer key to determine frames")
args= vars(parser.parse_args())

# Check if files exist
check_file(args['model'])
check_file(args['secret'])
check_file(args['cover'])

# Load the model
model_hide=load_model(args['model'],compile=False)

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
fps=24
# Break the video into frames
def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frames.append(frame)
    cap.release()
    return frames

frames = get_frames(args['cover'])

# Get the frame rate of the original video

#original_shape = frames[0].shape

# Divide the secret image into 10 equal parts
secret_image = Image.open(args['secret']).convert('RGB')
width, height = secret_image.size
secret_parts = [secret_image.crop((0, i*height//12, width, (i+1)*height//12)) for i in range(12)]

# Create a VideoWriter object with the same fps as the original video
container_outvid = cv2.VideoWriter('cover_outvid_224.avi',cv2.VideoWriter_fourcc('H','F','Y','U'), fps, (224, 224))
# Define the frames to be altered based on the key
np.random.seed(args['key'])
key_frames_indices = np.sort(np.random.choice(len(frames), 12, replace=False))

# Process each frame
for idx in tqdm(range(len(frames)), desc="Processing frames"):
    frame = frames[idx]
    frame = cv2.resize(frame, (224, 224))
    if idx in key_frames_indices:
        # Convert frame to RGB and reshape
        secret =secret_parts[list(key_frames_indices).index(idx)]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        secret = secret.resize((224, 224))
        coverin = np.array(frame.convert('RGB')).reshape(1,224,224,3)/255.0
        secretin = np.array(secret.convert('RGB')).reshape(1,224,224,3)/255.0
        # Predict the output       
        coverout=model_hide.predict([normalize_batch(secretin),normalize_batch(coverin)])

        # Postprocess the output
        coverout = denormalize_batch(coverout)
        coverout=np.squeeze(coverout)*255.0
        coverout=np.uint8(coverout)

        # Resize the output frame back to the original shape
        coverout = cv2.cvtColor(coverout, cv2.COLOR_RGB2BGR)
        # Write frame to video
        container_outvid.write(coverout)
    else:
        # Convert frame to RGB before writin
        container_outvid.write(frame)

container_outvid.release()