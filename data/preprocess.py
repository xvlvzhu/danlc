import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE
import keras
import os
import tifffile as tiff
from utils.utils import *

batch_size = 64
num_classes = 5
epochs = 10
num = 7000
# input image dimensions
img_rows, img_cols = 64, 64

source_x = np.zeros([num, img_rows, img_cols, 6])
source_y = []
target_x = np.zeros([num, img_rows, img_cols, 6])
target_y = []

source_dir = '../../dataset/paper_datasets_1/haihe_64'
target_dir = '../../dataset/paper_datasets_1/yellow_64'
i = 0
j = 0
for class_dir in os.listdir(source_dir):
    dir_path = os.path.join(source_dir, class_dir)
    # print(dir_path)
    for image in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image)
        img = tiff.imread(image_path)
        img = add_index_layer(img,append=True)
        source_x[j, :, :, :] = img
        source_y.append(i)
        j = j + 1
    i = i + 1

i = 0
j = 0
for class_dir in os.listdir(target_dir):
    dir_path = os.path.join(target_dir, class_dir)
    # print(dir_path)
    for image in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image)
        img = tiff.imread(image_path)
        img = add_index_layer(img,append=True)
        target_x[j, :, :, :] = img
        target_y.append(i)
        j = j + 1
    i = i + 1
# x = np.concatenate(x)
source_y = np.array(source_y).reshape(-1, 1)
target_y = np.array(target_y).reshape(-1, 1)

source_y = keras.utils.to_categorical(source_y, num_classes)
target_y = keras.utils.to_categorical(target_y, num_classes)

source_x = source_x.astype('float32')
target_x = target_x.astype('float32')
source_x /= 255
target_x /= 255
print(source_x.shape, 'train samples')
print(target_x.shape, 'test samples')

print(source_y.shape)
print(target_x.shape)

np.save('source_x_paper.npy',source_x)
np.save('target_x_paper.npy',target_x)
np.save('source_y_paper.npy',source_y)
np.save('target_y_paper.npy',target_y)