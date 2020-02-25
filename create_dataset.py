
from PIL import Image
import os, sys
from shutil import copyfile
import numpy as np
import datasets

# Resize images.
def resize(path, size=64):
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((size,size), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG')

for folder in ['./dataset_32x32/M/', './dataset_32x32/F/']:
    resize(folder, size=32)

datasets.save_data(data_dir='dataset_32x32', save_file='dataset_32x32.mat', target_size=(32,32))
