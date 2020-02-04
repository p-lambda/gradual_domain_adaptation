
from PIL import Image
import os, sys
from shutil import copyfile
import numpy as np
import datasets

# Make source and target datasets.
# num_train = 5000
# num_test = 1000
# per_gender_tr = int(num_train / 2)
# per_gender_te = int(num_test / 2)
# per_gender = per_gender_tr + per_gender_te
# female_photos = sorted(os.listdir('./F'))
# male_photos = sorted(os.listdir('./M'))
# old_f_photos, new_f_photos = female_photos[:per_gender], female_photos[-per_gender:]
# old_m_photos, new_m_photos = male_photos[:per_gender], male_photos[-per_gender:]
# for dataset_name, class_name, data_array in zip(
#     ['./old_images/', './new_images/', './old_images/', './new_images/'],
#     ['F/', 'F/', 'M/', 'M/'],
#     [old_f_photos, new_f_photos, old_m_photos, new_m_photos]):
#     image_names = data_array
#     np.random.shuffle(image_names)
#     train_names = image_names[:per_gender_tr]
#     test_names = image_names[per_gender_tr:]
#     assert(len(test_names) == per_gender_te)
#     for file in train_names:
#         copyfile('./' + class_name + file, dataset_name + 'train/' + class_name + file)
#     for file in test_names:
#         copyfile('./' + class_name + file, dataset_name + 'test/' + class_name + file)

# print(len(old_f_photos))
# for file in old_f_photos:
#     copyfile('./F/' + file, './old_images/F/' + file)
# for file in new_f_photos:
#     copyfile('./F/' + file, './new_images/F/' + file)
# for file in old_m_photos:
#     copyfile('./M/' + file, './old_images/M/' + file)
# for file in old_m_photos:
#     copyfile('./M/' + file, './new_images/M/' + file)

# # Resize images.
# def resize(path, size=64):
#     dirs = os.listdir(path)
#     for item in dirs:
#         if os.path.isfile(path+item):
#             im = Image.open(path+item)
#             f, e = os.path.splitext(path+item)
#             imResize = im.resize((size,size), Image.ANTIALIAS)
#             imResize.save(f + '.png', 'PNG')

# # for i in ['./old_images/', './new_images/']:
# #     for j in ['train/', 'test/']:
# #         for k in ['F/', 'M/']:
# #             resize(i + j + k)

# for folder in ['./dataset_64x64/M/', './dataset_64x64/F/']:
#     resize(folder, size=64)

datasets.save_data(data_dir='dataset_64x64', save_file='dataset_64x64.mat', target_size=(64,64))