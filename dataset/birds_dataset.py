import os
import cv2
import time
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

import torch.utils.data as data
from collections import Counter

SEED = 20190519
EVAL_RATIO = 0.05

ia.seed(int(time.time()))

seq = iaa.Sequential([
         iaa.Affine(rotate=(-45, 45)),
         iaa.CropAndPad(percent=(-0.10, 0.10)),
         iaa.Fliplr(0.5),
])
'''iaa.OneOf([
    iaa.ElasticTransformation(sigma=0.25, alpha=(0.01, 0.6)),
    iaa.PerspectiveTransform(scale=(0.01, 0.10)),
    iaa.PiecewiseAffine(scale=(0.01, 0.03)),
 ]),
   iaa.OneOf([
    iaa.GammaContrast(gamma=(0.8, 1.2)),
    iaa.SigmoidContrast(cutoff=(0.4, 0.6), gain=(5, 10)),
    iaa.Multiply((0.8, 1.2)),
    iaa.Add((-20, 20)),
 ]),
])'''


class ListLoader(object):
    def __init__(self, root_path, num_classes):
        np.random.seed(SEED)

        self.category_count = Counter()  # number of images for each category
        self.image_list = []
        self.labelmap = {}
        for directory in os.walk(root_path):
            for dir_name in directory[1]:  # All subdirectories
                pos = dir_name.find('.')
                type_id = int(dir_name[0:pos])
                type_name = dir_name[pos+1:]
                if type_id < 0 or type_id > num_classes:
                    print('Wrong directory: {}!'.format(dir_name))
                    continue
                self.labelmap[type_id] = type_name
                for image_file in os.listdir(os.path.join(root_path, dir_name)):
                    self.category_count[type_id] += 1
                    full_path = os.path.join(root_path, dir_name, image_file)
                    self.image_list.append((full_path, type_id))

        avg_count = sum(self.category_count.values()) / len(self.category_count)
        print('Avg count per category:', avg_count)
        self.category_multiple = {}
        small_cat = 0
        for type_id in self.category_count:
            multiple = int(avg_count / self.category_count[type_id])
            if multiple > 1:
                small_cat += 1
            self.category_multiple[type_id] = multiple
        print('Small categories:', small_cat)

    def image_indices(self):
        '''Return train/eval image files' list'''
        length = len(self.image_list)
        indices = np.random.permutation(length)
        point = int(length * EVAL_RATIO)
        eval_indices = indices[0:point]
        train_indices = indices[point:]

        # For categories which have small number of images, oversample it
        extra_train_indices = []
        for index in train_indices:
            _, type_id = self.image_list[index]
            multiple = self.category_multiple[type_id]
            if multiple > 1:
                for i in range(multiple):
                    extra_train_indices.append(index)
        train_indices = np.concatenate((train_indices, extra_train_indices))

        return self.image_list, train_indices, eval_indices

    def multiples(self):
        return self.category_multiple

    def export_labelmap(self, path='labelmap.csv'):
        with open(path, 'w') as fp:
            for type_id, type_name in self.labelmap.items():
                fp.write(str(type_id) + ',' + type_name + '\n')


class BirdsDataset(data.Dataset):
    """ All images and classes for birds through the world """
    def __init__(self, image_list, image_indices, category_multiple, is_training):
        self.image_list = image_list
        self.image_indices = image_indices
        self.category_multiple = category_multiple
        self.is_training = is_training

    def __getitem__(self, index):
        image_path, type_id = self.image_list[self.image_indices[index]]
        image = cv2.imread(image_path)
        if isinstance(type_id, str):
            print('What:', type_id)
        if self.is_training:
            multiple = self.category_multiple[type_id]
            # Only augment small categories
            if multiple > 1:
                image = seq.augment_image(image)
                # imgaug may fracture numpy array
                image = np.ascontiguousarray(image)
        return image, int(type_id)

    def __len__(self):
        return len(self.image_indices)


if __name__ == '__main__':
    list_loader = ListLoader('/media/data2/i18n/V1', 11000)
    img_list, train_lst, eval_lst = list_loader.image_indices()
    print('train_lst', train_lst, len(train_lst))
    print('eval_lst', eval_lst, len(eval_lst))

    bd = BirdsDataset(img_list, eval_lst)
    image, type_id = bd[1023]
    print('image', image)
    print('type_id', type_id, type(type_id))
