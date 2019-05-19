import os
import cv2

import numpy as np

import torch
import torch.utils.data as data

SEED = 20190519
EVAL_RATIO = 0.05

class ListLoader(object):
    def __init__(self, root_path, num_classes):
        np.random.seed(SEED)

        self.image_list = []
        self.labelmap = {}
        for directory in os.walk(root_path):
            for dir_name in directory[1]: # All subdirectories
                pos = dir_name.find('.')
                type_id = int(dir_name[0:pos])
                type_name = dir_name[pos+1:]
                if type_id < 0 or type_id > num_classes:
                    print('Wrong directory: {}!'.format(dir_name))
                    continue
                self.labelmap[type_id] = type_name
                for image_file in os.listdir(os.path.join(root_path, dir_name)):
                    full_path = os.path.join(root_path, dir_name, image_file)
                    self.image_list.append((full_path, type_id))
        self.image_list = np.asarray(self.image_list)

    def image_indices(self):
        '''Return train/eval image files' list'''
        length = len(self.image_list)
        indices = np.random.permutation(length)
        point = int(length * EVAL_RATIO)
        eval_indices = indices[0:point]
        train_indices = indices[point:]
        return self.image_list, train_indices, eval_indices

    def export_labelmap(self, path='labelmap.csv'):
        with open(path, 'w') as fp:
            for type_id, type_name in self.labelmap.items():
                fp.write(str(type_id) + ',' + type_name + '\n')


class BirdsDataset(data.Dataset):
    """ All images and classes for birds through the world """
    def __init__(self, image_list, image_indices):
        self.image_list = image_list
        self.image_indices = image_indices

    def __getitem__(self, index):
        image_path, type_id = self.image_list[self.image_indices[index]]
        image = cv2.imread(image_path)
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
