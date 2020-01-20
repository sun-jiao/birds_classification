import argparse
import cv2
import numpy as np
import os
import queue
import torch
import threading

from efficientnet_pytorch import EfficientNet

BATCH_SIZE = 32
NR_THREADS = 3
request_queue = queue.Queue(maxsize=256)


class FileThread(threading.Thread):
    def __init__(self, file_list, start, stride):
        threading.Thread.__init__(self)
        self._file_list = file_list
        self._start = start
        self._stride = stride

    def run(self):
        total = len(self._file_list)
        index = self._start
        while (index < total):
            pair = self._file_list[index]
            img = cv2.imread(pair[0])
            img = cv2.resize(img, (200, 200))
            request_queue.put((img, pair[1]))
            index += self._stride
        request_queue.put(("Finish", "Finish"))


class EvalThread(threading.Thread):
    def __init__(self, net, stride):
        threading.Thread.__init__(self)
        self._net = net
        self._stride = stride
        self._statistics = {}

    def _process(self, img_batch, label_batch):
        input = torch.from_numpy(np.asarray(img_batch))
        result = self._net(input.permute(0, 3, 1, 2).float())
        values, indices = torch.max(result, 1)
        print(indices, label_batch)

        for index in range(len(label_batch)):
            type_id = label_batch[index]
            predict = indices[index]

            if type_id not in self._statistics:
                self._statistics[type_id] = {}
                self._statistics[type_id]["total"] = 0
                self._statistics[type_id]["correct"] = 0

            entry = self._statistics[type_id]
            entry["total"] += 1
            if predict == type_id:
                entry["correct"] += 1

    def run(self):
        finish = 0
        img_batch = []
        label_batch = []

        while (True):
            obj = request_queue.get()
            img = obj[0]
            type_id = obj[1]
            if img == "Finish":
                finish += 1
                if finish >= self._stride:
                    if len(img_batch) > 0:
                        self._process(img_batch, label_batch)
                    break
            else:
                label_batch.append(type_id)
                img_batch.append(img)
                if len(img_batch) < BATCH_SIZE:
                    continue
                self._process(img_batch, label_batch)
                img_batch = []
                label_batch = []

    def dump(self):
        for type_name, values in self._statistics.items():
            percentage = values["correct"] / values["total"]
            print(type_name, values, percentage)


def get_file_list(root_path):
    file_list = []

    # limit = 0
    for directory in os.walk(root_path):
        for dir_name in directory[1]:  # All subdirectories
            pos = dir_name.find('.')
            type_id = int(dir_name[0:pos])
            # type_name = dir_name[pos+1:]

            count = 0
            for image_file in os.listdir(os.path.join(root_path, dir_name)):
                count += 1
            if count < 10:
                continue

            # limit += 1
            # if limit > 10:
            #    break

            for image_file in os.listdir(os.path.join(root_path, dir_name)):
                full_path = os.path.join(root_path, dir_name, image_file)
                file_list.append((full_path, type_id))

    return file_list


def eval_traindata(args):
    net = EfficientNet.from_name('efficientnet-b2', override_params={'num_classes': 11000})
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()

    file_list = get_file_list(args.dataset_root)

    thread_list = []
    for index in range(NR_THREADS):
        thread = FileThread(file_list, index, NR_THREADS)
        thread.start()
        thread_list.append(thread)

    eval_thread = EvalThread(net, NR_THREADS)
    eval_thread.start()

    eval_thread.join()
    eval_thread.dump()

    for thread in thread_list:
        thread.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/media/data2/i18n/V3',
                        type=str, help='Root path of data')
    parser.add_argument('--trained_model', default='ckpt/bird_cls_0.pth',
                        type=str, help='Trained ckpt file path to open')
    args = parser.parse_args()

    eval_traindata(args)
