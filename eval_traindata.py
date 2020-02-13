import argparse
import cv2
import numpy as np
import os
import sys
import queue
import torch
import time
import threading

from efficientnet_pytorch import EfficientNet

BATCH_SIZE = 32
NR_THREADS = 3
request_queue = queue.Queue(maxsize=128)


class FileThread(threading.Thread):
    def __init__(self, file_list, start, stride):
        threading.Thread.__init__(self)
        self._file_list = file_list
        self._start = start
        self._stride = stride

    def run(self):
        total = len(self._file_list)
        index = self._start - self._stride

        while (True):
            index += self._stride
            if index >= total:
                break
            pair = self._file_list[index]
            img = cv2.imread(pair[0])
            # img = cv2.resize(img, (200, 200))
            if img is None:
                print("[Error]", pair[0], file=sys.stderr)
                continue
            if img.shape != (300, 300, 3):
                print("[Warn]", file=sys.stderr)
                continue
            request_queue.put((img, pair[1], pair[0]))  # image_binary, label, image_path

        request_queue.put(("Finish", "Finish"))


class EvalThread(threading.Thread):
    def __init__(self, net, stride):
        threading.Thread.__init__(self)
        self._net = net
        self._stride = stride
        self._statistics = {}

    def _process(self, img_batch, label_batch, path_batch):
        input = torch.from_numpy(np.asarray(img_batch)).cuda()
        result = self._net(input.permute(0, 3, 1, 2).float())
        # values, indices = torch.max(result, 1)
        values, indices = torch.topk(result, 10)

        for index in range(len(label_batch)):
            type_id = label_batch[index]
            path = path_batch[index]
            predict = indices[index]

            if type_id not in self._statistics:
                self._statistics[type_id] = {}
                self._statistics[type_id]["total"] = 0
                self._statistics[type_id]["correct"] = 0

            entry = self._statistics[type_id]
            entry["total"] += 1
            if predict[0] == type_id:
                entry["correct"] += 1

            pred_str = ",".join(str(item) for item in predict.tolist())
            print("{},{},{}".format(path, type_id, pred_str))

    def run(self):
        finish = 0
        img_batch = []
        label_batch = []
        path_batch = []
        begin = 0
        verbose_period = 0

        while (True):
            obj = request_queue.get()
            img = obj[0]
            if isinstance(img, str) and img == "Finish":
                finish += 1
                if finish >= self._stride:
                    if len(img_batch) > 0:
                        self._process(img_batch, label_batch, path_batch)
                    break
            else:
                type_id = obj[1]
                path = obj[2]
                path_batch.append(path)
                label_batch.append(type_id)
                img_batch.append(img)
                if len(img_batch) < BATCH_SIZE:
                    continue
                self._process(img_batch, label_batch, path_batch)

                verbose_period += 1
                if verbose_period >= 100:
                    duration = time.time() - begin
                    print("Rate: {}".format(verbose_period*BATCH_SIZE/duration), file=sys.stderr)
                    verbose_period = 0
                    begin = time.time()
                img_batch = []
                label_batch = []
                path_batch = []

    def dump(self):
        with open("total.csv", "w") as fp:
            fp.write("id,total,correct,percent\n")
            for type_id, values in self._statistics.items():
                total = values["total"]
                correct = values["correct"]
                percentage = 100 * (correct / total)
                fp.write("{},{},{},{}\n".format(type_id, total, correct, percentage))


def get_file_list(root_path):
    file_list = []

    # limit = 0
    for directory in os.walk(root_path):
        for dir_name in directory[1]:  # All subdirectories
            pos = dir_name.find('.')
            type_id = int(dir_name[0:pos])

            count = 0
            for image_file in os.listdir(os.path.join(root_path, dir_name)):
                count += 1
            if count < 10:
                continue

            # limit += 1
            # if limit > 10:
            #    break

            limit = 0
            for image_file in os.listdir(os.path.join(root_path, dir_name)):
                full_path = os.path.join(root_path, dir_name, image_file)
                file_list.append((full_path, type_id))
                limit += 1
                if limit > 100:
                    break

    return file_list


def eval_traindata(args):
    net = EfficientNet.from_name('efficientnet-b2', override_params={
                                    'image_size': 300,
                                    'num_classes': 11000})
    net.load_state_dict(torch.load(args.trained_model))
    net.eval().cuda()

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
