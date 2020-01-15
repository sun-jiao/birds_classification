import argparse
import cv2
import os
import queue
import torch
import threading

from efficientnet_pytorch import EfficientNet

NR_THREADS = 3
request_queue = queue.Queue()


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
            request_queue.put((img, pair[1]))
            index += self._stride
        request_queue.put(("Finish", "Finish"))


class EvalThread(threading.Thread):
    def __init__(self, net, stride):
        threading.Thread.__init__(self)
        self._net = net
        self._stride = stride
        self._statistics = {}

    def run(self):
        finish = 0

        while (True):
            obj = request_queue.get()
            img = obj[0]
            type_name = obj[1]
            if img == "Finish":
                finish += 1
                if finish >= self._stride:
                    break
            else:
                input = torch.from_numpy(img)
                result = self._net(input.unsqueeze(0).permute(0, 3, 1, 2).float())
                values, indices = torch.max(result, 1)
                print(indices.item(), type_name)
                if type_name not in self._statistics:
                    self._statistics[type_name] = {}
                    self._statistics[type_name]["total"] = 0
                    self._statistics[type_name]["correct"] = 0

                entry = self._statistics[type_name]
                entry["total"] += 1
                if indices.item() == type_name:
                    entry["correct"] += 1

    def dump(self):
        for type_name, values in self._statistics.items():
            percentage = values["total"] / values["corret"]
            print(type_name, values, percentage)


def get_file_list(root_path):
    file_list = []

    limit = 0
    for directory in os.walk(root_path):
        for dir_name in directory[1]:  # All subdirectories
            pos = dir_name.find('.')
            type_id = int(dir_name[0:pos])
            # type_name = dir_name[pos+1:]

            count = 0
            for image_file in os.listdir(os.path.join(root_path, dir_name)):
                count += 1
            if count < 100:
                continue

            limit += 1
            if limit > 2:
                break

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
