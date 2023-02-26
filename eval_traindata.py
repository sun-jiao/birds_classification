import sys
import cv2
import time
import json
import queue
import torch
import argparse
import threading

import numpy as np

import pycls.core.builders as model_builder
from pycls.core.config import cfg

from dataset.birds_dataset import ListLoader

BATCH_SIZE = 32
NR_THREADS = 3
INCORRECT_DATA_FILE = "incorrect.txt"
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

        while True:
            index += self._stride
            if index >= total:
                break
            pair = self._file_list[index]
            img = cv2.imread(pair[0])
            if img is None:
                print("[Error]", pair[0], file=sys.stderr)
                continue
            if img.shape != (300, 300, 3):
                print("[Warn]", file=sys.stderr)
                continue
            # binary_image, label, image_path
            request_queue.put((img, pair[1], pair[0]))

        request_queue.put(("Finish", "Finish"))


class EvalThread(threading.Thread):
    def __init__(self, net, stride, fp):
        threading.Thread.__init__(self)
        self._net = net
        self._stride = stride
        self._statistics = {}
        self._statistics["correct"] = 0
        self._statistics["correct_top1_value"] = 0.0
        self._statistics["incorrect"] = 0
        self._statistics["incorrect_top1_value"] = 0.0
        self._fp = fp

    def _process(self, img_batch, label_batch, path_batch):
        input = torch.from_numpy(np.asarray(img_batch)).cuda()
        result = self._net(input.permute(0, 3, 1, 2).float())
        values, indices = torch.topk(result, 10)
        values = values.cpu()
        indices = indices.cpu()

        for index in range(len(label_batch)):
            type_id = label_batch[index]
            path = path_batch[index]
            predict = indices[index]
            top10 = values[index]

            if type_id not in self._statistics:
                self._statistics[type_id] = {}
                self._statistics[type_id]["total"] = 0
                self._statistics[type_id]["correct"] = 0

            entry = self._statistics[type_id]
            entry["total"] += 1
            if predict[0] == type_id:
                entry["correct"] += 1
                self._statistics["correct"] += 1
                self._statistics["correct_top1_value"] += top10[0].item()
            else:
                self._fp.write(f"({json.dumps(path)}, {type_id}, {predict[:5]})\n")
                self._statistics["incorrect"] += 1
                self._statistics["incorrect_top1_value"] += top10[0].item()

    def run(self):
        finish = 0
        img_batch = []
        label_batch = []
        path_batch = []
        begin = 0
        verbose_period = 0

        while True:
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
                if verbose_period >= 1000:
                    duration = time.time() - begin
                    print(
                        "Rate: {}".format(verbose_period * BATCH_SIZE / duration),
                        file=sys.stderr,
                    )
                    print(
                        "Corret: {}".format(
                            self._statistics["correct"]
                            / (
                                self._statistics["correct"]
                                + self._statistics["incorrect"]
                            )
                        ),
                        file=sys.stderr,
                    )
                    print(f"correct confidence: {self._statistics['correct_top1_value']/self._statistics['correct']}")
                    print(f"incorrect confidence: {self._statistics['incorrect_top1_value']/self._statistics['incorrect']}")
                    verbose_period = 0
                    begin = time.time()
                img_batch = []
                label_batch = []
                path_batch = []

    def dump(self):
        print(f"final correct confidence: {self._statistics['correct_top1_value']/self._statistics['correct']}")
        print(f"final incorrect confidence: {self._statistics['incorrect_top1_value']/self._statistics['incorrect']}")
        with open("total.csv", "w") as fp:
            fp.write("id,total,correct,percent\n")
            for type_id, values in self._statistics.items():
                if not isinstance(values, dict):
                    continue
                total = values["total"]
                correct = values["correct"]
                percentage = 100 * (correct / total)
                fp.write("{},{},{},{}\n".format(type_id, total, correct, percentage))


def get_file_list():
    output = []

    with open("train.txt", "r") as fp:
        for line in fp:
            path, type_id = eval(line)
            output.append((path, type_id))

    return output


def eval_traindata(args):
    state_net = torch.load(args.trained_model)
    cfg.merge_from_other_cfg(state_net["_config"])
    del state_net["_config"]
    net = model_builder.build_model()
    net.load_state_dict(state_net)
    net.eval().cuda()

    file_list = get_file_list()

    thread_list = []
    for index in range(NR_THREADS):
        thread = FileThread(file_list, index, NR_THREADS)
        thread_list.append(thread)
        thread.start()

    with open(INCORRECT_DATA_FILE, "w") as fp:
        eval_thread = EvalThread(net, 1, fp)
        eval_thread.start()

        eval_thread.join()
        eval_thread.dump()

    for thread in thread_list:
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default="/media/data2/i18n/V3",
        type=str,
        help="Root path of data",
    )
    parser.add_argument(
        "--trained_model",
        default="ckpt/bird_cls_0.pth",
        type=str,
        help="Trained ckpt file path to open",
    )
    args = parser.parse_args()

    list_loader = ListLoader(args.dataset_root, 11120, True)
    img_list, train_lst, eval_lst = list_loader.image_indices()
    train_set = set([str(img_list[ind]) for ind in train_lst])
    with open("train.txt", "w") as fp:
        for item in train_set:
            fp.write(item + "\n")
    eval_set = set([str(img_list[ind]) for ind in eval_lst])
    with open("eval.txt", "w") as fp:
        for item in eval_set:
            fp.write(item + "\n")

    eval_traindata(args)
