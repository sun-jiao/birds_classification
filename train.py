import time
import argparse

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from torch.autograd import Variable
from dataset.birds_dataset import BirdsDataset
from nets import resnet

cfg = {
    'nr_images': 3934740,
    'num_classes': 11000,
    'num_workers': 4,
    'verbose_steps': 100,
    }

def train(args):
    t0 = time.time()
    dataset = BirdsDataset(args.dataset_root, cfg['num_classes'])
    t1 = time.time()
    print('Load dataset with {} secs'.format(t1 - t0))
    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=cfg['num_workers'],
                                  shuffle=True, pin_memory=True)

    net = resnet.resnet18(num_classes=cfg['num_classes']).cuda()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    batch_iterator = iter(data_loader)
    for iteration in range(args.max_epoch * cfg['nr_images'] // args.batch_size):
        try:
            images, type_ids = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, type_ids = next(batch_iterator)
        except Exception as e:
            print('Loading data exception:', e)

        images = Variable(images.cuda())
        type_ids = Variable(type_ids.cuda())

        ids_one_hot = torch.zeros(len(type_ids), type_ids.max() + 1, device='cuda')
        ids_one_hot = ids_one_hot.scatter_(1, type_ids.unsqueeze(1), 1.)
        #print('ids_one_hot', ids_one_hot, ids_one_hot.shape)

        # forward
        t0 = time.time()
        out = net(images.permute(0, 3, 1, 2).float())
        #print('out:', out, out.shape)

        # backprop
        optimizer.zero_grad()
        loss = F.cross_entropy(out, type_ids)
        loss.backward()
        optimizer.step()
        t1 = time.time()

        if iteration % cfg['verbose_steps'] == 0:
            # accuracy
            _, predict = torch.max(out, 1)
            correct = (predict == type_ids)
            accuracy = correct.sum().item() / correct.size()[0]
            print('loss: %.4f | accuracy: %.4f | timer: %.4f sec.' % (loss.item(), accuracy, (t1 - t0)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--max_epoch', default=100, type=int, help='Maximum epoches for training')
    parser.add_argument('--dataset_root', default='/media/data2/i18n/V1', type=str, help='Root path of data')
    parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optimizer')
    args = parser.parse_args()

    train(args)
