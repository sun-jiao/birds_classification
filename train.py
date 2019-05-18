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
    'verbose_period': 100,
    'save_period': 20000,
    'save_folder': 'ckpt/',
    'ckpt_name': 'bird_cls',
    }

def save_ckpt(net, iteration):
    torch.save(net.state_dict(), cfg['save_folder'] + cfg['ckpt_name'] + '_' + str(iteration) + '.pth')

def train(args):
    t0 = time.time()
    dataset = BirdsDataset(args.dataset_root, cfg['num_classes'])
    t1 = time.time()
    print('Load dataset with {} secs'.format(t1 - t0))
    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=cfg['num_workers'],
                                  shuffle=True, pin_memory=True)

    net = resnet.resnext50_32x4d(num_classes=cfg['num_classes']).cuda()
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        net.load_weights(args.resume)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    batch_iterator = iter(data_loader)
    for iteration in range(args.max_epoch * cfg['nr_images'] // args.batch_size):
        t0 = time.time()
        try:
            images, type_ids = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, type_ids = next(batch_iterator)
        except Exception as e:
            print('Loading data exception:', e)

        images = Variable(images.cuda())
        type_ids = Variable(type_ids.cuda())

        # forward
        out = net(images.permute(0, 3, 1, 2).float())
        #print('out:', out, out.shape)

        # backprop
        optimizer.zero_grad()
        loss = F.cross_entropy(out, type_ids)
        loss.backward()
        optimizer.step()
        t1 = time.time()

        if iteration % cfg['verbose_period'] == 0:
            # accuracy
            _, predict = torch.max(out, 1)
            correct = (predict == type_ids)
            accuracy = correct.sum().item() / correct.size()[0]
            print('step: %d loss: %.4f | accuracy: %.4f | time: %.4f sec.' %
                  (iteration, loss.item(), accuracy, (t1 - t0)))

        if iteration % cfg['save_period'] == 0:
            # save checkpoint
            print('Saving state, iter:', iteration)
            save_ckpt(net, iteration)

    # final checkpoint
    save_ckpt(net, iteration)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--max_epoch', default=100, type=int, help='Maximum epoches for training')
    parser.add_argument('--dataset_root', default='/media/data2/i18n/V1', type=str, help='Root path of data')
    parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optimizer')
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint file to resume training from')
    args = parser.parse_args()

    train(args)
