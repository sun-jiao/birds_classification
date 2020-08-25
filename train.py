import time
import argparse

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn

import pycls.core.model_builder as model_builder
from pycls.core.config import cfg

from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from dataset.birds_dataset import BirdsDataset, ListLoader
# from efficientnet_pytorch import EfficientNet
from utils import augmentations
# from warmup_scheduler import GradualWarmupScheduler

import apex.amp as amp

config = {
    'num_classes': 11000,
    'num_workers': 4,
    'verbose_period': 2000,
    'eval_period': 40000,
    'save_period': 40000,
    'save_folder': 'ckpt/',
    'ckpt_name': 'bird_cls',
    }


def save_ckpt(net, iteration):
    torch.save(net.state_dict(), config['save_folder'] + config['ckpt_name'] + '_' + str(iteration) + '.pth')


def evaluate(net, eval_loader):
    total_loss = 0.0
    batch_iterator = iter(eval_loader)
    sum_accuracy = 0
    for iteration in range(len(eval_loader)):
        images, type_ids = next(batch_iterator)
        images = Variable(images.cuda())
        type_ids = Variable(type_ids.cuda())

        # forward
        out = net(images.permute(0, 3, 1, 2).float())
        # accuracy
        _, predict = torch.max(out, 1)
        correct = (predict == type_ids)
        sum_accuracy += correct.sum().item() / correct.size()[0]
        # loss
        loss = F.cross_entropy(out, type_ids)
        total_loss += loss.item()
    return total_loss / iteration, sum_accuracy / iteration


def warmup_learning_rate(optimizer, steps, warmup_steps):
    min_lr = args.lr / 100
    slope = (args.lr - min_lr) / warmup_steps

    lr = steps * slope + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, train_loader, eval_loader):
    """net = EfficientNet.from_name('efficientnet-b2', override_params={
                                     'image_size': IMAGE_SHAPE[0],
                                     'num_classes': config['num_classes'],
                                     'dropout_rate': 0.0,
                                     'drop_connect_rate': 0.0,
                                 }).cuda()"""
    cfg.MODEL.TYPE = "regnet"
    cfg.REGNET.DEPTH = 20
    cfg.REGNET.SE_ON = False
    cfg.REGNET.W0 = 128
    cfg.MODEL.NUM_CLASSES = config["num_classes"]
    net = model_builder.build_model()
    print("net", net)
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ckpt_file = config['save_folder'] + config['ckpt_name'] + '_' + str(args.resume) + '.pth'
        net.load_state_dict(torch.load(ckpt_file))

    if args.finetune:
        print('Finetuning......')
        # Freeze all layers
        for param in net.parameters():
            param.requires_grad = False
        # Unfreeze some layers
        # for index in [17, 18, 19, 20, 21, 22]:
        #    for param in net._blocks[index].parameters():
        #        param.requires_grad = True
        # net._conv_head.weight.requires_grad = True
        # net._fc.weight.requires_grad = True
        for layer in [net.s2.b5]:
            for param in layer.parameters():
                param.requies_grad = True
        net.head.fc.weight.requires_grad = True
        optimizer = optim.SGD(filter(lambda param: param.requires_grad, net.parameters()),
                              lr=args.lr, momentum=args.momentum, nesterov=False)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, nesterov=False)

    if args.finetune:
        scheduler = CosineAnnealingLR(optimizer, 10)
    else:
        scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2,
                                      verbose=True, threshold=1e-3, threshold_mode='abs')

    if args.fp16:
        net, optimizer = amp.initialize(net, optimizer, opt_level="O2")
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=100, total_epoch=4000,
    #                                          after_scheduler=scheduler)

    aug = augmentations.Augmentations().cuda()
    batch_iterator = iter(train_loader)
    sum_accuracy = 0
    step = 0
    for iteration in range(args.resume + 1, args.max_epoch * len(train_loader.dataset) // args.batch_size):
        t0 = time.time()
        try:
            images, type_ids = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(train_loader)
            images, type_ids = next(batch_iterator)
        except Exception as e:
            print('Loading data exception:', e)

        images = Variable(images.cuda()).permute(0, 3, 1, 2).float()
        type_ids = Variable(type_ids.cuda())

        one_hot = torch.cuda.FloatTensor(type_ids.shape[0], config['num_classes'])
        one_hot.fill_(4.54587e-5)
        one_hot.scatter_(1, type_ids.unsqueeze(1), 0.5)

        # augmentation
        if not args.finetune:
            images = aug(images)
        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss = torch.sum(- one_hot * F.log_softmax(out, -1), -1).mean()
        # loss = torch.sum(- one_hot * F.log_softmax(out, -1), -1)
        # loss, _ = loss.topk(k=(loss.shape[0] // 4))
        # loss = loss.mean()
        # loss = F.cross_entropy(out, type_ids)

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        t1 = time.time()

        if iteration % config['verbose_period'] == 0:
            # accuracy
            _, predict = torch.max(out, 1)
            correct = (predict == type_ids)
            accuracy = correct.sum().item() / correct.size()[0]
            print('iter: %d loss: %.4f | acc: %.4f | time: %.4f sec.' %
                  (iteration, loss.item(), accuracy, (t1 - t0)), flush=True)
            sum_accuracy += accuracy
            step += 1

        warmup_steps = config['verbose_period'] * 4
        if iteration < warmup_steps:
            warmup_learning_rate(optimizer, iteration, warmup_steps)

        if iteration % config['eval_period'] == 0 and iteration != 0 and step != 0:
            loss, accuracy = evaluate(net, eval_loader)
            print(f'Eval accuracy: {accuracy:.4f} | Train accuracy: {sum_accuracy/step:.4f}', flush=True)
            if args.finetune:
                scheduler.step()
            else:
                scheduler.step(accuracy)
            print("learning rate:", scheduler.get_lr())
            sum_accuracy = 0
            step = 0

        if iteration % config['save_period'] == 0 and iteration != 0:
            # save checkpoint
            print('Saving state, iter:', iteration, flush=True)
            save_ckpt(net, iteration)

    # final checkpoint
    save_ckpt(net, iteration)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--max_epoch', default=100, type=int, help='Maximum epoches for training')
    parser.add_argument('--dataset_root', default='/media/data2/i18n/V3', type=str, help='Root path of data')
    parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optimizer')
    parser.add_argument('--resume', default=0, type=int, help='Checkpoint steps to resume training from')
    parser.add_argument('--finetune', default=False, type=bool, help='Finetune model by using all categories')
    parser.add_argument('--fp16', default=False, type=bool, help='Use float16 precision to train')
    args = parser.parse_args()

    t0 = time.time()
    list_loader = ListLoader(args.dataset_root, config['num_classes'], args.finetune)
    list_loader.export_labelmap()
    image_list, train_indices, eval_indices = list_loader.image_indices()

    train_set = BirdsDataset(image_list, train_indices, list_loader.multiples(), True)
    eval_set = BirdsDataset(image_list, eval_indices, list_loader.multiples(), False)
    print('train set: {} eval set: {}'.format(len(train_set), len(eval_set)))

    train_loader = data.DataLoader(train_set, args.batch_size, num_workers=config['num_workers'],
                                   shuffle=True, pin_memory=True,
                                   collate_fn=BirdsDataset.my_collate)
    eval_loader = data.DataLoader(eval_set, args.batch_size // 4, num_workers=config['num_workers'],
                                  shuffle=False, pin_memory=True,
                                  collate_fn=BirdsDataset.my_collate)
    t1 = time.time()
    print('Load dataset with {} secs'.format(t1 - t0))

    train(args, train_loader, eval_loader)
