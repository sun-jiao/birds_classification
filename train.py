import time

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from torch.autograd import Variable
from dataset.birds_dataset import BirdsDataset
from nets import resnet

def train():
    dataset = BirdsDataset('/disk3/donghao/data/bird/')
    data_loader = data.DataLoader(dataset, 32, num_workers=8, shuffle=True, pin_memory=True)

    net = resnet.resnet18(num_classes=10900).cuda()

    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    batch_iterator = iter(data_loader)
    for iteration in range(1000):
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

        if iteration % 10 == 0:
            print('loss: %.4f || timer: %.4f sec.' % (loss.item(), t1 - t0))

if __name__ == '__main__':
    train()
