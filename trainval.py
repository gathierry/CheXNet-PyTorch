import os
import time
import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import DataParallel, BCELoss
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
import pytorch_utils
from cxr import CXR
from alternative_data.cxr2 import CXR2
from config import Config
from data_generator import DataGenerator
from densenet import DenseNet121
from loss import WeightedBCE, UnWeightedBCE
from lr_scheduler import LRScheduler
from test import compute_AUCs


def print_log(epoch, lr, train_metrics, train_time, val_metrics, val_time, save_dir, log_mode):
    train_loss, train_aucs = train_metrics
    val_loss, val_aucs = val_metrics
    str0 = 'Epoch %03d (lr %.7f)' % (epoch, lr)
    str11 = 'Train: time {:.1f}, loss {:.4f}'.format(train_time, train_loss)
    str12 = str(['{:.4f}'.format(auc) for auc in train_aucs])
    str21 = 'Val:   time {:.1f}, loss {:.4f}'.format(val_time, val_loss)
    str22 = str(['{:.4f}'.format(auc) for auc in val_aucs])

    print(str0)
    print(str11)
    print(str12)
    print(str21)
    print(str22 + '\n')
    if epoch > 1:
        log_mode = 'a'
    f = open(save_dir + 'train_log.txt', log_mode)
    f.write(str0 + '\n')
    f.write(str11 + '\n')
    f.write(str12 + '\n')
    f.write(str21 + '\n')
    f.write(str22 + '\n\n')
    f.close()
    return val_loss


def train_val(data_loader, net, loss, optimizer=None, lr=None, training=True):
    start_time = time.time()
    if training:
        net.train()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        net.eval()
    target_lst = []
    output_lst = []
    loss_lst = []
    for i, (idx, data, target) in enumerate(data_loader):
        data = data.cuda(async=True)
        target = target.cuda(async=True)
        output, _ = net(data)
        loss_output = loss(output, target)
        if training:
            optimizer.zero_grad()
            loss_output.backward()
            optimizer.step()
        loss_lst.append(loss_output.data.cpu().item())
        target_lst.append(target.data.cpu().numpy())
        output_lst.append((output.data.cpu().numpy()))
    target_lst = np.concatenate(target_lst, axis=0)
    output_lst = np.concatenate(output_lst, axis=0)
    aucs = compute_AUCs(output_lst, target_lst)
    end_time = time.time()
    metrics = [np.mean(loss_lst), aucs]
    return metrics, end_time - start_time


if __name__ == '__main__':
    config = Config()
    resume = False
    workers = config.train_workers
    n_gpu = pytorch_utils.setgpu(config.train_gpus)
    batch_size = config.train_batch_size_per_gpu * n_gpu
    epochs = config.epochs
    base_lr = config.base_lr
    save_dir = config.proj_dir + 'checkpoints/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    train_data = CXR2(config.data_dir, 'train')
    val_data = CXR2(config.data_dir, 'val')
    print('Train sample number: %d' % train_data.size())
    print('Val sample number: %d' % val_data.size())
    pos = train_data.get_occurrences()
    neg = [train_data.size() - x for x in pos]

    net = DenseNet121(num_classes=len(train_data.labels))
    # loss = WeightedBCE(pos, neg)
    loss = UnWeightedBCE()
    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True

    start_epoch = 1
    lr = base_lr
    best_val_loss = float('inf')
    log_mode = 'w'
    if resume:
        checkpoint = torch.load(save_dir + 'densenet_016.ckpt')
        start_epoch = checkpoint['epoch'] + 1
        lr = checkpoint['lr']
        best_val_loss = checkpoint['best_val_loss']
        net.load_state_dict(checkpoint['state_dict'])
        log_mode = 'a'
    net = DataParallel(net).cuda()

    train_generator = DataGenerator(config, train_data, phase='train')
    train_loader = DataLoader(train_generator,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers,
                              pin_memory=True)
    val_generator = DataGenerator(config, val_data, phase='val')
    val_loader = DataLoader(val_generator,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=workers,
                            pin_memory=True)
    # optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-5)

    lrs = LRScheduler(lr, epochs, patience=5, factor=0.1, min_lr=1e-5, best_loss=best_val_loss)
    for epoch in range(start_epoch, epochs + 1):
        train_metrics, train_time = train_val(train_loader, net, loss, optimizer, lr, training=True)
        with torch.no_grad():
            val_metrics, val_time = train_val(val_loader, net, loss, training=False)

        val_loss = print_log(epoch, lr, train_metrics, train_time, val_metrics, val_time, save_dir, log_mode)

        lr = lrs.update_by_rule(val_loss)
        if val_loss < best_val_loss or epoch % 10 == 0 or lr is None:
            best_val_loss = min(val_loss, best_val_loss)
            state_dict = net.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'lr': lr,
                'best_val_loss': best_val_loss},
                os.path.join(save_dir, 'densenet_%03d.ckpt' % epoch))

        if lr is None:
            print('Training is early-stopped')
            break


