import os
import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import sys
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
sys.path.append('../')
import pytorch_utils
from cxr import CXR
from alternative_data.cxr2 import CXR2
from config import Config
from data_generator import DataGenerator
from densenet import DenseNet121
from cam import cam


def compute_AUCs(output, target):
    """
    :param output: matrix nx14
    :param target: matrix nx14
    :return: list of 14 elements
    """
    aucs = []
    num_classes = output.shape[1]
    for i in range(num_classes):
        aucs.append(roc_auc_score(target[:, i], output[:, i]))
    return aucs


if __name__ == '__main__':
    config = Config()
    resume = False
    workers = config.test_workers
    n_gpu = pytorch_utils.setgpu(config.test_gpus)
    batch_size = config.test_batch_size_per_gpu * n_gpu

    test_data = CXR2(config.data_dir, 'test')
    print('Test sample number: %d' % test_data.size())

    net = DenseNet121(num_classes=len(test_data.labels))
    checkpoint = torch.load(os.path.join(config.proj_dir, 'checkpoints', 'densenet_024.ckpt'))  # must before cuda
    net.load_state_dict(checkpoint['state_dict'])
    net = net.cuda()
    cudnn.benchmark = True
    net = DataParallel(net).cuda()

    test_generator = DataGenerator(config, test_data, phase='test')
    test_loader = DataLoader(test_generator,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=workers,
                             pin_memory=True)

    net.eval()
    with torch.no_grad():
        target_lst = []
        output_lst = []
        for idx, data, target in tqdm(test_loader):
            data = data.cuda(async=True)
            target = target.cuda(async=True)
            output, fm = net(data)
            heatmaps = cam(fm, list(net.parameters())[-2])
            target_lst.append(target.data.cpu().numpy())
            output_lst.append((output.data.cpu().numpy()))
        target_lst = np.concatenate(target_lst, axis=0)
        output_lst = np.concatenate(output_lst, axis=0)
    aucs = compute_AUCs(output_lst, target_lst)
    for i, auc in enumerate(aucs):
        label = test_data.labels[i]
        print('{}: {:.4} ({})'.format(label, auc, config.benchmark[label]))
    print('AVG %.4f' % (sum(aucs) / float(len(aucs))))