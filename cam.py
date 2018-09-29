import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
import cv2
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from data_generator import DataGenerator
import pytorch_utils
from config import Config
from densenet import DenseNet121


def cam(fm, w):
    fm = fm[0]
    M = torch.mm(w, fm.view(fm.size(0), -1))
    M = M.view(M.size(0), fm.size(1), fm.size(2)).data.cpu().numpy()
    return M

def draw_heatmap(image, heatmap):
    alpha = 0.5
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, alpha, image, 1 - alpha, 0)
    return fin