import numpy as np
import torch
import cv2

def cam(fms, w):
    """
    :param fms: tensor [n, 1024, 7, 7]
    :param w: tensor [1, 1024]
    :return: a list of feature maps
    """
    Ms = []
    for fm in fms:
        M = torch.mm(w, fm.view(fm.size(0), -1))
        M = M.view(M.size(0), fm.size(1), fm.size(2)).data.cpu().numpy()
        Ms.append(M)
    return Ms

def draw_heatmap(image, heatmap):
    alpha = 0.5
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, alpha, image, 1 - alpha, 0)
    return fin
