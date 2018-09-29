import math
import numpy as np

class Config:

    def __init__(self):
        # custom configs
        self.proj_dir = 'YOUR PROJECT DIR HERE'  # where trained models are saved
        self.data_dir = 'YOUR DATA DIR HERE'
        self.train_batch_size_per_gpu = 48
        self.train_workers = 64
        self.train_gpus = '0,1,2,3'  # CUDA_DEVICES
        self.base_lr = 1e-3  # learning rate
        self.epochs = 1000

        self.test_batch_size_per_gpu = 80
        self.test_workers = 64
        self.test_gpus = '4,5'  # CUDA_DEVICES

        # Img
        self.img_max_size = 224
        self.mu = [0.485, 0.456, 0.406]
        self.sigma = [0.229, 0.224, 0.225]

        self.benchmark = {'Atelectasis': 0.8094,
                          'Cardiomegaly': 0.9248,
                          'Effusion': 0.8638,
                          'Infiltration': 0.7345,
                          'Mass': 0.8676,
                          'Nodule': 0.7802,
                          'Pneumonia': 0.7680,
                          'Pneumothorax': 0.8887,
                          'Consolidation': 0.7901,
                          'Edema': 0.8878,
                          'Emphysema': 0.9371,
                          'Fibrosis': 0.8047,
                          'Pleural_Thickening': 0.8062,
                          'Hernia': 0.9164}

