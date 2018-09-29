from torchvision import transforms
import torch
from torch.utils.data import Dataset
from PIL import Image


class DataGenerator(Dataset):
    def __init__(self, config, data, phase='train'):
        self.data = data
        trans = []
        if phase == 'train':
            trans.append(transforms.RandomResizedCrop(config.img_max_size,
                                                      ratio=(1.0, 1.0), interpolation=Image.BICUBIC))
            trans.append(transforms.RandomHorizontalFlip(0.5))
        else:
            trans.append(transforms.Resize(config.img_max_size, Image.BICUBIC))
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(config.mu, config.sigma))
        self.transform = transforms.Compose(trans)

    def __getitem__(self, idx):
        fn = self.data.get_image_path(idx)
        image = Image.open(fn).convert('RGB')
        label = self.data.get_label(idx)
        image = self.transform(image)
        return idx, image, torch.FloatTensor(label)

    def __len__(self):
        return self.data.size()


if __name__ == '__main__':
    from config import Config
    from torch.utils.data import DataLoader
    from cxr import CXR

    config = Config()
    train_data = CXR(config.data_dir, 'train')
    train_dataset = DataGenerator(config, train_data, phase='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=True)

    for i, (images, labels) in enumerate(train_loader):
        print(images.size(), labels.size())
