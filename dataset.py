from torchvision import transforms
import torch
from torch.utils.data import Dataset
import cv2
import PIL.Image as Image
import numpy as np


class FaceDataset(Dataset):
    def __init__(self, datapath, image_size=128, augment=False):
        self.augment = augment
        self.image_size = image_size
        self.trms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                             std=[0.5, 0.5, 0.5])])

        with open(datapath, 'r') as f:
            self.image_path_list = f.readlines()

    def __cv2ToPillow(self, image):
        assert type(image) == np.ndarray, 'input should be a ndarray!'
        return Image.fromarray(image)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_path_list[idx])
        # assert image.shape[:1] == (self.image_size, self.image_size), f'image size should be {self.image_size}'
        if self.augment:
            pass
        image = self.trms(image)
        label = torch.ones(1)
        return image, label

    def __len__(self):
        return len(self.image_path_list)

