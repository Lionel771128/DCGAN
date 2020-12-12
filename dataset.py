from torchvision import transforms
import torch
from torch.utils.data import Dataset
import cv2
import PIL.Image as Image
import numpy as np


class FaceDataset(Dataset):
    def __init__(self, datapath, image_size=128, augment=False, cache_image=False):
        self.augment = augment
        self.image_size = image_size
        self.cache_image = cache_image
        self.trms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                             std=[0.5, 0.5, 0.5])])
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        with open(datapath, 'r') as f:
            self.image_path_list = f.readlines()

        if cache_image:
            self.image_list = []
            for img_path in self.image_path_list:
                self.image_list.append(cv2.imread(img_path.strip()))

    def __cv2ToPillow(self, image):
        assert type(image) == np.ndarray, 'input should be a ndarray!'
        return Image.fromarray(image)

    def __getitem__(self, idx):
        if self.cache_image:
            image = self.image_list[idx]
        else:
            image = cv2.imread(self.image_path_list[idx].strip())
        # assert image.shape[:1] == (self.image_size, self.image_size), f'image size should be {self.image_size}'
        if self.augment:
            center = (image.shape[0] // 2, image.shape[1] // 2)
            dh, dw = self.image_size // 2, self.image_size // 2
            image = image[center[0] - dh: center[0] + dh, center[1] - dw: center[1] + dw, :]
            pass
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = self.trms(image)
        label = torch.ones(1)
        return image, label

    def __len__(self):
        return len(self.image_path_list)

