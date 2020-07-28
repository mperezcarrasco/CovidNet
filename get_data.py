import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import cv2
import numpy as np
from PIL import Image


def crop_top(img, percent=0.15):
    """
    Croping the top 15% of the image: it usually contains irrelevant information.
    """
    offset = int(img.shape[0] * percent)
    return img[offset:]

def central_crop(img):
    """
    Cropping into an squared version of the image.
    """
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_h:offset_h + size, offset_w:offset_w + size]

def get_processed_img(filepath, top_percent=0.15):
    """Return a center-croped version of the image.

    Args:
        filepath (str): path of the image to be preprocessed.
        top_percent (float): Percent of the image to be top-cropped. 
    """
    img = cv2.imread(filepath)
    img = crop_top(img, percent=top_percent)
    img = central_crop(img)
    return img


class CovidLoader(object):
    """
    Return preprocessed versions of the images on the fly.
    """
    def __init__(self, files, img_size=224):
        """Initialization.

        Args:
            files (str): list containing metadata for each image.
            img_size (int): Final size for the image.
        """
        self.files = files
        labels = [file.split(' ')[2] for file in files]
        self.mapping = {'normal': 0,
                        'pneumonia': 1,
                        'COVID-19': 2
                        }
        self.labels = [self.mapping[label.rstrip("\n")] for label in labels]

        self.transform  = transforms.Compose([
                          transforms.Resize(256),
                          transforms.CenterCrop(img_size),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225]),])
        self.size = img_size

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        return len(self.files)

    def __getitem__(self, index):
        """Return a preprocessed item from the dataset.

        Args:
            index (int): index of the element to be returned.

        Returns:
            img (torch.Tensor): torch.Tensor version of the preprocessed image.
            label (int): label of the image based on self.mapping.
        """
        info = self.files[index].split(' ')
        filepath = 'data/test/' + info[1]
        img = get_processed_img(filepath)
        label = self.mapping[info[2].rstrip("\n")]
        img = Image.fromarray(img)
        
        img = self.transform(img)
        return img, label
        

def get_dataloader(files, batch_size=64):
    """Build and return data loader."""
    data = CovidLoader(files)

    data_loader = DataLoader(dataset=data,
                             batch_size=batch_size,
                             shuffle=False)
    return data_loader