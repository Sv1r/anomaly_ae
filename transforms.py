import cv2
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

import settings


def get_train_aug(image_size):
    train_aug = A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=1.),
            A.VerticalFlip(p=1.)
        ], p=.2),
        A.OneOf([
            A.ElasticTransform(p=1.),
            A.GridDistortion(p=1.),
            A.OpticalDistortion(p=1.)
        ], p=.2),
        A.OneOf([
            A.CropAndPad(
                px=(-5, 5),
                pad_mode=cv2.BORDER_REPLICATE,
                p=1.
            ),
            A.Affine(
                translate_px=(-5, 5),
                rotate=(-45, 45),
                shear=(-10, 10),
                interpolation=cv2.INTER_CUBIC,
                mode=cv2.BORDER_REPLICATE,
                p=1.
            ),
            A.ShiftScaleRotate(
                shift_limit=.1,
                scale_limit=(-.1, .1),
                rotate_limit=(-30, 30),
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_REPLICATE,
                p=1.
            )
        ], p=.2),
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_CUBIC, p=1.),
        A.Normalize(mean=settings.MEAN, std=settings.STD, p=1.),
        ToTensorV2()
    ])
    return train_aug


def get_valid_aug(image_size):
    valid_aug = A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_CUBIC, p=1.),
        A.Normalize(mean=settings.MEAN, std=settings.STD, p=1.),
        ToTensorV2()
    ])
    return valid_aug
