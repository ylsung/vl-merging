from .utils import (
    inception_normalize,
)
from torchvision import transforms
from .randaugment import RandomAugment
from PIL import Image

from dall_e.utils import map_pixels
from .random_crop_two_pics import RandomResizedCropAndInterpolationWithTwoPic


def square_transform(size=224):
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )


def square_transform_randaug(size=224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            inception_normalize,
        ]
    )


class OperationList(object):
    def __init__(self, operation_list):
        self.operation_list = operation_list
        
    def __call__(self, tensors):
        outputs = [operation_list(tensor) for operation_list, tensor in zip(self.operation_list, tensors)]
        return outputs
    
    def __repr__(self):
        return self.__class__.__name__


def square_transform_mim(size=224, second_size=None):
    return transforms.Compose(
        [
            RandomResizedCropAndInterpolationWithTwoPic(size=size, second_size=second_size, scale=(1.0, 1.0)),
            OperationList([transforms.ToTensor(), transforms.ToTensor()]),
            OperationList([inception_normalize, map_pixels]),
        ]
    )


def square_transform_randaug_mim(size=224, second_size=None):
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']), 
            RandomResizedCropAndInterpolationWithTwoPic(size=size, second_size=second_size, scale=(0.5, 1.0)), # return two images
            OperationList([transforms.ToTensor(), transforms.ToTensor()]),
            OperationList([inception_normalize, map_pixels]),
        ]
    )