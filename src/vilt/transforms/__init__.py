from .pixelbert import (
    pixelbert_transform,
    pixelbert_transform_randaug,
)
from .square_transform import (
    square_transform,
    square_transform_mim,
    square_transform_randaug,
    square_transform_randaug_mim,
)

_transforms = {
    "pixelbert": pixelbert_transform,
    "pixelbert_randaug": pixelbert_transform_randaug,
    "square_transform": square_transform,
    "square_transform_mim": square_transform_mim,
    "square_transform_randaug": square_transform_randaug,
    "square_transform_randaug_mim": square_transform_randaug_mim,
}


def keys_to_transforms(keys: list, size=224):
    return [_transforms[key](size=size) for key in keys]

def keys_to_transforms_for_mim(keys: list, size=224, second_size=None):
    return [_transforms[key](size=size, second_size=second_size) for key in keys]
