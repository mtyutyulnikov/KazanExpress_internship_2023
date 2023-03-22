import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2


def compose(transforms_to_compose):
    result = albu.Compose(
        [item for sublist in transforms_to_compose for item in sublist]
    )
    return result


def hard_transforms():
    result = [
        albu.CoarseDropout(
            max_height=32, max_width=32, max_holes=4, mask_fill_value=1.0
        ),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(brightness_limit=0.07),
                albu.GridDistortion(distort_limit=0.1),
                albu.ColorJitter(),
            ],
            p=0.3,
        ),
        albu.HorizontalFlip(p=0.5),
    ]
    return result


def post_transforms():
    return [
        albu.Normalize(),
        albu.Resize(256, 256),
        ToTensorV2(),
    ]


def get_train_transforms():
    return compose([hard_transforms(), post_transforms()])


def get_test_transforms():
    return compose([post_transforms()])
