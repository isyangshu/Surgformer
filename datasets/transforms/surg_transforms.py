import cv2
import torch
import torchvision.transforms.functional as F
import warnings
import random
import numpy as np
import torchvision
from PIL import Image, ImageOps
import numbers
from imgaug import augmenters as iaa

class SurgTransforms(object):

    def __init__(self, input_size=224, scales=(0.0, 0.3)):
        self.scales = scales
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.aug = iaa.Sequential([
            # Resize to (252, 448)
            iaa.Resize({"height": 252, "width": 448}),
            # Crop with Scale [0.8 - 1.0]
            iaa.Crop(percent=scales, keep_size=False),
            # Resize to (224, 224)
            iaa.Resize({"height": input_size, "width": input_size}),
            # Random Augment Surgery
            iaa.SomeOf((0, 2), [
                iaa.pillike.EnhanceSharpness(),
                iaa.pillike.Autocontrast(),
                iaa.pillike.Equalize(),
                iaa.pillike.EnhanceContrast(),
                iaa.pillike.EnhanceColor(),
                iaa.pillike.EnhanceBrightness(),
                iaa.Rotate((-30, 30)),
                iaa.ShearX((-20, 20)),
                iaa.ShearY((-20, 20)),
                iaa.TranslateX(percent=(-0.1, 0.1)),
                iaa.TranslateY(percent=(-0.1, 0.1))
                ]),
            iaa.Sometimes(0.3, iaa.AddToHueAndSaturation((-50, 50), per_channel=True)),
            # Horizontally flip 50% of all images
            iaa.Fliplr(0.5)])


    def __call__(self, img_tuple):
        images, label = img_tuple

        # 给定裁剪起始及裁剪尺寸进行裁剪及Resize，处理过程中维持视频序列相同处理方案
        augDet = self.aug.to_deterministic()
        augment_images = []
        for _, img in enumerate(images):
            img_aug = augDet.augment_image(np.array(img))
            augment_images.append(img_aug)

        # for index, img in enumerate(augment_images):
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     cv2.imshow(str(index), img)
        #     cv2.waitKey()
        return (augment_images, label)
    

class SurgStack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        if img_group[0].shape[2] == 1:
            return (np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2), label)
        elif img_group[0].shape[2] == 3:
            if self.roll:
                return (np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2), label)
            else:
                return (np.concatenate(img_group, axis=2), label)


if __name__ == '__main__':

    class SurgTransforms(object):

        def __init__(self, input_size=224, scales=(0.0, 0.3)):
            self.scales = scales
            self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
            self.aug = iaa.Sequential([
                # Resize to (252, 448)
                iaa.Resize({"height": 252, "width": 448}),
                # Crop with Scale [0.8 - 1.0]
                iaa.Crop(percent=scales, keep_size=False),
                # Resize to (224, 224)
                iaa.Resize({"height": input_size, "width": input_size}),
                # Random Augment Surgery
                iaa.SomeOf((0, 2), [
                    iaa.pillike.EnhanceSharpness(),
                    iaa.pillike.Autocontrast(),
                    iaa.pillike.Equalize(),
                    iaa.pillike.EnhanceContrast(),
                    iaa.pillike.EnhanceColor(),
                    iaa.pillike.EnhanceBrightness(),
                    iaa.Rotate((-30, 30)),
                    iaa.ShearX((-20, 20)),
                    iaa.ShearY((-20, 20)),
                    iaa.TranslateX(percent=(-0.1, 0.1)),
                    iaa.TranslateY(percent=(-0.1, 0.1))
                    ]),
                iaa.Sometimes(0.3, iaa.AddToHueAndSaturation((-50, 50), per_channel=True)),
                # Horizontally flip 50% of all images
                iaa.Fliplr(0.5)
                ])


        def __call__(self, images):

            # 给定裁剪起始及裁剪尺寸进行裁剪及Resize，处理过程中维持视频序列相同处理方案
            augDet = self.aug.to_deterministic()
            augment_images = []
            for _, img in enumerate(images):
                img_aug = augDet.augment_image(img)
                augment_images.append(img_aug)

            return augment_images
    
    A = SurgTransforms()
    origin_images = cv2.imread("data/cholec80/frames/train/video01/0.jpg")
    origin_images = cv2.cvtColor(origin_images, cv2.COLOR_BGR2RGB)

    images = np.array([origin_images for _ in range(4)], dtype=np.uint8)
    img_1 = A(images)
    for index, img in enumerate(img_1):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        print(img.shape)
        cv2.imshow(str(index), img)
        cv2.waitKey()
    img_2 = A(images)
    for index, img in enumerate(img_2):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        print(img.shape)
        cv2.imshow(str(index)+'2', img)
        cv2.waitKey()
    img_3 = A(images)
    for index, img in enumerate(img_3):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        print(img.shape)
        cv2.imshow(str(index)+'3', img)
        cv2.waitKey()
