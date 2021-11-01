# -*- coding: utf-8 -*-
# @File : transforms.py
# @Author: Runist
# @Time : 2021/9/26 10:56
# @Software: PyCharm
# @Brief: torchvision transforms implement

from utils.parser import args
import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import torchvision.transforms as transforms


class ResizeImage(object):
    """Resize image and pad image."""
    def __init__(self, size=(256, 256), padding=(0, 0)):
        self.size = size
        self.padding = padding

    def __call__(self, sample):
        img_l = sample['image_l']
        img_r = sample['image_r']
        label = sample['label']

        height, width = self.size
        pad_h, pad_w = self.padding

        img_l = img_l.resize((width, height), Image.BICUBIC)
        img_r = img_r.resize((width, height), Image.BICUBIC)
        label = label.resize((width, height), Image.NEAREST)

        img_l_padded = Image.new('RGB', (width + pad_w, height + pad_h), (0, 0, 0))
        img_r_padded = Image.new('RGB', (width + pad_w, height + pad_h), (0, 0, 0))
        label_padded = Image.new('L', (width + pad_w, height + pad_h), 0)

        img_l_padded.paste(img_l, (pad_w, pad_h // 2, width, height + pad_h // 2))
        img_r_padded.paste(img_r, (pad_w, pad_h // 2, width, height + pad_h // 2))
        label_padded.paste(label, (pad_w, pad_h // 2, width, height + pad_h // 2))

        return {'image_l': img_l_padded, 'image_r': img_r_padded, 'label': label_padded}


class LabelRemap(object):
    """Label image pixel value remap to index."""
    def __call__(self, sample):
        img_l = sample['image_l']
        img_r = sample['image_r']
        label = sample['label']

        label = np.array(label).astype(np.float32) / 255
        label = Image.fromarray(label)

        return {'image_l': img_l, 'image_r': img_r, 'label': label}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation."""
    def __call__(self, sample):
        img_l = sample['image_l']
        img_r = sample['image_r']
        label = sample['label']

        img_l = np.array(img_l).astype(np.float32) / 255.
        img_r = np.array(img_r).astype(np.float32) / 255.
        label = np.array(label).astype(np.float32)

        return {'image_l': img_l, 'image_r': img_r, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_l = sample['image_l']
        img_r = sample['image_r']
        label = sample['label']

        img_l = np.array(img_l).astype(np.float32).transpose((2, 0, 1))
        img_r = np.array(img_r).astype(np.float32).transpose((2, 0, 1))
        label = np.array(label).astype(np.float32)
        label = np.expand_dims(label, axis=0)

        img_l = torch.from_numpy(img_l).float()
        img_r = torch.from_numpy(img_r).float()
        label = torch.from_numpy(label).float()

        return {'image_l': img_l, 'image_r': img_r, 'label': label}


class RandomHorizontalFlip(object):
    """Random horizontal flip image."""
    def __call__(self, sample):
        img_l = sample['image_l']
        img_r = sample['image_r']
        label = sample['label']
        
        if random.random() < 0.5:
            img_l = img_l.transpose(Image.FLIP_LEFT_RIGHT)
            img_r = img_r.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image_l': img_l, 'image_r': img_r, 'label': label}


class RandomGaussianBlur(object):
    """Random add Gaussian blur to the image."""
    def __call__(self, sample):
        img_l = sample['image_l']
        img_r = sample['image_r']
        label = sample['label']
        
        if random.random() < 0.5:
            img_l = img_l.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            img_r = img_r.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image_l': img_l, 'image_r': img_r, 'label': label}


class RandomScaleCrop(object):
    """Random rescale image and crop fix size image."""
    def __init__(self, base_size, crop_size, fill=255):
        self.base_size = min(base_size)
        self.crop_size_h, self.crop_size_w = crop_size
        self.fill = fill

    def __call__(self, sample):
        img_l = sample['image_l']
        img_r = sample['image_r']
        label = sample['label']

        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img_l.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img_l = img_l.resize((ow, oh), Image.BILINEAR)
        img_r = img_r.resize((ow, oh), Image.BILINEAR)
        label = label.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < self.crop_size_h:
            padh = self.crop_size_h - oh if oh < self.crop_size_h else 0
            padw = self.crop_size_w - ow if ow < self.crop_size_w else 0
            img_l = ImageOps.expand(img_l, border=(0, 0, padw, padh), fill=0)
            img_r = ImageOps.expand(img_r, border=(0, 0, padw, padh), fill=0)
            label = ImageOps.expand(label, border=(0, 0, padw, padh), fill=self.fill)

        # random crop crop_size
        w, h = img_l.size
        x1 = random.randint(0, w - self.crop_size_w)
        y1 = random.randint(0, h - self.crop_size_h)
        img_l = img_l.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))
        img_r = img_r.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))
        label = label.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))

        return {'image_l': img_l, 'image_r': img_r, 'label': label}


class RandomBrightness(object):
    """Set random brightness for the image."""
    def __init__(self, low, high):
        self.high = high
        self.low = low

    def __call__(self, sample):
        img_l = sample['image_l']
        img_r = sample['image_r']
        label = sample['label']
        
        if random.random() < 0.5:
            img_l = ImageEnhance.Brightness(img_l)
            img_l = img_l.enhance(random.uniform(self.low, self.high))
    
            img_r = ImageEnhance.Brightness(img_r)
            img_r = img_r.enhance(random.uniform(self.low, self.high))

        return {'image_l': img_l, 'image_r': img_r, 'label': label}


class RandomColor(object):
    """Set random color for the image."""
    def __init__(self, low, high):
        self.high = high
        self.low = low

    def __call__(self, sample):
        img_l = sample['image_l']
        img_r = sample['image_r']
        label = sample['label']
        
        if random.random() < 0.5:
            img_l = ImageEnhance.Color(img_l)
            img_l = img_l.enhance(random.uniform(self.low, self.high))
    
            img_r = ImageEnhance.Color(img_r)
            img_r = img_r.enhance(random.uniform(self.low, self.high))

        return {'image_l': img_l, 'image_r': img_r, 'label': label}


class RandomContrast(object):
    """Set random contrast for the image."""
    def __init__(self, low, high):
        self.high = high
        self.low = low

    def __call__(self, sample):
        img_l = sample['image_l']
        img_r = sample['image_r']
        label = sample['label']
        
        if random.random() < 0.5:
            img_l = ImageEnhance.Contrast(img_l)
            img_l = img_l.enhance(random.uniform(self.low, self.high))
    
            img_r = ImageEnhance.Contrast(img_r)
            img_r = img_r.enhance(random.uniform(self.low, self.high))

        return {'image_l': img_l, 'image_r': img_r, 'label': label}


class RandomSharpness(object):
    """Set random sharpness for the image."""
    def __init__(self, low, high):
        self.high = high
        self.low = low

    def __call__(self, sample):
        img_l = sample['image_l']
        img_r = sample['image_r']
        label = sample['label']
        
        if random.random() < 0.5:
            img_l = ImageEnhance.Sharpness(img_l)
            img_l = img_l.enhance(random.uniform(self.low, self.high))
    
            img_r = ImageEnhance.Sharpness(img_r)
            img_r = img_r.enhance(random.uniform(self.low, self.high))

        return {'image_l': img_l, 'image_r': img_r, 'label': label}


class RandomGaussianNoise(object):
    """Random add Gaussian noise to the image."""
    def __init__(self, mean=0.0, std=5.0):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img_l = sample['image_l']
        img_r = sample['image_r']
        label = sample['label']
        
        if random.random() < 0.5:
            img_r = np.array(img_r)
            noise = np.random.normal(self.mean, self.std, img_r.shape)
            img_r = img_r + noise
            img_r = np.clip(img_r, 0.0, 255)
            img_r = Image.fromarray(img_r.astype(np.uint8))

        return {'image_l': img_l, 'image_r': img_r, 'label': label}


class SingleImageResizeImage(object):
    def __init__(self, size=(256, 256), padding=(0, 0), mode="RGB"):
        self.size = size
        self.padding = padding
        self.mode = mode

    def __call__(self, image, ):
        height, width = self.size
        pad_h, pad_w = self.padding

        if self.mode == "RGB":
            image = image.resize((width, height), Image.BICUBIC)
            img_padded = Image.new(self.mode, (width + pad_w, height + pad_h), (0, 0, 0))
        else:
            image = image.resize((width, height), Image.NEAREST)
            img_padded = Image.new(self.mode, (width + pad_w, height + pad_h), (0))
        img_padded.paste(image, (pad_w, pad_h // 2, width, height + pad_h // 2))

        return img_padded


class SingleImageNormalize(object):
    """Normalize a tensor image."""
    def __call__(self, image):
        image = image
        image = np.array(image).astype(np.float32) / 255.

        return image


class SingleImageToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = np.array(image).astype(np.float32).transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float()

        return image


class SingleImageToNumpy(object):
    """Convert PIL.Image in sample to ndarray."""

    def __call__(self, image):
        image = np.array(image).astype(np.uint8)

        return image


train_transforms = transforms.Compose([
    ResizeImage(size=(args.h_size, args.w_size), padding=(args.padding_h_size, args.padding_w_size)),
    LabelRemap(),
    RandomHorizontalFlip(),
    RandomGaussianBlur(),
    RandomBrightness(0.6, 1.4),
    RandomColor(0.7, 1.3),
    RandomContrast(0.7, 1.3),
    RandomSharpness(0, 2.0),
    RandomGaussianNoise(),
    RandomScaleCrop(base_size=(args.h_size, args.w_size),
                    crop_size=(args.h_size + args.padding_h_size, args.w_size + args.padding_w_size),
                    fill=255),
    Normalize(),
    ToTensor()])

test_transforms = transforms.Compose([
    ResizeImage(),
    LabelRemap(),
    Normalize(),
    ToTensor()])
