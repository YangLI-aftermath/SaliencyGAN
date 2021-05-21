# -*- coding=utf-8 -*-
import os
import cv2
import numpy as np
"""
image preprogressing resize ro same wd and ht
"""
def image_resize(image=0, image_size = 256, image_channel = 3):
    ht, wd = image.shape[0:2]
    src_shape = image.shape
    new_ht = 0
    new_wd = 0
    imagec = 0
    if ht > wd:
        new_ht = image_size
        new_wd = wd * new_ht / ht
        new_wd = int(new_wd)
        new_ht = int(new_ht)
        if image.ndim == 2:
          pad_zeros = np.zeros((image_size, image_size - new_wd), dtype = image.dtype)
        else:
          pad_zeros = np.zeros((image_size, image_size - new_wd, image_channel), dtype = image.dtype)
        # print(new_ht, " ", new_wd)
        imagec = cv2.resize(image, (new_wd, new_ht))  # make sure max(wd,ht) = cg.image_size
        imagec = np.hstack((imagec, pad_zeros))
    else:
        new_wd = image_size
        new_ht = ht * new_wd / wd
        new_wd = int(new_wd)
        new_ht = int(new_ht)
        if image.ndim == 2:
          pad_zeros = np.zeros((image_size - new_ht, image_size), dtype=image.dtype)
        else: 
          pad_zeros = np.zeros((image_size-new_ht, image_size, image_channel), dtype=image.dtype)
        # print(new_ht, " ", new_wd)
        imagec = cv2.resize(image, (new_wd, new_ht))  # make sure max(wd,ht) = cg.image_size
        imagec = np.vstack((imagec, pad_zeros))
    resized_shape = imagec.shape
    return imagec, src_shape, resized_shape

"""
  image normalize with mask
"""
def img_normalize_with_mask(images, mask):
  images[:, :, 2] -= np.mean(images[:,:,2])
  images[:, :, 1] -= np.mean(images[:,:,1])
  images[:, :, 0] -= np.mean(images[:,:,0])

  images[:, :, 2] /= ( np.std(images[:,:,2]) + 1e-12)
  images[:, :, 1] /= ( np.std(images[:,:,1]) + 1e-12)
  images[:, :, 0] /= ( np.std(images[:,:,0]) + 1e-12)

  return images, mask

"""
  image normalize
"""
def img_normalize(images):
  images[:, :, 2] -= np.mean(images[:,:,2])
  images[:, :, 1] -= np.mean(images[:,:,1])
  images[:, :, 0] -= np.mean(images[:,:,0])

  images[:, :, 2] /= ( np.std(images[:,:,2]) + 1e-12)
  images[:, :, 1] /= ( np.std(images[:,:,1]) + 1e-12)
  images[:, :, 0] /= ( np.std(images[:,:,0]) + 1e-12)
  return images