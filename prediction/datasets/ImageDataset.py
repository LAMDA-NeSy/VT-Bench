from typing import Tuple
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.io import read_image

from utils.utils import grab_hard_eval_image_augmentations, grab_soft_eval_image_augmentations, grab_image_augmentations

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Normalize # <--- 确保有这一行

def convert_to_float(x):
  return x.float()

def convert_to_ts(x, **kwargs):
  x = np.clip(x, 0, 255) / 255
  x = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x

def convert_to_ts_01(x, **kwargs):
  x = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x


class ImageDataset(Dataset):
  """
  Dataset for the evaluation of images
  """
  def __init__(self, data: str, labels: str, delete_segmentation: bool, eval_train_augment_rate: float, img_size: int, target: str, train: bool, live_loading: bool, task: str,
                dataset_name:str='dvm', augmentation_speedup:bool=False) -> None:
      super(ImageDataset, self).__init__()
      self.train = train
      self.eval_train_augment_rate = eval_train_augment_rate
      self.live_loading = live_loading
      self.task = task

      self.dataset_name = dataset_name
      self.augmentation_speedup = augmentation_speedup

      self.data = torch.load(data)
      self.labels = torch.load(labels)

      if delete_segmentation:
          for im in self.data:
              im[0,:,:] = 0

      self.transform_train = grab_hard_eval_image_augmentations(img_size, target, augmentation_speedup=self.augmentation_speedup)

      if self.augmentation_speedup:
          target_lower = self.dataset_name.lower()
          
          if target_lower == 'dvm':
              self.transform_val = A.Compose([
                  A.Resize(height=img_size, width=img_size),
                  A.Lambda(name='convert2tensor', image=convert_to_ts)
              ])
              print('Using dvm transform for val transform in ImageDataset')
          
          elif target_lower == 'cardiac':
              self.transform_val = A.Compose([
                  A.Resize(height=img_size, width=img_size),
                  A.Lambda(name='convert2tensor', image=convert_to_ts_01)
              ])
              print('Using cardiac transform for val transform in ImageDataset')
          
          elif target_lower in ['celeba', 'adoption', 'pawpularity', 'anime']:
              print(f'Using standard (0-255 -> 0-1) transform for {self.dataset_name} (Albumentations)')
              self.transform_val = A.Compose([
                  A.Resize(height=img_size, width=img_size),
                  A.Normalize(
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
                      max_pixel_value=255.0
                  ),
                  ToTensorV2() 
              ])
          
          # === [新增] Pneumonia 分支 ===
          elif target_lower in ['pneumonia', 'los', 'rr']:
              print(f'Using Pneumonia transform for val transform (Resize + ImageNet Norm)')
              self.transform_val = A.Compose([
                  A.Resize(height=img_size, width=img_size),
                  # Pneumonia 使用标准的 ImageNet 归一化
                  A.Normalize(
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
                      max_pixel_value=255.0
                  ),
                  ToTensorV2()
              ])
          # ============================

          elif target_lower == 'breast_cancer': 
              print(f'Using Breast Cancer transform (Resize + L-to-RGB + NORMALIZE + ToTensor)')
              self.transform_val = A.Compose([
                  A.Resize(height=img_size, width=img_size),
                  A.ToRGB(p=1.0),
                  A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
                  ToTensorV2()
              ])
          
          elif target_lower == 'skin_cancer': 
              print(f'Using Skin Cancer transform (Resize + 0-1 Norm)')
              self.transform_val = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=(0.0, 0.0, 0.0),
                            std=(255.0, 255.0, 255.0),
                            max_pixel_value=255.0),
                ToTensorV2()
            ])
              
          else:
              # [修复] 原代码 raise print(...) 是语法错误，修正为 raise ValueError
              raise ValueError(f'Unsupported dataset: {self.dataset_name}. Supported: dvm, cardiac, celeba, adoption, pneumonia, etc.')
      else:
          self.transform_val = transforms.Compose([
              transforms.Resize(size=(img_size,img_size)),
              transforms.Lambda(convert_to_float)
          ])


  def __getitem__(self, indx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns an image for evaluation purposes.
    If training, has {eval_train_augment_rate} chance of being augmented.
    If val, never augmented.
    """
    im = self.data[indx]
    if self.live_loading:
      if self.augmentation_speedup:
        im = np.load(im[:-4]+'.npy', allow_pickle=True)
      else:
        im = read_image(im)
        im = im / 255

    if self.train and (random.random() <= self.eval_train_augment_rate):
      im = self.transform_train(image=im)['image'] if self.augmentation_speedup else self.transform_train(im)
    else:
      im = self.transform_val(image=im)['image'] if self.augmentation_speedup else self.transform_val(im)
    
    # label = self.labels[indx]
    # return (im), label
    raw_label = self.labels[indx]
    
    if self.task == 'regression':
        # 回归任务强制转为 float32
        label = torch.tensor(raw_label, dtype=torch.float)
    else:
        # 分类任务强制转为 long (int64)
        label = torch.tensor(raw_label, dtype=torch.long)
        
    return (im), label

  def __len__(self) -> int:
    return len(self.labels)
