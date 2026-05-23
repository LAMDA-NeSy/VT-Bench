'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
* Based on MMCL codebase https://github.com/paulhager/MMCL-Tabular-Imaging/blob/main/datasets/ImagingAndTabularDataset.py
'''
from typing import List, Tuple
import random
import csv
import copy

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import transforms
from torchvision.io import read_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Normalize # <--- 确保有这一行
import numpy as np
import os
import sys
from os.path import join

current_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from utils.utils import grab_hard_eval_image_augmentations

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

def to_hwc_for_convert(x, **kwargs):
    # HxW -> HxWx1
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return x[:, :, None]
    return x

def scale_to_01_if_uint8(x, **kwargs):
    if isinstance(x, np.ndarray) and x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    return x

class ImagingAndTabularDataset(Dataset):
  """
  Multimodal dataset that imaging and tabular data for evaluation.
  Load mask csv to imitate missing tabular data
  missing_strategy: value or feature
  missing_rate: 0.0 to 1.0

  The imaging view has {eval_train_augment_rate} chance of being augmented.
  The tabular view corruption rate to be augmented.
  """
  def __init__(
      self,
      data_path_imaging: str, delete_segmentation: bool, eval_train_augment_rate: float, 
      data_path_tabular: str, field_lengths_tabular: str, eval_one_hot: bool,
      labels_path: str, img_size: int, live_loading: bool, train: bool, target: str,
      corruption_rate: float, data_base: str, missing_tabular: str=False, missing_strategy: str='value', missing_rate: float=0.0, augmentation_speedup: bool=False, 
      algorithm_name: str=None,
      task: str='classification'
      ) -> None:

    # Imaging
    self.task = task
    self.missing_tabular = missing_tabular
    self.data_imaging = torch.load(data_path_imaging)

    self.delete_segmentation = delete_segmentation
    self.eval_train_augment_rate = eval_train_augment_rate
    self.live_loading = live_loading
    self.augmentation_speedup = augmentation_speedup
    self.dataset_name = target.lower()
    # self.dataset_name = data_path_tabular.split('/')[-1].split('_')[0]

    if self.delete_segmentation:
      for im in self.data_imaging:
        im[0,:,:] = 0

    self.transform_train = grab_hard_eval_image_augmentations(img_size, target, augmentation_speedup=augmentation_speedup)

    if augmentation_speedup:
      if self.dataset_name == 'dvm':
        self.default_transform = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts)
        ])
        print('Using dvm transform for default transform')
      elif self.dataset_name in ['cardiac', 'cad', 'infarction']:
        self.default_transform = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts_01)
        ])
        print('Using cardiac transform for default transform in ImagingAndTabularDataset')
      elif self.dataset_name in ['pneumonia', 'los', 'rr']:
        self.default_transform = A.Compose([
            # 1. 尺寸
            A.Resize(height=img_size, width=img_size),
            # 2. 强制转 RGB (匹配 ImageNet 模型输入需求)
            A.ToRGB(p=1.0), 
            # 3. [必须与训练代码一致] 使用 ImageNet 统计量
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0
            ),
            # 4. 转 Tensor
            ToTensorV2(),
            A.Lambda(name="make_contig", image=lambda x, **k: x.contiguous()),
        ])

      elif self.dataset_name == 'adoption': # <-- 确保你的文件名解析出来是 'adoption'
        print(f'Using adoption transform for default transform (Albumentations)')
        # --- 修改这里 ---
        self.default_transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            ToTensorV2() # <--- 添加 (您之前的代码里漏了这行)
        ])
        # --- 修改结束 ---
      elif self.dataset_name in ['celeba', 'pawpularity', 'anime']:
          print(f'Using standard (0-255 -> 0-1) transform for CelebA (Albumentations)')
          self.default_transform = A.Compose([
              A.Resize(height=img_size, width=img_size),
              A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
              ToTensorV2() # 自动处理 [0, 255] -> [0.0, 1.0] 和 HWC -> CHW
          ])
      elif self.dataset_name == 'breast_cancer': # <-- 替换为您数据集的名称
          print(f'Using Breast Cancer transform (Resize + L-to-RGB + NORMALIZE + ToTensor)')
          
          self.default_transform = A.Compose([
              A.Resize(height=img_size, width=img_size),
              A.ToRGB(p=1.0),
              # --- [修复] 添加下面这一行 ---
              # A.Normalize 会自动将 uint8 [0, 255] 转为 float32 并除以 255.0
              A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
              # --- 修复结束 ---
              ToTensorV2()
          ])
      elif self.dataset_name == 'skin_cancer': 
        self.default_transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            ToTensorV2()
        ])

      else:
          # 修正一下这里的报错方式
          raise ValueError(f'Unsupported dataset: {self.dataset_name}.')  
    else:
      self.default_transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ])

    # Tabular
    self.data_tabular = np.array(self.read_and_parse_csv(data_path_tabular))
    self.generate_marginal_distributions()
    self.field_lengths_tabular = np.array(torch.load(field_lengths_tabular))  # 原始顺序
    self.eval_one_hot = eval_one_hot
    self.c = corruption_rate if corruption_rate else None

    # Missing mask
    self.missing_strategy = missing_strategy
    self.algorithm_name = algorithm_name
    if self.missing_tabular:
      tabular_name = data_path_tabular.split('/')[-1].split('.')[0]
      missing_mask_path = join(data_base, 'missing_mask', f'{tabular_name}_{target}_{missing_strategy}_{missing_rate}.npy')
      self.missing_mask_data = np.load(missing_mask_path)
      print(f'Load missing mask from {missing_mask_path}')
      assert len(self.data_imaging) == self.missing_mask_data.shape[0]

      if self.eval_one_hot and missing_strategy in set(['feature', 'MI', 'LI']):
        # 先在 field_lengths_tabular 上做“按列剔除”
        self.field_lengths_tabular = self.field_lengths_tabular[~self.missing_mask_data[0]]
        print('Onehot input tabular feature size: ', len(self.field_lengths_tabular), int(np.sum(self.field_lengths_tabular)))
      else:
        print('Transformer input tabular feature size: ', len(self.field_lengths_tabular), len(self.field_lengths_tabular))

    # === 新增：根据（可能已经被 mask 过的）field_lengths_tabular 计算 cat/con 索引，并构造重排顺序 ===
    self.cat_indices = [i for i, fl in enumerate(self.field_lengths_tabular) if fl > 1]
    self.con_indices = [i for i, fl in enumerate(self.field_lengths_tabular) if fl == 1]
    self.reorder_indices = np.array(self.cat_indices + self.con_indices, dtype=np.int64)

    # 对应的重排后的 field_lengths，用于 one-hot（非 missing_tabular 情况）
    self.field_lengths_reordered = self.field_lengths_tabular[self.reorder_indices]

    print(f"[ImagingAndTabularDataset] num_cat (from lengths>1): {len(self.cat_indices)}")
    print(f"[ImagingAndTabularDataset] num_con (from lengths==1): {len(self.con_indices)}")
    
    # Classifier
    self.labels = torch.load(labels_path)

    self.train = train
    print("self.train:",self.train)
    assert len(self.data_imaging) == len(self.data_tabular) == len(self.labels) 
  
  def read_and_parse_csv(self, path_tabular: str) -> List[List[float]]:
    """
    Does what it says on the box.
    """
    with open(path_tabular,'r') as f:
      reader = csv.reader(f)
      data = []
      for r in reader:
        r2 = [float(r1) for r1 in r]
        data.append(r2)
    return data

  def generate_marginal_distributions(self) -> None:
    """
    Generates empirical marginal distribution by transposing data
    """
    data = np.array(self.data_tabular)
    self.marginal_distributions = np.transpose(data)

  def corrupt(self, subject: List[float]) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = copy.deepcopy(subject)
    subject = np.array(subject)

    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    for i in indices:
      marg_dist = self.marginal_distributions[i][~self.missing_mask_data[:,i]] if self.missing_tabular else self.marginal_distributions[i]
      if marg_dist.size != 0:
        value = np.random.choice(marg_dist, size=1)
        subject[i] = value
    return subject

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.eval_one_hot:
      return int(np.sum(self.field_lengths_tabular))
    else:
      return len(self.field_lengths_tabular)

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features.

    - 如果 missing_tabular=False：
        subject 已经在 __getitem__ 里重排为 [cat, ..., con]，
        对应的 field_lengths 使用 self.field_lengths_reordered。
    - 如果 missing_tabular=True 且 strategy in ['feature','MI','LI'] 且 eval_one_hot:
        subject 会先按 missing_mask 删掉整列，
        field_lengths_tabular 在 __init__ 时已经做了同样的列过滤，此时两者顺序一致。
    """

    # 1) 根据是否使用 “feature-level missing” 决定 field_lengths 使用哪个版本
    if self.missing_tabular and self.missing_strategy in set(['feature', 'MI', 'LI']) and self.eval_one_hot:
      # 这里 subject 还是“原始顺序”的子集（按 missing_mask 删列后）
      mask_row = torch.from_numpy(self.missing_mask_data[0]).to(subject.device)  # shape: [num_features]
      subject = subject[~mask_row]  # 删掉整列 missing 的特征
      field_lengths = self.field_lengths_tabular  # 已在 __init__ 里做过相同的过滤
    else:
      # 没有 feature-level missing，subject 已在 __getitem__ 里重排为 cat-first
      field_lengths = self.field_lengths_reordered if hasattr(self, 'field_lengths_reordered') else self.field_lengths_tabular

    # 2) 逐列做 one-hot / 保留数值
    out = []
    assert len(subject) == len(field_lengths), \
        f"subject length {len(subject)} != field_lengths length {len(field_lengths)}"

    for i in range(len(subject)):
      field_len = int(field_lengths[i])
      if field_len == 1:
        # 连续特征，直接保留数值
        out.append(subject[i].unsqueeze(0))
      else:
        # 类别特征：做 one-hot，先 clamp 防止极端越界
        idx = subject[i].long()
        idx = torch.clamp(idx, min=0, max=field_len - 1)
        out.append(torch.nn.functional.one_hot(idx, num_classes=field_len).to(subject.dtype))

    return torch.cat(out)
  
  def _reorder_subject_np(self, subject_np: np.ndarray) -> np.ndarray:
    """
    将原始顺序的 1D numpy 向量 subject_np
    重排为 [所有类别特征 | 所有连续特征] 的顺序。
    """
    return subject_np[self.reorder_indices]


  def __getitem__(self, index: int):
        try:
            # 1. 获取路径
            im_path = self.data_imaging[index]
            # [关键] 强制转为 Python 原生字符串，防止 pathlib 对象或 numpy str 引发 collate 问题
            path = str(im_path) 

            # ================== Part 1: 读取图片 ==================
            if self.live_loading:
                if self.augmentation_speedup:
                    # 加载 .npy
                    # [关键] 加上 .copy()，确保 numpy 数组拥有一块独立的、连续的内存，不与文件句柄挂钩
                    im = np.load(im_path[:-4]+'.npy', allow_pickle=True).copy()
                else:
                    # 读取原始图片
                    im_tensor = read_image(im_path) 
                    # [关键] .numpy().copy() 切断与 PyTorch 底层 allocator 的联系
                    im = im_tensor.permute(1, 2, 0).numpy().copy()
            
            # 2. 获取 Tabular 数据
            # [关键] 强制 copy，防止 view 引用
            subj_np = self.data_tabular[index].copy()
            
            # ================== Part 2: 应用增强 ==================
            if self.train and (random.random() <= self.eval_train_augment_rate):
                res = self.transform_train(image=im)
                im = res['image']
                if self.c and self.c > 0:
                    tab_np = self.corrupt(subj_np)
                else:
                    tab_np = subj_np.copy()
            else:
                res = self.default_transform(image=im)
                im = res['image']
                tab_np = subj_np.copy()

            # ================== Part 3: 终极清洗 (Paranoid Mode) ==================
            
            # --- 处理图片 (Image) ---
            # 如果是 Tensor，先转 numpy 再转回来，或者直接 clone，确保斩断联系
            if isinstance(im, torch.Tensor):
                im = im.detach().cpu().numpy() # 先退回 numpy
            
            # 此时 im 必须是 numpy (H, W, C)
            # 强制检查维度，防止 (H, W) 灰度图混入
            if im.ndim == 2:
                im = np.expand_dims(im, axis=-1) # (H, W) -> (H, W, 1)
                im = np.repeat(im, 3, axis=-1)   # (H, W, 1) -> (H, W, 3) 暴力转 RGB
            elif im.ndim == 3 and im.shape[2] == 1:
                im = np.repeat(im, 3, axis=-1)   # 灰度通道复制
            
            # 创建全新的 Tensor，不共享内存
            im_tensor = torch.tensor(im, dtype=torch.float32)
            
            # 调整为 (C, H, W)
            if im_tensor.shape[0] != 3 and im_tensor.shape[2] == 3:
                im_tensor = im_tensor.permute(2, 0, 1)
            
            # 归一化
            if im_tensor.max() > 1.0:
                 im_tensor = im_tensor / 255.0

            # 最终的连续化
            im_tensor = im_tensor.contiguous()

            # --- 处理表格 (Tabular) ---
            if not self.missing_tabular:
                tab_np = self._reorder_subject_np(tab_np)
            
            # 创建全新的 Tensor
            tab_tensor = torch.tensor(tab_np, dtype=torch.float32).contiguous()

            if self.eval_one_hot:
                tab_tensor = self.one_hot_encode(tab_tensor).to(torch.float).contiguous()

            # --- 处理 Label ---
            if self.task == 'regression':
                label = torch.tensor(self.labels[index], dtype=torch.float)
            else:
                # 即使是 scalar 也要 clone
                label = torch.tensor(self.labels[index], dtype=torch.long).clone().detach()

            # ================== Part 4: 返回结果 ==================
            if self.missing_tabular:
                # [关键] 这里的 missing_mask 是最容易出问题的
                # 务必使用 torch.tensor(..., copy=True) 或者是 .clone()
                # 确保它不是 Bool 类型 (DataLoader 对 BoolTensor 支持有时有 bug) -> 转 Float 或 Long
                mask_data = self.missing_mask_data[index]
                missing_mask = torch.tensor(mask_data, dtype=torch.float32).contiguous()
                
                return (im_tensor, tab_tensor, missing_mask, path), label
            else:
                return (im_tensor, tab_tensor, path), label

        except Exception as e:
            # 这是一个非常有用的 Debug 手段
            # 如果某个样本处理失败，它会打印具体的索引和错误，而不是让 DataLoader 吞掉错误
            print(f"!!! Error loading index {index}: {e}")
            print(f"Path: {self.data_imaging[index]}")
            raise e

  def __len__(self) -> int:
    return len(self.data_tabular)
  
  
if __name__ == '__main__':
  dataset = ImagingAndTabularDataset(
    data_path_imaging='/bigdata/siyi/data/DVM/features/val_paths_all_views.pt', delete_segmentation=False, eval_train_augment_rate=0.8, 
          data_path_tabular='/bigdata/siyi/data/DVM/features/dvm_features_val_noOH_all_views_physical_jittered_50_reordered.csv', 
          field_lengths_tabular='/bigdata/siyi/data/DVM/features/tabular_lengths_all_views_physical_reordered.pt', eval_one_hot=False,
          labels_path='/bigdata/siyi/data/DVM/features/labels_model_all_val_all_views.pt', img_size=128, live_loading=True, train=True, target='dvm',
          corruption_rate=0.3, data_base='/bigdata/siyi/data/DVM/features', missing_tabular=True, 
          missing_strategy='feature', missing_rate=0.7, augmentation_speedup=True, algorithm_name='DAFT'
  )
  print(dataset[0][0][2], dataset[0][0][2].dtype)
  print(dataset.missing_mask_data.sum()/dataset.missing_mask_data.size)

  # tab=self.one_hot_encode(tab).to(torch.float)