'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
* Based on MMCL codebase https://github.com/paulhager/MMCL-Tabular-Imaging/blob/main/datasets/ContrastiveImagingAndTabularDataset.py
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
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
from PIL import Image


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


def scale_to_01_if_uint8(x, **kwargs):
    if isinstance(x, np.ndarray) and x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    return x

class ContrastiveReconstructImagingAndTabularDataset(Dataset):
  """
  Multimodal dataset that generates multiple views of imaging and tabular data for contrastive learning.
  The first imaging view is always augmented. The second has {augmentation_rate} chance of being augmented.
  The first tabular view is never augmented. The second view is masked and replaced with mask_rate and replace_rate
  with values chosen from the empirical marginal distribution of that feature.
  """
  def __init__(
      self, 
      data_path_imaging: str, delete_segmentation: bool, augmentation: transforms.Compose, augmentation_rate: float, 
      data_path_tabular: str, corruption_rate: float, replace_random_rate: float, replace_special_rate: float, field_lengths_tabular: str, one_hot_tabular: bool,
      labels_path: str, img_size: int, live_loading: bool, augmentation_speedup: bool=False, target: str='none',
      task: str='classification' # <--- [新增] 任务类型参数
      ) -> None:
            
    # Imaging
    self.task = task # <--- [新增] 保存任务类型
    self.data_imaging = torch.load(data_path_imaging)
    self.transform = augmentation


    self.delete_segmentation = delete_segmentation
    self.augmentation_rate = augmentation_rate
    self.live_loading = live_loading
    self.augmentation_speedup = augmentation_speedup
   
    self.dataset_name = target.lower()
    # self.dataset_name = data_path_tabular.split('/')[-1].split('_')[0]

    if self.delete_segmentation:
      for im in self.data_imaging:
        im[0,:,:] = 0

    print(f"!!!!!!!!!!!!!!!!!!!!!!!!dataset_name: {self.dataset_name} && augmentation_speedup:{augmentation_speedup}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    if augmentation_speedup:
      if self.dataset_name == 'dvm':
        self.default_transform = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts)
        ])
        print(f'Using dvm transform for default transform in ContrastiveReconstructImagingAndTabularDataset')
      elif self.dataset_name in ['cardiac', 'cad', 'infarction']:
        self.default_transform = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts_01)
        ])
        print(f'Using cardiac transform for default transform in ContrastiveReconstructImagingAndTabularDataset')   
      # 将 pneumonia, los, rr 统一处理
      elif self.dataset_name in ['pneumonia', 'los', 'rr']:
        print(f'Using {self.dataset_name} transform (Resize + RGB + ImageNet Norm)')
        
        self.default_transform = A.Compose([
            # 1. 强制固定尺寸
            A.Resize(height=img_size, width=img_size),
            
            # 2. [关键] 强制转 RGB (3通道)
            # 即使原图是灰度，也会复制成 3 通道，满足模型输入要求
            A.ToRGB(p=1.0),
            
            # 3. [关键] 使用 ImageNet 归一化 (与训练代码保持一致)
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0
            ),
            
            # 4. 转 Tensor (自动处理 HWC -> CHW)
            ToTensorV2()
        ])

      elif self.dataset_name in ['celeba', 'adoption', 'pawpularity', 'anime']:
          print(f'Using standard (0-255 -> 0-1) transform for CelebA (Albumentations)')
          self.default_transform = A.Compose([
              A.Resize(height=img_size, width=img_size),
              A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
              ToTensorV2() # 自动处理 [0, 255] -> [0.0, 1.0] 和 HWC -> CHW
          ])
      elif self.dataset_name == 'breast_cancer': # <-- 替换为您数据集的名称
          print(f'Using Breast Cancer transform (Resize + L-to-RGB + 0-1 Norm)')
          
          self.default_transform = A.Compose([
              A.Resize(height=img_size, width=img_size),
              A.ToRGB(p=1.0),
              ToTensorV2()
          ])   
      elif self.dataset_name == 'skin_cancer':
        print(f'Using Skin Cancer transform (Resize + 0-1 Norm)')
        self.default_transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.0, 0.0, 0.0),
                        std=(255.0, 255.0, 255.0),  # 这里的 std=255 刚好等价于除以 255
                        max_pixel_value=255.0),
            ToTensorV2()
        ])
      else:
          # 修正一下这里的报错方式
          raise ValueError(f'Unsupported dataset: {self.dataset_name}.')  
    else:
      self.default_transform = transforms.Compose([
        transforms.Resize(size=(img_size,img_size)),
        transforms.Lambda(convert_to_float)
      ])

    # Tabular
    # self.data_tabular = self.read_and_parse_csv(data_path_tabular)
    # self.generate_marginal_distributions()
    print("Loading tabular data from CSV...")
    print(f"--- [DEBUG] Loading tabular file: {data_path_tabular} ---") # <--- 添加这一行
    data_df = pd.read_csv(data_path_tabular, header=None, dtype=np.float32) 
    self.data_tabular = data_df.values 
    self.generate_marginal_distributions(data_df) 
    print("Tabular data loaded.")
    self.c = corruption_rate
    self.field_lengths_tabular = torch.load(field_lengths_tabular)
    # === 新增：根据 field_lengths_tabular 计算类别/连续列的索引 ===
    #  TIP 的假设：field_len == 1 -> 连续特征； >1 -> 类别特征
    self.cat_indices = [i for i, fl in enumerate(self.field_lengths_tabular) if fl > 1]
    self.con_indices = [i for i, fl in enumerate(self.field_lengths_tabular) if fl == 1]
    
    # === 新增：按照 cat-first 的顺序重排 field_lengths，用于 one_hot ===
    reordered_indices = self.cat_indices + self.con_indices
    self.field_lengths_reordered = torch.tensor(
        [int(self.field_lengths_tabular[i]) for i in reordered_indices],
        dtype=torch.long
    )
    self.one_hot_tabular = one_hot_tabular
    self.replace_random_rate = replace_random_rate
    self.replace_special_rate = replace_special_rate
    
    # Classifier
    self.labels = torch.load(labels_path)
    if not isinstance(self.labels, torch.Tensor):
      self.labels = torch.as_tensor(np.asarray(self.labels))

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

  def generate_marginal_distributions(self, data_df: pd.DataFrame) -> None:
    """
    Generates empirical marginal distribution by transposing data
    """
    # 算法的 'corrupt' 部分需要 list, 所以这里 .tolist() 是必须的
    # self.marginal_distributions = data_df.transpose().values.tolist()
    self.marginal_distributions = data_df.transpose().values


  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.one_hot_tabular:
      return int(sum(self.field_lengths_tabular))
    else:
      return len(self.field_lengths_tabular)

  def corrupt(self, subject: List[float]) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = copy.deepcopy(subject)
    subject = np.array(subject)

    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    pick_value_positions = np.random.choice(self.marginal_distributions.shape[1], size=len(indices))
    subject[indices] = self.marginal_distributions[indices, pick_value_positions]
    return subject
  
# --- 用这个替换你的 Dataset 类中的 mask 函数 ---
  def mask(self, subject: List[float]) -> List[float]:
    subject = copy.deepcopy(subject)
    subject = np.array(subject)

    # 1. 找到所有 *可以被 mask* 的有效索引池
    possible_indices = []
    for i in range(len(subject)):
      field_len = self.field_lengths_tabular[i]
      value = subject[i]
      if field_len == 1:
        if not np.isnan(value):
          possible_indices.append(i)
      else:
        if value >= 0 and value < field_len:
          possible_indices.append(i)

    # 2. 计算要 mask 的总数
    total_rate = self.replace_random_rate + self.replace_special_rate
    if total_rate == 0 or len(possible_indices) == 0:
        indices = []
        num_random = 0
    else:
        # ✅ 更合理：基于 “可被 mask 的字段数” 计算数量
        num_to_mask = round(len(possible_indices) * total_rate)
        num_to_mask = min(num_to_mask, len(possible_indices))
        indices = random.sample(possible_indices, num_to_mask)
        num_random = int(len(indices) * self.replace_random_rate / total_rate) if total_rate > 0 else 0


    num_special = len(indices) - num_random
    
    if num_random > 0: 
      pick_value_positions = np.random.choice(self.marginal_distributions.shape[1], size=num_random)
      subject[indices[:num_random]] = self.marginal_distributions[indices[:num_random], pick_value_positions]

    mask, mask_random, mask_special = np.zeros_like(subject, dtype=bool), np.zeros_like(subject, dtype=bool), np.zeros_like(subject, dtype=bool)
    
    if indices: 
      mask[indices] = True
      mask_random[indices[:num_random]] = True
      mask_special[indices[num_random:]] = True
    
    assert np.sum(mask) == np.sum(mask_random) + np.sum(mask_special)
    
    return subject, mask, mask_special, mask_random

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    
    """
    out = []
    for i in range(len(subject)):
        field_len = int(self.field_lengths_reordered[i])
        if field_len == 1:
            out.append(subject[i].unsqueeze(0))
        else:
            out.append(
                torch.nn.functional.one_hot(
                    subject[i].long(), num_classes=field_len
                ).float()  # 转成 float，方便后面拼别的特征
            )
    return torch.cat(out)

  def generate_imaging_views(self, index: int) -> List[torch.Tensor]:
    """
    Generates two views of a subjects image. Also returns original image resized to required dimensions.
    The first is always augmented. The second has {augmentation_rate} chance to be augmented.
    """
    im = self.data_imaging[index]
    if self.live_loading:
      if self.augmentation_speedup:
        # im = np.load(im[:-4]+'.npy', allow_pickle=True)
        im = np.load(im, allow_pickle=True)
      else:
        im = read_image(im)
        im = im / 255 if self.dataset_name == 'dvm' else im
    ims = [self.transform(image=im)['image']] if self.augmentation_speedup else [self.transform(im)]
    if random.random() < self.augmentation_rate:
      ims.append(self.transform(image=im)['image'] if self.augmentation_speedup else self.transform(im))
    else:
      ims.append(self.default_transform(image=im)['image'] if self.augmentation_speedup else self.default_transform(im))

    orig_im = self.default_transform(image=im)['image'] if self.augmentation_speedup else self.default_transform(im)
    
    if orig_im.dtype == torch.uint8:
        # print(f"--- [FIXING orig_im, index {index}] Manually converting uint8 -> float32 ---")
        orig_im = orig_im.float() / 255.0

    if ims[0].dtype == torch.uint8:
        # print(f"--- [FIXING ims[0], index {index}] Manually converting uint8 -> float32 ---")
        ims[0] = ims[0].float() / 255.0

    if ims[1].dtype == torch.uint8:
        # print(f"--- [FIXING ims[1], index {index}] Manually converting uint8 -> float32 ---")
        ims[1] = ims[1].float() / 255.0

    return ims, orig_im
  

  def _reorder_subject(self, subject: torch.Tensor) -> torch.Tensor:
      """
      将长度为 num_features 的 1D 向量 subject
      从 [原始列顺序] 重排为 [所有类别列 | 所有连续列]
      """
      # subject: 1D tensor, shape [num_features]
      cat_part = subject[self.cat_indices]
      con_part = subject[self.con_indices]
      return torch.cat([cat_part, con_part], dim=0)

  def _reorder_mask(self, mask: torch.Tensor) -> torch.Tensor:
      """
      同样方式重排 mask / mask_special（1D bool tensor）
      """
      cat_part = mask[self.cat_indices]
      con_part = mask[self.con_indices]
      return torch.cat([cat_part, con_part], dim=0)


  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
      imaging_views, unaugmented_image = self.generate_imaging_views(index)

      # ---- 先拿到原始 numpy 向量 ----
      subj_np = self.data_tabular[index]  # shape: [num_features]
      
      # ---- 1) origin / corrupted view ----
      if self.c > 0:
        corrupted_np = self.corrupt(subj_np)  # 仍按原顺序做 corruption
        corrupted = torch.tensor(corrupted_np, dtype=torch.float)
      else:
        corrupted = torch.tensor(subj_np, dtype=torch.float)
      
      # ---- 2) masked view + masks ----
      masked_view_np, mask_np, mask_special_np, mask_random_np = self.mask(subj_np)
      masked_view = torch.from_numpy(masked_view_np).float()
      mask = torch.from_numpy(mask_np.astype(bool))
      mask_special = torch.from_numpy(mask_special_np.astype(bool))
      # mask_random 目前 Trainer 不直接用，但你如果后续要用也可以类似处理
      # mask_random = torch.from_numpy(mask_random_np.astype(bool))

      # ---- 3) 统一重排顺序：cat-first, con-last ----
      corrupted = self._reorder_subject(corrupted)
      masked_view = self._reorder_subject(masked_view)
      mask = self._reorder_mask(mask)
      mask_special = self._reorder_mask(mask_special)

      tabular_views = [corrupted, masked_view, mask, mask_special]

      if self.one_hot_tabular:
        tabular_views = [self.one_hot_encode(tv) for tv in tabular_views]

      # ---- 4) label ----
      # label = self.labels[index].clone().detach().to(torch.long)
      x = self.labels[index]  # 现在肯定是 Tensor

      if self.task == "regression":
          label = x.to(torch.float32)   # 或 torch.float
      else:
          label = x.to(torch.long)

      # ---- 5) unaugmented_tabular 给在线评估用，同样要重排 ----
      unaugmented_tabular = torch.tensor(subj_np, dtype=torch.float)
      unaugmented_tabular = self._reorder_subject(unaugmented_tabular)

      return imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular


  def __len__(self) -> int:
    return len(self.data_tabular)
  

if __name__ == '__main__':
  dataset = ContrastiveReconstructImagingAndTabularDataset(
    data_path_imaging='/bigdata/siyi/data/DVM/features/val_paths_all_views.pt', delete_segmentation=False, augmentation=transforms.Compose([transforms.Resize(size=(128,128)),transforms.Lambda(convert_to_float)]), augmentation_rate=0.5,
    data_path_tabular='/bigdata/siyi/data/DVM/features/dvm_features_val_noOH_all_views_physical_jittered_50_reordered.csv', corruption_rate=0.15, replace_random_rate=0.0, replace_special_rate=0.50, 
    field_lengths_tabular='/bigdata/siyi/data/DVM/features/tabular_lengths_all_views_physical_reordered.pt', one_hot_tabular=False,
    labels_path='/bigdata/siyi/data/DVM/features/labels_model_all_val_all_views.pt', img_size=128, live_loading=True, augmentation_speedup=False
  )
  a = list(range(17))
  x = dataset[3]

# data_path_imaging 这是一个 PyTorch 张量文件，其中存储的是一个字符串列表。列表中的每一个字符串都是指向训练集（train set）中某一张车辆图片的.npy文件的完整文件路径。
# data_path_tabular 所有样本的完整表格特征
# labels_path 这是一个 PyTorch 张量文件，里面存储的是一个整数列表。这个列表包含了验证集中每一个样本对应的标签（Label）。
# field_lengths_tabular 这是一个 PyTorch 张量文件，里面存储的是一个整数列表，这个列表描述了上述表格特征文件中每一列的特征维度。