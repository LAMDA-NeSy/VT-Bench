from typing import List, Tuple
import random
import csv
import copy
import numpy as np

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import transforms
from torchvision.io import read_image
import albumentations as A
from albumentations import Normalize
from albumentations.pytorch import ToTensorV2

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

class ContrastiveImagingAndTabularDataset(Dataset):
  """
  Multimodal dataset that generates multiple views of imaging and tabular data for contrastive learning.

  The first imaging view is always augmented. The second has {augmentation_rate} chance of being augmented.
  The first tabular view is never augmented. The second view is corrupted by replacing {corruption_rate} features
  with values chosen from the empirical marginal distribution of that feature.
  """
  def __init__(
      self, 
      data_path_imaging: str, delete_segmentation: bool, augmentation: transforms.Compose, augmentation_rate: float, 
      data_path_tabular: str, corruption_rate: float, field_lengths_tabular: str, one_hot_tabular: bool,
      labels_path: str, img_size: int, live_loading: bool, augmentation_speedup: bool=False, target: str='none',
      task: str='classification' # <--- [新增] 任务参数
      ) -> None:
            
    # Imaging
    self.task = task # <--- [新增] 保存任务类型
    self.data_imaging = torch.load(data_path_imaging)
    self.transform = augmentation
    self.delete_segmentation = delete_segmentation
    self.augmentation_rate = augmentation_rate
    self.live_loading = live_loading
    self.augmentation_speedup = augmentation_speedup
    # self.dataset_name = data_path_tabular.split('/')[-1].split('_')[0]
    self.dataset_name = target.lower()

    if self.delete_segmentation:
      for im in self.data_imaging:
        im[0,:,:] = 0

    if augmentation_speedup:
      if self.dataset_name == 'dvm':
        self.default_transform = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts)
        ])
        print(f'Using dvm transform for default transform in ContrastiveImagingAndTabularDataset')
      elif self.dataset_name in ['cardiac', 'cad', 'infarction']:
        self.default_transform = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts_01)
        ])
        print(f'Using cardiac transform for default transform in ContrastiveImagingAndTabularDataset')
        # --- 修改结束 ---
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
      elif self.dataset_name in ['celeba', 'adoption', 'pawpularity','anime']:
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
              A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
              ToTensorV2()
          ])   
      elif self.dataset_name == 'skin_cancer':
        print(f'Using Skin Cancer transform (Resize + 0-1 Norm)')
        self.default_transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            ToTensorV2()   # 就够了，默认会把 uint8 → float32 /255
        ])
      else:
          # 修正一下这里的报错方式
          raise ValueError(f'Unsupported dataset: {self.dataset_name}.')  
    else:
      self.default_transform = transforms.Compose([
        transforms.Resize(size=(img_size,img_size)),
        # transforms.Lambda(lambda x : x.float())
        transforms.Lambda(convert_to_float)
      ])

    # Tabular
    # self.data_tabular = self.read_and_parse_csv(data_path_tabular)
    # self.generate_marginal_distributions(data_path_tabular)
    print("Loading tabular data from CSV...")
    data_df = pd.read_csv(data_path_tabular, header=None, dtype=np.float32)
    self.data_tabular = data_df.values 
    self.generate_marginal_distributions(data_df) 
    print("Tabular data loaded.")
    self.c = corruption_rate
    self.field_lengths_tabular = torch.load(field_lengths_tabular)
    self.one_hot_tabular = one_hot_tabular

    # Change the order of features to categorical, continuous 
    
    # Classifier
    self.labels = torch.load(labels_path)
    if not isinstance(self.labels, torch.Tensor):
      self.labels = torch.as_tensor(np.asarray(self.labels))
    
  
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
    # data = np.array(self.data_tabular)
    # self.marginal_distributions = np.transpose(data)
    # data_df = pd.read_csv(data_path, header=None)
    self.marginal_distributions = data_df.transpose().values.tolist()

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.one_hot_tabular:
      return int(sum(self.field_lengths_tabular))
    else:
      return len(self.data_tabular[0])

  def corrupt(self, subject: List[float]) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = copy.deepcopy(subject)

    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    for i in indices:
      subject[i] = random.sample(self.marginal_distributions[i],k=1)[0] 
    return subject

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(subject[i].long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def generate_imaging_views(self, index: int) -> List[torch.Tensor]:
    """
    Generates two views of a subjects image. Also returns original image resized to required dimensions.
    The first is always augmented. The second has {augmentation_rate} chance to be augmented.
    """
    im = self.data_imaging[index]
    if self.live_loading:
      if self.augmentation_speedup:
        im = np.load(im[:-4]+'.npy', allow_pickle=True)
      else:
        im = read_image(im)
        im = im / 255 if self.dataset_name == 'dvm' else im
    # ims = [self.transform(image=im)['image']] if self.augmentation_speedup else [self.transform(im)]
    ims = [torch.tensor(0, dtype=torch.float)] # Placeholder
    if random.random() < self.augmentation_rate:
      ims.append(self.transform(image=im)['image'] if self.augmentation_speedup else self.transform(im))
    else:
      ims.append(self.default_transform(image=im)['image'] if self.augmentation_speedup else self.default_transform(im))

    orig_im = self.default_transform(image=im)['image'] if self.augmentation_speedup else self.default_transform(im)
    
    if isinstance(orig_im, torch.Tensor) and orig_im.dtype == torch.uint8:
        orig_im = orig_im.float() / 255.0
    for k in range(len(ims)):
        if isinstance(ims[k], torch.Tensor) and ims[k].dtype == torch.uint8:
            ims[k] = ims[k].float() / 255.0

    return ims, orig_im

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    imaging_views, unaugmented_image = self.generate_imaging_views(index)
    # tabular_views = [torch.tensor(self.data_tabular[index], dtype=torch.float), torch.tensor(self.corrupt(self.data_tabular[index]), dtype=torch.float)]
    tabular_row_numpy = self.data_tabular[index]
    tabular_row_list = list(tabular_row_numpy) 
    tabular_views = [
        torch.tensor(tabular_row_numpy, dtype=torch.float), # 原始视图 (从 numpy 创建)
        torch.tensor(self.corrupt(tabular_row_list), dtype=torch.float) # 损坏的视图 (从 list 创建)
    ]
    if self.one_hot_tabular:
      tabular_views = [self.one_hot_encode(tv) for tv in tabular_views]
    # label = torch.tensor(self.labels[index], dtype=torch.long)
    # --- [修改] 根据任务类型转换 Label ---
    x = self.labels[index]  # 现在肯定是 Tensor

    if self.task == "regression":
        label = x.to(torch.float32)   # 或 torch.float
    else:
        label = x.to(torch.long)
        
    return imaging_views, tabular_views, label, unaugmented_image

  def __len__(self) -> int:
    return len(self.data_tabular)