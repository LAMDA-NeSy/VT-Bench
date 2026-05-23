from typing import List, Tuple
from os.path import join
import os
import sys
from torch import nn
import albumentations as A
from albumentations import Normalize  # <--- 添加
from albumentations.pytorch import ToTensorV2  # <--- 添加
from torchvision import transforms # <--- 确保这个也在

import torch
import numpy as np

def create_logdir(name: str, resume_training: bool, wandb_logger):
    """
    创建一个安全且有权限的日志目录。
    日志将保存在项目根目录下的 'results' 文件夹中。
    """
    # --- 修改部分 开始 ---
    
    # 获取此文件(utils.py)所在的目录
    # __file__ 是一个指向当前文件路径的 Python 内置变量
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 假设项目根目录是 'utils' 文件夹的上一级目录
    project_root = os.path.dirname(current_file_dir)
    
    # 将所有结果保存在项目根目录下的 "results" 文件夹中
    # 这样可以避免权限问题，并且让项目文件更有条理
    basepath = join(project_root, 'results', 'runs', name)
    
    # --- 修改部分 结束 ---

    # 后续逻辑保持不变
    run_name = wandb_logger.name
    logdir = join(basepath, run_name)
    
    if os.path.exists(logdir) and not resume_training:
        raise Exception(f'Run {run_name} already exists. Please delete the folder {logdir} or choose a different run name.')
    
    # exist_ok=True 确保在目录已存在时不会报错，这对于多进程或并发任务是安全的
    os.makedirs(logdir, exist_ok=True)
    
    return logdir

def convert_to_float(x):
  return x.float()


def convert_to_ts(x, **kwargs):
  x = np.clip(x, 0, 255) / 255
  x = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x

def convert_to_ts_01(x, **kwargs):
  x  = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x

def to_hwc_for_convert(x, **kwargs):
    """
    保证输入给 convert_to_ts_01 的是 HWC
    - 灰度 HxW -> HxWx1
    - 彩色 HxWxC -> 原样
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 2:
        x = x[:, :, None]
    return x

def scale_to_01_if_uint8(x, **kwargs):
    """
    不改 convert_to_ts_01 的前提下，保证 uint8 输入转成 [0,1] float32
    - 若输入已经是 float（比如已归一化/窗宽窗位），则不动
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    return x



def grab_image_augmentations(img_size: int, target: str, augmentation_speedup: bool = False, crop_scale_lower: float = 0.08) -> transforms.Compose:
    """
    Defines augmentations to be used with images during contrastive training.
    """
    
    # [新增] 辅助函数：确保 Albumentations 输入是 HWC 且 3 通道
    def _ensure_hwc3(image, **kwargs):
        if image is None: return image
        if image.ndim == 2:   # H x W
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1: # H x W x 1
            image = np.repeat(image, 3, axis=2)
        return image

    # [新增] 辅助函数：Torchvision 转换
    def _to_tensor_if_numpy(x):
        if isinstance(x, np.ndarray):
            import torch
            t = torch.from_numpy(x)
            if t.ndim == 2: t = t.unsqueeze(-1)
            if t.shape[-1] == 1: t = t.repeat(1, 1, 3)
            return t.permute(2, 0, 1).contiguous().float().div(255.0)
        return x

    target = target.lower()

    # ================= DVM =================
    if target == 'dvm':
        if augmentation_speedup:
            transform = A.Compose([
                A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, p=0.8),
                A.ToGray(p=0.2),
                A.GaussianBlur(blur_limit=(29,29), sigma_limit=(0.1,2.0), p=0.5),
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(crop_scale_lower, 1.0), ratio=(0.75, 1.3333333333333333)),
                A.HorizontalFlip(p=0.5),
                ToTensorV2() # 替换 convert_to_ts
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=29, sigma=(0.1, 2.0))],p=0.5),
                transforms.RandomResizedCrop(size=(img_size,img_size), scale=(crop_scale_lower, 1.0), ratio=(0.75, 1.3333333333333333)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor() # 替换 lambda
            ])
        print('Using dvm transform for train augmentation')
    
    # ================= CARDIAC =================
    elif target in ['cardiac','cad', 'infarction']:
        if augmentation_speedup:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=45),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.2, 1.0)),
                ToTensorV2() # 替换 convert_to_ts_01
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.RandomResizedCrop(size=img_size, scale=(0.2,1)),
                transforms.ToTensor()
            ])
        print('Using cardiac transform for train augmentation')

    # ================= PNEUMONIA (重点修复) =================
    elif target in ['pneumonia', 'los', 'rr']:
        if augmentation_speedup:
            # Albumentations 分支
            transform = A.Compose([
                # 1. [关键修复] 强制转为 HWC 3通道！防止单通道报错
                A.Lambda(name="ensure_hwc3", image=_ensure_hwc3),
                
                # 2. [关键修复] 使用 RandomResizedCrop 代替 Pad+Affine
                # 这是训练阶段最标准的增强，既改变了尺寸又增加了变异，且保证输出绝对是 img_size
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
                
                # 其他增强保持不变
                A.Affine(rotate=(-10, 10), translate_percent=(-0.02, 0.02), shear=(-5, 5), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.15, p=0.6),
                A.RandomGamma(gamma_limit=(85, 115), p=0.3),
                A.GaussNoise(var_limit=(1.0, 10.0), p=0.2),
                A.GaussianBlur(blur_limit=(3, 3), p=0.1),
                A.CoarseDropout(max_holes=6, max_height=int(img_size * 0.07), max_width=int(img_size * 0.07),
                                min_holes=1, fill_value=0, p=0.15),

                # Normalize (针对 3 通道)
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                            max_pixel_value=255.0),
                
                # 转 Tensor
                ToTensorV2()
            ])

        else:
            # Torchvision 分支
            transform = transforms.Compose([
                transforms.Lambda(_to_tensor_if_numpy),
                
                # 强制 3 通道 (你之前的代码里有，这里保留)
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x),
                
                transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.0)), # 统一使用 RRC
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02), shear=5),
                transforms.ColorJitter(brightness=0.10, contrast=0.15),
                
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    elif target in ['celeba', 'adoption', 'pawpularity', 'anime']:
        # ...
        if augmentation_speedup:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=45),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.2, 1.0)),
                
                # vvvv 修复：添加下面这块 vvvv
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
                # ^^^^ 修复：添加上面这块 ^^^^
                
                ToTensorV2() 
            ])

    elif target.lower() == 'breast_cancer':
      print('Using breast_cancer transform for pretrain augmentation')
      if augmentation_speedup:
          transform = A.Compose([
                  A.HorizontalFlip(p=0.5),
                  A.Rotate(limit=45),
                  A.ToRGB(p=1.0), # 修复 permute 错误
                  A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                  A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.6, 1.0)),
                  A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0), # 修复 ByteTensor
                  ToTensorV2() # 修复 ByteTensor
              ])
      else:
          # 同样修复非 speedup 的 torchvision 分支
          transform = transforms.Compose([
              transforms.RandomHorizontalFlip(),
              transforms.RandomRotation(45),
              transforms.Grayscale(num_output_channels=3), # torchvision 版的 ToRGB
              transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
              transforms.RandomResizedCrop(size=img_size, scale=(0.6,1)),
              transforms.ToTensor() # 自动处理 [0, 255] -> [0.0, 1.0]
          ])
    
    elif target.lower() == 'skin_cancer':
      print(f'Using skin_cancer transform (RGB + 0-1 Norm) for train augmentation')
      if augmentation_speedup:
          transform = A.Compose([
              A.HorizontalFlip(p=0.5),
              A.Rotate(limit=45),
              A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
              A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.2, 1.0)),
              ToTensorV2()
          ])
      else:
          transform = transforms.Compose([
              transforms.RandomHorizontalFlip(),
              transforms.RandomRotation(45),
              transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
              transforms.RandomResizedCrop(size=img_size, scale=(0.2,1)),
              transforms.Lambda(convert_to_float)   # 不会崩，只是 0-255 float
          ])
    else:
        # [新增] 捕获未定义的
        raise ValueError(f"No augmentations defined in grab_image_augmentations for target: {target}")

    return transform

def grab_soft_eval_image_augmentations(img_size: int, target: str, augmentation_speedup: bool=False) -> transforms.Compose:
  """
  Defines augmentations to be used during evaluation of contrastive encoders. Typically a less sever form of contrastive augmentations.
  """
  # --- helper: ensure image is 3-channel HWC for albumentations ---
  def _ensure_hwc3(image, **kwargs):
      # image is expected to be numpy array
      if image is None:
          return image
      if image.ndim == 2:  # H x W
          image = np.stack([image, image, image], axis=-1)
      elif image.ndim == 3 and image.shape[2] == 1:  # H x W x 1
          image = np.repeat(image, 3, axis=2)
      return image

  # --- helper: ensure image is CHW float tensor for torchvision pipeline ---
  def _to_tensor_if_numpy(x):
      # If your Dataset returns numpy HWC uint8, this makes torchvision pipeline work.
      # If your Dataset already returns PIL or Tensor, it will pass through.
      if isinstance(x, np.ndarray):
          # HWC uint8 -> float tensor CHW in [0,1]
          import torch
          t = torch.from_numpy(x)
          if t.ndim == 2:
              t = t.unsqueeze(-1)
          if t.shape[-1] == 1:
              t = t.repeat(1, 1, 3)
          t = t.permute(2, 0, 1).contiguous().float().div(255.0)
          return t
      return x

  if target.lower() == 'dvm':
    if augmentation_speedup:
      transform = A.Compose([
                  A.HorizontalFlip(p=0.5),
                  A.Rotate(limit=20),
                  A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
                  A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
                  A.Lambda(name='convert2tensor', image=convert_to_ts)
      ])
    else:
      transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.RandomResizedCrop(size=img_size, scale=(0.8,1)),
        transforms.Lambda(convert_to_float)
      ])
    print('Using dvm transform for soft eval augmentation')
  elif target.lower() in ['pneumonia', 'los', 'rr']:
        if augmentation_speedup:
            # Albumentations: 不再 ToRGB、不再 Resize/Crop（你离线已 resize 到 IMG_SIZE）
            # 仅做：确保 3 通道(防御) + Normalize + ToTensor
            transform = A.Compose([
                A.Lambda(name="ensure_hwc3", image=_ensure_hwc3),
                # 可选：如果你不完全确定离线都是 img_size，可保留 PadIfNeeded（不插值，不改变已有像素）
                # A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0),

                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # Torchvision: 不再 Grayscale(3)、不再 Resize/Crop（避免重复插值）
            # 仅做：把 numpy HWC -> tensor CHW(防御) + Normalize
            transform = transforms.Compose([
                transforms.Lambda(_to_tensor_if_numpy),
                # 如果你的 Dataset 返回的是 PIL Image（不是 numpy），上面 Lambda 会原样返回，
                # 那你需要再加 ToTensor()。为了同时兼容 numpy 和 PIL，可用下面这种写法：
                transforms.Lambda(lambda x: x if hasattr(x, "shape") and str(type(x)).find("torch") != -1
                                  else transforms.ToTensor()(x)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        print('Using pneumonia transform for soft eval augmentation')

  else:
    if augmentation_speedup:
      transform = A.Compose([
                  A.HorizontalFlip(p=0.5),
                  A.Rotate(limit=20),
                  A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
                  A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
                  A.Lambda(name='convert2tensor', image=convert_to_ts_01)
      ])
    else:
      transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.RandomResizedCrop(size=img_size, scale=(0.8,1)),
        transforms.Lambda(convert_to_float)
      ])
    print('Using cardiac transform for soft eval augmentation')
  return transform


def grab_hard_eval_image_augmentations(img_size: int, target: str, augmentation_speedup: bool = False) -> transforms.Compose:
    """
    Defines augmentations to be used during evaluation of contrastive encoders.
    """
    
    # --- helper for albumentations ---
    def _ensure_hwc3(image, **kwargs):
        if image is None: return image
        # 强制转为 HWC (3通道)
        if image.ndim == 2:   # H x W
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1: # H x W x 1
            image = np.repeat(image, 3, axis=2)
        return image

    # --- helper for torchvision ---
    def _to_tensor_if_numpy(x):
        if isinstance(x, np.ndarray):
            import torch
            t = torch.from_numpy(x)
            if t.ndim == 2:
                t = t.unsqueeze(-1)
            if t.shape[-1] == 1:
                t = t.repeat(1, 1, 3)
            # HWC -> CHW, 归一化
            t = t.permute(2, 0, 1).contiguous().float()
            if t.max() > 1.0:
                 t = t.div(255.0)
            return t
        return x

    # ================= PNEUMONIA (重点修复) =================
    if target.lower() in ['pneumonia', 'los', 'rr']:
        print(f'Using [FIXED] pneumonia transform for hard eval augmentation (Speedup={augmentation_speedup})')
        if augmentation_speedup:
            transform = A.Compose([
                # 1. 确保输入是 HWC 且 3 通道
                A.Lambda(name="ensure_hwc3", image=_ensure_hwc3),
                
                # 2. [关键修复] 强制 Resize 到固定大小，防止尺寸不一致
                # 之前只有 PadIfNeeded，如果原图 > img_size 或者 Affine 导致尺寸变化就会报错
                A.Resize(height=img_size, width=img_size),

                # 3. 增强操作
                A.Affine(
                    rotate=(-10, 10),
                    translate_percent=(-0.02, 0.02),
                    scale=(0.95, 1.05),
                    shear=(-5, 5),
                    p=0.5
                ),
                A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.15, p=0.5),

                # 4. 归一化 + 转 Tensor (CHW)
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                            max_pixel_value=255.0),
                ToTensorV2(),
            ])
        else:
            # Torchvision 分支
            transform = transforms.Compose([
                transforms.Lambda(_to_tensor_if_numpy),
                transforms.Resize((img_size, img_size)), # 强制 Resize
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.ColorJitter(brightness=0.10, contrast=0.15),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    # ================= DVM =================
    elif target.lower() == 'dvm':
        if augmentation_speedup:
            transform = A.Compose([
                A.Resize(height=img_size, width=img_size), # 确保有 Resize
                A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, p=0.8),
                A.ToGray(p=0.2),
                A.GaussianBlur(blur_limit=(29,29), sigma_limit=(0.1,2.0), p=0.5),
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333333333333333)),
                A.HorizontalFlip(p=0.5),
                ToTensorV2() # 替换 convert_to_ts
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=29, sigma=(0.1, 2.0))],p=0.5),
                transforms.RandomResizedCrop(size=(img_size,img_size), scale=(0.6, 1.0), ratio=(0.75, 1.3333333333333333)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
        print('Using dvm transform for hard eval augmentation')

    # ================= Breast Cancer =================
    elif target.lower() == 'breast_cancer':
        print('Using breast_cancer transform for hard eval augmentation')
        if augmentation_speedup:
            transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=45),
                A.ToRGB(p=1.0),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.6, 1.0)),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
                ToTensorV2()
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.Grayscale(num_output_channels=3),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.RandomResizedCrop(size=img_size, scale=(0.6,1)),
                transforms.ToTensor()
            ])

    # ================= DEFAULT (CelebA, etc) =================
    elif target.lower() in ['celeba','pawpularity', 'anime', 'adoption']:
        print(f'Using DEFAULT (ImageNet Norm) transform for hard eval augmentation for target: {target}')
        if augmentation_speedup:
            transform = A.Compose([
                A.Resize(height=img_size, width=img_size), # 先 Resize
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=45),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.6, 1.0)),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                ToTensorV2()
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.RandomResizedCrop(size=img_size, scale=(0.6,1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # ================= Cardiac / Others =================
    else:
        print('Using cardiac (default) transform for hard eval augmentation')
        if augmentation_speedup:
            transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=45),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.6, 1.0)),
                ToTensorV2() # 替换 convert_to_ts_01
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.RandomResizedCrop(size=img_size, scale=(0.6,1)),
                transforms.ToTensor()
            ])
            
    return transform

def grab_wids(category: str):
  # boat
  wids_b = ['n02951358', 'n03447447', 'n04612504', 'n03344393', 'n03662601', 'n04273569'] 
  # domestic cat
  wids_c = ['n02123597', 'n02123159', 'n02123045', 'n02124075', 'n02123394']
  # domestic dog
  wids_d = ['n02102480', 'n02096585', 'n02093256', 'n02091831', 'n02086910', 'n02100735', 'n02102040', 'n02085936', 'n02097130', 'n02097047', 'n02106662', 'n02110958', 'n02097209', 'n02092002', 'n02107142', 'n02099712', 'n02093754', 'n02112018', 'n02105412', 'n02096437', 'n02105251', 'n02108089', 'n02108551', 'n02095889', 'n02113624', 'n02093428', 'n02088238', 'n02100877', 'n02099849', 'n02108422', 'n02098413', 'n02086240', 'n02107574', 'n02101556', 'n02099429', 'n02098105', 'n02087394', 'n02108000', 'n02106166', 'n02107683', 'n02091244', 'n02101388', 'n02111889', 'n02093647', 'n02102973', 'n02101006', 'n02109961', 'n02085782', 'n02091635', 'n02112706', 'n02090622', 'n02110063', 'n02113712', 'n02110341', 'n02086079', 'n02089973', 'n02112350', 'n02113799', 'n02105162', 'n02108915', 'n02104029', 'n02089867', 'n02098286', 'n02105505', 'n02110627', 'n02106550', 'n02105641', 'n02100583', 'n02090721', 'n02093859', 'n02113978', 'n02088466', 'n02095570', 'n02099267', 'n02099601', 'n02106030', 'n02112137', 'n02089078', 'n02092339', 'n02088632', 'n02102177', 'n02096051', 'n02096294', 'n02096177', 'n02093991', 'n02110185', 'n02111277', 'n02090379', 'n02111500', 'n02088364', 'n02088094', 'n02094114', 'n02104365', 'n02111129', 'n02109525', 'n02097658', 'n02113186', 'n02095314', 'n02113023', 'n02087046', 'n02094258', 'n02100236', 'n02097298', 'n02105855', 'n02085620', 'n02106382', 'n02091032', 'n02110806', 'n02086646', 'n02094433', 'n02091134', 'n02107312', 'n02107908', 'n02097474', 'n02091467', 'n02102318', 'n02105056', 'n02109047']

  if category == 'Boat':
    return wids_b
  elif category == 'DomesticCat':
    return wids_c
  elif category == 'DomesticDog':
    return wids_d
  else:
    raise ValueError('Category not recognized')

def grab_arg_from_checkpoint(args: str, arg_name: str):
  """
  Loads a lightning checkpoint and returns an argument saved in that checkpoints hyperparameters
  """
  if args.checkpoint:
    ckpt = torch.load(args.checkpoint)
    load_args = ckpt['hyper_parameters']
  else:
    load_args = args
  return load_args[arg_name]

def chkpt_contains_arg(ckpt_path: str, arg_name: str):
  """
  Checks if a checkpoint contains a given argument.
  """
  ckpt = torch.load(ckpt_path)
  return arg_name in ckpt['hyper_parameters']

def prepend_paths(hparams):
  db = hparams.data_base
  
  for hp in [
    'labels_train', 'labels_val', 
    'data_train_imaging', 'data_val_imaging', 
    'data_val_eval_imaging', 'labels_val_eval_imaging', 
    'train_similarity_matrix', 'val_similarity_matrix', 
    'data_train_eval_imaging', 'labels_train_eval_imaging',
    'data_train_tabular', 'data_val_tabular', 
    'data_val_eval_tabular', 'labels_val_eval_tabular', 
    'data_train_eval_tabular', 'labels_train_eval_tabular',
    'field_indices_tabular', 'field_lengths_tabular',
    'data_test_eval_tabular', 'labels_test_eval_tabular',
    'data_test_eval_imaging', 'labels_test_eval_imaging',
    ]:
    if hp in hparams and hparams[hp]:
      hparams['{}_short'.format(hp)] = hparams[hp]
      hparams[hp] = join(db, hparams[hp])

  return hparams

def re_prepend_paths(hparams):
  db = hparams.data_base
  
  for hp in [
    'labels_train', 'labels_val', 
    'data_train_imaging', 'data_val_imaging', 
    'data_val_eval_imaging', 'labels_val_eval_imaging', 
    'train_similarity_matrix', 'val_similarity_matrix', 
    'data_train_eval_imaging', 'labels_train_eval_imaging',
    'data_train_tabular', 'data_val_tabular', 
    'data_val_eval_tabular', 'labels_val_eval_tabular', 
    'data_train_eval_tabular', 'labels_train_eval_tabular',
    'field_indices_tabular', 'field_lengths_tabular',
    'data_test_eval_tabular', 'labels_test_eval_tabular',
    'data_test_eval_imaging', 'labels_test_eval_imaging',
    ]:
    if hp in hparams and hparams[hp]:
      hparams[hp] = join(db, hparams['{}_short'.format(hp)])

  return hparams

def cos_sim_collate(data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
  """
  Collate function to use when cosine similarity of embeddings is relevant. Takes the embeddings returned by the dataset and calculates the cosine similarity matrix for them.
  """
  view_1, view_2, labels, embeddings, thresholds = zip(*data)
  view_1 = torch.stack(view_1)
  view_2 = torch.stack(view_2)
  labels = torch.tensor(labels)
  threshold = thresholds[0]

  cos = torch.nn.CosineSimilarity(dim=0)
  cos_sim_matrix = torch.zeros((len(embeddings),len(embeddings)))
  for i in range(len(embeddings)):
      for j in range(i,len(embeddings)):
          val = cos(embeddings[i],embeddings[j]).item()
          cos_sim_matrix[i,j] = val
          cos_sim_matrix[j,i] = val

  if threshold:
    cos_sim_matrix = torch.threshold(cos_sim_matrix,threshold,0)

  return view_1, view_2, labels, cos_sim_matrix

def calc_logits_labels(out0, out1, temperature=0.1):
  out0 = nn.functional.normalize(out0, dim=1)
  out1 = nn.functional.normalize(out1, dim=1)

  logits = torch.matmul(out0, out1.T) / temperature
  labels = torch.arange(len(out0), device=out0.device)

  return logits, labels