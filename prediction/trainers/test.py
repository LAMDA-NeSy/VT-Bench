from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from datasets.ImageDataset import ImageDataset
from datasets.TabularDataset import TabularDataset
from datasets.ImagingAndTabularDataset import ImagingAndTabularDataset
from datasets.TabularDataset import TabularDataset
from models.Evaluator import Evaluator
from omegaconf import OmegaConf
import pandas as pd
from os.path import join
from utils.utils import grab_arg_from_checkpoint, grab_hard_eval_image_augmentations, grab_wids, create_logdir


def test(hparams, wandb_logger=None):
  """
  Tests trained models. 
  
  IN
  hparams:      All hyperparameters
  """
  pl.seed_everything(hparams.seed)
  
  if hparams.eval_datatype == 'imaging':
      test_dataset = ImageDataset(hparams.data_test_eval_imaging, hparams.labels_test_eval_imaging, hparams.delete_segmentation, 0, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading, task=hparams.task,
                                  dataset_name=hparams.dataset_name, augmentation_speedup=hparams.augmentation_speedup)
      hparams.transform_test = test_dataset.transform_val.__repr__()
  elif hparams.eval_datatype == 'multimodal':
    assert hparams.strategy == 'tip'
    test_dataset = ImagingAndTabularDataset(
    hparams.data_test_eval_imaging, hparams.delete_segmentation, 0, 
    hparams.data_test_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
    hparams.labels_test_eval_imaging, grab_arg_from_checkpoint(hparams, 'img_size'), hparams.live_loading, train=False, target=hparams.target, corruption_rate=0,
    data_base=hparams.data_base, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,
    augmentation_speedup=hparams.augmentation_speedup
  )
    hparams.input_size = test_dataset.get_input_size()
  elif hparams.eval_datatype == 'tabular':
    test_dataset = TabularDataset(hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular, 0, 0, train=False, 
                                eval_one_hot=hparams.eval_one_hot, field_lengths_tabular=hparams.field_lengths_tabular,
                                data_base=hparams.data_base, 
                                strategy=hparams.strategy, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate)
    hparams.input_size = test_dataset.get_input_size()
  elif hparams.eval_datatype == 'imaging_and_tabular':
    test_dataset = ImagingAndTabularDataset(
      hparams.data_test_eval_imaging, hparams.delete_segmentation, 0, 
      hparams.data_test_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
      hparams.labels_test_eval_imaging, hparams.img_size, hparams.live_loading, train=False, target=hparams.target,
      corruption_rate=0.0, data_base=hparams.data_base, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,
      augmentation_speedup=hparams.augmentation_speedup)
    hparams.input_size = test_dataset.get_input_size()
  else:
    raise Exception('argument dataset must be set to imaging, tabular or multimodal')
  
  drop = ((len(test_dataset)%hparams.batch_size)==1)

  test_loader = DataLoader(
    test_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, drop_last=drop, persistent_workers=True)

  logdir = create_logdir('test', hparams.resume_training, wandb_logger)
  hparams.dataset_length = len(test_loader)

  tmp_hparams = OmegaConf.create(OmegaConf.to_container(hparams, resolve=True))
  tmp_hparams.checkpoint = None
  model = Evaluator(tmp_hparams)
  model.freeze()
  trainer = Trainer.from_argparse_args(hparams, gpus=1, logger=wandb_logger)
  test_results = trainer.test(model, test_loader, ckpt_path=hparams.checkpoint)
  df = pd.DataFrame(test_results)
  df.to_csv(join(logdir, 'test_results.csv'), index=False)

# from torch.utils.data import DataLoader
# import pytorch_lightning as pl
# from pytorch_lightning import Trainer
# from datasets.ImageDataset import ImageDataset
# from datasets.TabularDataset import TabularDataset
# from datasets.ImagingAndTabularDataset import ImagingAndTabularDataset
# from datasets.TabularDataset import TabularDataset
# from models.Evaluator import Evaluator
# from omegaconf import OmegaConf  # 保留，以防 hparams 需要
# import pandas as pd
# from os.path import join
# from utils.utils import grab_arg_from_checkpoint, grab_hard_eval_image_augmentations, grab_wids, create_logdir
# import torch

# def test(hparams, wandb_logger=None):
#     """
#     Tests trained models. 
    
#     IN
#     hparams:      All hyperparameters
#     """
#     pl.seed_everything(hparams.seed)
    
#     # --- [第 1 部分：数据集加载] ---
#     # 这部分保持不变，因为我们仍然需要正确加载测试数据
    
#     if hparams.eval_datatype == 'imaging':
#         test_dataset = ImageDataset(hparams.data_test_eval_imaging, hparams.labels_test_eval_imaging, hparams.delete_segmentation, 0, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading, task=hparams.task,
#                                         dataset_name=hparams.dataset_name, augmentation_speedup=hparams.augmentation_speedup)
#         hparams.transform_test = test_dataset.transform_val.__repr__()
#     elif hparams.eval_datatype == 'multimodal':
#         assert hparams.strategy == 'tip'
#         test_dataset = ImagingAndTabularDataset(
#         hparams.data_test_eval_imaging, hparams.delete_segmentation, 0, 
#         hparams.data_test_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
#         hparams.labels_test_eval_imaging, grab_arg_from_checkpoint(hparams, 'img_size'), hparams.live_loading, train=False, target=hparams.target, corruption_rate=0,
#         data_base=hparams.data_base, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,
#         augmentation_speedup=hparams.augmentation_speedup
#         )
#         hparams.input_size = test_dataset.get_input_size()
#     elif hparams.eval_datatype == 'tabular':
#         test_dataset = TabularDataset(hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular, 0, 0, train=False, 
#                                         eval_one_hot=hparams.eval_one_hot, field_lengths_tabular=hparams.field_lengths_tabular,
#                                         data_base=hparams.data_base, 
#                                         strategy=hparams.strategy, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate)
#         hparams.input_size = test_dataset.get_input_size()
#     elif hparams.eval_datatype == 'imaging_and_tabular':
#         test_dataset = ImagingAndTabularDataset(
#             hparams.data_test_eval_imaging, hparams.delete_segmentation, 0, 
#             hparams.data_test_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
#             hparams.labels_test_eval_imaging, hparams.img_size, hparams.live_loading, train=False, target=hparams.target,
#             corruption_rate=0.0, data_base=hparams.data_base, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,
#             augmentation_speedup=hparams.augmentation_speedup)
#         hparams.input_size = test_dataset.get_input_size()
#     else:
#         raise Exception('argument dataset must be set to imaging, tabular or multimodal')
    
#     drop = ((len(test_dataset)%hparams.batch_size)==1)

#     test_loader = DataLoader(
#         test_dataset,
#         num_workers=hparams.num_workers, batch_size=hparams.batch_size,   
#         pin_memory=True, shuffle=False, drop_last=drop, persistent_workers=True)

#     # --- [第 2 部分：模型加载和测试（已修改）] ---
    
#     logdir = create_logdir('test', hparams.resume_training, wandb_logger)
#     hparams.dataset_length = len(test_loader)

#     print(f"Manually loading model from: {hparams.checkpoint}")

#     # 1. 加载 checkpoint 文件
#     checkpoint_dict = torch.load(hparams.checkpoint, map_location="cpu")
    
#     # 2. 提取 .ckpt 中保存的 "旧" hparams (这是一个普通字典)
#     model_hparams_from_ckpt_dict = checkpoint_dict['hyper_parameters']

#     # 3. *** 关键修复 ***
#     #    我们必须合并 "新" "旧" 两个配置
    
#     # 3a. 将 "新" hparams (来自命令行的 OmegaConf 对象) 作为基础
#     #    它包含了 'share_weights=False'
#     final_hparams = hparams.copy()

#     # 3b. 将 "旧" hparams (来自.ckpt的字典) 转换为 OmegaConf 对象
#     model_hparams_conf = OmegaConf.create(model_hparams_from_ckpt_dict)
    
#     # 3c. 合并它们。
#     #    .merge_with() 会用 "旧" 配置中的值覆盖 "新" 配置。
#     #    这能确保模型结构(如 num_layers)是正确的。
#     #    同时，"新" 配置中独有的 'share_weights' 会被保留。
#     OmegaConf.set_struct(final_hparams, False) # 允许合并
#     final_hparams.merge_with(model_hparams_conf)
    
#     # 4. *** 关键修复 2 (来自之前的步骤) ***
#     #    在合并后，我们 *仍然* 需要覆盖那个指向预训练文件的 'checkpoint' 键
#     print(f"Original 'checkpoint' hparam was: {final_hparams.checkpoint}")
#     final_hparams.checkpoint = None
#     print("Set internal 'checkpoint' hparam to None.")

#     # 5. 使用这个 *完整且已修复* 的 'final_hparams' 来初始化模型
#     #    现在它既有正确的模型结构，又有 'share_weights'
#     model = Evaluator(hparams=final_hparams)
    
#     # 6. 手动将权重加载到模型骨架中
#     model.load_state_dict(checkpoint_dict['state_dict'])
    
#     print("Model initialized with MERGED hparams and weights loaded successfully.")

#     model.freeze()
#     trainer = Trainer.from_argparse_args(hparams, gpus=1, logger=wandb_logger)

#     # --- [修改后的测试调用] ---
#     # 因为模型已经从 .ckpt 加载了权重，
#     # 我们不再需要向 .test() 传递 ckpt_path
#     print("Starting test...")
#     test_results = trainer.test(model, test_loader)
#     # --- [修改] ---
    
#     print(f"Test complete. Results: {test_results}")
#     df = pd.DataFrame(test_results)
#     df.to_csv(join(logdir, 'test_results.csv'), index=False)
#     print(f"Test results saved to {join(logdir, 'test_results.csv')}")