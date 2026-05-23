import os 
import sys
import time
from datetime import datetime
import random
from multiprocessing import Queue

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, open_dict, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from utils.utils import grab_arg_from_checkpoint, prepend_paths, re_prepend_paths

# Ensure the prediction directory is in sys.path for relative imports
PREDICTION_DIR = os.path.dirname(os.path.abspath(__file__))
if PREDICTION_DIR not in sys.path:
    sys.path.insert(0, PREDICTION_DIR)

PREDICTION_SRC = os.path.join(os.path.dirname(__file__), "src")
if PREDICTION_SRC not in sys.path:
  sys.path.insert(0, PREDICTION_SRC)

from vlms_prediction import VLMs_prediction


torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
hydra.HYDRA_FULL_ERROR = 1


#@hydra.main(config_path='./configs', config_name='config', version_base=None)
def run(args: DictConfig):
  from trainers.pretrain import pretrain
  from trainers.evaluate import evaluate
  from trainers.test import test
  now = datetime.now()
  start = time.time()
  pl.seed_everything(args.seed)

  args = prepend_paths(args)
  time.sleep(random.randint(1,5)) # Prevents multiple runs getting the same version when launching many jobs at once

  if args.resume_training:
    # if args.wandb_id:
    #   wandb_id = args.wandb_id
    wandb_id = args.wandb_id if args.wandb_id else None
    tmp_data_base = args.data_base
    checkpoint = args.checkpoint
    ckpt = torch.load(args.checkpoint)
    args = ckpt['hyper_parameters']
    args = OmegaConf.create(args)
    #with open_dict(args):
    args.checkpoint = checkpoint
    args.resume_training = True
    if not 'wandb_id' in args or not args.wandb_id:
      args.wandb_id = wandb_id
    # Run prepend again in case we move to another server and need to redo the paths
    args.data_base = tmp_data_base
    args = re_prepend_paths(args)
  
  if args.generate_embeddings:
    if not args.datatype:
      args.datatype = grab_arg_from_checkpoint(args, 'dataset')
    generate_embeddings(args)
    return args
  
  # base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
  # base_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'result')

  project_root = os.path.dirname(os.path.abspath(__file__))
  base_dir = os.path.join(project_root, 'results')
  
  exp_name = f'{args.exp_name}_{args.target}_{now.strftime("%m%d_%H%M")}'
  if args.use_wandb:
    if args.resume_training and args.wandb_id:
      wandb_logger = WandbLogger(name=exp_name, project=args.wandb_project, entity=args.wandb_entity, save_dir=base_dir, offline=args.offline, id=args.wandb_id, resume='allow')
    else:
      wandb_logger = WandbLogger(name=exp_name, project=args.wandb_project, entity=args.wandb_entity, save_dir=base_dir, offline=args.offline)
    args.wandb_id = wandb_logger.version
  else:
    # === 这是新的、更健壮的解决方案 ===
    print(f"--- Wandb is DISABLED. Using CSVLogger at {base_dir}/{exp_name} ---")
    
    # 使用 CSVLogger 作为备用
    wandb_logger = CSVLogger(
        save_dir=base_dir, 
        name=exp_name
    )
    
    # 确保 wandb_id 为 None
    args.wandb_id = None
    # === 解决方案结束 ===
  # args.wandb_id = wandb_logger.version

  if args.checkpoint and not args.resume_training:
    if not args.datatype:
      args.datatype = grab_arg_from_checkpoint(args, 'datatype')
  
  print('Comment: ', args.comment)
  print(f'Pretrain LR: {args.lr}, Decay: {args.weight_decay}')
  print(f'Finetune LR: {args.lr_eval}, Decay: {args.weight_decay_eval}')
  print(f'Corruption rate: {args.corruption_rate}, temperature: {args.temperature}')
  if args.algorithm_name == 'TIP':
    print('Special Replace Ratio: ', args.replace_special_rate)
    
  if args.pretrain:
    print('=================================================================================\n')
    print('Start pretraining\n')  
    print(f'Target is {args.target}')
    print('=================================================================================')
    torch.cuda.empty_cache()
    args.checkpoint = pretrain(args, wandb_logger)
  
  if args.test:
    test(args, wandb_logger)
  elif args.evaluate:
    print('=================================================================================\n')
    print('Start Finetuning')  
    print(f'Target is {args.target}')
    print('=================================================================================\n')
    torch.cuda.empty_cache()
    args.checkpoint = evaluate(args, wandb_logger)

  wandb.finish()
  del wandb_logger

  end = time.time()
  time_elapsed = end-start
  print('Total running time: {:.0f}h {:.0f}m'.
      format(time_elapsed // 3600, (time_elapsed % 3600)//60))
  
  return args.checkpoint 

@property
def exception(self):
  if self._pconn.poll():
    self._exception = self._pconn.recv()
  return self._exception

@hydra.main(config_path='./configs', config_name='config_biomedia', version_base=None)
def control(args: DictConfig):
  run(args)

# def call_with_specific_config(config_name, model_id):
#     with initialize(version_base=None, config_path="./configs"):
#         cfg = compose(
#             config_name=config_name, 
#         )
#         checkpoints = run(cfg)
#         return checkpoints

def call_with_specific_config(config_name, model_id, diagnosis="full"):
    """
    Dispatches the experiment to the correct model handler based on model_id.
    Standardizes config names into either 'config_{dataset}_image' or 'config_{dataset}_Tabular'.
    """
    
    # 1. Parse the dataset name from the incoming config_name (expected: config_{dataset}_{model_id})
    # Example: config_UKB_vit -> dataset is 'UKB'
    model_id = model_id.strip()
    parts = config_name.split('_')
    if len(parts) < 3:
        raise ValueError(f"Invalid config_name format: {config_name}. Expected 'config_{{dataset}}_{{model_id}}'")
    
    dataset_name = parts[1]

    # 2. Determine the target config file name based on model_id
    if model_id in ['vit', 'resnet']:
        target_config = f"config_{dataset_name}_image"
    elif model_id in ['tabtransformer', 'tabpfn', 'lightgbm']:
        target_config = f"config_{dataset_name}_Tabular"
    elif model_id in [
        'Qwen/Qwen3-VL-8B-Instruct',
        'SpursgoZmy/table-llava-v1.5-7b-hf',
    ]:
        target_config = None
    else:
        target_config = config_name

    print(f"[*] Dispatching task | Model: {model_id} | Dataset: {dataset_name}")
    if target_config:
        print(f"[*] Loading target config: {target_config}.yaml")

    # 3. Initialize Hydra and Compose the configuration
    # Use global_config_path if needed. 'with' ensures cleanup of Hydra state.
    if target_config:
        with initialize(version_base=None, config_path="./configs"):
            cfg = compose(config_name=target_config)

    # 4. Routing logic: Call the specific experiment function
    checkpoints = None

    

    if model_id == 'vit':
        # Result filename will be vit_results_{target}.txt
        from unimodal.vit import run_vit_experiment
        checkpoints = run_vit_experiment(cfg)
        
    elif model_id == 'tabpfn':
        # Result will be saved to result/tabpfn_results.json
        from unimodal.eval_tabpfn import run_tabpfn_experiment
        checkpoints = run_tabpfn_experiment(cfg)
        
    elif model_id == 'lightgbm':
        # Result will be lgb_results_{dataset}.txt
        from unimodal.LightGBM import run_lgb_experiment
        checkpoints = run_lgb_experiment(cfg)
        
    elif model_id == 'tabtransformer':
        # Result will be result/fttrans.txt
        from unimodal.transformer import run_ftt_experiment
        checkpoints = run_ftt_experiment(cfg)

    elif model_id in [
        'Qwen/Qwen3-VL-8B-Instruct',
        'SpursgoZmy/table-llava-v1.5-7b-hf',
    ]:
        result = VLMs_prediction(
            data=dataset_name,
            model=model_id,
            diagnosis=diagnosis,
        )
        output_dirs = result.get("output_dirs", [])
        checkpoints = output_dirs[0] if output_dirs else None
    else:
       checkpoints = run(cfg)
    return checkpoints        

if __name__ == "__main__":
  call_with_specific_config()

