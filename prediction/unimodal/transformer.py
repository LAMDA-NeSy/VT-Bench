import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import rtdl
from sklearn.preprocessing import StandardScaler, LabelEncoder
# [修改 1] 引入 mean_absolute_error 和 r2_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import time
import os
import random
import sys
import json

# --- Hydra 导入 ---
import hydra
from hydra import initialize, compose
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path

# --- 0. 配置与设置随机种子 ---
def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 自定义模型类 ---
class MyFTTransformer(nn.Module):
    def __init__(self, ft_transformer_module):
        super().__init__()
        self.ft_transformer = ft_transformer_module

    def forward(self, x_num, x_cat):
        if x_num is not None and x_num.shape[1] == 0:
            x_num = None
            
        out = self.ft_transformer(x_num, x_cat)

        if isinstance(out, tuple):
            x_embed, x_out = out   
        else:
            x_out = out           

        # 回归任务 d_out=1 时，做个 squeeze
        if x_out.shape[-1] == 1:
            return x_out.squeeze(-1)
        return x_out


# --- 1. 数据加载与预处理 ---
def load_and_preprocess_data(cfg: DictConfig, batch_size: int):
    print(f"--- 1. 正在为数据集: '{cfg.target}' (通用加载器) 加载数据 ---")

    try:
        # --- 1. 加载字段长度 (基数) ---
        all_field_lengths = torch.load(cfg.field_lengths_tabular)
        
        if isinstance(all_field_lengths, torch.Tensor):
            all_field_lengths = all_field_lengths.cpu().tolist()
            
        con_indices = [i for i, length in enumerate(all_field_lengths) if length == 1]
        cat_indices = [i for i, length in enumerate(all_field_lengths) if length > 1]
        
        num_con = len(con_indices)
        num_cat = len(cat_indices)
        
        cat_cardinalities = [all_field_lengths[i] for i in cat_indices]

        print(f"    自动检测结果: {num_con} 个连续特征, {num_cat} 个分类特征。")

        # --- 2. 加载特征 (CSV) ---
        train_df = pd.read_csv(cfg.data_train_eval_tabular, header=None)
        val_df = pd.read_csv(cfg.data_val_eval_tabular, header=None)
        test_df = pd.read_csv(cfg.data_test_eval_tabular, header=None)

        if train_df.shape[1] != len(all_field_lengths):
            print(f"🔴 错误：CSV 列数 ({train_df.shape[1]}) 与 field_lengths 长度 ({len(all_field_lengths)}) 不一致！")
            sys.exit(1)

        # --- 3. 加载标签 ---
        y_train = torch.load(cfg.labels_train_eval_tabular)
        y_val = torch.load(cfg.labels_val_eval_tabular)
        y_test = torch.load(cfg.labels_test_eval_tabular)

    except Exception as e:
        print(f"🔴 加载数据时发生错误: {e}")
        sys.exit(1)

    # --- 4. 拆分特征并转换为 Tensors ---
    def split_and_convert_to_tensors(df, y):
        # 数值/类别特征拆分
        X_num_df = df.iloc[:, con_indices] if len(con_indices) > 0 else None
        X_cat_df = df.iloc[:, cat_indices] if len(cat_indices) > 0 else None

        # 数值特征 -> float32
        if X_num_df is not None:
            X_num_tensor = torch.as_tensor(X_num_df.to_numpy(), dtype=torch.float32)
        else:
            X_num_tensor = None

        # 类别特征 -> int64（若没有类别列，返回 None）
        if X_cat_df is not None:
            X_cat_tensor = torch.as_tensor(X_cat_df.to_numpy(), dtype=torch.long)
        else:
            X_cat_tensor = None

        # 标签 -> 按任务类型转 dtype（关键修复：先转成 Tensor）
        if isinstance(y, torch.Tensor):
            y_tensor = y.detach().cpu()
        else:
            y_tensor = torch.as_tensor(np.asarray(y))

        y_tensor = y_tensor.reshape(-1)  # 保证是 (N,)

        if cfg.task == "classification":
            y_tensor = y_tensor.to(torch.long)
        else:
            y_tensor = y_tensor.to(torch.float32)

        return X_num_tensor, X_cat_tensor, y_tensor


    print("    正在根据 field_lengths 拆分并转换数据...")
    
    X_train_num, X_train_cat, y_train = split_and_convert_to_tensors(train_df, y_train)
    X_val_num, X_val_cat, y_val = split_and_convert_to_tensors(val_df, y_val)
    X_test_num, X_test_cat, y_test = split_and_convert_to_tensors(test_df, y_test)
    
    # DataLoader 处理 None 的情况
    if X_train_num is None: X_train_num = torch.empty((len(y_train), 0))
    if X_val_num is None: X_val_num = torch.empty((len(y_val), 0))
    if X_test_num is None: X_test_num = torch.empty((len(y_test), 0))

    try:
        train_dataset = TensorDataset(X_train_num, X_train_cat, y_train)
        val_dataset = TensorDataset(X_val_num, X_val_cat, y_val)
        test_dataset = TensorDataset(X_test_num, X_test_cat, y_test)
    except Exception as e:
        print(f"🔴 创建 Dataset 出错: {e}")
        sys.exit(1)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    d_out = cfg.num_classes
    
    model_inputs = {
        "n_num_features": num_con,
        "cat_cardinalities": cat_cardinalities,
        "d_out": d_out,
        "task": cfg.task
    }
    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    
    return loaders, model_inputs

# --- 3. 定义模型创建函数 ---
def create_model(params, n_num_features, cat_cardinalities, d_out, device):
    base_ft_transformer = rtdl.FTTransformer(
        feature_tokenizer=rtdl.modules.FeatureTokenizer(
            n_num_features=n_num_features, 
            cat_cardinalities=cat_cardinalities, 
            d_token=params['d_token']
        ),
        transformer=rtdl.modules.Transformer(
            d_token=params['d_token'],
            n_blocks=params['n_blocks'],
            attention_dropout=params['attention_dropout'],
            ffn_d_hidden=params['ffn_d_hidden'],
            ffn_dropout=params['ffn_dropout'],
            residual_dropout=params['residual_dropout'],
            attention_n_heads=8,
            attention_initialization='kaiming',
            attention_normalization='LayerNorm',
            ffn_activation='ReLU',
            ffn_normalization='LayerNorm',
            prenormalization=True,
            first_prenormalization=False,
            last_layer_query_idx=[-1],
            n_tokens=None,
            kv_compression_ratio=None,
            kv_compression_sharing=None,
            head_activation=nn.Identity,
            head_normalization=nn.Identity,
            d_out=d_out,
        ),
    )

    model = MyFTTransformer(
        ft_transformer_module=base_ft_transformer,
    ).to(device)

    return model


# --- 4. 辅助函数 ---
def create_loss_fn(task, device):
    if task == 'classification':
        return nn.CrossEntropyLoss().to(device)
    elif task == 'regression':
        return nn.MSELoss().to(device)
    else:
        raise ValueError(f"未知的任务类型: {task}")

def get_scoring_info(task):
    if task == 'classification':
        return 'accuracy', 'max'
    elif task == 'regression':
        return 'rmse', 'min'
    else:
        raise ValueError(f"未知的任务类型: {task}")

# --- 5. 定义训练与评估函数 ---

def search_for_best_params(param_combinations, cfg, seed, loaders, model_inputs, device):
    print("\n" + "-"*10 + f" [种子 {seed}] 阶段一：快速超参数搜索 (15 epochs) " + "-"*10)
    
    train_loader, val_loader = loaders['train'], loaders['val']
    n_num, cats, d_out, task = model_inputs.values()
    
    scoring_metric, mode = get_scoring_info(task)
    best_score = -float('inf') if mode == 'max' else float('inf')
    best_params = None
    
    for i, params in enumerate(param_combinations):
        print(f"\n--- [试验 {i+1}/{len(param_combinations)}] ---")
        print(f"测试参数: {params}")
        
        set_seed(seed)
        model = create_model(params, n_num, cats, d_out, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        loss_fn = create_loss_fn(task, device)
        
        for epoch in range(15):
            model.train()
            for x_num_batch, x_cat_batch, y_batch in train_loader:
                x_num_batch, x_cat_batch, y_batch = x_num_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(x_num_batch, x_cat_batch)
                
                # [修改 2] 修复 Loss 类型：回归用 float，分类用 long
                target = y_batch.long() if task == 'classification' else y_batch.float()
                loss = loss_fn(y_pred, target)
                
                loss.backward()
                optimizer.step()

        # 在验证集上评估
        model.eval()
        val_preds_proba = []
        val_labels = []
        with torch.no_grad():
            for x_num_batch, x_cat_batch, y_batch in val_loader:
                x_num_batch, x_cat_batch = x_num_batch.to(device), x_cat_batch.to(device)
                y_pred = model(x_num_batch, x_cat_batch)
                
                if task == 'classification':
                    val_preds_proba.append(y_pred.softmax(dim=1).cpu().numpy())
                else:
                    val_preds_proba.append(y_pred.cpu().numpy())
                val_labels.append(y_batch.cpu().numpy())
        
        val_preds_proba = np.concatenate(val_preds_proba)
        val_labels = np.concatenate(val_labels)
        
        current_score = 0.0
        if task == 'classification':
            val_preds_class = np.argmax(val_preds_proba, axis=1)
            current_score = accuracy_score(val_labels, val_preds_class)
        elif task == 'regression':
            current_score = np.sqrt(mean_squared_error(val_labels, val_preds_proba.squeeze()))
        
        print(f"试验 {i+1} 验证集 {scoring_metric}: {current_score:.4f}")
        
        if (mode == 'max' and current_score > best_score) or \
           (mode == 'min' and current_score < best_score):
            best_score = current_score
            best_params = params
            print(f"  (发现新的最佳参数!)")
            
    print("\n" + "-"*10 + " 阶段一搜索完成 " + "-"*10)
    print(f"最佳验证集 {scoring_metric}: {best_score:.4f}")
    print(f"选定的最佳参数: {best_params}")
    return best_params

def train_final_model(best_params, cfg, seed, loaders, model_inputs, device, 
                      patience: int, max_epochs: int):
    print("\n" + "-"*10 + f" [种子 {seed}] 阶段二：使用早停机制充分训练最佳模型 " + "-"*10)
    
    train_loader, val_loader = loaders['train'], loaders['val']
    n_num, cats, d_out, task = model_inputs.values()
    
    set_seed(seed)
    model = create_model(best_params, n_num, cats, d_out, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    loss_fn = create_loss_fn(task, device)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = f'best_model_seed_{seed}.pt' 

    for epoch in range(max_epochs):
        model.train()
        for x_num_batch, x_cat_batch, y_batch in train_loader:
            x_num_batch, x_cat_batch, y_batch = x_num_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_num_batch, x_cat_batch)
            
            # [修改 2] 修复 Loss 类型：回归用 float，分类用 long
            target = y_batch.long() if task == 'classification' else y_batch.float()
            loss = loss_fn(y_pred, target)
            
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_num_batch, x_cat_batch, y_batch in val_loader:
                x_num_batch, x_cat_batch, y_batch = x_num_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
                y_pred = model(x_num_batch, x_cat_batch)
                
                target = y_batch.long() if task == 'classification' else y_batch.float()
                val_loss += loss_fn(y_pred, target).item()
                
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{max_epochs}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  (验证集损失连续 {patience} 个epoch未改善，触发早停!)")
                break
    
    print(f"加载在验证集上性能最佳的模型 (来自 {best_model_path})...")
    model.load_state_dict(torch.load(best_model_path))
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        
    return model

# 阶段三函数：在测试集上评估
def evaluate_final_model(cfg, final_model, test_loader, task, device):
    final_model.eval()
    all_preds_proba, all_labels = [], []
    with torch.no_grad():
        for x_num_batch, x_cat_batch, y_batch in test_loader:
            x_num_batch, x_cat_batch = x_num_batch.to(device), x_cat_batch.to(device)
            
            y_pred = final_model(x_num_batch, x_cat_batch)
            
            if task == 'classification':
                all_preds_proba.append(y_pred.softmax(dim=1).cpu().numpy())
            else:
                all_preds_proba.append(y_pred.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            
    all_preds_proba = np.concatenate(all_preds_proba)
    all_labels = np.concatenate(all_labels)

    metrics_dict = {}
    result_line = ""

    if task == 'classification':
        all_preds_class = np.argmax(all_preds_proba, axis=1)
        acc = accuracy_score(all_labels, all_preds_class)
        if cfg.num_classes == 2:
            auc = roc_auc_score(all_labels, all_preds_proba[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_preds_proba, multi_class='ovr', average='macro')
        macro_f1 = f1_score(all_labels, all_preds_class, average='macro')
        
        metrics_dict = {'acc': acc, 'auc': auc, 'macro-F1': macro_f1}
        result_line = f"acc:{acc:.4f},auc:{auc:.4f},macro-F1:{macro_f1:.4f}"
        
    elif task == 'regression':
        # [修改 3] 新增 MAE 和 R2 计算逻辑
        preds = all_preds_proba.squeeze() # 确保是一维数组
        
        rmse = np.sqrt(mean_squared_error(all_labels, preds))
        mae = mean_absolute_error(all_labels, preds)
        r2 = r2_score(all_labels, preds)
        
        metrics_dict = {'rmse': rmse, 'mae': mae, 'r2': r2}
        result_line = f"rmse:{rmse:.4f}, mae:{mae:.4f}, r2:{r2:.4f}"

    print(f"最终测试集性能: {result_line}")
    return metrics_dict, result_line

# --- 6. 主执行流程 ---
def call_ftt_with_config(config_name: str):
    """
    FT-Transformer 的函数式调用接口。
    """
    # 1. 初始化并加载配置
    # 注意: config_path 需指向你存放 yaml 文件的真实相对路径
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=config_name)
    
    # 2. 执行核心实验逻辑
    return run_ftt_experiment(cfg)

def run_ftt_experiment(cfg: DictConfig):
    """
    FT-Transformer 核心训练与评估逻辑。
    """
    # --- 1.A. 路径解析逻辑 ---
    print("--- 1.A. 正在解析数据路径 ---")
    data_root = cfg.get('data_base')

    if data_root is not None:
        print(f"    检测到 'data_root'，将为所有数据文件添加前缀: {data_root}")
        path_keys = [
            'labels_train', 'labels_val',
            'data_train_imaging', 'data_val_imaging',
            'data_train_tabular', 'data_val_tabular',
            'field_lengths_tabular',
            'data_train_eval_tabular', 'labels_train_eval_tabular',
            'data_val_eval_tabular', 'labels_val_eval_tabular',
            'data_test_eval_tabular', 'labels_test_eval_tabular',
            'data_train_eval_imaging', 'labels_train_eval_imaging',
            'data_val_eval_imaging', 'labels_val_eval_imaging',
            'data_test_eval_imaging', 'labels_test_eval_imaging'
        ]
        for key in path_keys:
            if key in cfg and cfg[key] is not None:
                original_path = cfg[key]
                cfg[key] = os.path.join(data_root, original_path)
    else:
        print("    未提供 'data_root'。将假定路径已正确。")
    
    print("--- 最终配置: ---")
    print(OmegaConf.to_yaml(cfg))

    # --- 设备与配置加载 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'正在使用设备: {device}')
    
    # 在非 Hydra 主程序模式下，如果 get_original_cwd 报错，则回退到 os.getcwd()
    try:
        base_path = get_original_cwd()
    except Exception:
        base_path = os.getcwd()

    model_config_filename = 'ftt_model_config.json'
    model_config_path = os.path.join(base_path, model_config_filename)
    
    print(f"--- 正在从 {model_config_path} 加载模型配置 ---")
    try:
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        search_space = model_config['search_space']
        hyperparams = model_config['hyperparams']
        print("模型配置加载成功。")
    except Exception as e:
        print(f"错误: 加载模型配置失败: {e}")
        return None

    # 参数解析
    seeds = list(hyperparams['seeds'])
    batch_size = hyperparams['batch_size']
    D_TOKEN = hyperparams['d_token']
    LEARNING_RATE = hyperparams['learning_rate']
    WEIGHT_DECAY = hyperparams['weight_decay']
    N_TRIALS = hyperparams['n_trials']
    patience = hyperparams['patience']
    max_epochs = hyperparams['max_epochs']

    final_results_summary = []

    # 加载数据
    loaders, model_inputs = load_and_preprocess_data(cfg, batch_size)

    # --- 实验循环 ---
    for seed in seeds:
        print("\n" + "="*30 + f" 开始执行，随机种子: {seed} " + "="*30)
        set_seed(seed)

        # 随机搜索空间构建
        param_combinations = []
        for _ in range(N_TRIALS):
            params = {
                'n_blocks': random.choice(search_space['n_blocks']),
                'ffn_d_hidden': random.choice(search_space['ffn_d_hidden']),
                'residual_dropout': random.uniform(*search_space['residual_dropout']),
                'attention_dropout': random.uniform(*search_space['attention_dropout']),
                'ffn_dropout': random.uniform(*search_space['ffn_dropout']),
                'd_token': D_TOKEN, 
                'learning_rate': LEARNING_RATE, 
                'weight_decay': WEIGHT_DECAY,
            }
            param_combinations.append(params)
        
        # 寻找最优参数
        best_params = search_for_best_params(
            param_combinations, cfg, seed, loaders, model_inputs, device
        )
        
        # 训练最终模型
        final_model = train_final_model(
            best_params, cfg, seed, loaders, model_inputs, device,
            patience=patience, max_epochs=max_epochs
        )
        
        print("\n" + "-"*10 + f" [种子 {seed}] 阶段三：测试集评估 " + "-"*10)
        metrics_dict, result_line = evaluate_final_model(
            cfg, final_model, loaders['test'], model_inputs['task'], device
        )
        
        result_dict = {
            'seed': seed, 
            'best_params': best_params, 
            'result_line': result_line,
            **metrics_dict
        }
        final_results_summary.append(result_dict)

    # --- 结果记录 ---
    output_file_path = 'result/fttrans.txt'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'a') as f:
        f.write(f"--- 实验配置: {cfg.get('target', 'unknown')} ---\n")
        f.write(OmegaConf.to_yaml(cfg)) 
        f.write("\n--- 模型配置 ---\n")
        f.write(json.dumps(model_config, indent=2))
        f.write("\n\n" + "="*30 + " 最终总结 " + "="*30 + "\n")
        
        for final_result in final_results_summary:
            line = f"种子: {final_result['seed']} | {final_result['result_line']}"
            f.write(line + "\n")

        params_header = f"\n最佳参数 (种子 {final_results_summary[-1]['seed']}):"
        f.write(params_header + "\n" + str(final_results_summary[-1]['best_params']) + "\n")
        f.write("="*80 + "\n")

    print(f"\n实验完成，结果已保存至: {output_file_path}")
    return final_results_summary

