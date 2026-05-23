# eval_tabpfn.py
# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import torch
import random

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# [修改] 引入回归指标
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# [修改] 引入 Regressor
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.many_class.many_class_classifier import ManyClassClassifier

from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# =========================
# 工具函数
# =========================

TEXT_LENGTH_DROP_THRESHOLD = 30
HIGH_CARDINALITY_THRESHOLD = 200
N_ENSEMBLE_CONFIGURATIONS = 16

def load_data(cfg: DictConfig):
    """
    (重构版 - 基于 field_lengths 自动判断列类型)
    """
    import sys
    import numpy as np
    import pandas as pd
    import torch

    target = cfg.target
    print(f"[INFO] 正在加载 target: {target} (自动推断列类型)")

    def to_numpy(x):
        """把 torch.Tensor / numpy.ndarray / list 等统一转成 numpy.ndarray"""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)

    def postprocess_y(y_np, task: str):
        """按任务类型统一 y 的 dtype/shape"""
        y_np = to_numpy(y_np)

        if task == "regression":
            # 回归：float32，保留原 shape（常见为 (N,) 或 (N,1)）
            return y_np.astype(np.float32)

        # 分类：希望是 (N,) 的 int64
        y_np = y_np
        if y_np.ndim > 1 and y_np.shape[-1] == 1:
            y_np = y_np.reshape(-1)

        # 如果标签被存成 float（0.0/1.0/2.0），转成 int 更稳
        if np.issubdtype(y_np.dtype, np.floating):
            y_np = y_np.astype(np.int64)
        else:
            y_np = y_np.astype(np.int64, copy=False)

        return y_np

    try:
        # --- 1. 加载数据 ---
        X_train_full = pd.read_csv(cfg.data_train_eval_tabular, header=None)
        y_train_obj = torch.load(cfg.labels_train_eval_tabular, weights_only=False)
        y_train_full = postprocess_y(y_train_obj, cfg.task)

        X_test_full = pd.read_csv(cfg.data_test_eval_tabular, header=None)
        y_test_obj = torch.load(cfg.labels_test_eval_tabular, weights_only=False)
        y_test_full = postprocess_y(y_test_obj, cfg.task)

        # --- 2. 加载 field_lengths 并计算索引 ---
        field_lengths_path = cfg.field_lengths_tabular
        print(f"[INFO] 读取字段长度文件: {field_lengths_path}")

        try:
            field_lengths_obj = torch.load(field_lengths_path, weights_only=False)
            field_lengths = to_numpy(field_lengths_obj)
        except Exception:
            field_lengths = np.load(field_lengths_path)

        field_lengths = np.array(field_lengths).flatten()

        n_cols_data = X_train_full.shape[1]
        n_cols_lengths = len(field_lengths)
        if n_cols_data != n_cols_lengths:
            print(f"🔴 错误：CSV 列数 ({n_cols_data}) 与 field_lengths 长度 ({n_cols_lengths}) 不匹配！")
            sys.exit(1)

        con_indices = [i for i, fl in enumerate(field_lengths) if fl == 1]
        cat_indices = [i for i, fl in enumerate(field_lengths) if fl > 1]

        print(f"[INFO] 自动检测结果:")
        print(f"      - 数值列数量: {len(con_indices)}")
        print(f"      - 类别列数量: {len(cat_indices)}")

        # --- 3. 定义列名 ---
        all_col_names = [f"col_{i}" for i in range(n_cols_data)]
        X_train_full.columns = all_col_names
        X_test_full.columns = all_col_names

        num_cols = [all_col_names[i] for i in con_indices]
        cat_cols = [all_col_names[i] for i in cat_indices]

        # --- 4. 标签处理 (1-indexed -> 0-indexed) ---
        # 只有分类任务才执行
        if cfg.task == "classification":
            label_min = int(np.min(y_train_full)) if y_train_full.size > 0 else 0
            label_max = int(np.max(y_train_full)) if y_train_full.size > 0 else 0
            if label_min == 1 and label_max == cfg.num_classes:
                print(f"    [!] 警告：检测到 1-indexed 标签，正在修复...")
                y_train_full = y_train_full - 1
                y_test_full = y_test_full - 1

        # --- 5. 强制类型转换 ---
        if cat_cols:
            for col in cat_cols:
                X_train_full[col] = X_train_full[col].astype(str)
                X_test_full[col] = X_test_full[col].astype(str)

        if num_cols:
            for col in num_cols:
                X_train_full[col] = pd.to_numeric(X_train_full[col], errors="coerce").fillna(0)
                X_test_full[col] = pd.to_numeric(X_test_full[col], errors="coerce").fillna(0)

        # （可选）调试输出，确认类型和形状，确认没问题后可删
        print(f"[DEBUG] X_train: {X_train_full.shape}, y_train: {y_train_full.shape}, {y_train_full.dtype}")
        print(f"[DEBUG] X_test : {X_test_full.shape}, y_test : {y_test_full.shape}, {y_test_full.dtype}")
        print(f"[DEBUG] num_cols={len(num_cols)}, cat_cols={len(cat_cols)}")

        return X_train_full, y_train_full, X_test_full, y_test_full, num_cols, cat_cols

    except Exception as e:
        print(f"🔴 加载数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def build_preprocess(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
    
    return ColumnTransformer(transformers=transformers, remainder='drop', verbose_feature_names_out=False)

def get_subsample_indices(y, sample_size, seed, task):
    """
    [修改] 通用采样函数：
    - 分类任务：分层采样
    - 回归任务：随机采样
    """
    sample_size = int(sample_size)
    if len(y) <= sample_size:
        return np.arange(len(y))
    
    # 1. 回归任务直接随机采样
    if task == 'regression':
        np.random.seed(seed)
        return np.random.choice(np.arange(len(y)), sample_size, replace=False)

    # 2. 分类任务逻辑
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2 or (counts < 2).any():
        np.random.seed(seed)
        return np.random.choice(np.arange(len(y)), sample_size, replace=False)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=seed)
    idx_all = np.arange(len(y))
    try:
        for sub_idx, _ in sss.split(idx_all, y):
            return sub_idx
    except ValueError:
        np.random.seed(seed)
        return np.random.choice(idx_all, sample_size, replace=False)

def evaluate_metrics(y_true, y_pred, task, y_proba=None):
    """
    [修改] 支持回归和分类指标
    """
    res = {}
    
    if task == 'classification':
        res["accuracy"] = accuracy_score(y_true, y_pred)
        res["macro_f1"] = f1_score(y_true, y_pred, average='macro')
        res["weighted_f1"] = f1_score(y_true, y_pred, average='weighted')
        
        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:
                    res["auc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    res["auc_macro_ovr"] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            except:
                pass
                
    elif task == 'regression':
        # [新增] 回归指标
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        res["rmse"] = rmse
        res["mae"] = mae
        res["r2"] = r2
        
    return res

# =========================
# 主流程
# =========================

def call_with_specific_config(config_name: str):
    """
    Functional interface to run the experiment with a specific config file.
    """
    # 1. Initialize Hydra and compose the configuration
    # config_path is relative to this python file
    with initialize(version_base=None, config_path="../configs"):
        # We load the specific config_name passed as an argument
        cfg = compose(config_name=config_name)
        
    # 2. Call the core logic (original main function logic)
    return run_tabpfn_experiment(cfg)

def run_tabpfn_experiment(cfg: DictConfig):
    """
    Core logic extracted from the original main function.
    """
    seeds = [2022, 2023, 2024]
    results_all = []

    # Ensure 'task' field exists in cfg
    if 'task' not in cfg:
        print("⚠️ Missing 'task' in config, defaulting to 'classification'")
        cfg.task = 'classification'

    for seed in seeds:
        print(f"\n🚀 Running seed = {seed} | Task: {cfg.task}")
        cfg.seed = seed

        # --- Path Resolution ---
        data_root = cfg.get('data_base')
        if data_root:
            path_keys = [
                'labels_train_eval_tabular', 'labels_test_eval_tabular',
                'data_train_eval_tabular', 'data_test_eval_tabular',
                'field_lengths_tabular'
            ]
            for key in path_keys:
                if key in cfg and cfg[key] and not os.path.isabs(cfg[key]):
                    cfg[key] = os.path.join(data_root, cfg[key])

        TRAIN_SAMPLE_THRESHOLD = cfg.get('train_sample_max', 10000)
        TEST_SAMPLE_THRESHOLD = cfg.get('test_sample_max', 10000)

        # 1. Load Data
        X_train_full, y_train_full, X_test_full, y_test_full, num_cols, cat_cols = load_data(cfg)

        # 2. Preprocessing
        preprocess = build_preprocess(num_cols, cat_cols)

        # 3. Subsampling
        sample_size = min(len(y_train_full), TRAIN_SAMPLE_THRESHOLD)
        sub_idx = get_subsample_indices(y_train_full, sample_size, seed, cfg.task)
        X_train_sampled = X_train_full.iloc[sub_idx]
        y_train_sampled = y_train_full[sub_idx]

        # 4. Feature Transformation
        print("Preprocessing features...")
        X_train_np = preprocess.fit_transform(X_train_sampled)
        X_test_np  = preprocess.transform(X_test_full)

        # 5. Model Initialization
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if cfg.task == 'classification':
            if cfg.num_classes > 10:
                base_clf = TabPFNClassifier(n_estimators=N_ENSEMBLE_CONFIGURATIONS, device=device)
                clf = ManyClassClassifier(estimator=base_clf, alphabet_size=10, random_state=seed)
            else:
                clf = TabPFNClassifier(n_estimators=N_ENSEMBLE_CONFIGURATIONS, device=device)
        elif cfg.task == 'regression':
            print("Initializing TabPFNRegressor...")
            clf = TabPFNRegressor(n_estimators=N_ENSEMBLE_CONFIGURATIONS, device=device)
        else:
            raise ValueError(f"Unknown task type: {cfg.task}")

        # 6. Training
        clf.fit(X_train_np, y_train_sampled)

        # 7. Evaluation Sampling
        X_test_eval, y_test_eval = X_test_np, y_test_full
        if len(X_test_np) > TEST_SAMPLE_THRESHOLD:
            stratify_target = y_test_full if cfg.task == 'classification' else None
            X_test_eval, _, y_test_eval, _ = train_test_split(
                X_test_np, y_test_full,
                train_size=TEST_SAMPLE_THRESHOLD,
                stratify=stratify_target,
                random_state=seed
            )
        
        # 8. Prediction
        test_proba = None
        if cfg.task == 'classification':
            test_proba = clf.predict_proba(X_test_eval)
            test_pred  = np.argmax(test_proba, axis=1)
        else:
            test_pred = clf.predict(X_test_eval)
        
        # 9. Metrics
        metrics = evaluate_metrics(y_test_eval, test_pred, cfg.task, test_proba)
        results_all.append({"seed": seed, "results": metrics})

    # Save results
    output_file = cfg.get('output_file', "result/tabpfn_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results_all, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Return path or metrics if needed for higher-level logic
    return output_file
