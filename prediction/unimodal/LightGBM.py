import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
# --- 修改点 1: 引入 mean_absolute_error 和 r2_score ---
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
import sys
import torch
import os
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

def load_and_preprocess_data(cfg: DictConfig):
    task = cfg.get('task', 'classification')
    print(f"--- 1. 正在为 Target: '{cfg.target}' (任务: {task}) (通用LGBM加载器) 加载数据 ---")

    try:
        # --- 1. 加载数据 ---
        X_train = pd.read_csv(cfg.data_train_eval_tabular, header=None)
        X_test = pd.read_csv(cfg.data_test_eval_tabular, header=None)
        
        # 加载标签
        # y_train = torch.load(cfg.labels_train_eval_tabular).numpy()
        # y_test = torch.load(cfg.labels_test_eval_tabular).numpy()
        y_train = torch.load(cfg.labels_train_eval_tabular, map_location="cpu")

        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()
        else:
            y_train = np.asarray(y_train)

        y_train = y_train.reshape(-1)  # (N,1) -> (N,)

        y_test = torch.load(cfg.labels_test_eval_tabular, map_location="cpu")

        if isinstance(y_test, torch.Tensor):
            y_test = y_test.detach().cpu().numpy()
        else:
            y_test = np.asarray(y_test)

        y_test = y_test.reshape(-1)



        print("    数据加载成功。")

        # --- 2. 加载字段长度 (用于自动识别类别特征) ---
        # [!] 关键修改：读取 field_lengths
        all_field_lengths = torch.load(cfg.field_lengths_tabular)
        if isinstance(all_field_lengths, torch.Tensor):
            all_field_lengths = all_field_lengths.tolist()

        # 简单的校验
        if X_train.shape[1] != len(all_field_lengths):
            print(f"🔴 错误：CSV 列数 ({X_train.shape[1]}) 与 field_lengths 长度 ({len(all_field_lengths)}) 不一致！")
            sys.exit(1)

        # --- 3. 标签处理 (1-indexed -> 0-indexed) ---
        if task == 'classification':
            label_min = np.min(y_train)
            label_max = np.max(y_train)
            if label_min == 1 and label_max == cfg.num_classes:
                print(f"    [!] 警告：检测到 1-indexed 标签，正在修正...")
                y_train = y_train - 1
                y_test = y_test - 1

        # --- 4. 转换分类特征 (核心修改) ---
        
        # 自动识别：长度 > 1 的是类别特征
        cat_indices = [i for i, length in enumerate(all_field_lengths) if length > 1]
        
        print(f"    自动检测到 {len(cat_indices)} 个类别特征 (根据 field_lengths > 1)。")

        if len(cat_indices) > 0:
            # 为了避免 pandas 的 SettingWithCopyWarning 或类型混淆，
            # 建议给列重命名为字符串，这样处理起来更清晰
            X_train.columns = [str(i) for i in range(X_train.shape[1])]
            X_test.columns  = [str(i) for i in range(X_test.shape[1])]

            # 仅将检测到的类别列转换为 'category' 类型
            for idx in cat_indices:
                col_name = str(idx)
                # 转换为 category
                X_train[col_name] = X_train[col_name].astype('category')
                
                # 对齐测试集 (处理未知类别)
                # set_categories 确保测试集即使有未见过的类别也不会报错(会变成NaN)，
                # 或者确保其类别列表与训练集一致
                X_test[col_name] = pd.Categorical(X_test[col_name], categories=X_train[col_name].cat.categories, ordered=False)
            
            print("    已将类别特征转换为 pandas 'category' dtype。LightGBM 将自动识别它们。")
        else:
            print("    未检测到类别特征，所有列将作为数值处理。")

    except Exception as e:
        print(f"🔴 加载数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 5. 确定问题类型 (保持不变) ---
    print("-" * 30)
    if task == 'classification':
        num_classes = cfg.get('num_classes', len(np.unique(y_train))) 
        if num_classes == 2:
            problem_type = 'binary'; objective = 'binary'; num_class_param = {}; scoring_metric = 'roc_auc'
        else:
            problem_type = 'multiclass'; objective = 'multiclass'; num_class_param = {'num_class': num_classes}; scoring_metric = 'accuracy'
    elif task == 'regression':
        problem_type = 'regression'; objective = 'regression_l2'; num_class_param = {}; scoring_metric = 'neg_root_mean_squared_error'
    else:
        print(f"错误: 不支持的任务类型 '{task}'"); sys.exit(1)

    print(f"LGBM Objective: {objective}, Scoring: {scoring_metric}")
    
    return X_train, y_train, X_test, y_test, problem_type, objective, num_class_param, scoring_metric


def get_model_and_grid(problem_type, objective, num_class_param, seed):
    """
    根据问题类型获取LGBM模型和参数网格。
    """
    if problem_type in ['binary', 'multiclass']:
        model = lgb.LGBMClassifier(
            objective=objective,
            **num_class_param,
            random_state=seed,
            n_jobs=1,
            
            # --- ↓↓↓ 关键修改：添加下面一行 ↓↓↓ ---
            bagging_freq=1 # 只需要在这里激活 bagging
        )
    elif problem_type == 'regression':
        model = lgb.LGBMRegressor(
            objective=objective,
            random_state=seed,
            n_jobs=1,
            
            # --- ↓↓↓ 关键修改：添加下面一行 ↓↓↓ ---
            bagging_freq=1 # 只需要在这里激活 bagging
        )
    
    # --- ↓↓↓ 关键修改：修改 param_grid ↓↓↓ ---
    param_grid = {
        'num_leaves': [31, 127],
        'learning_rate': [0.01, 0.1],
        'min_child_samples': [20, 50, 100],
        'min_sum_hessian_in_leaf': [1e-3, 1e-2, 1e-1],
        
        # --- 将采样参数添加到网格搜索中 ---
        'feature_fraction': [0.8, 0.9], # 搜索 80% 或 90% 的特征
        'bagging_fraction': [0.8, 0.9]  # 搜索 80% 或 90% 的数据
    }
    
    return model, param_grid


def run_experiment(X_train, y_train, X_test, y_test, problem_type, objective, num_class_param, scoring_metric, seed):
    """
    使用给定的随机种子运行一次模型训练和评估。
    """
    print(f"\n{'='*25} ---------------- 随机种子: {seed} ---------------- {'='*25}")
    
    model, param_grid = get_model_and_grid(problem_type, objective, num_class_param, seed)

    print(f"开始进行网格搜索 (评分指标: {scoring_metric})...")
    
    if problem_type == 'regression':
        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    else:
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring_metric,
        cv=cv_splitter,
        n_jobs=-1, # GridSearchCV 使用所有核心
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print("网格搜索完成！")
    print(f"找到的最佳超参数: {grid_search.best_params_}")
    print(f"在交叉验证中的最佳 {scoring_metric}: {grid_search.best_score_:.4f}")
    print("-" * 30)

    print("使用最佳模型在测试集(验证集)上进行最终评估...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    if problem_type in ['binary', 'multiclass']:
        y_pred_proba = best_model.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        if problem_type == 'binary':
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        
        result_line = f"acc:{acc:.4f},auc:{auc:.4f},macro-F1:{macro_f1:.4f}"
    
    elif problem_type == 'regression':
        # --- 修改点 2: 增加 MAE 和 R2 的计算 ---
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        result_line = f"rmse:{rmse:.4f},mae:{mae:.4f},r2:{r2:.4f}"

    print("评估结果:")
    print(result_line)
    
    return result_line, grid_search.best_params_

def call_lgb_with_config(config_name: str):
    """
    Functional interface to run the LightGBM experiment with a specific config.
    """
    # 1. Initialize and Compose the configuration
    # Ensure config_path points correctly to your yaml directory
    with initialize(version_base=None, config_path="../configs"):
        # Load the specific config file
        cfg = compose(config_name=config_name)
    
    # 2. Run the core experiment logic
    return run_lgb_experiment(cfg)

def run_lgb_experiment(cfg: DictConfig):
    """
    Core logic for LightGBM path resolution and experiment execution.
    """
    
    # --- 1.A. Parse Data Paths ---
    print("--- 1.A. Parsing Data Paths ---")
    data_root = cfg.get('data_base') 
    
    if data_root is not None:
        print(f"    Detected 'data_root', prepending to all path keys: {data_root}")
        
        # Define all path keys that require prefixing
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
        
        # Traverse keys and update paths if they exist
        for key in path_keys:
            if key in cfg and cfg[key] is not None:
                original_path = cfg[key]
                cfg[key] = os.path.join(data_root, original_path)
    else:
        print("    No 'data_root' provided. Assuming paths are already absolute or relative to CWD.")

    print("\n--- Final Configuration (Paths Resolved): ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------")
    print(f"Current Working Directory: {os.getcwd()}")
    print("--------------------")

    # 1. Load and preprocess data
    X_train, y_train, X_test, y_test, problem_type, objective, num_class_param, scoring_metric = load_and_preprocess_data(cfg)

    # --- Handle Dataset Name ---
    # When using Compose API, HydraConfig might not be populated automatically
    try:
        dataset_name = HydraConfig.get().runtime.choices.get("dataset", "unknown_dataset")
    except Exception:
        # Fallback: try to get dataset name from config or use 'manual_run'
        dataset_name = cfg.get('target', "manual_run")

    output_filename = os.path.join("result", f"lgb_results_{dataset_name}.txt")
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    seeds = [2022, 2023, 2024]
    
    # 3. Open file to write results
    print(f"\nPreparing to write results to: {output_filename}")
    
    with open(output_filename, 'a') as f:
        f.write("--- Final Config ---\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("-" * 30 + "\n\n")

        # 4. Iterate through random seeds
        for seed in seeds:
            result_line, best_params = run_experiment(
                X_train, y_train, X_test, y_test,
                problem_type, objective, num_class_param, scoring_metric,
                seed
            )
            
            # 5. Log results
            print(f"Writing results for seed {seed} to {output_filename}...")
            f.write(f"seed:{seed}\n")
            f.write(f"best_params: {best_params}\n")
            f.write(result_line + "\n\n")

    print(f"\nTask Complete! Results saved at: '{os.path.join(os.getcwd(), output_filename)}'")
    return output_filename

