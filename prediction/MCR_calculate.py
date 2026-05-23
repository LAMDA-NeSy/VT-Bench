import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients
import numpy as np
from tqdm import tqdm
import warnings
import os
import sys
from argparse import Namespace
from collections import OrderedDict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pytorch_lightning as pl

# Ensure the prediction directory is in sys.path for imports
_prediction_dir = os.path.dirname(os.path.abspath(__file__))
if _prediction_dir not in sys.path:
    sys.path.insert(0, _prediction_dir)

# --- Import your models and Datasets ---
from datasets.ImagingAndTabularDataset import ImagingAndTabularDataset
from models.Tip_utils.pieces import DotDict
from models.Evaluator import Evaluator
from models.Evaluator_regression import Evaluator_Regression
from models.MultimodalModel import MultimodalModel
from models.Tip_utils.Tip_downstream import TIPBackbone
from models.Tip_utils.Tip_downstream_ensemble import TIPBackboneEnsemble
from models.DAFT import DAFT


# =========================================================================
#
# PART 2: Model Adapter
# (The key to resolving API inconsistency across different models)
#
# =========================================================================

class ModelAdapter:
    """
    Wraps different models (TIP, TIP-Ensemble, DAFT, CONCAT/MAX) 
    to provide a unified API for analysis functions.
    """
    def __init__(self, model: nn.Module, hparams: Namespace):
        self.model = model
        self.hparams = hparams
        self.model_type = None

        if hparams.eval_datatype == 'multimodal':
            # --- [New] Logic for handling Ensemble models ---
            if hparams.finetune_ensemble:
                self.model_type = "TIP_Ensemble"
                assert isinstance(model, TIPBackboneEnsemble)
                self.image_encoder = model.encoder_imaging
                self.tabular_encoder = model.encoder_tabular
                self.fusion_classifier_func = self._tip_ensemble_fusion_classifier
            else:
                self.model_type = "TIP_Standard"
                assert isinstance(model, TIPBackbone)
                self.image_encoder = model.encoder_imaging
                self.tabular_encoder = model.encoder_tabular
                self.fusion_classifier_func = self._tip_fusion_classifier
            
        elif hparams.eval_datatype == 'imaging_and_tabular':
            if hparams.algorithm_name == 'DAFT':
                self.model_type = "DAFT"
                assert isinstance(model, DAFT)
                self.image_encoder = model.imaging_encoder
                self.tabular_encoder = model.tabular_encoder
                self.fusion_classifier_func = self._daft_fusion_classifier
            elif hparams.algorithm_name in ['CONCAT', 'MAX']:
                self.model_type = hparams.algorithm_name
                assert isinstance(model, MultimodalModel)
                self.image_encoder = model.imaging_model.encoder
                self.tabular_encoder = model.tabular_model.encoder
                self.fusion_classifier_func = self._multimodal_fusion_classifier
            else:
                # Exclude MUL (incompatible) and MultimodalModelTransformer
                raise ValueError(f"Unsupported algorithm_name: {hparams.algorithm_name}")
        else:
            raise TypeError(f"Unsupported eval_datatype: {hparams.eval_datatype}")
            
        print(f"ModelAdapter initialized for model type: {self.model_type}")

    def get_image_embedding(self, img_input: torch.Tensor) -> torch.Tensor:
        """Returns z_i (Image Embedding)"""
        # TIP and TIP_Ensemble share the same logic
        if self.model_type in ["TIP_Standard", "TIP_Ensemble"]:
            return self.image_encoder(img_input)[-1]
        elif self.model_type == "DAFT":
            return self.image_encoder(img_input)[-1]
        elif self.model_type in ["CONCAT", "MAX"]:
            return self.image_encoder(img_input)[0].squeeze()
        
    def get_tabular_embedding(self, tab_input: torch.Tensor) -> torch.Tensor:
        """Returns z_t (Tabular Embedding)"""
        # TIP, TIP_Ensemble, and CONCAT/MAX share the same logic
        if self.model_type in ["TIP_Standard", "TIP_Ensemble"]:
            return self.tabular_encoder(tab_input)
        elif self.model_type == "DAFT":
            return self.tabular_encoder(tab_input) # DAFT's encoder is Identity
        elif self.model_type in ["CONCAT", "MAX"]:
            return self.tabular_encoder(tab_input).squeeze()

    # --- Internal wrappers (for Integrated Gradients) ---
    
    def _tip_fusion_classifier(self, img_embed, tab_embed):
        """Fusion/classification path for the standard TIP model"""
        multimodal_features = self.model.encoder_multimodal(
            x=tab_embed, image_features=img_embed
        )
        return self.model.classifier(multimodal_features[:, 0, :])

    # --- [New] Fusion function for Ensemble ---
    def _tip_ensemble_fusion_classifier(self, img_embed, tab_embed):
        """Fusion/classification path for the Ensemble model."""
        # 1. Multimodal path
        x_m = self.model.encoder_multimodal(x=tab_embed, image_features=img_embed)
        out_m = self.model.classifier_multimodal(x_m[:,0,:])
        
        # 2. Image-only path
        if self.model.encoder_imaging_type == 'resnet':
            out_i = self.model.classifier_imaging(F.adaptive_avg_pool2d(img_embed, (1, 1)).flatten(1))
        elif self.model.encoder_imaging_type == 'vit':
            out_i = self.model.classifier_imaging(img_embed[:,0,:])
        
        # 3. Tabular-only path
        out_t = self.model.classifier_tabular(tab_embed[:,0,:])
        
        # 4. Average Ensemble
        x = (out_i + out_t + out_m) / 3.0
        return x

    def _daft_fusion_classifier(self, img_embed, tab_embed):
        """Fusion/classification path for the DAFT model"""
        x_fused = self.model.daft(x_im=img_embed, x_tab=tab_embed)
        x_fused = self.model.residual(x_fused)
        x_fused = x_fused + self.model.shortcut(img_embed)
        x_fused = self.model.act(x_fused)
        x_pool = F.adaptive_avg_pool2d(x_fused, (1, 1)).flatten(1)
        return self.model.head(x_pool)

    def _multimodal_fusion_classifier(self, img_embed, tab_embed):
        """Fusion/classification path for CONCAT/MAX models"""
        if self.model_type == 'CONCAT':
            x_fused = torch.cat([img_embed, tab_embed], dim=1)
        elif self.model_type == 'MAX':
            x_im_proj = self.model.imaging_proj(img_embed)
            x_fused = torch.stack([x_im_proj, tab_embed], dim=1)
            x_fused, _ = torch.max(x_fused, dim=1)
        return self.model.head(x_fused)


# =========================================================================
#
# PART 3: Adapter-driven Analysis Functions
# (No changes needed here as the Adapter handles all complexities)
#
# =========================================================================

def calculate_full_ablation_performance(adapter, test_loader, device, img_baseline, tab_baseline, task='classification'):
    """
    Calculates performance when both modalities are ablated (x'n).
    Model inputs consist entirely of mean training set embeddings.
    """
    adapter.model.to(device)
    adapter.model.eval()
    
    all_labels, all_preds = [], []
    
    print(f"--- Running Full Ablation (x'n) Evaluation ---")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            img_input = batch[0][0].to(device)
            tab_input = batch[0][1].to(device)
            labels = batch[1].to(device)
            
            batch_size = img_input.shape[0]

            # --- Logic fix: Dynamically match dimensions for expansion ---
            img_shape = [batch_size] + [-1] * (img_baseline.dim() - 1)
            img_embed_ablated = img_baseline.expand(*img_shape)
            
            if adapter.model_type == "DAFT":
                tab_embed_ablated = torch.zeros_like(tab_input) 
            else:
                tab_shape = [batch_size] + [-1] * (tab_baseline.dim() - 1)
                tab_embed_ablated = tab_baseline.expand(*tab_shape)

            logits = adapter.fusion_classifier_func(img_embed_ablated, tab_embed_ablated)
            
            # 3. Collect results
            if task == 'regression':
                preds = logits.squeeze()
            else:
                preds = torch.argmax(logits, dim=1)
                
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    print("\n" + "="*40)
    print(f"RESULTS FOR FULL ABLATION (x'n)")
    print("="*40)
    
    if task == 'classification':
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {f1:.4f}")
        return {'accuracy': acc, 'f1_macro': f1}
    else:
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {np.sqrt(mse):.4f}")
        print(f"R2:   {r2:.4f}")
        return {'mae': mae, 'rmse': np.sqrt(mse), 'r2': r2}


# --- 3.1: Shared Baseline Function (Fixed shape errors) ---
def compute_mean_embeddings(adapter: ModelAdapter, 
                            data_loader: DataLoader, 
                            device: torch.device,
                            baseline_file_path: str):
    """
    Calculates mean embeddings (z_i, z_t) for the dataset.
    Uses accumulation to avoid Out of Memory (OOM) errors.
    """
    
    # --- [Check if file exists] ---
    if os.path.exists(baseline_file_path):
        print(f"--- Loading existing baseline from file ---")
        print(f"Path: {baseline_file_path}")
        try:
            baselines = torch.load(baseline_file_path, map_location=device)
            print("Baseline loaded successfully.")
            return baselines['img_embed'], baselines['tab_embed']
        except Exception as e:
            print(f"Failed to load baseline (file might be corrupted): {e}")
            print("Recomputing baseline...")
    # ----------------------------

    adapter.model.to(device)
    adapter.model.eval()
    
    # --- Modification: Use accumulators instead of storing full lists ---
    running_img_sum = None
    running_tab_sum = None
    total_samples = 0
    # -----------------------------------------------

    print(f"Calculating mean embeddings baseline from {len(data_loader)} batches...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            img_input = batch[0][0].to(device)
            tab_input = batch[0][1].to(device)
            
            if img_input.dtype == torch.uint8:
                img_input = img_input.float() / 255.0
            
            # Get current batch embeddings
            img_embed = adapter.get_image_embedding(img_input)
            tab_embed = adapter.get_tabular_embedding(tab_input)
            
            # --- Accumulation logic ---
            batch_size = img_embed.shape[0]
            
            # 1. Initialize accumulators (if None)
            if running_img_sum is None:
                # Accumulate on CPU to save GPU memory
                # Shape of img_embed[0] might be [C, H, W] or [D]
                running_img_sum = torch.zeros_like(img_embed[0], device='cpu', dtype=torch.float32)
                running_tab_sum = torch.zeros_like(tab_embed[0], device='cpu', dtype=torch.float32)

            # 2. Add current batch sum to running totals
            # sum(dim=0) collapses the batch dimension
            running_img_sum += img_embed.sum(dim=0).cpu()
            running_tab_sum += tab_embed.sum(dim=0).cpu()
            
            total_samples += batch_size
            # ----------------
                
    # --- Calculate final average ---
    mean_img_embed = running_img_sum / total_samples
    mean_tab_embed = running_tab_sum / total_samples
    
    # 2. Add back batch dimension (dim=0) to result in [1, ...]
    mean_img_embed = mean_img_embed.unsqueeze(0) 
    mean_tab_embed = mean_tab_embed.unsqueeze(0) 
    
    print(f"Mean embeddings calculated from {total_samples} samples.")

    try:
        print(f"Saving new baseline to: {baseline_file_path}")
        baselines_to_save = {
            'img_embed': mean_img_embed.to(device),
            'tab_embed': mean_tab_embed.to(device)
        }
        torch.save(baselines_to_save, baseline_file_path)
        print("Baseline saved successfully.")
    except Exception as e:
        print(f"Warning: Failed to save baseline: {e}")

    return mean_img_embed.to(device), mean_tab_embed.to(device)

def evaluate_model_with_ablation(adapter: ModelAdapter, 
                                 data_loader: DataLoader, 
                                 device: torch.device,
                                 img_embed_baseline: torch.Tensor,
                                 tab_embed_baseline: torch.Tensor,
                                 ablate_image: bool = False,
                                 ablate_tabular: bool = False,
                                 task: str='classification'):
    """Evaluates model performance using modality ablation."""
    adapter.model.to(device)
    adapter.model.eval()
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            img_input = batch[0][0].to(device)
            tab_input = batch[0][1].to(device)
            labels = batch[1].to(device)
            
            if img_input.dtype == torch.uint8:
                img_input = img_input.float() / 255.0

            img_embed = adapter.get_image_embedding(img_input)
            tab_embed = adapter.get_tabular_embedding(tab_input)
            
            if ablate_image:
                img_embed = img_embed_baseline.expand_as(img_embed)
            if ablate_tabular:
                if adapter.model_type == "DAFT":
                    tab_embed = torch.zeros_like(tab_embed)
                else:
                    tab_embed = tab_embed_baseline.expand_as(tab_embed)

            logits = adapter.fusion_classifier_func(img_embed, tab_embed)
            if task == 'regression':
                preds = logits.squeeze()
            elif task == 'classification':
                preds = torch.argmax(logits, dim=1)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    if task == 'classification':
        print("Classification")
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        return {'accuracy': accuracy, 'f1_macro': f1_macro}
    else:
        print("Regression")
        mse = mean_squared_error(all_labels, all_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}



def run_ablation_analysis(adapter: ModelAdapter, 
                          data_loader: DataLoader, 
                          device: torch.device,
                          img_embed_baseline: torch.Tensor,
                          tab_embed_baseline: torch.Tensor,
                          task: str='classification'
                          ):
    """Runs a complete ablation study."""
    print("\n--- [ANALYSIS 1/2] Running Modality Ablation ---")
    
    print("Evaluating: Full Model (Baseline)")
    metrics_full = evaluate_model_with_ablation(
        adapter, data_loader, device, img_embed_baseline, tab_embed_baseline, task=task 
    )
    print("Evaluating: Image-Only (Tabular Ablated)")
    metrics_img_only = evaluate_model_with_ablation(
        adapter, data_loader, device, img_embed_baseline, tab_embed_baseline, ablate_tabular=True, task=task
    )
    print("Evaluating: Tabular-Only (Image Ablated)")
    metrics_tab_only = evaluate_model_with_ablation(
        adapter, data_loader, device, img_embed_baseline, tab_embed_baseline, ablate_image=True, task=task
    )
    if task == 'classification':
        base_f1 = metrics_full['accuracy']
        acc_loss_tab_ablated = base_f1 - metrics_img_only['accuracy']
        acc_loss_img_ablated = base_f1 - metrics_tab_only['accuracy']
        total_loss = np.abs(acc_loss_tab_ablated) + np.abs(acc_loss_img_ablated) + 1e-10
        img_contrib_pct = (acc_loss_img_ablated / total_loss) * 100.0
        tab_contrib_pct = (acc_loss_tab_ablated / total_loss) * 100.0
        print("\n--- Ablation Analysis Results (Global Contribution) ---")
        print(f"{'Experiment':<20} | {'Accuracy':<10} | {'Macro F1':<10} | {'Δ (ACC)':<10}")
        print("-" * 56)
        print(f"{'Full Model':<20} | {metrics_full['accuracy']:<10.4f} | {metrics_full['f1_macro']:<10.4f} | {'-':<10}")
        print(f"{'Image-Only':<20} | {metrics_img_only['accuracy']:<10.4f} | {metrics_img_only['f1_macro']:<10.4f} | {-acc_loss_tab_ablated:<10.4f}")
        print(f"{'Tabular-Only':<20} | {metrics_tab_only['accuracy']:<10.4f} | {metrics_tab_only['f1_macro']:<10.4f} | {-acc_loss_img_ablated:<10.4f}")
        print("-" * 56)
        print(f"Global Contribution (F1-based):")
        print(f"  > Image Modality: {img_contrib_pct:.2f}%")
        print(f"  > Tabular Modality: {tab_contrib_pct:.2f}%")
        print("-" * 56)
    else:
        base_mae = metrics_full['mae']
    
        # Calculate Error Increase
        # Positive = Error increases after removal = Modality is Helpful
        # Negative = Error decreases after removal = Modality is Harmful
        mae_increase_no_tab = metrics_img_only['mae'] - base_mae
        mae_increase_no_img = metrics_tab_only['mae'] - base_mae
        
        loss_tab = mae_increase_no_tab  
        loss_img = mae_increase_no_img  
        
        # --- [Key Logic: Use sum of absolute values as denominator] ---
        # Ensures denominator is positive, preserving the sign of contribution.
        # e.g., +10 / 30 = +33% (Helpful), -20 / 30 = -66% (Harmful)
        total_magnitude = abs(loss_tab) + abs(loss_img) + 1e-9
        
        img_contrib_pct = (loss_img / total_magnitude) * 100.0
        tab_contrib_pct = (loss_tab / total_magnitude) * 100.0

        print("\n=== Ablation Results (MAE - Lower is Better) ===")
        print(f"{'Experiment':<20} | {'MAE':<10} | {'RMSE':<10} | {'MAE Increase'}")
        print("-" * 60)
        print(f"{'Full Model':<20} | {metrics_full['mae']:<10.4f} | {metrics_full['rmse']:<10.4f} | {'-':<10}")
        print(f"{'Image-Only':<20} | {metrics_img_only['mae']:<10.4f} | {metrics_img_only['rmse']:<10.4f} | {mae_increase_no_tab:+.4f}")
        print(f"{'Tabular-Only':<20} | {metrics_tab_only['mae']:<10.4f} | {metrics_tab_only['rmse']:<10.4f} | {mae_increase_no_img:+.4f}")
        print("-" * 60)
        print(f"Global Contribution (Allowing Negative):")
        print(f"  > Image:    {img_contrib_pct:+.2f}%")
        print(f"  > Tabular: {tab_contrib_pct:+.2f}%")
        print("============================================\n")
    

def calculate_attribution_magnitude(attributions_tensor):
    return torch.sum(torch.abs(attributions_tensor))

def calculate_relative_contribution(img_attr_score, tab_attr_score, epsilon=1e-10):
    """
    Converts absolute attribution scores of two modalities to relative percentages (r_i, r_t).
    [Fixed: Added .item() to move back to CPU]
    """
    total_attribution = img_attr_score + tab_attr_score + epsilon
    img_percent = (img_attr_score / total_attribution) * 100.0
    tab_percent = (tab_attr_score / total_attribution) * 100.0
    
    # --- [Key Fix] ---
    # .item() converts scalar tensors to Python numbers, handling CUDA-to-CPU transfer.
    return img_percent.item(), tab_percent.item()
    # ---------------------

def get_ig_contribution(adapter: ModelAdapter, 
                        img_sample: torch.Tensor, 
                        tab_sample: torch.Tensor, 
                        img_embed_baseline: torch.Tensor, 
                        tab_embed_baseline: torch.Tensor, 
                        target_class: int,
                        m_steps=100,
                        task: str='classification'):
    """Calculates Integrated Gradients (IG) for a single sample."""
    ig = IntegratedGradients(adapter.fusion_classifier_func)
    adapter.model.eval()
    with torch.no_grad():
        if img_sample.dtype == torch.uint8:
            img_sample = img_sample.float() / 255.0
        img_embed_sample = adapter.get_image_embedding(img_sample)
        tab_embed_sample = adapter.get_tabular_embedding(tab_sample)

    inputs_tuple = (img_embed_sample, tab_embed_sample)
    
    if adapter.model_type == "DAFT":
        tab_baseline_final = torch.zeros_like(tab_embed_sample)
    else:
        tab_baseline_final = tab_embed_baseline
    baselines_tuple = (img_embed_baseline, tab_baseline_final)

    if task == 'classification':
        attributions_tuple = ig.attribute(
            inputs=inputs_tuple, baselines=baselines_tuple,
            target=target_class, n_steps=m_steps,
            return_convergence_delta=False
        )
    else:
        attributions_tuple = ig.attribute(
            inputs=inputs_tuple,
            baselines=baselines_tuple,
            target=None, # Regression output is scalar, no target index needed
            n_steps=m_steps,
            return_convergence_delta=False
        )
    img_attributions = attributions_tuple[0].squeeze(0)
    tab_attributions = attributions_tuple[1].squeeze(0)
    img_score_l1 = calculate_attribution_magnitude(img_attributions)
    tab_score_l1 = calculate_attribution_magnitude(tab_attributions)
    return calculate_relative_contribution(img_score_l1, tab_score_l1)


def run_ig_analysis_on_dataset(adapter: ModelAdapter, 
                               data_loader: DataLoader, 
                               device: torch.device,
                               img_embed_baseline: torch.Tensor,
                               tab_embed_baseline: torch.Tensor,
                               task: str='classification'):
    """Runs a complete Integrated Gradients analysis."""
    print("\n--- [ANALYSIS 2/2] Running Integrated Gradients ---")
    adapter.model.to(device)
    adapter.model.eval()
    all_results = []
    
    if data_loader.batch_size != 1:
        warnings.warn("IG analysis requires batch_size=1. Skipping.")
        return

    print(f"Calculating IG contributions for {len(data_loader)} samples...")
    
    for i, batch in enumerate(tqdm(data_loader)):
        img_sample = batch[0][0].to(device)
        tab_sample = batch[0][1].to(device)
        target_class = batch[1].item()
        
        # No try/except block here to facilitate debugging
        img_pct, tab_pct = get_ig_contribution(
            adapter, img_sample, tab_sample,
            img_embed_baseline, tab_embed_baseline,
            target_class, m_steps=100,
            task=task
        )
        all_results.append((img_pct, tab_pct))


    print("\n--- IG Analysis Results (Average Local Contribution) ---")
    results_array = np.array(all_results)
    mean_img_contrib = np.nanmean(results_array[:, 0])
    mean_tab_contrib = np.nanmean(results_array[:, 1])
    std_img = np.nanstd(results_array[:, 0])
    n_samples = results_array.shape[0]
    ci_img = 1.96 * (std_img / np.sqrt(n_samples))
    ci_tab = 1.96 * (np.nanstd(results_array[:, 1]) / np.sqrt(n_samples))

    print(f"Total samples: {n_samples}")
    print(f"  > Image Modality: {mean_img_contrib:.2f}% (± {ci_img:.2f} 95% CI)")
    print(f"  > Tabular Modality: {mean_tab_contrib:.2f}% (± {ci_tab:.2f} 95% CI)")
    print("-" * 56)


# =========================================================================
#
# PART 4: Execution Script
# (Updated to use Evaluator.load_from_checkpoint)
#
# =========================================================================

def grab_arg_from_checkpoint(args: Namespace, arg_name: str):
    """
    Safely retrieves a hyperparameter from a checkpoint file or Namespace.
    """
    if args.checkpoint:
        try:
            ckpt = torch.load(args.checkpoint) 
            load_args = ckpt['hyper_parameters']
        except FileNotFoundError:
            load_args = args
    else:
        load_args = args
    if isinstance(load_args, Namespace):
        return getattr(load_args, arg_name, None) # .get-like safety
    else:
        return load_args.get(arg_name, None)

def load_analysis_datasets(cfg: Namespace):
    """
    Loads DataLoaders using corrected 'test_dataset' logic.
    """
    
    # Common arguments
    shared_args = {
        'delete_segmentation': cfg.delete_segmentation,
        'eval_one_hot': cfg.eval_one_hot, 'train': False, 'target': cfg.target,
        'corruption_rate': 0.0, 'data_base': cfg.data_base,
        'missing_tabular': False, 'missing_strategy': 'None',
        'missing_rate': 0.0,
        'augmentation_speedup': cfg.augmentation_speedup,
        'algorithm_name': cfg.algorithm_name
    }
    
    # --- 1. Create Training Dataset (for baseline calculation) ---
    if cfg.eval_datatype == 'multimodal':
        img_size_train = grab_arg_from_checkpoint(cfg, 'img_size')
        aug_rate_train = 0.0
    else: # imaging_and_tabular
        img_size_train = cfg.img_size
        aug_rate_train = 0.0
        
    train_dataset = ImagingAndTabularDataset(
        data_path_imaging=cfg.data_train_eval_imaging,
        eval_train_augment_rate=aug_rate_train, 
        data_path_tabular=cfg.data_train_eval_tabular,
        field_lengths_tabular=cfg.field_lengths_tabular,
        labels_path=cfg.labels_train_eval_imaging, 
        img_size=img_size_train, 
        live_loading=cfg.live_loading,
        **shared_args
    )
    
    # --- 2. Create Test Dataset (for analysis) ---
    if cfg.eval_datatype == 'multimodal':
        img_size_test = grab_arg_from_checkpoint(cfg, 'img_size')
        aug_rate_test = 0.0
    else: # imaging_and_tabular
        img_size_test = cfg.img_size
        aug_rate_test = 0.0
        
    test_dataset = ImagingAndTabularDataset(
        data_path_imaging=cfg.data_test_eval_imaging,
        eval_train_augment_rate=aug_rate_test, 
        data_path_tabular=cfg.data_test_eval_tabular,
        field_lengths_tabular=cfg.field_lengths_tabular,
        labels_path=cfg.labels_test_eval_imaging, 
        img_size=img_size_test,
        live_loading=cfg.live_loading,
        **shared_args
    )
    
    # --- 3. Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader_ablation = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader_ig = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers
    )
    print("DataLoaders prepared.")
    return train_loader, test_loader_ablation, test_loader_ig

# ----------------------------------------------------
# Main Execution Logic
# ----------------------------------------------------

def load_and_run_from_checkpoint(checkpoints:str):
    """
    Loads configuration, instantiates objects, and runs analysis.
    """
    
    # --- 1. Load Config ---
    checkpoints_path = checkpoints
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading checkpoint from: {checkpoints_path}")
    checkpoint = torch.load(checkpoints_path, map_location=DEVICE)
    
    hparams = Namespace(**checkpoint['hyper_parameters'])
    state_dict = checkpoint['state_dict']
    
    # --- 2. Apply Hacks/Patches ---
    hparams.missing_tabular = False

    # --- 3. Instantiate Evaluator (Manual instantiation to avoid load errors) ---
    print("Instantiating Evaluator (this will load the model)...")
    print(f"Task: {hparams.task}")
    if hparams.task == 'classification':
        eval_module = Evaluator(hparams).to(DEVICE)
    else:
        eval_module = Evaluator_Regression(hparams).to(DEVICE)
    
    # --- 4. Manually load State Dict ---
    print("Loading state_dict into Evaluator...")
    eval_module.load_state_dict(state_dict)
    eval_module.eval()
    print("Model loaded successfully.")

    # --- 5. Get model and create Adapter ---
    model_to_analyze = eval_module.model
    adapter = ModelAdapter(model_to_analyze, hparams)

    # --- 6. Load Data ---
    train_loader, test_loader_ablation, test_loader_ig = load_analysis_datasets(hparams)

    # --- 7. Run Full Analysis ---
    print("\n--- Computing Shared Baselines ---")
    model_id_name = hparams.algorithm_name if hparams.eval_datatype == 'imaging_and_tabular' else 'TIP'
    dataset_name = hparams.target
    BASELINE_FILE_PATH = f"./{dataset_name}_{model_id_name}_baselines.pt"
    # --------------------------

    print("\n--- Computing Shared Baselines ---")
    img_baseline, tab_baseline = compute_mean_embeddings(
        adapter, 
        train_loader, 
        DEVICE,
        BASELINE_FILE_PATH  # Passing file path
    )
    
    run_ablation_analysis(
        adapter, test_loader_ablation, DEVICE,
        img_baseline, tab_baseline, task=hparams.task
    )
    
    # run_ig_analysis_on_dataset(
    #     adapter, test_loader_ig, DEVICE,
    #     img_baseline, tab_baseline, task=hparams.task
    # )


    # x_prime_n_metrics = calculate_full_ablation_performance(
    #     adapter, test_loader_ablation, DEVICE,
    #     img_baseline, tab_baseline, task=hparams.task
    # )
    
    
    print("\n\n--- Full Analysis Complete ---")

if __name__ == "__main__":
    
    print("\n--- Script ready for execution ---")

    checkpoints_path = [
        '/data1/jiazy/mytip/results/runs/eval/Infarction_TIP_lr_1e-3_Infarction_0120_0143/checkpoint_best_auc.ckpt',
    ]
    for checkpoint in checkpoints_path:
        load_and_run_from_checkpoint(checkpoint)