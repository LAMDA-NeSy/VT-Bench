from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_warn
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional.classification import binary_auroc, multiclass_auroc, binary_accuracy, multiclass_accuracy
from torchmetrics.functional import mean_squared_error, mean_absolute_error, r2_score

from pl_bolts.models.self_supervised.evaluator import SSLEvaluator


class SSLOnlineEvaluator(Callback):  # pragma: no cover
    """Attaches a MLP for fine-tuning using the standard self-supervised protocol."""

    def __init__(
        self,
        z_dim: int,
        drop_p: float = 0.2,
        hidden_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        swav: bool = False,
        multimodal: bool = False,
        strategy: str = None,
        task: str = 'classification'
    ):
        super().__init__()

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.drop_p = drop_p

        self.optimizer: Optional[Optimizer] = None
        self.online_evaluator: Optional[SSLEvaluator] = None
        self.num_classes: Optional[int] = num_classes
        self.dataset: Optional[str] = None
        self.swav = swav
        self.multimodal = multimodal
        self.strategy = strategy
        self.task = task 

        self._recovered_callback_state: Optional[Dict[str, Any]] = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        if self.num_classes is None:
            # [修正 3] 如果是回归任务，强制 num_classes 为 1，忽略 DataModule 的设置
            if self.task == 'regression':
                self.num_classes = 1
            else:
                self.num_classes = trainer.datamodule.num_classes

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.online_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim,
        ).to(pl_module.device)

        accel = (
            trainer.accelerator_connector
            if hasattr(trainer, "accelerator_connector")
            else trainer._accelerator_connector
        )
        if accel.is_distributed:
            if accel._strategy_flag in ["ddp", "ddp2", "ddp_spawn"]:
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.online_evaluator = DDP(self.online_evaluator, device_ids=[pl_module.device])
            elif trainer._strategy_flag == "dp":
                from torch.nn.parallel import DataParallel as DP
                self.online_evaluator = DP(self.online_evaluator, device_ids=[pl_module.device])
            else:
                rank_zero_warn(
                    "Does not support this type of distributed accelerator. The online evaluator will not sync."
                )

        self.optimizer = torch.optim.Adam(self.online_evaluator.parameters(), lr=1e-4)

        if self._recovered_callback_state is not None:
            self.online_evaluator.load_state_dict(self._recovered_callback_state["state_dict"])
            self.optimizer.load_state_dict(self._recovered_callback_state["optimizer_state"])

    def to_device(self, batch: Sequence, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:
        if self.swav:
            x, y = batch
            x = x[0]
        elif self.multimodal and self.strategy == 'comparison':
            x_i, _, y, x_orig = batch
            x = x_orig
        elif self.multimodal and self.strategy == 'tip':
            x_i, _, y, x_orig, x_t_orig = batch 
            x = x_orig
            x_t = x_t_orig
        else:
            _, x, y = batch
        
        if self.strategy == 'comparison':
            x = x.to(device)
            y = y.to(device)
            return x, y, None
        elif self.strategy == 'tip':
            x = x.to(device)
            y = y.to(device)
            x_t = x_t.to(device)
            return x, y, x_t
        else:
            # 添加默认处理，防止报错
            x = x.to(device)
            y = y.to(device)
            return x, y, None

    def shared_step(
        self,
        pl_module: LightningModule,
        batch: Sequence,
    ):
        with torch.no_grad():
            with set_training(pl_module, False):
                x, y, x_t = self.to_device(batch, pl_module.device)
                representations = pl_module(x, tabular=x_t) if x_t is not None else pl_module(x)

        # forward pass
        mlp_logits = self.online_evaluator(representations)  # type: ignore[operator]
        
        if self.task == 'regression':
            # [修正 2] 强制转 float，防止 Double 报错
            y = y.float() 
            
            # [修正 1] 使用 view(-1) 代替 squeeze()，处理 BatchSize=1 的情况更安全
            mlp_logits = mlp_logits.view(-1)
            y = y.view(-1)
            
            mlp_loss = F.mse_loss(mlp_logits, y)
            
            rmse = mean_squared_error(mlp_logits, y, squared=False)
            mae = mean_absolute_error(mlp_logits, y)
            r2 = r2_score(mlp_logits, y)
            
            return rmse, r2, mae, mlp_loss
        else:
            # 分类任务
            mlp_loss = F.cross_entropy(mlp_logits, y)
            mlp_logits_sm = mlp_logits.softmax(dim=1)
            if self.num_classes == 2:
                auc = binary_auroc(mlp_logits_sm[:, 1], y)
                acc = binary_accuracy(mlp_logits_sm[:, 1], y)
            else:
                auc = multiclass_auroc(mlp_logits_sm, y, self.num_classes)
                acc = multiclass_accuracy(mlp_logits_sm, y, self.num_classes)

            return acc, auc, None, mlp_loss

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int
    ) -> None:
        
        if self.task == 'regression':
            train_rmse, train_r2, train_mae, mlp_loss = self.shared_step(pl_module, batch)
            
            mlp_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            pl_module.log("classifier.train.loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("classifier.train.rmse", train_rmse, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("classifier.train.mae", train_mae, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("classifier.train.r2", train_r2, on_step=False, on_epoch=True, sync_dist=True)
        else:
            train_acc, train_auc, _, mlp_loss = self.shared_step(pl_module, batch)

            mlp_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            pl_module.log("classifier.train.loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("classifier.train.auc", train_auc, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("classifier.train.acc", train_acc, on_step=False, on_epoch=True, sync_dist=True)


    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        
        if self.task == 'regression':
            val_rmse, val_r2, val_mae, mlp_loss = self.shared_step(pl_module, batch)
            pl_module.log("classifier.val.loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("classifier.val.rmse", val_rmse, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("classifier.val.mae", val_mae, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("classifier.val.r2", val_r2, on_step=False, on_epoch=True, sync_dist=True)
        else:
            val_acc, val_auc, _, mlp_loss = self.shared_step(pl_module, batch)
            pl_module.log("classifier.val.loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("classifier.val.auc", val_auc, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("classifier.val.acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> dict:
        return {"state_dict": self.online_evaluator.state_dict(), "optimizer_state": self.optimizer.state_dict()}

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, callback_state: Dict[str, Any]) -> None:
        self._recovered_callback_state = callback_state


@contextmanager
def set_training(module: nn.Module, mode: bool):
    """Context manager to set training mode."""
    original_mode = module.training

    try:
        module.train(mode)
        yield module
    finally:
        module.train(original_mode)