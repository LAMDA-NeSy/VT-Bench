from typing import Tuple, List, Any

import torch
import torchmetrics
import pytorch_lightning as pl
from torchmetrics import R2Score, MeanSquaredError

# 引入你的模型类 (保持不变)
from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.MultimodalModel import MultimodalModel
from models.Tip_utils.Tip_downstream import TIPBackbone
from models.Tip_utils.Tip_downstream_ensemble import TIPBackboneEnsemble
from models.DAFT import DAFT
from models.MultimodalModelMUL import MultimodalModelMUL
from models.MultimodalModelTransformer import MultimodalModelTransformer

class Evaluator_Regression(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # 1. 确保 hparams 被正确保存
        self.save_hyperparameters(hparams)

        # --- 模型选择逻辑 ---
        if self.hparams.eval_datatype == 'imaging':
            self.model = ImagingModel(self.hparams)
        elif self.hparams.eval_datatype == 'multimodal':
            assert self.hparams.strategy == 'tip'
            if self.hparams.finetune_ensemble == True:
                self.model = TIPBackboneEnsemble(self.hparams)
            else:
                self.model = TIPBackbone(self.hparams)
        elif self.hparams.eval_datatype == 'tabular':
            self.model = TabularModel(self.hparams)
        elif self.hparams.eval_datatype == 'imaging_and_tabular':
            if self.hparams.algorithm_name == 'DAFT':
                self.model = DAFT(self.hparams)
            elif self.hparams.algorithm_name in set(['CONCAT','MAX']):
                if self.hparams.strategy == 'tip':
                    self.model = MultimodalModelTransformer(self.hparams)
                else:
                    self.model = MultimodalModel(self.hparams)
            elif self.hparams.algorithm_name == 'MUL':
                # 2. 安全检查：MUL 方法必须配合大 ResNet
                if 'resnet18' in self.hparams.model or 'resnet34' in self.hparams.model:
                     print("Warning: MUL method usually requires resnet50/101 due to channel dimensions!")
                self.model = MultimodalModelMUL(self.hparams)
    
        self.criterion = torch.nn.MSELoss()

        # --- Metrics 定义 ---
        self.mae_train = torchmetrics.MeanAbsoluteError()
        self.mae_val = torchmetrics.MeanAbsoluteError()
        
        # 测试指标
        self.mae_test = torchmetrics.MeanAbsoluteError()
        self.rmse_test = MeanSquaredError(squared=False)
        self.r2_test = R2Score()
        self.pcc_test = torchmetrics.PearsonCorrCoef(num_outputs=1)
        
        self.best_val_score = float('inf') 

        print(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        x, y = batch
        
        # 3. 关键修改：强制转换标签为 float32，防止 Double 报错
        y = y.float() 

        y_hat = self.forward(x)
        
        # 拍扁
        y_hat = y_hat.view(-1)
        y = y.view(-1)

        loss = self.criterion(y_hat, y)
        
        # Metric 计算不需要 detach，torchmetrics 会自己处理，但为了显存安全 detach 也无妨
        # 注意：不要在 step 里 detach loss 用于反向传播，但这里你返回的是 loss，所以没问题
        self.mae_train(y_hat.detach(), y)

        self.log('eval.train.loss', loss, on_epoch=True, on_step=False)
        self.log('eval.train.mae', self.mae_train, on_epoch=True, on_step=False)

        return loss
  
    # 4. 说明：PyTorch Lightning 新版中，如果 log 设置了 on_epoch=True，
    # Metrics 会自动 reset，training_epoch_end 其实可以省略，但保留也没错。
    def training_epoch_end(self, _) -> None:
        self.mae_train.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        x, y = batch
        y = y.float() # 类型转换

        y_hat = self.forward(x)
        
        y_hat = y_hat.view(-1)
        y = y.view(-1)

        loss = self.criterion(y_hat, y)
        
        self.mae_val(y_hat.detach(), y)
        
        self.log('eval.val.loss', loss, on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # 注意：sanity check 期间不要更新 best score
        if self.trainer.sanity_checking:
            return  

        epoch_mae_val = self.mae_val.compute()

        self.log('eval.val.mae', epoch_mae_val, on_epoch=True, on_step=False)
        
        if epoch_mae_val < self.best_val_score:
            self.best_val_score = epoch_mae_val
        
        self.mae_val.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        x, y = batch 
        y = y.float() # 类型转换

        y_hat = self.forward(x)
        
        y_hat = y_hat.view(-1)
        y = y.view(-1)
            
        y_hat = y_hat.detach()

        self.mae_test(y_hat, y)
        self.rmse_test(y_hat, y)
        
        # 获取 Batch Size 的逻辑是正确的
        if isinstance(x, torch.Tensor):
            batch_size = x.size(0)
        elif isinstance(x, list):
            batch_size = x[0].size(0)
        else:
            batch_size = 0 # Fallback

        if batch_size >= 2:
            self.r2_test(y_hat, y)
            self.pcc_test(y_hat, y)

    def test_epoch_end(self, _) -> None:
        test_mae = self.mae_test.compute()
        test_rmse = self.rmse_test.compute()
        test_r2 = self.r2_test.compute()
        test_pcc = self.pcc_test.compute()

        self.log('test.mae', test_mae)
        self.log('test.rmse', test_rmse)
        self.log('test.r2', test_r2)
        self.log('test.pcc.mean', test_pcc)
        
        self.mae_test.reset()
        self.rmse_test.reset()
        self.r2_test.reset()
        self.pcc_test.reset()

    def configure_optimizers(self):
        # 5. 确保这里使用的是正确的 LR 参数名
        lr = self.hparams.lr_eval 
        
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=self.hparams.weight_decay_eval
        )
        
        # 修改 min_lr 逻辑，防止参数名不存在
        min_lr_value = lr * 0.0001
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=int(10/self.hparams.check_val_every_n_epoch), 
            min_lr=min_lr_value
        )
        
        return {
           "optimizer": optimizer, 
           "lr_scheduler": {
             "scheduler": scheduler,
             "monitor": 'eval.val.mae', 
             "strict": False
           }
         }