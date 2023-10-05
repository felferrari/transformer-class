from typing import Any
import torch
import lightning as L
from pydoc import locate
from torchmetrics.classification import MulticlassF1Score

class ModelModule(L.LightningModule):
    def __init__(self, training_params):
        super().__init__()
        self.save_hyperparameters()
        loss_params = training_params['loss_fn']['params']
        if 'weight' in loss_params.keys():
            loss_params['weight'] = torch.tensor(loss_params['weight'])
        if 'alpha' in loss_params.keys():
            loss_params['alpha'] = torch.tensor(loss_params['alpha'])
        self.loss = locate(training_params['loss_fn']['module'])(**loss_params)
        self.train_metric = MulticlassF1Score(num_classes = training_params['n_classes'], average= 'none')
        self.val_metric = MulticlassF1Score(num_classes = training_params['n_classes'], average= 'none')

        self.optimizer_cfg = training_params['optimizer']

    def training_step(self, batch, batch_idx):
        x, y = batch
        def_target = y[0]
        def_prev = self.forward(x)
        loss_batch = self.loss(def_prev, def_target)
        self.log("train_loss", loss_batch, prog_bar=True, logger = True, on_step=True, on_epoch=True)
        if batch_idx % 10 == 0:
            self.train_metric.cpu()
            f1 = self.train_metric(def_prev.detach().cpu(), def_target.detach().cpu())
            self.log("train_f1_class_0",f1[0].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
            self.log("train_f1_class_1",f1[1].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
            self.log("train_f1_class_2",f1[2].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
        return loss_batch
    
    def on_train_epoch_end(self) -> None:
        self.train_metric.reset()
        return super().on_train_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        def_target = y[0]
        def_prev = self.forward(x)
        loss_batch = self.loss(def_prev, def_target)
        self.log("val_loss", loss_batch, prog_bar=True, logger = True, on_step=True, on_epoch=True)
        if batch_idx % 10 == 0:
            self.val_metric.cpu()
            f1 = self.val_metric(def_prev.detach().cpu(), def_target.detach().cpu())
            self.log("val_f1_class_0",f1[0].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
            self.log("val_f1_class_1",f1[1].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
            self.log("val_f1_class_2",f1[2].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
        return loss_batch
    
    def on_validation_epoch_end(self) -> None:
        self.val_metric.reset()
        return super().on_validation_epoch_end()
    
    def configure_optimizers(self):
        optimizer = locate(self.optimizer_cfg['module'])(self.parameters(), lr = self.optimizer_cfg['params']['lr'])
        return optimizer
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.forward(batch[0])
    
class ModelModuleMultiTask(L.LightningModule):
    def __init__(self, training_params):
        super().__init__()
        self.save_hyperparameters()
        loss_def_params = training_params['loss_fn_def']['params']
        self.loss_def = locate(training_params['loss_fn_def']['module'])(**loss_def_params)

        loss_cloud_params = training_params['loss_fn_cloud']['params']
        self.loss_cloud = locate(training_params['loss_fn_cloud']['module'])(**loss_cloud_params)
        #self.register_parameter ("loss_cloud_weight", torch.tensor(training_params['loss_fn_cloud']['weight'], requires_grad=False))
        self.register_buffer ("loss_cloud_weight", torch.tensor(training_params['loss_fn_cloud']['weight'], requires_grad=False))
        #self.loss_cloud_weight = torch.tensor(training_params['loss_fn_cloud']['weight'])

        self.train_metric_def = MulticlassF1Score(num_classes = training_params['n_classes'], average= 'none')
        self.val_metric_def = MulticlassF1Score(num_classes = training_params['n_classes'], average= 'none')

        self.optimizer_cfg = training_params['optimizer']

    def training_step(self, batch, batch_idx):
        x, y = batch
        def_target = y[0]
        cloud_target = torch.squeeze(y[1], dim= 2)
        prev = self.forward(x)
        #self.loss_cloud_weight = self.loss_cloud_weight.type_as(x[0], device=self.device)

        loss_def_batch = self.loss_def(prev[0], def_target)
        loss_cloud_batch = self.loss_cloud(prev[1], cloud_target)
        loss_batch = loss_def_batch + self.loss_cloud_weight * loss_cloud_batch

        self.log("train_loss", loss_batch, prog_bar=True, logger = True, on_step=True, on_epoch=True)
        self.log("train_loss_def", loss_def_batch, prog_bar=True, logger = True, on_step=True, on_epoch=True)
        self.log("train_loss_cloud", loss_cloud_batch, prog_bar=True, logger = True, on_step=True, on_epoch=True)
        if batch_idx % 10 == 0:
            self.train_metric_def.cpu()
            f1 = self.train_metric_def(prev[0].detach().cpu(), def_target.detach().cpu())
            self.log("train_f1_class_0",f1[0].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
            self.log("train_f1_class_1",f1[1].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
            self.log("train_f1_class_2",f1[2].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
        return loss_batch
    
    def on_train_epoch_end(self) -> None:
        self.train_metric_def.reset()
        return super().on_train_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        def_target = y[0]
        cloud_target = torch.squeeze(y[1], dim= 2)
        prev = self.forward(x)
        #self.loss_cloud_weight = self.loss_cloud_weight.type_as(x[0], device=self.device)

        loss_def_batch = self.loss_def(prev[0], def_target)
        loss_cloud_batch = self.loss_cloud(prev[1], cloud_target)
        loss_batch = loss_def_batch + self.loss_cloud_weight * loss_cloud_batch

        self.log("val_loss", loss_batch, prog_bar=True, logger = True, on_step=True, on_epoch=True)
        self.log("val_loss_def", loss_def_batch, prog_bar=True, logger = True, on_step=True, on_epoch=True)
        self.log("val_loss_cloud", loss_cloud_batch, prog_bar=True, logger = True, on_step=True, on_epoch=True)
        if batch_idx % 10 == 0:
            self.val_metric_def.cpu()
            f1 = self.val_metric_def(prev[0].detach().cpu(), def_target.detach().cpu())
            self.log("val_f1_class_0",f1[0].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
            self.log("val_f1_class_1",f1[1].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
            self.log("val_f1_class_2",f1[2].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
        return loss_batch
    
    def on_validation_epoch_end(self) -> None:
        self.val_metric_def.reset()
        return super().on_validation_epoch_end()
    
    def configure_optimizers(self):
        optimizer = locate(self.optimizer_cfg['module'])(self.parameters(), lr = self.optimizer_cfg['params']['lr'])
        return optimizer
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        #x, y = batch
        return self.forward(batch[0])[0]