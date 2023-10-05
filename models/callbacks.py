from typing import Any, Literal
import numpy as np
from einops import rearrange
from lightning.pytorch.callbacks import BasePredictionWriter, Callback
from lightning.pytorch.core import LightningModule
from lightning.pytorch.trainer import Trainer
from einops import rearrange
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

class PredictedImageWriter(BasePredictionWriter):
    def __init__(self, shape, patch_size, n_classes, remove_border, write_interval: Literal['batch', 'epoch', 'batch_and_epoch'] = "batch") -> None:
        self.shape = shape
        self.patch_size = patch_size
        self.padded_shape = (shape[0]+2*patch_size, shape[1]+2*patch_size)
        self.n_classes = n_classes
        self.remove_border = remove_border
        super().__init__(write_interval)

    def restart_image(self):
        self.predicted = np.zeros(self.padded_shape + (self.n_classes, ))
        self.count = np.zeros(self.padded_shape)

        self.predicted = rearrange(self.predicted, 'h w c -> (h w) c')
        self.count = rearrange(self.count, 'h w -> (h w)')

    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        predictions = outputs.cpu().numpy()
        predictions = rearrange(predictions, 'n c h w -> n h w c')
        for pred_i, prediction in enumerate(predictions):
            pred_resized = prediction[self.remove_border:-self.remove_border, self.remove_border:-self.remove_border]
            index_resized = batch[1][pred_i][self.remove_border:-self.remove_border, self.remove_border:-self.remove_border].cpu().numpy()
            self.predicted[index_resized] += pred_resized
            self.count[index_resized] += 1

        #return super().on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
    def predicted_image(self):
        pred = rearrange(self.predicted, '(h w) c -> h w c', h = self.padded_shape[0], w = self.padded_shape[1])
        count = rearrange(self.count, '(h w) -> h w', h = self.padded_shape[0], w = self.padded_shape[1])
        pred = pred[self.patch_size:-self.patch_size, self.patch_size:-self.patch_size]
        count = count[self.patch_size:-self.patch_size, self.patch_size:-self.patch_size]
        count = np.expand_dims(count, axis=-1)

        return pred/count
    
class TrainSamples(Callback):
    def __init__(self, visual_path, max_samples, model_idx) -> None:
        self.visual_path = visual_path
        self.max_samples = max_samples
        self.model_idx = model_idx
        super().__init__()



    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0) -> None:
        if batch_idx > self.max_samples:
            return
        epoch = trainer.current_epoch
        pred = rearrange(trainer.model(batch[0]).cpu().numpy(), 'b c h w -> b h w c')
        pred = np.argmax(pred, axis=-1)
        opt_data = rearrange(batch[0][0].cpu().numpy(), 'b n c h w -> b n h w c')[:,:,:,:,[2,1,0]]
        label = batch[1][0].cpu().numpy()
        for i in range(1): # range(label.shape[0]):
            pred_sample = pred[i]
            opt_sample = opt_data[i]
            label_sample = label[i]

            opt_img_file = self.visual_path / f'sample_{self.model_idx}_{batch_idx}_{i}_{epoch}_0_opt.png'
            fig = px.imshow(opt_sample, facet_col=0, contrast_rescaling = 'minmax')
            fig.write_image(opt_img_file)

            pred_img_file = self.visual_path / f'sample_{self.model_idx}_{batch_idx}_{i}_{epoch}_1_pred.png'
            fig = px.imshow(pred_sample, range_color= [0, 2])
            fig.write_image(pred_img_file)

            label_img_file = self.visual_path / f'sample_{self.model_idx}_{batch_idx}_{i}_{epoch}_2_label.png'
            fig = px.imshow(label_sample, range_color= [0, 2])
            fig.write_image(label_img_file)

            
