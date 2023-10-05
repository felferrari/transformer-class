from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from tqdm import tqdm
import torch
import os
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
#from torch.nn.functional import one_hot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from einops import rearrange
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import lightning as L

class ModelTrainer(L.LightningModule):
    def __init__(self, model, class_weights):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.class_weights = class_weights

    def to(self, *args, **kargs):
        super().to(*args, **kargs)
        self.class_weights = self.class_weights.to(*args, **kargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        def_target = y[0]
        def_prev = self.model(x)
        loss = torch.nn.functional.cross_entropy(def_prev, def_target, weight=self.class_weights, ignore_index = 2)
        self.log("train_loss", loss, prog_bar=True, logger = True, on_step=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        def_target = y[0]
        def_prev = self.model(x)
        loss = torch.nn.functional.cross_entropy(def_prev, def_target, weight=self.class_weights, ignore_index = 2)
        self.log("test_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        def_target = y[0]
        def_prev = self.model(x)
        loss = torch.nn.functional.cross_entropy(def_prev, def_target, weight=self.class_weights, ignore_index = 2)
        self.log("val_loss", loss, prog_bar=True, logger = True, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

def train_model(model, training_params, early_stop, loss_fn, optimizer, train_dl, val_dl, lr_scheduler):
    for epoch in range(training_params['max_epochs']):
        print(f"-------------------------------\nEpoch {epoch}")
        model.train()
        loss, f1_0, f1_1 = train_loop(epoch, train_dl, model, loss_fn, optimizer, training_params)
        model.eval()
        val_loss, f1_0, f1_1 = val_loop(epoch, val_dl, model, loss_fn, training_params)

        if early_stop.testEpoch(model = model, val_value = val_loss):
            min_val = early_stop.better_value
            break

        lr_scheduler.step()

    return epoch, min_val

def train_loop(epoch, dataloader, model, loss_fn, optimizer, params):
    """Executes a train loop epoch

    Args:
        dataloader (Dataloader): Pytorch Dataloader to extract train data
        model (Module): model to be trained
        loss_fn (Module): Loss Criterion
        optimizer (Optimizer): Optimizer to adjust the model's weights

    Returns:
        float: average loss of the epoch
    """
    train_loss, steps = 0, 0
    pbar = tqdm(dataloader)
    metric = MulticlassF1Score(num_classes=params['n_classes'], average = None)
    
    for X, y, cloud in pbar:
        optimizer.zero_grad()
        
        pred = model(X)
        loss = loss_fn(pred, y)
        steps += 1
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        f1 = metric(pred.cpu(), y.cpu())
        m_train_loss = train_loss/steps
        pbar.set_description(f'Train Loss: {m_train_loss:.4f}, F1-Score Classes: 0:{f1[0].item():.4f}, 1:{f1[1].item():.4f}, 2:{f1[2].item():.4f}')
        

    train_loss /= steps
    f1 = metric.compute()
    print(f'Train Loss: {train_loss/steps:.4f}, Acc: 0:{f1[0].item():.4f}, 1:{f1[1].item():.4f}, 2:{f1[2].item():.4f}')
    metric.reset()
    return train_loss, f1[0].item(), f1[1].item()

def val_loop(epoch, dataloader, model, loss_fn, params):
    """Evaluates a validation loop epoch

    Args:
        dataloader (Dataloader): Pytorch Dataloader to extract validation data
        model (Module): model to be evaluated
        loss_fn (Module): Loss Criterion

    Returns:
        float: average loss of the epoch
    """
    val_loss, steps = 0, 0
    #f1 = MulticlassF1Score(num_classes=params['n_classes'], ignore_index = params['loss_fn']['ignore_index']).to(device)
    metric = MulticlassF1Score(num_classes=params['n_classes'], average = None)

    with torch.no_grad():
        pbar = tqdm(dataloader)
        for X, y, cloud in pbar:
            pred = model(X)
            loss = loss_fn(pred, y)
            steps += 1
            val_loss += loss.item()
            f1 = metric(pred.cpu(), y.cpu())
            m_val_loss = val_loss/steps
            pbar.set_description(f'Validation Loss: {m_val_loss:.4f}, F1-Score Classes: 0:{f1[0].item():.4f}, 1:{f1[1].item():.4f}, 2:{f1[2].item():.4f}')

    val_loss /= steps
    f1 = metric.compute()
    print(f'Validation Loss: {val_loss/steps:.4f}, F1-Score Classes: 0:{f1[0].item():.4f}, 1:{f1[1].item():.4f}, 2:{f1[2].item():.4f}')
    return val_loss, f1[0].item(), f1[1].item()

def sample_figures_loop(dataloader, model, n_batches, epoch, path_to_samples, model_idx):

    pbar = tqdm(desc = f'Sampling figures', total = n_batches)
    for i_sample, sample in enumerate(dataloader):
        if i_sample >= n_batches:
            break
        label = sample[1]
        x = sample[0]
        pred = model(x).argmax(axis=1)
        cmap = plt.get_cmap('tab20', 3)

        plt.close('all')

        #for i, l in enumerate(label):
        figure, ax = plt.subplots(nrows=2, ncols=4, figsize = (10,5))
        p = pred[0]
        l = label[0]
        #cmap = plt.get_cmap('tab20', 1)
        img_opt_0 = rearrange(x[0][0][0].cpu().numpy(), 'c h w -> h w c')[:,:,[3,2,1]]
        img_opt_1 = rearrange(x[0][-1][0].cpu().numpy(), 'c h w -> h w c')[:,:,[3,2,1]]

        img_sar_0 = rearrange(x[1][0][0].cpu().numpy(), 'c h w -> h w c')[:,:,0]
        img_sar_1 = rearrange(x[1][-1][0].cpu().numpy(), 'c h w -> h w c')[:,:,1]

        ax[0, 0].imshow(img_opt_0*5)
        ax[0, 0].title.set_text('OPT 0')

        ax[0, 1].imshow(img_opt_1*5)
        ax[0, 1].title.set_text('OPT 1')

        ax[1, 0].imshow(img_sar_0, cmap = 'gray')
        ax[1, 0].title.set_text('SAR 0')

        ax[1, 1].imshow(img_sar_1, cmap = 'gray')
        ax[1, 1].title.set_text('SAR 1')

        l0 = ax[0, 2].imshow(l.cpu(), cmap = cmap, vmin=0, vmax = 2)
        ax[0, 2].title.set_text('Label')
        divider = make_axes_locatable(ax[0, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        figure.colorbar(l0, cax=cax, orientation='vertical', ticks = np.arange(3))

        l1 = ax[1, 2].imshow(l.cpu(), cmap = cmap, vmin=0, vmax = 2)
        ax[1, 2].title.set_text('Label')
        divider = make_axes_locatable(ax[1, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        figure.colorbar(l1, cax=cax, orientation='vertical', ticks = np.arange(3))

        p0 = ax[0, 3].imshow(p.cpu(), cmap = cmap, vmin=0, vmax = 2)
        ax[0, 3].title.set_text('Prediction')
        divider = make_axes_locatable(ax[0, 3])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        figure.colorbar(p0, cax=cax, orientation='vertical', ticks = np.arange(3))
        
        p1 = ax[1, 3].imshow(p.cpu(), cmap = cmap, vmin=0, vmax = 2)
        ax[1, 3].title.set_text('Prediction')
        divider = make_axes_locatable(ax[1, 3])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        figure.colorbar(p1, cax=cax, orientation='vertical', ticks = np.arange(3))

        plt.setp(ax, xticks=[], yticks=[])
        figure.suptitle(f'Epoch {epoch}')

        fig_path = path_to_samples / f'model_{model_idx}'
        fig_path.mkdir(exist_ok=True)
        fig_path = fig_path / f'sample_{i_sample}_{epoch}.png'
        figure.savefig(fig_path, bbox_inches='tight')
        figure.clf()
        plt.close()

        pbar.update(1)

class EarlyStop():
    def __init__(self, train_patience, path_to_save, min_delta = 0, min_epochs = None) -> None:

        self.train_pat = train_patience
        self.no_change_epochs = 0
        self.better_value = None
        self.path_to_save = path_to_save
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.decorred_epochs = 0

    def testEpoch(self, model, val_value):
        self.decorred_epochs+=1
        if self.min_epochs is not None:
            if self.decorred_epochs <= self.min_epochs:
                print(f'Epoch {self.decorred_epochs} from {self.min_epochs} minimum epochs. Validation value:{val_value:.4f}' )
                return False
        if self.better_value is None:
            self.no_change_epochs += 1
            self.better_value = val_value
            print(f'First Validation Value {val_value:.4f}. Saving model in {self.path_to_save}' )
            torch.save(model.state_dict(), self.path_to_save)
            return False
        delta = -(val_value - self.better_value)
        if delta > self.min_delta:
            self.no_change_epochs = 0
            print(f'Validation value improved from {self.better_value:.4f} to {val_value:.4f}. Saving model in {self.path_to_save}' )
            torch.save(model.state_dict(), self.path_to_save)
            self.better_value = val_value
            return False
        else:
            self.no_change_epochs += 1
            print(f'No improvement for {self.no_change_epochs}/{self.train_pat} epoch(s). Better Validation value is {self.better_value:.4f}' )
            if self.no_change_epochs >= self.train_pat:
                return True
