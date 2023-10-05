import argparse
from utils.ops import count_parameters, save_yaml, load_yaml
import torch
from torch.multiprocessing import Process, freeze_support
from torch.utils.data import DataLoader
import time
from  pathlib import Path
from utils.datasets import TrainDataset, ValDataset#, to_gpu
from pydoc import locate
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from models.callbacks import TrainSamples
#from fvcore.nn import flop_count_table, FlopCountAnalysis, flop_count

parser = argparse.ArgumentParser(
    description='Train NUMBER_MODELS models based in the same parameters'
)

parser.add_argument( # The path to the config file (.yaml)
    '-c', '--cfg',
    type = Path,
    default = 'cfg.yaml',
    help = 'Path to the config file (.yaml).'
)

parser.add_argument( # Experiment number
    '-e', '--experiment',
    type = int,
    default = 99,
    help = 'The number of the experiment'
)

parser.add_argument( # Model number
    '-m', '--model',
    type = int,
    default = -1,
    help = 'Number of the model to be retrained'
)

parser.add_argument( # Model number
    '-i', '--start-model',
    type = int,
    default = -1,
    help = 'Number of the model to be retrained'
)

parser.add_argument( # Accelerator
    '-d', '--devices',
    type = int,
    nargs='+',
    default = [0],
    help = 'Accelerator devices to be used'
)

parser.add_argument( # Log in neptune
    '-n', '--neptune-log',
    action='store_true',
    help = 'Log in neptune'
)

parser.add_argument( # specific site location number
    '-s', '--site',
    type = int,
    default=2,
    help = 'Site location number'
)

args = parser.parse_args()

cfg = load_yaml(args.cfg)
site_cfg = load_yaml(f'site_{args.site}.yaml')

training_params = cfg['training_params']
preparation_params = cfg['preparation_params']
experiment_params = cfg['experiments'][f'exp_{args.experiment}']
paths_params = cfg['paths']
general_params = cfg['general_params']

experiments_folders = general_params['experiments_folders']

original_opt_imgs = site_cfg['original_data']['opt_imgs']
original_sar_imgs = site_cfg['original_data']['sar_imgs']

experiments_paths = paths_params['experiments']

#create experiment folder structure
exp_path = Path(experiments_paths) / f'exp_{args.experiment}'

models_path = exp_path / experiments_folders['models']
logs_path = exp_path / experiments_folders['logs']
visual_path = exp_path / experiments_folders['visual']
predicted_path = exp_path / experiments_folders['predicted']
results_path = exp_path / experiments_folders['results']
visual_logs_path = exp_path / experiments_folders['visual_logs']

exp_path.mkdir(exist_ok=True)
models_path.mkdir(exist_ok=True)
logs_path.mkdir(exist_ok=True)
visual_path.mkdir(exist_ok=True)
predicted_path.mkdir(exist_ok=True)
results_path.mkdir(exist_ok=True)
visual_logs_path.mkdir(exist_ok=True)

#setting up prepared data source
prepared_folder = Path(paths_params['prepared_data'])

train_folder = prepared_folder / preparation_params['train_folder']
val_folder = prepared_folder / preparation_params['validation_folder']
prepared_patches_file = prepared_folder / preparation_params['prepared_data']
prepared_patches = load_yaml(prepared_patches_file)


patch_size = general_params['patch_size']
n_classes = general_params['n_classes']

batch_size = training_params['batch_size'] // len(args.devices)
min_val_loss = training_params['min_val_loss']

if 'override_train_params' in experiment_params.keys():
    training_params.update(experiment_params['override_train_params'])


def run(model_idx):
    last_val_loss = float('inf')
    
    while last_val_loss > min_val_loss:
        torch_seed = int(1000*time.time())
        torch.manual_seed(torch_seed)
        torch.set_float32_matmul_precision('high')

        model = locate(experiment_params['model'])(experiment_params, training_params)

        if 'pretrained_encoders' in experiment_params.keys():
            pretrained_encoder_params = experiment_params['pretrained_encoders']
            pretrained_opt_exp = pretrained_encoder_params['opt_exp']
            pretrained_sar_exp = pretrained_encoder_params['sar_exp']
            opt_exp_params = cfg['experiments'][f'exp_{pretrained_opt_exp}']
            sar_exp_params = cfg['experiments'][f'exp_{pretrained_sar_exp}']

            opt_exp_path = Path(experiments_paths) / f'exp_{pretrained_opt_exp}'
            sar_exp_path = Path(experiments_paths) / f'exp_{pretrained_sar_exp}'

            opt_models_path = opt_exp_path / experiments_folders['models']
            sar_models_path = sar_exp_path / experiments_folders['models']

            opt_logs_path = opt_exp_path / experiments_folders['logs']
            sar_logs_path = sar_exp_path / experiments_folders['logs']

            opt_pred_results = load_yaml(opt_logs_path / f'model_{model_idx}' / 'train_results.yaml')
            sar_pred_results = load_yaml(sar_logs_path / f'model_{model_idx}' / 'train_results.yaml')

            opt_model = locate(opt_exp_params['model'])
            sar_model = locate(sar_exp_params['model'])

            opt_model_path = opt_models_path / opt_pred_results['model_file']
            sar_model_path = sar_models_path / sar_pred_results['model_file']

            opt_model = opt_model.load_from_checkpoint(opt_model_path)
            sar_model = sar_model.load_from_checkpoint(sar_model_path)

            model.encoder_opt.load_state_dict(opt_model.encoder.state_dict())
            model.encoder_sar.load_state_dict(sar_model.encoder.state_dict())

        if 'pretrained_encoders_decoders' in experiment_params.keys():
            pretrained_encoder_params = experiment_params['pretrained_encoders_decoders']
            pretrained_opt_exp = pretrained_encoder_params['opt_exp']
            pretrained_sar_exp = pretrained_encoder_params['sar_exp']
            opt_exp_params = cfg['experiments'][f'exp_{pretrained_opt_exp}']
            sar_exp_params = cfg['experiments'][f'exp_{pretrained_sar_exp}']

            opt_exp_path = Path(experiments_paths) / f'exp_{pretrained_opt_exp}'
            sar_exp_path = Path(experiments_paths) / f'exp_{pretrained_sar_exp}'

            opt_models_path = opt_exp_path / experiments_folders['models']
            sar_models_path = sar_exp_path / experiments_folders['models']

            opt_logs_path = opt_exp_path / experiments_folders['logs']
            sar_logs_path = sar_exp_path / experiments_folders['logs']

            opt_pred_results = load_yaml(opt_logs_path / f'model_{model_idx}' / 'train_results.yaml')
            sar_pred_results = load_yaml(sar_logs_path / f'model_{model_idx}' / 'train_results.yaml')

            opt_model = locate(opt_exp_params['model'])
            sar_model = locate(sar_exp_params['model'])

            opt_model_path = opt_models_path / opt_pred_results['model_file']
            sar_model_path = sar_models_path / sar_pred_results['model_file']

            opt_model = opt_model.load_from_checkpoint(opt_model_path)
            sar_model = sar_model.load_from_checkpoint(sar_model_path)

            model.encoder_opt.load_state_dict(opt_model.encoder.state_dict())
            model.decoder_opt.load_state_dict(opt_model.decoder.state_dict())
            model.encoder_sar.load_state_dict(sar_model.encoder.state_dict())
            model.decoder_sar.load_state_dict(sar_model.decoder.state_dict())

        tb_logger = TensorBoardLogger(
            save_dir = logs_path,
            name = f'model_{model_idx}',
            version = ''
        )

        loggers = [tb_logger]

        log_cfg = Path('loggers.yaml')
        if log_cfg.exists() and args.neptune_log:
            log_cfg = load_yaml(log_cfg)
            if 'neptune' in log_cfg.keys():
                neptune_cfg = log_cfg['neptune']
                neptune_logger = locate(neptune_cfg['module'])(
                    project = neptune_cfg['project'],
                    api_key = neptune_cfg['api_key']
                )
                params = {
                    'experiment': args.experiment,
                    'model_idx': model_idx,
                    'model': experiment_params['model']
                }
                neptune_logger.log_hyperparams(params=params)
                loggers.append(neptune_logger)


        train_ds = TrainDataset(experiment_params, train_folder, prepared_patches['train'])
        val_ds = ValDataset(experiment_params, val_folder, prepared_patches['val'])

        train_dl = DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True
        )

        val_dl = DataLoader(
            dataset=val_ds,
            batch_size=batch_size,
            num_workers=8,
            persistent_workers=True
        )
        #inputs = next(iter(val_dl))[0]
        #print(flop_count_table(FlopCountAnalysis(model, inputs)))
        #flops = flop_count(model, inputs)

        early_stop_callback = EarlyStopping(monitor="val_loss", verbose = True, mode="min", **training_params['early_stop'])

        best_model_callback = ModelCheckpoint(
            str(models_path), 
            monitor = 'val_loss', 
            verbose = True, 
            filename = f'model_{model_idx}'
            )
        
        history_model_callback = ModelCheckpoint(
            str(logs_path / f'model_{model_idx}'), 
            verbose = False, 
            save_top_k = -1,
            filename = '{epoch}'
            )
        
        #train_samples = TrainSamples(visual_logs_path, 20, model_idx)
        
        profiler = SimpleProfiler(
            dirpath = logs_path,
            filename = f'model_{model_idx}',
            )
        
        trainer = pl.Trainer(
            accelerator  = 'gpu',
            devices = args.devices,
            limit_train_batches = training_params['max_train_batches'], 
            limit_val_batches = training_params['max_val_batches'], 
            max_epochs = training_params['max_epochs'], 
            callbacks = [early_stop_callback, best_model_callback], 
            logger = loggers,
            log_every_n_steps = 1,
            profiler = profiler,
            #num_sanity_val_steps = 0
            enable_progress_bar = True
            )
        
        t0 = time.time()
        trainer.fit(model = model, train_dataloaders=train_dl, val_dataloaders=val_dl) #, datamodule=data_module)
        train_time = time.time() - t0
        
        last_val_loss = best_model_callback.best_model_score.item()

        if last_val_loss <= min_val_loss:
            model_file = Path(best_model_callback.best_model_path)
            run_results = {
                'model_file': model_file.name,
                'total_train_time': train_time,
                'train_per_epoch': train_time / trainer.current_epoch,
                'n_paramters': count_parameters(model), 
                'converged': True
            }
            save_yaml(run_results, logs_path / f'model_{model_idx}' / 'train_results.yaml')
            break
        else:
            print('Model didn\'t converged. Repeating the training...')
            model_file = Path(best_model_callback.best_model_path)
            model_file.unlink()

if __name__=="__main__":
    freeze_support()
    
    if args.model == -1:
        if args.start_model == -1:
            for model_idx in range(training_params['n_models']):
                p = Process(target=run, args=(model_idx,))
                p.start()
                p.join()
        else:
            for model_idx in range(args.start_model, training_params['n_models']):
                p = Process(target=run, args=(model_idx,))
                p.start()
                p.join()
    
    else:
        run(args.model)


    