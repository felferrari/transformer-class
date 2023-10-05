import argparse
from  pathlib import Path
import importlib
#from conf import default, general, paths
import os
import time
import sys
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from osgeo import ogr, gdal, gdalconst
from utils.ops import save_geotiff, load_sb_image
from multiprocessing import Process
from skimage.morphology import area_opening
import pandas as pd
import yaml
from multiprocessing import Pool
from itertools import combinations, product, repeat

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
    default = 1,
    help = 'The number of the experiment'
)

parser.add_argument( # Generate the Max Cloud Map Geotiff
    '-s', '--cloud-max-map',
    #type = bool,
    action='store_true',
    help = 'Generate the Max Cloud Map Geotiff'
)



args = parser.parse_args()

with open(args.cfg, 'r') as file:
    cfg = yaml.load(file, Loader=yaml.Loader)

prediction_params = cfg['prediction_params']
preparation_params = cfg['preparation_params']
experiment_params = cfg['experiments'][f'exp_{args.experiment}']
label_params = cfg['label_params']
previous_def_params = cfg['previous_def_params']
original_data_params = cfg['original_data']

experiments_paths = prediction_params['experiments_paths']
prepared_folder = Path(preparation_params['folder'])

exp_path = Path(experiments_paths['folder']) / f'exp_{args.experiment}'

models_path = exp_path / experiments_paths['models']
logs_path = exp_path / experiments_paths['logs']
visual_path = exp_path / experiments_paths['visual']
predicted_path = exp_path / experiments_paths['predicted']
results_path = exp_path / experiments_paths['results']

patch_size = prediction_params['patch_size']
n_classes = prediction_params['n_classes']
prediction_prefix = experiment_params['prefixs']['prediction']
cloud_prefix = experiment_params['prefixs']['cloud']
n_models = prediction_params['n_models']

n_opt_imgs_groups = len(experiment_params['test_opt_imgs'])
n_sar_imgs_groups = len(experiment_params['test_sar_imgs'])

opt_folder = Path(original_data_params['opt']['folder'])
base_data = opt_folder / original_data_params['opt']['imgs']['test'][0]

opt_imgs_groups = experiment_params['test_opt_imgs']
original_opt_files = original_data_params['opt']['imgs']['test']
#sar_imgs_groups = experiment_params['test_sar_imgs']

#mean prediction
def eval_prediction(opt_imgs_groups_idx, sar_imgs_groups_idx, models_l):
    label = load_sb_image(Path(label_params['test_path'])).astype(np.uint8)

    pred_prob = np.zeros_like(label, dtype=np.float16)
    for model_idx in models_l:
        pred_prob_file = predicted_path / f'{prediction_prefix}_prob_{opt_imgs_groups_idx}_{sar_imgs_groups_idx}_{model_i}.npy'
        pred_prob += np.load(pred_prob_file)
    pred_prob = pred_prob / len(models_l)

    pred_b = (pred_prob > 0.5).astype(np.uint8)
    pred_b[label == 2] = 0
    pred_red = pred_b - area_opening(pred_b, 625)

    pred_b[label == 2] = 2
    pred_b[pred_red == 1] = 2
    label[pred_red == 1] = 2

    error_map = np.zeros_like(label, dtype=np.uint8)
    error_map[np.logical_and(pred_b == 0, label == 0)] = 0 #tn
    error_map[np.logical_and(pred_b == 1, label == 1)] = 1 #tp
    error_map[np.logical_and(pred_b == 0, label == 1)] = 2 #fn
    error_map[np.logical_and(pred_b == 1, label == 0)] = 3 #fp

    if model_idx is None:
        error_map_file = visual_path / f'{prediction_prefix}_error_map_{args.experiment}_{opt_imgs_groups_idx}_{sar_imgs_groups_idx}.tif'
        save_geotiff(base_data, error_map_file, error_map, dtype = 'byte')
    else:
        error_map_file = visual_path / f'{prediction_prefix}_error_map_{args.experiment}_{opt_imgs_groups_idx}_{sar_imgs_groups_idx}_{model_idx}.tif'
        save_geotiff(base_data, error_map_file, error_map, dtype = 'byte')
    
    
    #load cloud maps
    opt_imgs_idxs = opt_imgs_groups[opt_imgs_groups_idx]
    cloud_max = np.concatenate(
        [np.expand_dims(load_sb_image(opt_folder / f'{cloud_prefix}_{original_opt_files[opt_img_idx]}'), axis=-1) for opt_img_idx in opt_imgs_idxs],
        axis=-1
    ).max(axis=-1)

    cloud_pixels = np.zeros_like(cloud_max, dtype=np.uint8)
    cloud_pixels[cloud_max>0.5] = 1

    no_cloud_tns = np.logical_and(error_map == 0, cloud_pixels == 0).sum()
    no_cloud_tps = np.logical_and(error_map == 1, cloud_pixels == 0).sum()
    no_cloud_fns = np.logical_and(error_map == 2, cloud_pixels == 0).sum()
    no_cloud_fps = np.logical_and(error_map == 3, cloud_pixels == 0).sum()

    cloud_tns = np.logical_and(error_map == 0, cloud_pixels == 1).sum()
    cloud_tps = np.logical_and(error_map == 1, cloud_pixels == 1).sum()
    cloud_fns = np.logical_and(error_map == 2, cloud_pixels == 1).sum()
    cloud_fps = np.logical_and(error_map == 3, cloud_pixels == 1).sum()

    global_tns = no_cloud_tns + cloud_tns
    global_tps = no_cloud_tps + cloud_tps
    global_fns = no_cloud_fns + cloud_fns
    global_fps = no_cloud_fps + cloud_fps

    no_cloud_precision = no_cloud_tps / (no_cloud_tps + no_cloud_fps)
    no_cloud_recall = no_cloud_tps / (no_cloud_tps + no_cloud_fns)
    no_cloud_f1 = 2 * no_cloud_precision * no_cloud_recall / (no_cloud_precision + no_cloud_recall)

    cloud_precision = cloud_tps / (cloud_tps + cloud_fps)
    cloud_recall = cloud_tps / (cloud_tps + cloud_fns)
    cloud_f1 = 2 * cloud_precision * cloud_recall / (cloud_precision + cloud_recall)

    global_precision = global_tps / (global_tps + global_fps)
    global_recall = global_tps / (global_tps + global_fns)
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall)
  
    return [
        opt_imgs_groups_idx, 
        sar_imgs_groups_idx, 
        len(model_idx),
        model_idx,
        no_cloud_f1,
        no_cloud_precision,
        no_cloud_recall,
        cloud_f1,
        cloud_precision,
        cloud_recall,
        global_f1,
        global_precision,
        global_recall,
        no_cloud_tns,
        no_cloud_tps,
        no_cloud_fns,
        no_cloud_fps,
        cloud_tns, 
        cloud_tps, 
        cloud_fns, 
        cloud_fps,
        global_tns, 
        global_tps, 
        global_fns, 
        global_fps
    ]


if __name__=="__main__":
    
    imgs_groups_idxs = []
    for opt_imgs_group_idx in range(n_opt_imgs_groups):
        for sar_imgs_group_idx in range(n_sar_imgs_groups):
            for model_i in range(n_models):
                imgs_groups_idxs.append([opt_imgs_group_idx, sar_imgs_group_idx, model_i])

    with Pool(8) as pool:
        metrics = pool.starmap(eval_prediction, imgs_groups_idxs)

    headers =  [
        'opt_imgs_groups_idx', 
        'sar_imgs_groups_idx', 
        'n_models',
        'model_idx',
        'no_cloud_f1',
        'no_cloud_precision',
        'no_cloud_recall',
        'cloud_f1',
        'cloud_precision',
        'cloud_recall',
        'global_f1',
        'global_precision',
        'global_recall',
        'no_cloud_tns',
        'no_cloud_tps',
        'no_cloud_fns',
        'no_cloud_fps',
        'cloud_tns', 
        'cloud_tps', 
        'cloud_fns', 
        'cloud_fps',
        'global_tns', 
        'global_tps', 
        'global_fns', 
        'global_fps'
        ]
    
    results_df = pd.DataFrame(metrics, columns=headers)

    mean_results_df = results_df.groupby(['model_idx'], dropna=False)[[
            'no_cloud_tns',
            'no_cloud_tps',
            'no_cloud_fns',
            'no_cloud_fps',
            'cloud_tns', 
            'cloud_tps', 
            'cloud_fns', 
            'cloud_fps',
            'global_tns',
            'global_tps',
            'global_fns',
            'global_fps',
        ]].apply(sum)
    
    for cloud_cond in ['no_cloud', 'cloud', 'global']:
        mean_results_df[f'{cloud_cond}_precision'] = mean_results_df[f'{cloud_cond}_tps'] / (mean_results_df[f'{cloud_cond}_tps'] + mean_results_df[f'{cloud_cond}_fps'])
        mean_results_df[f'{cloud_cond}_recall'] = mean_results_df[f'{cloud_cond}_tps'] / (mean_results_df[f'{cloud_cond}_tps'] + mean_results_df[f'{cloud_cond}_fns'])
        mean_results_df[f'{cloud_cond}_f1'] = 2 * (mean_results_df[f'{cloud_cond}_precision'] * mean_results_df[f'{cloud_cond}_recall']) / (mean_results_df[f'{cloud_cond}_precision'] + mean_results_df[f'{cloud_cond}_recall'])

    comb_results = mean_results_df.groupby(['n_models'])[[
            'no_cloud_f1',
            'no_cloud_precision',
            'no_cloud_recall',
            'cloud_f1', 
            'cloud_precision', 
            'cloud_recall', 
            'global_f1',
            'global_precision',
            'global_recall',
        ]].agg([np.mean, np.std])
    
    
    results_file = results_path / f'results_{args.experiment}.xlsx'
    with pd.ExcelWriter(results_file) as writer:     
        comb_results.to_excel(writer, sheet_name='combination results')
        results_df.to_excel(writer, sheet_name='general results')
        mean_results_df.to_excel(writer, sheet_name='models results')
    



        