import argparse
from  pathlib import Path
import numpy as np
from utils.ops import save_geotiff, load_sb_image, load_yaml, load_ml_image
from skimage.morphology import area_opening
import pandas as pd
from multiprocessing import Pool
import tqdm
import sqlite3
import datetime

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
    default = 2,
    help = 'The number of the experiment'
)

parser.add_argument( # Generate the Max Cloud Map Geotiff
    '-m', '--cloud-max-map',
    #type = bool,
    action='store_true',
    help = 'Generate the Max Cloud Map Geotiff'
)

parser.add_argument( # specific site location number
    '-s', '--site',
    type = int,
    default=2,
    help = 'Site location number'
)

parser.add_argument( # specific site location number
    '-p', '--processes',
    type = int,
    default=9,
    help = 'Number of paralel threads'
)

args = parser.parse_args()

cfg = load_yaml(args.cfg)
site_cfg = load_yaml(f'site_{args.site}.yaml')

preparation_params = cfg['preparation_params']
experiment_params = cfg['experiments'][f'exp_{args.experiment}']
paths_params = cfg['paths']
general_params = cfg['general_params']

experiments_paths = general_params['experiments_folders']
#prepared_folder = Path(paths_params['prepared_data'])

exp_path = Path(paths_params['experiments']) / f'exp_{args.experiment}'

models_path = exp_path / experiments_paths['models']
logs_path = exp_path / experiments_paths['logs']
visual_path = exp_path / experiments_paths['visual']
predicted_path = exp_path / experiments_paths['predicted']
results_path = exp_path / experiments_paths['results']

results_sqlite_path = Path(paths_params['results_sqlite'])

patch_size = general_params['patch_size']
n_classes = general_params['n_classes']
prediction_prefix = general_params['prefixs']['prediction']
cloud_prefix = general_params['prefixs']['cloud']
n_models = general_params['n_models']

n_opt_imgs_groups = len(experiment_params['test_opt_imgs'])
n_sar_imgs_groups = len(experiment_params['test_sar_imgs'])

original_opt_imgs = site_cfg['original_data']['opt_imgs']
original_sar_imgs = site_cfg['original_data']['sar_imgs']

min_area = general_params['min_area']

imgs_groups_idxs = []
for opt_imgs_group_idx in range(n_opt_imgs_groups):
     for sar_imgs_group_idx in range(n_sar_imgs_groups):
          imgs_groups_idxs.append([opt_imgs_group_idx, sar_imgs_group_idx, None])
          for model_i in range(n_models):
              imgs_groups_idxs.append([opt_imgs_group_idx, sar_imgs_group_idx, model_i])
              


opt_folder = Path(paths_params['opt_data'])
base_data = opt_folder / original_opt_imgs['test'][0]

opt_imgs_groups = experiment_params['test_opt_imgs']
original_opt_files = original_opt_imgs['test']
#sar_imgs_groups = experiment_params['test_sar_imgs']

#mean prediction
def eval_prediction(data):

    opt_imgs_groups_idx, sar_imgs_groups_idx, model_idx = data

    label = load_sb_image(Path(paths_params['label_test'])).astype(np.uint8)

    if model_idx is None:
        pred_prob = np.zeros_like(label, dtype=np.float16)
        #pred_prob = None
        for model_i in range(n_models):
            pred_prob_file = predicted_path / f'{prediction_prefix}_{args.experiment}_{opt_imgs_groups_idx}_{sar_imgs_groups_idx}_{model_i}.npz'
            pred_prob += np.load(pred_prob_file)['pred']
            #pred_prob_file = visual_path /f'{prediction_prefix}_{args.experiment}_{opt_imgs_groups_idx}_{sar_imgs_groups_idx}_{model_i}.tif'
            #if pred_prob is None:
            #    pred_prob = load_sb_image(pred_prob_file).astype(np.float16)
            #else:
            #    pred_prob += load_sb_image(pred_prob_file).astype(np.float16)
        pred_prob = pred_prob / n_models
        mean_prob_file = visual_path / f'{prediction_prefix}_mean_prob_{args.experiment}_{opt_imgs_groups_idx}_{sar_imgs_groups_idx}.tif'
        #save_geotiff(base_data, mean_prob_file, pred_prob, dtype = 'float')
        ep = 1e-7
        cliped_pred_prob = np.clip(pred_prob.astype(np.float32), ep, 1-ep)
        entropy = (-1/2) * (cliped_pred_prob * np.log(cliped_pred_prob) + (1-cliped_pred_prob) * np.log(1-cliped_pred_prob))
        entropy_tif_file = visual_path / f'entropy_{args.experiment}_{opt_imgs_groups_idx}_{sar_imgs_groups_idx}.tif'
        save_geotiff(base_data, entropy_tif_file, entropy, dtype = 'float')

    else:
        pred_prob_file = predicted_path / f'{prediction_prefix}_{args.experiment}_{opt_imgs_groups_idx}_{sar_imgs_groups_idx}_{model_idx}.npz'
        pred_prob = np.load(pred_prob_file)['pred']
        #pred_prob_file = visual_path /f'{prediction_prefix}_{args.experiment}_{opt_imgs_groups_idx}_{sar_imgs_groups_idx}_{model_idx}.tif'
        #pred_prob = load_sb_image(pred_prob_file).astype(np.float16)


    #pred_b = (np.argmax(pred_prob, -1)==1).astype(np.uint8)
    #pred_b[pred_b == 2] = 0
    #pred_b[label == 2] = 2

    pred_b = (pred_prob > 0.5).astype(np.uint8)
    pred_b[label == 2] = 0
    pred_b[label == 3] = 0
    pred_red = pred_b - area_opening(pred_b, min_area)

    pred_b[label == 2] = 3
    pred_b[label == 3] = 3
    pred_b[pred_red == 1] = 3
    label[pred_red == 1] = 3


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
        #save_geotiff(base_data, error_map_file, error_map, dtype = 'byte')
    
    
    #load cloud maps
    opt_imgs_idxs = opt_imgs_groups[opt_imgs_groups_idx]
    cloud_max = np.concatenate(
        [np.expand_dims(load_sb_image(opt_folder / f'{cloud_prefix}_{original_opt_files[opt_img_idx]}'), axis=-1) for opt_img_idx in opt_imgs_idxs],
        axis=-1
    ).max(axis=-1)

    cloud_pixels = np.zeros_like(cloud_max, dtype=np.uint8)
    cloud_pixels[cloud_max>50] = 1

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
    

    with Pool(args.processes) as pool:
        #metrics = pool.imap(eval_prediction, imgs_groups_idxs)
        metrics = list(tqdm.tqdm(pool.imap(eval_prediction, imgs_groups_idxs), total=len(imgs_groups_idxs), desc = f'Evaluating Predictions from experiment {args.experiment}'))

    headers =  [
        'opt_imgs_groups_idx', 
        'sar_imgs_groups_idx', 
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

    sum_results_df = results_df.groupby(['model_idx'], dropna=False)[[
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
        sum_results_df[f'{cloud_cond}_precision'] = sum_results_df[f'{cloud_cond}_tps'] / (sum_results_df[f'{cloud_cond}_tps'] + sum_results_df[f'{cloud_cond}_fps'])
        sum_results_df[f'{cloud_cond}_recall'] = sum_results_df[f'{cloud_cond}_tps'] / (sum_results_df[f'{cloud_cond}_tps'] + sum_results_df[f'{cloud_cond}_fns'])
        sum_results_df[f'{cloud_cond}_f1'] = 2 * (sum_results_df[f'{cloud_cond}_precision'] * sum_results_df[f'{cloud_cond}_recall']) / (sum_results_df[f'{cloud_cond}_precision'] + sum_results_df[f'{cloud_cond}_recall'])

    sum_results_df = sum_results_df.reset_index()
    final_results_df = sum_results_df[sum_results_df['model_idx'].isnull()].reset_index()

    results_file = results_path / f'final_results_{args.experiment}.xlsx'
    with pd.ExcelWriter(results_file) as writer:     
        results_df.to_excel(writer, sheet_name='general results')
        sum_results_df.to_excel(writer, sheet_name='models results')
        final_results_df.to_excel(writer, sheet_name='final results')

    final_results_file = results_path / f'results_{args.experiment}.data'
    final_results_df.to_pickle(final_results_file)

    models_results_file = results_path / f'models_results_{args.experiment}.data'
    sum_results_df.to_pickle(models_results_file)

    timest = datetime.datetime.now()
    
    final_results = final_results_df.drop(['index', 'model_idx'], axis=1)
    final_results.insert(0, 'site', args.site)
    final_results.insert(1, 'experiment_n', args.experiment)
    final_results.insert(2, 'time', timest)
    final_results['exp_params'] = str(experiment_params)

    sum_results_df.insert(0, 'site', args.site)
    sum_results_df.insert(1, 'experiment_n', args.experiment)
    sum_results_df.insert(3, 'time', timest)
    sum_results_df = sum_results_df.dropna()
    sum_results_df = sum_results_df.astype({'model_idx': int})

    con = sqlite3.connect(results_sqlite_path)
    final_results.to_sql('experiments', con, if_exists = 'append', index = False)
    sum_results_df.to_sql('all_models', con, if_exists = 'append', index = False)



        