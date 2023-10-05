import argparse
from utils.ops import load_opt_image, save_geotiff, load_yaml
import numpy as np
from  pathlib import Path
import yaml
from s2cloudless import S2PixelCloudDetector
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Generate .tif with training (0) and validation (1) areas.'
)

parser.add_argument( # The path to the config file (.yaml)
    '-c', '--cfg',
    type = Path,
    default = 'cfg.yaml',
    help = 'Path to the config file (.yaml).'
)

parser.add_argument( # specific site location number
    '-s', '--site',
    default = 1,
    type = int,
    help = 'Site location number'
)

args = parser.parse_args()

cfg = load_yaml(args.cfg)
site_cfg = load_yaml(f'site_{args.site}.yaml')

paths_params = cfg['paths']
general_params = cfg['general_params']

cloud_prefix = general_params['prefixs']['cloud']

opt_folder = Path(paths_params['opt_data'])

original_data = site_cfg['original_data']

base_image = opt_folder/ original_data['opt_imgs']['train'][0]

cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=True)

opt_files = list(set(original_data['opt_imgs']['train']).union(set(original_data['opt_imgs']['test'])))
pbar = tqdm(opt_files, desc = 'Generating Clouds for OPT files')
for opt_img_file in pbar:
    pbar.set_description(f'Generating cloud map for {opt_img_file}')
    opt_img = load_opt_image(opt_folder / opt_img_file) 
    cloud_map = np.squeeze(cloud_detector.get_cloud_probability_maps(np.expand_dims(opt_img, axis=0)))

    cloud_tif_file = opt_folder / f'{cloud_prefix}_{opt_img_file}'
    save_geotiff(base_image, cloud_tif_file, cloud_map, dtype = 'float')

# for opt_img_file in tqdm(original_data['opt_imgs']['train'], desc = 'Generating Clouds for Trainig files'):
#     opt_img = load_opt_image(opt_folder / opt_img_file)
#     cloud_map = np.squeeze(cloud_detector.get_cloud_probability_maps(np.expand_dims(opt_img, axis=0)))

#     cloud_tif_file = opt_folder / f'{cloud_prefix}_{opt_img_file}'
#     save_geotiff(base_image, cloud_tif_file, cloud_map, dtype = 'float')

# for opt_img_file in tqdm(original_data['opt_imgs']['test'], desc = 'Generating Clouds for Testing files'):
#     opt_img = load_opt_image(opt_folder / opt_img_file) / 10000
#     cloud_map = np.squeeze(cloud_detector.get_cloud_probability_maps(np.expand_dims(opt_img, axis=0)))

#     cloud_tif_file = opt_folder / f'{cloud_prefix}_{opt_img_file}'
#     save_geotiff(base_image, cloud_tif_file, cloud_map, dtype = 'float')
