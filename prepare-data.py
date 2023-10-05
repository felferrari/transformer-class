import argparse
from  pathlib import Path
import yaml
import logging
from utils.ops import load_opt_image, load_SAR_image, load_sb_image, save_yaml, load_yaml, remove_outliers
import numpy as np
from skimage.util import view_as_windows
from tqdm import tqdm
import pandas as pd
from shutil import rmtree
import h5py

parser = argparse.ArgumentParser(
    description='Prepare the original files, generating .h5 files to be used in the training/testing steps'
)

parser.add_argument( # The path to the config file (.yaml)
    '-c', '--cfg',
    type = Path,
    default = 'cfg.yaml',
    help = 'Path to the config file (.yaml).'
)

parser.add_argument( # Create training/validation data.
    '-r', '--train-data',
    default=False,
    action=argparse.BooleanOptionalAction,
    help = 'Create training/validation data.'
)

parser.add_argument( # Create test data.
    '-t', '--test-data',
    default=False,
    action=argparse.BooleanOptionalAction,
    help = 'Create test data.'
)

parser.add_argument( # (Re)Generate statistics file.
    '-a', '--statistics',
    default=False,
    action=argparse.BooleanOptionalAction,
    help = '(Re)Generate statistics file.'
)

parser.add_argument( # specific site location number
    '-s', '--site',
    type = int,
    default=1,
    help = 'Site location number'
)

parser.add_argument( # Clear training prepared folder before prepare new data.
    '-x', '--clear-train-folder',
    action=argparse.BooleanOptionalAction,
    default=False,
    help = 'Clear training prepared folder before prepare new data.'
)

parser.add_argument( # Clear test prepared folder before prepare new data.
    '-z', '--clear-test-folder',
    action=argparse.BooleanOptionalAction,
    default=False,
    help = 'Clear test prepared folder before prepare new data.'
)

args = parser.parse_args()

cfg = load_yaml(args.cfg)
site_cfg = load_yaml(f'site_{args.site}.yaml')

general_params = cfg['general_params']
preparation_params = cfg['preparation_params']
paths_params = cfg['paths']

tiles_params = site_cfg['tiles_params']

original_opt_imgs = site_cfg['original_data']['opt_imgs']
original_sar_imgs = site_cfg['original_data']['sar_imgs']

prefix_params = general_params['prefixs']

opt_bands = general_params['opt_bands']
#list_opt_bands = general_params['list_opt_bands']
sar_bands = general_params['sar_bands']

patch_size = general_params['patch_size']
patch_overlap = preparation_params['patch_overlap']
min_def_proportion = preparation_params['min_def_proportion']

opt_prefix = prefix_params['opt']
sar_prefix = prefix_params['sar']
label_prefix = prefix_params['label']
previous_prefix = prefix_params['previous']
cloud_prefix = prefix_params['cloud']

prepared_folder = Path(paths_params['prepared_data'])
train_folder = prepared_folder / preparation_params['train_folder']
validation_folder = prepared_folder / preparation_params['validation_folder']
test_folder = prepared_folder / preparation_params['test_folder']

prepared_folder.mkdir(exist_ok=True)
if (args.clear_train_folder or args.train_data) and train_folder.exists():
    rmtree(train_folder)
    rmtree(validation_folder)
    train_folder.mkdir()
    validation_folder.mkdir()
if (args.clear_test_folder  or args.test_data) and test_folder.exists():
    rmtree(test_folder)
    test_folder.mkdir()

train_folder.mkdir(exist_ok=True)
validation_folder.mkdir(exist_ok=True)
test_folder.mkdir(exist_ok=True)

prepared_patches_file = prepared_folder / preparation_params['prepared_data']
statistics_file = prepared_folder / preparation_params['statistics_data']

#Generating patches idx
tiles = load_sb_image(Path(paths_params['tiles_path']))
shape = tiles.shape
tiles = tiles.flatten()

#def_prev = def_prev.flatten()
idx = np.arange(shape[0] * shape[1]).reshape(shape)
window_shape = (patch_size, patch_size)
slide_step = int((1-patch_overlap)*patch_size)
idx_patches = view_as_windows(idx, window_shape, slide_step).reshape((-1, patch_size, patch_size))

np.random.seed(123)

if args.statistics:

    opt_path = Path(paths_params['opt_data'])
    sar_path = Path(paths_params['sar_data'])

    opt_means, opt_stds, opt_maxs, opt_mins = [], [], [], []
    sar_means, sar_stds, sar_maxs, sar_mins = [], [], [], []
    pbar = tqdm(list(set(original_opt_imgs['test'] + original_opt_imgs['train'])), desc = 'Generating OPT statistics')
    for opt_img_i, opt_img_file in enumerate(pbar):
        data_file = opt_path / opt_img_file
        data = load_opt_image(data_file)
        data = remove_outliers(data)
        opt_means.append(data.mean(axis=(0,1)))
        opt_stds.append(data.std(axis=(0,1)))
        opt_maxs.append(data.max(axis=(0,1)))
        opt_mins.append(data.min(axis=(0,1)))
    pbar = tqdm(list(set(original_sar_imgs['test'] + original_sar_imgs['train'])), desc = 'Generating SAR statistics')
    for sar_img_i, sar_img_file in enumerate(pbar):
        data_file = sar_path / sar_img_file
        data = load_SAR_image(data_file)
        data = remove_outliers(data)
        sar_means.append(data.mean(axis=(0,1)))
        sar_stds.append(data.std(axis=(0,1)))
        sar_maxs.append(data.max(axis=(0,1)))
        sar_mins.append(data.min(axis=(0,1)))

    opt_means = np.array(opt_means).mean(axis=0)
    opt_stds = np.array(opt_stds).mean(axis=0)
    opt_maxs = np.array(opt_maxs).max(axis=0)
    opt_mins = np.array(opt_mins).min(axis=0)

    sar_means = np.array(sar_means).mean(axis=0)
    sar_stds = np.array(sar_stds).mean(axis=0)
    sar_maxs = np.array(sar_maxs).max(axis=0)
    sar_mins = np.array(sar_mins).min(axis=0)

    statistics = {
        'opt_means': opt_means.tolist(),
        'opt_stds': opt_stds.tolist(),
        'opt_maxs': opt_maxs.tolist(),
        'opt_mins': opt_mins.tolist(),
        'sar_means': sar_means.tolist(),
        'sar_stds': sar_stds.tolist(),
        'sar_maxs': sar_maxs.tolist(),
        'sar_mins': sar_mins.tolist(),
    }

    save_yaml(statistics, statistics_file)

data = None
#training patches
if args.train_data:
    statistics = load_yaml(statistics_file)

    # opt_means = statistics['opt_means']
    # opt_stds = statistics['opt_stds']
    # sar_means = statistics['sar_means']
    # sar_stds = statistics['sar_stds']

    outfile = prepared_folder / 'train-data-prep.txt'
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename=outfile,
            filemode='w'
            )
    log = logging.getLogger('preparing')

    train_label = load_sb_image(Path(paths_params['label_train'])).astype(np.uint8).flatten()
    previous_map = load_sb_image(Path(paths_params['previous_train'])).astype(np.float16)

    keep = ((train_label[idx_patches] == 1).sum(axis=(1,2)) / patch_size**2) >= min_def_proportion
    keep_args = np.argwhere(keep == True).flatten() #args with at least min_prop deforestation
    no_keep_args = np.argwhere(keep == False).flatten() #args with less than min_prop of deforestation

    no_keep_args = np.random.choice(no_keep_args, (keep==True).sum())

    keep_final = np.concatenate((keep_args, no_keep_args))

    all_idx_patches = idx_patches[keep_final]

    keep_train = np.squeeze(np.argwhere(np.all(tiles[all_idx_patches] == 1, axis=(1,2))))
    keep_val = np.squeeze(np.argwhere(np.all(tiles[all_idx_patches] == 0, axis=(1,2))))

    log.info(f'Train patches: {keep_train.sum()}')
    log.info(f'Validation patches: {keep_val.sum()}')
    
    train_idx_patches = all_idx_patches[keep_train]
    val_idx_patches = all_idx_patches[keep_val]

    prepared_patches = {
        'train': len(train_idx_patches),
        'val': len(val_idx_patches)
    }
    
    save_yaml(prepared_patches, prepared_patches_file)

    np.random.shuffle(val_idx_patches)
    np.random.shuffle(train_idx_patches)

    opt_path = Path(paths_params['opt_data'])
    sar_path = Path(paths_params['sar_data'])

    opt_imgs = []
    sar_imgs = []
    cloud_imgs = []
    pbar = tqdm(original_opt_imgs['train'], desc = 'Reading OPT Training files')
    for opt_img_i, opt_img_file in enumerate(pbar):
        pbar.set_description(f'Reading OPT training file {opt_img_file}')
        data_file = opt_path / opt_img_file
        data = load_opt_image(data_file)
        data = remove_outliers(data)
        # data = (data - opt_means) / opt_stds
        opt_imgs.append(data.astype(np.float16).reshape((-1, opt_bands)))

    pbar = tqdm(original_opt_imgs['train'], desc = 'Reading Cloud Training files')
    for opt_img_i, opt_img_file in enumerate(pbar):
        pbar.set_description(f'Reading Cloud training file {cloud_prefix}_{opt_img_file}')
        data_file = opt_path / f'{cloud_prefix}_{opt_img_file}'
        data = load_sb_image(data_file)/100
        cloud_imgs.append(data.astype(np.float16).reshape((-1, 1)))

    pbar = tqdm(original_sar_imgs['train'], desc = 'Reading SAR Training files')
    for sar_img_i, sar_img_file in enumerate(pbar):
        pbar.set_description(f'Reading SAR training file {sar_img_file}')
        data_file = sar_path / sar_img_file
        data = load_SAR_image(data_file)
        data = remove_outliers(data)
        # data = (data - sar_means) / sar_stds 
        sar_imgs.append(data.astype(np.float16).reshape((-1, sar_bands)))

    previous_map = previous_map.astype(np.float16).reshape(-1, 1)
    del data
    
    for patch_i, idx_patch in enumerate(tqdm(train_idx_patches, desc='Training Patches')):
        opt_patchs = []
        cloud_patchs = []
        sar_patchs = []
        for opt_img_i, opt_img in enumerate(opt_imgs):
            data_i = opt_img[idx_patch]
            opt_patchs.append(data_i)
        for cloud_img_i, cloud_img in enumerate(cloud_imgs):
            data_i = cloud_img[idx_patch]
            cloud_patchs.append(data_i)
        for sar_img_i, sar_img in enumerate(sar_imgs):
            data_i = sar_img[idx_patch]
            sar_patchs.append(data_i)

        opt_patchs = np.array(opt_patchs)
        cloud_patchs = np.array(cloud_patchs)
        sar_patchs = np.array(sar_patchs)

        label_patch = train_label[idx_patch]
        
        #np.save(label_patch_file, label_patch)

        previous_patch = previous_map[idx_patch]
        patch_file = train_folder / f'{patch_i:d}.h5'

        with h5py.File(patch_file, "w") as f:
            f.create_dataset('opt', data=opt_patchs, compression='lzf', chunks=(1, patch_size, patch_size, 1))
            f.create_dataset('cloud', data=cloud_patchs, compression='lzf', chunks=(1, patch_size, patch_size, 1))
            f.create_dataset('sar', data=sar_patchs, compression='lzf', chunks=(1, patch_size, patch_size, 1))
            f.create_dataset('previous', data=previous_patch, compression='lzf', chunks=(patch_size, patch_size, 1))
            f.create_dataset('label', data=label_patch, compression='lzf', chunks=(patch_size, patch_size))
        
        #train_dataset['patch_idx'].append(patch_i)

    for patch_i, idx_patch in enumerate(tqdm(val_idx_patches, desc='Validation Patches')):
        opt_patchs = []
        cloud_patchs = []
        sar_patchs = []
        for opt_img_i, opt_img in enumerate(opt_imgs):
            data_i = opt_img[idx_patch]
            opt_patchs.append(data_i)
        for cloud_img_i, cloud_img in enumerate(cloud_imgs):
            data_i = cloud_img[idx_patch]
            cloud_patchs.append(data_i)
        for sar_img_i, sar_img in enumerate(sar_imgs):
            data_i = sar_img[idx_patch]
            sar_patchs.append(data_i)

        opt_patchs = np.array(opt_patchs)
        cloud_patchs = np.array(cloud_patchs)
        sar_patchs = np.array(sar_patchs)

        label_patch = train_label[idx_patch]
        
        #np.save(label_patch_file, label_patch)

        previous_patch = previous_map[idx_patch]
        patch_file = validation_folder / f'{patch_i:d}.h5'
        with h5py.File(patch_file, "w") as f:
            f.create_dataset('opt', data=opt_patchs, compression='lzf', chunks=(1, patch_size, patch_size, 1))
            f.create_dataset('cloud', data=cloud_patchs, compression='lzf', chunks=(1, patch_size, patch_size, 1))
            f.create_dataset('sar', data=sar_patchs, compression='lzf', chunks=(1, patch_size, patch_size, 1))
            f.create_dataset('previous', data=previous_patch, compression='lzf', chunks=(patch_size, patch_size, 1))
            f.create_dataset('label', data=label_patch, compression='lzf', chunks=(patch_size, patch_size))
        
    del train_label
    data = None

#prediction/test image preparation
if args.test_data:

    # statistics = load_yaml(statistics_file)

    # opt_means = statistics['opt_means']
    # opt_stds = statistics['opt_stds']
    # sar_means = statistics['sar_means']
    # sar_stds = statistics['sar_stds']

    opt_path = Path(paths_params['opt_data'])
    sar_path = Path(paths_params['sar_data'])

    for opt_img_i, opt_img_file in enumerate(tqdm(original_opt_imgs['test'], desc = 'Converting OPT Testing files')):
        data_file = opt_path / opt_img_file
        data = load_opt_image(data_file)
        data = remove_outliers(data)
        #data = (data - opt_means) / opt_stds
        #data = data / 10000
        data_patch_file = test_folder / f'{opt_prefix}_{opt_img_i}.h5'
        with h5py.File(data_patch_file, "w") as f:
            f.create_dataset('opt', data=data.astype(np.float16), compression='lzf')
    for sar_img_i, sar_img_file in enumerate(tqdm(original_sar_imgs['test'], desc = 'Converting SAR Testing files')):
        data_file = sar_path / sar_img_file
        data = load_SAR_image(data_file)
        data = remove_outliers(data)
        #data = (data - sar_means) / sar_stds
        data_patch_file = test_folder / f'{sar_prefix}_{sar_img_i}.h5'
        with h5py.File(data_patch_file, "w") as f:
            f.create_dataset('sar', data=data.astype(np.float16), compression='lzf')

    previous_map = load_sb_image(Path(paths_params['previous_test'])).astype(np.float16)
    data_patch_file = test_folder / f'{previous_prefix}.h5'
    with h5py.File(data_patch_file, "w") as f:
        f.create_dataset('previous', data=previous_map.astype(np.float16), compression='lzf')

    test_label = load_sb_image(Path(paths_params['label_test'])).astype(np.uint8)
    data_patch_file = test_folder / f'{label_prefix}.h5'
    with h5py.File(data_patch_file, "w") as f:
        f.create_dataset('label', data=test_label, compression='lzf')