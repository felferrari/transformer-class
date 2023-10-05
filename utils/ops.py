import json
import numpy as np
import os
import sys
from osgeo import gdal_array
from osgeo import gdal, gdalconst
from pathlib import Path
from typing import Union
import yaml 
from multiprocessing import Pool

def remove_outliers(img, signficance = 0.01):
    outliers = np.quantile(img, [signficance, 1-signficance], axis = (0,1))
    for channel in range(img.shape[-1]):
        img[:,:,channel] = np.clip(img[:,:,channel], outliers[0, channel],  outliers[1, channel])
    return img


def load_json(fp):
    with open(fp) as f:
        return json.load(f)
    
def save_json(dict_:dict, file_path: Union[str, Path]) -> None:
    """Save a dictionary into a file

    Args:
        dict_ (dict): Dictionary to be saved
        file_path (Union[str, Path]): file path
    """

    with open(file_path, 'w') as f:
        json.dump(dict_, f, indent=4)

def save_yaml(dict_:dict, file_path: Union[str, Path]) -> None:
    """Save a dictionary into a file

    Args:
        dict_ (dict): Dictionary to be saved
        file_path (Union[str, Path]): file path
    """

    with open(file_path, 'w') as f:
        yaml.dump(dict_, f, default_flow_style=False)

def load_yaml(file_path: Union[str, Path]) -> None:
    """Save a dictionary into a file

    Args:
        dict_ (dict): Dictionary to be saved
        file_path (Union[str, Path]): file path
    """

    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
    
def load_opt_image(img_file):
    """load optical data.

    Args:
        img_file (str): path to the geotiff optical file.

    Returns:
        array:numpy array of the image.
    """
    img = gdal_array.LoadFile(str(img_file)).astype(np.float16)
    img[np.isnan(img)] = 0
    if len(img.shape) == 2 :
        img = np.expand_dims(img, axis=0)
    return np.moveaxis(img, 0, -1) / 10000

def load_sb_image(img_file):
    """load a single band geotiff image.

    Args:
        img_file (str): path to the geotiff file.

    Returns:
        array:numpy array of the image. Channels Last.
    """
    img = gdal_array.LoadFile(str(img_file))
    return img

def load_ml_image(img_file):
    """load a single band geotiff image.

    Args:
        img_file (str): path to the geotiff file.

    Returns:
        array:numpy array of the image. Channels Last.
    """
    img = gdal_array.LoadFile(str(img_file)).astype(np.float16)
    img[np.isnan(img)] = 0
    if len(img.shape) == 2 :
        img = np.expand_dims(img, axis=0)
    return np.moveaxis(img, 0, -1)


def load_SAR_image(img_file):
    """load SAR image, converting from Db to DN.

    Args:
        img_file (str): path to the SAR geotiff file.

    Returns:
        array:numpy array of the image. Channels Last.
    """
    img = gdal_array.LoadFile(str(img_file))
    #img = 10**(img/10) 
    img[np.isnan(img)] = 0
    return np.moveaxis(img, 0, -1)

def save_feature_map(path_to_file, tensor, index = None):
    if index is not None:
        fm = tensor[index]

def create_exps_paths(exp_n):
    exps_path = 'exps'

    exp_path = os.path.join(exps_path, f'exp_{exp_n}')
    models_path = os.path.join(exp_path, 'models')

    results_path = os.path.join(exp_path, 'results')
    predictions_path = os.path.join(results_path, 'predictions')
    visual_path = os.path.join(results_path, 'visual')

    logs_path = os.path.join(exp_path, 'logs')

    
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

    return exps_path, exp_path, models_path, results_path, predictions_path, visual_path, logs_path

def load_exp(exp_n = None):
    if exp_n is None:
        if len(sys.argv)==1:
            return None
        else:
            return load_json(os.path.join('conf', 'exps', f'exp_{sys.argv[1]}.json'))
    else:
        return load_json(os.path.join('conf', 'exps', f'exp_{exp_n}.json'))
    

def save_geotiff(base_image_path, dest_path, data, dtype):
    """Save data array as geotiff.
    Args:
        base_image_path (str): Path to base geotiff image to recovery the projection parameters
        dest_path (str): Path to geotiff image
        data (array): Array to be used to generate the geotiff
        dtype (str): Data type of the destiny geotiff: If is 'byte' the data is uint8, if is 'float' the data is float32
    """
    base_image_path = str(base_image_path)
    base_data = gdal.Open(base_image_path, gdalconst.GA_ReadOnly)

    geo_transform = base_data.GetGeoTransform()
    x_res = base_data.RasterXSize
    y_res = base_data.RasterYSize
    crs = base_data.GetSpatialRef()
    proj = base_data.GetProjection()
    dest_path = str(dest_path)

    if len(data.shape) == 2:
        if dtype == 'byte':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, 1, gdal.GDT_Byte)
            data = data.astype(np.uint8)
        elif dtype == 'float':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, 1, gdal.GDT_Float32)
            data = data.astype(np.float32)
        elif dtype == 'uint16':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, 1, gdal.GDT_UInt16)
            data = data.astype(np.uint16)
    elif len(data.shape) == 3:
        if dtype == 'byte':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, data.shape[-1], gdal.GDT_Byte)
            data = data.astype(np.uint8)
        elif dtype == 'float':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, data.shape[-1], gdal.GDT_Float32)
            data = data.astype(np.float32)
        elif dtype == 'float':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, data.shape[-1], gdal.GDT_UInt16)
            data = data.astype(np.uint16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetSpatialRef(crs)
    target_ds.SetProjection(proj)

    if len(data.shape) == 2:
        target_ds.GetRasterBand(1).WriteArray(data)
    elif len(data.shape) == 3:
        for band_i in range(1, data.shape[-1]+1):
            target_ds.GetRasterBand(band_i).WriteArray(data[:,:,band_i-1])
    target_ds = None

def count_parameters_old(model):
    """Count the number of model parameters
    Args:
        model (Module): Model
    Returns:
        int: Number of Model's parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):
    total_params = 0
    text = ''
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params+=params
        text += f'{name}: {params:,}\n'
    text+=f'Total: {total_params:,}\n'
    return total_params


