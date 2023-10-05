import ee
import time

def export_opt(img, roi, name):
    task_config = {
        'scale': 10, 
        #'crs': crs['crs'],
        'folder':'GEE_imgs_opt',
        #'crsTransform': crs['transform'],
        'fileFormat': 'GeoTIFF',
        'region': roi,
        'maxPixels' : 1e12
        
    }
    #print(f'Start download of {name} RGB OPT image')
    task = ee.batch.Export.image(img.select([
        #'B1', 
        'B2',
        'B3',
        'B4',
        'B5',
        'B6',
        'B7',
        'B8',
        'B8A',
        #'B9',
        'B11',
        'B12'
        ]), f'{name}', task_config)

    task.start()
    return task

def export_opt_l1(img, roi, name):
    task_config = {
        'scale': 10, 
        #'crs': crs['crs'],
        'folder':'GEE_imgs_opt_l1',
        #'crsTransform': crs['transform'],
        'fileFormat': 'GeoTIFF',
        'region': roi,
        'maxPixels' : 1e12
        
    }
    #print(f'Start download of {name} RGB OPT image')
    task = ee.batch.Export.image(img.select([
        'B1', 
        'B2',
        'B3',
        'B4',
        'B5',
        'B6',
        'B7',
        'B8',
        'B8A',
        'B9',
        'B10',
        'B11',
        'B12'
        ]), f'{name}', task_config)

    task.start()
    return task

def export_maskcloud(img, roi, name):
    task_config = {
        'scale': 10, 
        #'crs': crs['crs'],
        'folder':'GEE_imgs_opt',
        #'crsTransform': crs['transform'],
        'fileFormat': 'GeoTIFF',
        'region': roi,
        'maxPixels' : 1e12
        
    }
    #print(f'Start download of {name} RGB OPT image')
    task = ee.batch.Export.image(img.select([
        'MSK_CLDPRB'
        ]), f'{name}', task_config)

    task.start()
    return task


def export_sar(img, roi, name):
    task_config = {
        'scale': 10, 
        #'crs': crs['crs'],
        'folder':'GEE_imgs_sar',
        #'crsTransform': crs['transform'],
        'fileFormat': 'GeoTIFF',
        'region': roi,
        'maxPixels' : 1e12
        
    }
    task = ee.batch.Export.image(img.select([
        'VV', 
        'VH'
        ]), name, task_config)

    task.start()