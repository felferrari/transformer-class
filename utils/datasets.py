from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms.functional import vflip, hflip
import torch
import random
from skimage.util import view_as_windows
import h5py
from einops import rearrange
from .ops import load_opt_image, load_SAR_image, load_sb_image
from einops import rearrange

class GenericTrainDataset(Dataset):
    def __init__(self, params, data_folder, n_patches):
        #self.device = device
        self.params = params
        self.n_patches = n_patches
        self.data_folder = data_folder

        self.n_opt_img_groups = len(params['train_opt_imgs'])
        self.opt_imgs = params['train_opt_imgs']
        
        self.n_sar_img_groups = len(params['train_sar_imgs'])
        self.sar_imgs = params['train_sar_imgs']

    def __len__(self):
        return self.n_patches * self.n_opt_img_groups * self.n_sar_img_groups
    
    def augment_data(self, *args):
        return args
    
    
    def __getitem__(self, index):
        patch_index = index // (self.n_opt_img_groups * self.n_sar_img_groups)

        group_idx = index % (self.n_opt_img_groups * self.n_sar_img_groups)
        opt_group_index = group_idx // self.n_sar_img_groups
        sar_group_index = group_idx % self.n_sar_img_groups

        opt_images_idx = self.opt_imgs[opt_group_index]
        sar_images_idx = self.sar_imgs[sar_group_index]

        data = h5py.File(self.data_folder / f'{patch_index:d}.h5', 'r', rdcc_nbytes = 10*(1024**2))

        opt_patch = torch.from_numpy(data['opt'][()][opt_images_idx].astype(np.float32)).moveaxis(-1, -3)#.to(self.device)
        sar_patch = torch.from_numpy(data['sar'][()][sar_images_idx].astype(np.float32)).moveaxis(-1, -3)#.to(self.device)
        previous_patch = torch.from_numpy(data['previous'][()].astype(np.float32)).moveaxis(-1, -3)#.to(self.device)
        cloud_patch = torch.from_numpy(data['cloud'][()][opt_images_idx].astype(np.float32)).moveaxis(-1, -3)#.to(self.device)
        label_patch = torch.from_numpy(data['label'][()].astype(np.int64))#.to(self.device)


        opt_patch, sar_patch, previous_patch, cloud_patch, label_patch = self.augment_data(opt_patch, sar_patch, previous_patch, cloud_patch, label_patch)

        return [
            opt_patch,
            sar_patch,
            previous_patch,
        ], (label_patch, cloud_patch)

class TrainDataset(GenericTrainDataset):
    def augment_data(self, opt_patch, sar_patch, previous_patch, cloud_patch, label_patch):
        k = random.randint(0, 3)
        opt_patch = torch.rot90(opt_patch, k=k, dims=[-2, -1])
        sar_patch = torch.rot90(sar_patch, k=k, dims=[-2, -1])
        previous_patch = torch.rot90(previous_patch, k=k, dims=[-2, -1])
        cloud_patch = torch.rot90(cloud_patch, k=k, dims=[-2, -1])
        label_patch = torch.rot90(label_patch, k=k, dims=[-2, -1])

        if bool(random.getrandbits(1)):
            opt_patch = hflip(opt_patch)
            sar_patch = hflip(sar_patch)
            previous_patch = hflip(previous_patch)
            cloud_patch = hflip(cloud_patch)
            label_patch = hflip(label_patch)

        if bool(random.getrandbits(1)):
            opt_patch = vflip(opt_patch)
            sar_patch = vflip(sar_patch)
            previous_patch = vflip(previous_patch)
            cloud_patch = vflip(cloud_patch)
            label_patch = vflip(label_patch)
        
        return opt_patch, sar_patch, previous_patch, cloud_patch, label_patch

class ValDataset(GenericTrainDataset):
    pass


class PredDataset(Dataset):
    def __init__(self, patch_size, params, opt_files, sar_files, prev_file): #, statistics) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.params = params

        # opt_means = statistics['opt_means'] #TODO CHANGED
        # opt_stds = statistics['opt_stds']
        # sar_means = statistics['sar_means']
        # sar_stds = statistics['sar_stds']

        previous = load_sb_image(prev_file)
        self.original_shape = previous.shape

        pad_shape = ((patch_size, patch_size),(patch_size, patch_size))

        previous = np.pad(previous, pad_shape, mode='reflect')
        self.padded_shape = previous.shape

        pad_shape = ((self.patch_size, self.patch_size),(self.patch_size, self.patch_size), (0, 0))

        self.opt_data = [
            rearrange(np.pad((load_opt_image(opt_file)).astype(np.float16), pad_shape, mode='reflect'), 'h w c -> (h w) c')
            #rearrange(np.pad(((load_opt_image(opt_file) - opt_means) / opt_stds).astype(np.float16), pad_shape, mode='reflect'), 'h w c -> (h w) c') 
            for opt_file in opt_files
        ]

        self.sar_data = [
            rearrange(np.pad((load_SAR_image(sar_file)).astype(np.float16), pad_shape, mode='reflect'), 'h w c -> (h w) c')
            #rearrange(np.pad(((load_SAR_image(sar_file) - sar_means) / sar_stds).astype(np.float16), pad_shape, mode='reflect'), 'h w c -> (h w) c') 
            for sar_file in sar_files
        ]

        self.previous = rearrange(previous, 'h w -> (h w)')

        #previous = np.load(self.data_folder / f'{self.params["prefixs"]["previous"]}.npy').astype(np.float16)
        #previous = h5py.File(self.data_folder / f'{self.params["prefixs"]["previous"]}.h5')['previous'][()]

        #label = np.load(self.data_folder / f'{self.params["prefixs"]["label"]}.npy').astype(np.uint8)
        #label = h5py.File(self.data_folder / f'{self.params["prefixs"]["label"]}.h5')['label'][()].astype(np.uint8)
        #self.original_label = label

        #self.original_shape = label.shape


        #self.previous = torch.from_numpy(np.expand_dims(previous, axis=0)).to(self.device)

    # def load_opt_data(self, opt_images_idx):
    #     pad_shape = ((self.patch_size, self.patch_size),(self.patch_size, self.patch_size), (0, 0))

    #     self.opt_data = [
    #         np.pad(h5py.File(self.data_folder / f'{self.params["prefixs"]["opt"]}_{opt_idx}.h5')['opt'][()].astype(np.float16), pad_shape, mode='reflect').reshape((-1, self.params['opt_bands']))
    #         for opt_idx in opt_images_idx
    #         ]

    
    # def load_sar_data(self, sar_images_idx):
    #     pad_shape = ((self.patch_size, self.patch_size),(self.patch_size, self.patch_size), (0, 0))
    #     self.sar_data = [
    #         np.pad(h5py.File(self.data_folder / f'{self.params["prefixs"]["sar"]}_{sar_idx}.h5')['sar'][()].astype(np.float16), pad_shape, mode='reflect').reshape((-1, self.params['sar_bands']))
    #         for sar_idx in sar_images_idx
    #         ]

    def generate_overlap_patches(self, overlap):
        window_shape = (self.patch_size, self.patch_size)
        slide_step = int((1-overlap) * self.patch_size)
        idx_patches = np.arange(self.padded_shape[0]*self.padded_shape[1]).reshape(self.padded_shape)
        self.idx_patches = view_as_windows(idx_patches, window_shape, slide_step).reshape((-1, self.patch_size, self.patch_size))

    def __len__(self):
        return len(self.idx_patches)
    
    def __getitem__(self, index):
        patch_idx = self.idx_patches[index]

        opt_patch = torch.stack([ 
            torch.from_numpy(np.moveaxis(p[patch_idx], 2, 0).astype(np.float32))#.to(self.device)
            for p in self.opt_data 
            ])
        
        sar_patch = torch.stack([ 
            torch.from_numpy(np.moveaxis(p[patch_idx], 2, 0).astype(np.float32))#.to(self.device)
            for p in self.sar_data 
            ])
        previous_patch = np.expand_dims(self.previous[patch_idx], axis=0)
        #label_patch = self.label[patch_idx]

        return [
            opt_patch,
            sar_patch,
            torch.from_numpy(previous_patch.astype(np.float32))#.to(self.device)
        ], patch_idx
        
