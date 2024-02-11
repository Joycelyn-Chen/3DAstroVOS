import os
from os import path

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization


import torch
from pathlib import Path

class AstroVideoReader(Dataset):
    def __init__(self, vid_name, image_dir, mask_dir, num_frames=50, size = -1, to_save=None, use_all_mask=False, size_dir=None):
        self.vid_name = vid_name
        self.image_dir = image_dir 
        self.mask_dir = mask_dir
        self.num_frames = num_frames
        self.to_save = to_save
        self.use_all_mask = use_all_mask
        self.size_dir = size_dir if size_dir is not None else self.image_dir

        self.frames = self.__read_frames(self.image_dir) #sorted(os.listdir(self.image_dir))[:self.num_frames]  # Limit to num_frames
        
        timestamp_dir = os.path.join(mask_dir, sorted(os.listdir(mask_dir))[0])
        mask_filename = os.listdir(timestamp_dir)[0]
        self.palette = Image.open(os.path.join(timestamp_dir, mask_filename)).getpalette()
        self.first_gt_path = os.path.join(self.mask_dir, sorted(os.listdir(self.mask_dir))[0])



        if size < 0:
            self.im_transform = transforms.Compose([
                # transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                im_normalization,
            ])
            self.gt_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Grayscale(),
                im_normalization,
                transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
            ])
            self.gt_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.size = size


    def __getitem__(self, idx):
        info = {}
        data = {}       #{'rgb': torch.zeros((self.num_frames, *self.size))}  # Initialize the tensor for grayscale images
        info['frame'] = self.frames[idx]            # self.frames[idx * self.num_frames: (idx + 1) * self.num_frames]  # Get a batch of frames
        

        img_names = self.frames[idx]
        timestamp =  str(int(img_names[0].split("_")[-2]))  # sn34_smd132_bx5_pe300_hdf5_plt_cnt_0201_z643.jpg

        info['save'] = (self.to_save is None) or (timestamp in self.to_save) #any(idx in self.to_save for frame in info['frame'])

        frame_images = []
        frame_masks = []
        for i, img_name in enumerate(img_names):
            img_path = os.path.join(self.image_dir, timestamp, img_name)
            img = Image.open(img_path).convert('RGB')

            img = self.im_transform(img)  # Apply transformations
            frame_images.append(img)

            gt_path_dir = os.path.join(self.mask_dir, timestamp)
            if self.use_all_mask or (gt_path_dir == self.first_gt_path):
                gt_path = os.path.join(gt_path_dir, img_name[:-4] + '.png')
                if os.path.exists(gt_path):
                    mask = Image.open(gt_path).convert('P')
                    mask = self.gt_transform(mask)
                    # mask = np.array(mask, dtype=np.uint8)
                    frame_masks.append(mask)

        if self.use_all_mask or (gt_path_dir == self.first_gt_path):
            data['mask'] = torch.stack(frame_masks, 0)[:, 0, :, :] 
            with open("tmp.txt", "a+") as f:
                f.write(f"data['mask']: {data['mask'].shape}\n")
            

        
        data['rgb'] = torch.stack(frame_images, 0)[:, 0, :, :]  # Stack the transformed frame
        
        
        if self.image_dir == self.size_dir:
            shape = np.array(frame_images).shape[:3]
        else:
            size_path = os.path.join(self.size_dir, timestamp, img_name)
            size_im = Image.open(size_path).convert('RGB')
            shape = np.array(size_im).shape[:3]

        info['shape'] = shape
        info['need_resize'] = not (self.size < 0)
        data['info'] = info

        return data

    def get_palette(self):
        return self.palette

    def __len__(self):
        return len(self.frames) #// self.num_frames  # Number of batches
    
    def __read_frames(self, clip_path):
        frames = []
        for timestamp in sorted(os.listdir(clip_path)): 
            timestamp_path = os.path.join(clip_path, timestamp)    
            #read all images within time timestamp_path and store into frame
            frames.append(sorted(os.listdir(timestamp_path)))

        # then append the list to frames
        return frames


#+------------------------------------------older version-------------------------------------------------------------+


class VideoReader(Dataset):
    """
    This class is used to read a video, one frame at a time
    """
    def __init__(self, vid_name, image_dir, mask_dir, size=-1, to_save=None, use_all_mask=False, size_dir=None):
        """
        image_dir - points to a directory of jpg images
        mask_dir - points to a directory of png masks
        size - resize min. side to size. Does nothing if <0.
        to_save - optionally contains a list of file names without extensions 
            where the segmentation mask is required
        use_all_mask - when true, read all available mask in mask_dir.
            Default false. Set to true for YouTubeVOS validation.
        """
        self.vid_name = vid_name
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.to_save = to_save
        self.use_all_mask = use_all_mask
        if size_dir is None:
            self.size_dir = self.image_dir
        else:
            self.size_dir = size_dir

        self.frames = sorted(os.listdir(self.image_dir))
        self.palette = Image.open(path.join(mask_dir, sorted(os.listdir(mask_dir))[0])).getpalette()
        self.first_gt_path = path.join(self.mask_dir, sorted(os.listdir(self.mask_dir))[0])

        if size < 0:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
            ])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
                transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
            ])
        self.size = size


    def __getitem__(self, idx):
        frame = self.frames[idx]
        info = {}
        data = {}
        info['frame'] = frame
        info['save'] = (self.to_save is None) or (frame[:-4] in self.to_save)

        im_path = path.join(self.image_dir, frame)
        img = Image.open(im_path).convert('RGB')

        if self.image_dir == self.size_dir:
            shape = np.array(img).shape[:2]
        else:
            size_path = path.join(self.size_dir, frame)
            size_im = Image.open(size_path).convert('RGB')
            shape = np.array(size_im).shape[:2]

        gt_path = path.join(self.mask_dir, frame[:-4]+'.png')
        img = self.im_transform(img)

        load_mask = self.use_all_mask or (gt_path == self.first_gt_path)
        if load_mask and path.exists(gt_path):
            mask = Image.open(gt_path).convert('P')
            mask = np.array(mask, dtype=np.uint8)
            data['mask'] = mask

        info['shape'] = shape
        info['need_resize'] = not (self.size < 0)
        data['rgb'] = img       # [3, 1000, 1000]
        data['info'] = info
        
        return data

    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)), 
                    mode='nearest')

    def get_palette(self):
        return self.palette

    def __len__(self):
        return len(self.frames)