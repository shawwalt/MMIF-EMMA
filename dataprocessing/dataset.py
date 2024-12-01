import os
import re
import sys
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

def sorted_key(filename):
    # 使用正则表达式提取数字部分
    match = re.match(r"(\d+)[a-zA-Z]*\.png", filename)
    if match:
        return int(match.group(1))  # 返回数字部分作为排序依据
    return float('inf')  # 如果不匹配，放在最后

transform_train = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))  # 标准化参数
    transforms.RandomCrop((128, 128))
])

transform_val= transforms.Compose([
    transforms.ToTensor()
])

class MMIFDataSet(Dataset):
    def __init__(self, vis_dir, ir_dir, transform=None, from_file=False, file_path=''):
        super(Dataset, self).__init__()
        self.vis_dir = vis_dir
        self.ir_dir = ir_dir
        if not from_file:
            self.vi_paths = sorted(os.listdir(vis_dir), key=sorted_key)
            self.ir_paths = sorted(os.listdir(ir_dir), key=sorted_key)
        else:
            assert file_path != '', "train val file path not specified"
            self.vi_paths = sorted([line.strip() for line in open(file_path, 'r').readlines()])
            self.ir_paths = self.vi_paths

        self.transform = transform
        assert len(self.vi_paths) == len(self.ir_paths), "can not construct image pairs"

    def __getitem__(self, index):
        vi_path = self.vi_paths[index]
        ir_path = self.ir_paths[index]
        vi_image = cv2.imread(os.path.join(self.vis_dir, vi_path), cv2.IMREAD_GRAYSCALE)  # 确保图像为YCbCr格式, 只融合Y通道
        # vi_image = cv2.cvtColor(vi_image, cv2.COLOR_BGR2YCrCb)
        # vi_image, cb, cr = cv2.split(vi_image) # 只融合可见光的Y通道
        ir_image = cv2.imread(os.path.join(self.ir_dir, ir_path), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            vi_image = self.transform(vi_image)
            ir_image = self.transform(ir_image)
        return (vi_image, ir_image)
    
    def __len__(self):
        return len(self.vi_paths)
    
if __name__ == "__main__":
    vi, ir = "/home/Shawalt/Demos/ImageFusion/DataSet/MSRS/train/vi", "/home/Shawalt/Demos/ImageFusion/DataSet/MSRS/train/ir"
    file_path = './MMIF-EMMA/configs/MSRS/train_val_pair_1/train_paths.txt'
    data = MMIFDataSet(vi, ir, transform=transform_train, from_file=True, file_path=file_path)
    train_loader = DataLoader(data, batch_size=16, shuffle=True, num_workers=8)
    for batch in train_loader:
        print(batch)
    len(data)