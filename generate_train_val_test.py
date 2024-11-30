from  dataprocessing.dataset import MMIFDataSet, transform_val
import random

# 写入文件函数
def write_to_txt(file_list, file_path):
    """
    将文件列表写入到指定的txt文件中。
    
    参数:
        file_list (list): 文件路径列表。
        file_path (str): 要保存的txt文件路径。
    """
    with open(file_path, "w") as f:
        for line in file_list:
            f.write(f"{line}\n")

vi, ir = "/home/Shawalt/Demos/ImageFusion/DataSet/MSRS/test/vi", "/home/Shawalt/Demos/ImageFusion/DataSet/MSRS/test/ir"
data = MMIFDataSet(vi, ir, transform=transform_val)

index = list(range(len(data)))
random.shuffle(index)

train_portion, val_portion = 0.8, 0.2
train_len = int(len(data) * 0.8)
val_len = len(data) - train_len

train_indices = index[:train_len]
val_indices = index[train_len:]

ir_paths = data.ir_paths
vis_paths = data.vi_paths

train_ir_paths = [ir_paths[i] for i in train_indices]
train_vis_paths = [vis_paths[i] for i in train_indices]
val_ir_paths = [ir_paths[i] for i in val_indices]
val_vis_paths = [vis_paths[i] for i in val_indices]

write_to_txt(ir_paths, "MMIF-EMMA/configs/MSRS/test.txt")

# 保存训练集和验证集文件列表到txt文件
# write_to_txt(train_ir_paths, "MMIF-EMMA/configs/MSRS/train_val_pair_5/train_paths.txt")
# write_to_txt(val_ir_paths, "MMIF-EMMA/configs/MSRS/train_val_pair_5/val_paths.txt")


