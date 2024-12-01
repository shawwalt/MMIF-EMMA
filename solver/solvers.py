import sys
import os
import time

# 获取上级目录的路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将上级目录添加到 sys.path
sys.path.append(parent_dir)
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from dataprocessing.dataset import *
from metrics.Evaluator import Evaluator
import yaml
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from nets.Ufuser import *
from nets.Unet5 import *
from utils import Transformer, loss_fusion
from tqdm import tqdm

class MMIFExpSolver:
    def __init__(self, config_path):
        # 加载配置文件
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # 初始化设备
        self.gpu = self.config["gpu"]
        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() and self.config["use_cuda"] else "cpu")
        self.algorithm = self.config["algorithm"]
        self.num_epochs = self.config['dataset']['num_epochs']
        self.validate_step = self.config['dataset']['val_step']
        
        # 加载数据集
        self.vis_path = self.config["dataset"]["vis_path"]
        self.ir_path = self.config["dataset"]["ir_path"]
        self.test_vis_path = self.config["dataset"]["test_vis_path"]
        self.test_ir_path = self.config["dataset"]["test_ir_path"]
        if "train_paths" in self.config["dataset"]:
            self.train_paths = self.config["dataset"]["train_paths"]
        if "val_paths" in self.config["dataset"]:
            self.val_paths = self.config["dataset"]["val_paths"]
        if "test_paths" in self.config["dataset"]:
            self.test_paths = self.config["dataset"]["test_paths"]
        self.train_dataset = MMIFDataSet(self.vis_path, self.ir_path, from_file=True, file_path=self.train_paths, transform=transform_train)
        self.val_dataset = MMIFDataSet(self.vis_path, self.ir_path, from_file=True, file_path=self.val_paths, transform=transform_val)
        self.test_dataset = MMIFDataSet(self.test_vis_path, self.test_ir_path, from_file=True, file_path=self.test_paths, transform=transform_val)

        self.train_batch_size = self.config["dataset"]["train_batch_size"]
        self.val_batch_size = self.config["dataset"]["val_batch_size"]
        self.num_workers = self.config["dataset"]["num_workers"]
        
        self.trainloader = DataLoader(self.train_dataset,batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers)
        self.valloader = DataLoader(self.val_dataset,batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)
        self.testloader = DataLoader(self.test_dataset,batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)

    def _initialize_model(self):
        raise NotImplemented

    def train(self):
        raise NotImplemented

    def validate(self):
        raise NotImplemented
    
    def run(self):
        raise NotImplemented
    
class EMMAExpSolver(MMIFExpSolver):
    def __init__(self, config_path):
        super(EMMAExpSolver, self).__init__(config_path)
        self.set_outstream() # 设置命令行输出路径
        self.fuser = Ufuser().to(self.device)
        self.F2Vmodel = UNet5().to(self.device) 
        self.F2Imodel = UNet5().to(self.device)

        self.visulization_path = self.config['visualization_path']
        shift_num=self.config['others']['transformer']['shift_num']
        rotate_num=self.config['others']['transformer']['rotate_num']
        flip_num=self.config['others']['transformer']['flip_num']
        self.trans = Transformer(shift_num, rotate_num, flip_num)

        self._initialize_model()
        self._initialize_train_setting()

    def set_outstream(self):
        # 打开文件
        self.start_date = time.strftime("<%Y-%m-%d>-<%H:%M:%S>", time.localtime(time.time())) # 训练开始时间
        self.save_root = os.path.join(self.config['save_path'], self.config['algorithm'] + '_' + self.start_date)
        os.mkdir(os.path.join('.', self.save_root))
        self.f = open(os.path.join(self.save_root, f'log_{self.start_date}.txt'), 'w')
        # 将标准输出重定向到文件
        sys.stdout = self.f
        sys.stderr = self.f

    def _initialize_model(self):
        if "fuser" in self.config["others"]["EMMA"]['pretrained']:
            self.fuser.load_state_dict(torch.load(self.config["others"]["EMMA"]['pretrained']["fuser"]))

        if "Av" in self.config["others"]["EMMA"]['pretrained']:
            self.F2Vmodel.load_state_dict(torch.load(self.config["others"]["EMMA"]['pretrained']["Av"]))
            
        if "Ai" in self.config["others"]["EMMA"]['pretrained']:
            self.F2Imodel.load_state_dict(torch.load(self.config["others"]["EMMA"]['pretrained']["Ai"]))

    def _initialize_train_setting(self):
        self.lr = float(self.config['train']['optimizer']['lr'])
        self.alpha = float(self.config['train']['loss']['alpha'])
        self.step_size = int(self.config['train']['scheduler']['step_size'])
        self.gamma = float(self.config['train']['scheduler']['gamma'])
        self.weight_decay = float(self.config['train']['optimizer']['weight_decay'])

        if self.config['train']['optimizer']['type'] == 'adam':
            self.optimizer = torch.optim.Adam(self.fuser.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
        self.loss = loss_fusion()

    def train(self):
        self.fuser.train()
        self.F2Imodel.eval()
        self.F2Vmodel.eval()
        for epoch in range(self.num_epochs):
            ''' train '''
            process_bar = tqdm(range(len(self.train_dataset)), desc=f'Epoch {epoch+1} Training...')
            loss_sum = 0.0
            iter_num = 0
            for i, batch in enumerate(self.trainloader):
                data_VIS, data_IR = batch[0].to(self.device), batch[1].to(self.device)
                F=self.fuser(data_IR, data_VIS)  # F
                Ft = self.trans.apply(F)
                restore_ir, restore_vi = self.F2Imodel(Ft), self.F2Vmodel(Ft)
                Ft_caret= self.fuser(restore_ir, restore_vi) # Ft_caret
                self.optimizer.zero_grad()
                loss_total=self.loss(self.F2Vmodel(F),data_VIS)+self.loss(self.F2Imodel(F),data_IR)+self.alpha*self.loss(Ft,Ft_caret)
                # loss_total=self.loss(self.F2Vmodel(F),data_VIS)+self.loss(self.F2Imodel(F),data_IR) # w/o Equivariant Loss
                loss_total.backward()
                self.optimizer.step()
                process_bar.update(data_VIS.shape[0])
                loss_sum += loss_total.item()
                process_bar.set_postfix(loss=loss_total.item())
                iter_num += 1
            loss_sum /= iter_num
            process_bar.set_postfix(loss=loss_sum)
            self.scheduler.step()

            if (epoch+1) % self.validate_step == 0:
                self.validate()
                self.fuser.train() # return to train mode
    
    def validate(self):
        self.fuser.eval()
        process_bar = tqdm(range(len(self.val_dataset)), desc="Validating...")
        metric_result = np.zeros((8))
        data_range = 255.0 # 用于指定SSIM的输入图像动态范围
        for i, batch in enumerate(self.valloader):
            vi, ir = batch[0].to(self.device), batch[1].to(self.device)
            process_bar.update(vi.shape[0])
            fi = None
            with torch.no_grad():
                fi = self.fuser(ir, vi)
            fi_s = fi.cpu().numpy()
            vi = vi.cpu().numpy()
            ir = ir.cpu().numpy()
            for i, v, fi in zip(ir, vi, fi_s):
                # 先转为uint8是为了将小数位截断
                i, v, fi = (i.squeeze(0) * 255).astype('uint8'), (v.squeeze(0) * 255).astype('uint8'), (fi.squeeze(0) * 255).astype('uint8')
                i, v, fi = i.astype('float32'), v.astype('float32'), fi.astype('float32')
                metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                        , Evaluator.SF(fi), Evaluator.MI(fi, i, v)
                                        , Evaluator.SCD(fi, i, v), Evaluator.VIFF(fi, i, v)
                                        , Evaluator.Qabf(fi, i, v), Evaluator.SSIM(fi, i, v, data_range=data_range)])
        metric_result /= len(self.val_dataset)
        print('\n')
        print("\t\t EN\t SD\t SF\t MI\t SCD\t VIF\t Qabf\t SSIM\n")
        print(self.algorithm+'\t'+str(np.round(metric_result[0], 2))+'\t'
            +str(np.round(metric_result[1], 2))+'\t'
            +str(np.round(metric_result[2], 2))+'\t'
            +str(np.round(metric_result[3], 2))+'\t'
            +str(np.round(metric_result[4], 2))+'\t'
            +str(np.round(metric_result[5], 2))+'\t'
            +str(np.round(metric_result[6], 2))+'\t'
            +str(np.round(metric_result[7], 2))
            )
        print("="*80)

    def test(self):
        self.fuser.eval()
        process_bar = tqdm(range(len(self.test_dataset)), desc="Testing...")
        metric_result = np.zeros((8))
        data_range = 255.0 # 用于指定SSIM的输入图像动态范围
        for i, batch in enumerate(self.testloader):
            vi, ir = batch[0].to(self.device), batch[1].to(self.device)
            process_bar.update(vi.shape[0])
            fi = None
            with torch.no_grad():
                fi = self.fuser(ir, vi)
            fi_s = fi.cpu().numpy()
            vi = vi.cpu().numpy()
            ir = ir.cpu().numpy()
            for i, v, fi in zip(ir, vi, fi_s):
                # 先转为uint8是为了将小数位截断
                i, v, fi = (i.squeeze(0) * 255).astype('uint8'), (v.squeeze(0) * 255).astype('uint8'), (fi.squeeze(0) * 255).astype('uint8')
                i, v, fi = i.astype('float32'), v.astype('float32'), fi.astype('float32')
                metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                        , Evaluator.SF(fi), Evaluator.MI(fi, i, v)
                                        , Evaluator.SCD(fi, i, v), Evaluator.VIFF(fi, i, v)
                                        , Evaluator.Qabf(fi, i, v), Evaluator.SSIM(fi, i, v, data_range=data_range)])
        metric_result /= len(self.test_dataset)
        print('\n')
        print("\t\t EN\t SD\t SF\t MI\t SCD\t VIF\t Qabf\t SSIM\n")
        print(self.algorithm+'\t'+str(np.round(metric_result[0], 2))+'\t'
            +str(np.round(metric_result[1], 2))+'\t'
            +str(np.round(metric_result[2], 2))+'\t'
            +str(np.round(metric_result[3], 2))+'\t'
            +str(np.round(metric_result[4], 2))+'\t'
            +str(np.round(metric_result[5], 2))+'\t'
            +str(np.round(metric_result[6], 2))+'\t'
            +str(np.round(metric_result[7], 2))
            )
        print("="*80)

    def visulization(self):
        assert self.visulization_path is not None
        self.fuser.eval()
        vi_paths = sorted(os.listdir(self.vis_dir), key=sorted_key)
        ir_paths = sorted(os.listdir(self.ir_dir), key=sorted_key)
        for vi, ir in zip(vi_paths, ir_paths):
            vi_image = transform_val(cv2.imread(os.path.join(self.vis_dir, vi), cv2.IMREAD_GRAYSCALE))
            ir_image = transform_val(cv2.imread(os.path.join(self.vis_dir, ir), cv2.IMREAD_GRAYSCALE))
            fuse_image = self.fuser(ir_image, vi_image)
            vi_ycrcb = cv2.imread(os.path.join(self.vis_dir, vi), cv2.COLOR_BGR2YCrCb)
            vi_ycrcb[0] = (fuse_image * 255).astype('uint8')
            vi_image = cv2.cvtColor(vi_ycrcb, cv2.COLOR_YCR_CB2RGB)
            cv2.imwrite(self.visulization_path + vi, vi_image)
            print(f'img out {vi} in dir {self.visulization_path}')


    def save_checkpoint(self): # 保存训练记录
        torch.save(self.fuser.state_dict(),os.path.join(self.save_root, 'fuser.pth'))
        torch.save(self.F2Imodel.state_dict(), os.path.join(self.save_root, 'Ai.pth'))
        torch.save(self.F2Vmodel.state_dict(), os.path.join(self.save_root, 'Av.pth'))
        with open(os.path.join(self.save_root, 'option.yaml'), 'w') as file:
            yaml.safe_dump(self.config, file)
        print("checkpoint saved in " + self.save_root)
            
if __name__ == "__main__":
    solver = EMMAExpSolver("./MMIF-EMMA/option.yaml")
    # solver.train()
    solver.validate()

