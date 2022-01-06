from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from u2net import U2NET
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
from torchvision.datasets.folder import default_loader
import os

class MydataSet(Dataset):
    def __init__(self, transform=None, loader=default_loader):
        super(MydataSet, self).__init__()
        self.ground_truth_path = '/home/gaotiegang01/liuhao/dataset/CASIAv2/CASIA2.0_Groundtruth'

        # Au:0 Tp:1
        self.img_tag = []

        self.au_path = '/home/gaotiegang01/liuhao/dataset/CASIAv2/CASIA2.0_revised/Au'
        au_path_list = '/home/gaotiegang01/liuhao/dataset/CASIAv2/CASIA2.0_revised/au_list.txt'
        self.imgs_name = []
        f = open(au_path_list, "r")
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            # 天知道这个txt从哪冒出来的
            if line.endswith('.txt'):
                continue
            self.imgs_name.append(line)
            self.img_tag.append(0)

        self.tp_path = '/home/gaotiegang01/liuhao/dataset/CASIAv2/CASIA2.0_revised/Tp'
        tp_path_list = '/home/gaotiegang01/liuhao/dataset/CASIAv2/CASIA2.0_revised/tp_list.txt'
        f = open(tp_path_list, "r")
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            if line.endswith('.txt'):
                continue
            self.imgs_name.append(line)
            self.img_tag.append(1)

        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.imgs_name[index]
        img_tag = self.img_tag[index]

        if img_tag == 0:  # 未篡改图像，生成全黑GroundTruth
            img = self.loader(os.path.join(self.au_path, img_name))
            img = self.transform(img)
            ground_truth = torch.zeros(size=[1, img.shape[1], img.shape[2]])
        else:  # 篡改图像，读取对应的GroundTruth
            img = self.loader(os.path.join(self.tp_path, img_name))
            img = self.transform(img)
            ground_truth = Image.open(os.path.join(self.ground_truth_path, self.ground_truth_name(img_name))).convert('L')
            ground_truth = self.transform(ground_truth)

        return img, ground_truth

    def __len__(self):
        return len(self.imgs_name)

    def ground_truth_name(self, img_name):
        names = img_name.split('.')
        return names[0] + "_gt.png"


if __name__ == '__main__':
    net = U2NET()
    weight = torch.load('netpig.pt', map_location='cpu')
    net.load_state_dict(weight)
    net.cuda()

    # 进行验证
