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


class MydataSet(Dataset):
    def __init__(self, transform=None, loader=default_loader):
        super(MydataSet, self).__init__()
        self.ground_truth_path = '/home/gaotiegang01/liuhao/dataset/CASIAv2/CASIA2.0_Groundtruth'

        # Au:0 Tp:1
        self.img_tag = []

        self.au_path = '/home/gaotiegang01/liuhao/dataset/CASIAv2/CASIA2.0_revised/Au'
        au_path_list = '/home/gaotiegang01/liuhao/dataset/CASIAv2/CASIA2.0_revised/au_list.txt'
        self.imgs_name = []
        with open(au_path_list, "r") as f:
            data = f.readline()
            self.imgs_name.append(data)
            self.img_tag.append(0)

        self.tp_path = '/home/gaotiegang01/liuhao/dataset/CASIAv2/CASIA2.0_revised/Tp'
        tp_path_list = '/home/gaotiegang01/liuhao/dataset/CASIAv2/CASIA2.0_revised/tp_list.txt'
        with open(tp_path_list, "r") as f:
            data = f.readline()
            self.imgs_name.append(data)
            self.img_tag.append(1)

        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.imgs_name[index]
        img_tag = self.img_tag[index]

        if img_tag == 0:  # 未篡改图像，生成全黑GroundTruth
            img = self.loader(os.path.join(self.au_path, img_name))
            img = self.transform(img)
            ground_truth = torch.zeros(size=img.shape)
        else:  # 篡改图像，读取对应的GroundTruth
            img = self.loader(os.path.join(self.tp_path, img_name))
            img = self.transform(img)
            ground_truth = self.loader(os.path.join(self.ground_truth_path, img_name))

        return img, ground_truth

    def __len__(self):
        return len(self.imgs_name)


if __name__ == '__main__':
    net = U2NET()
    net.cuda()
    dataset = MydataSet()
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=4)
    Lossfuction = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    net.train()
    for epoch in range(300):
        for data, label in dataloader:
            data = data.cuda()
            label = label.cuda()
            o0, o1, o2, o3, o4, o5, o6 = net(data)
            loss0 = Lossfuction(o0, label)
            loss1 = Lossfuction(o1, label)
            loss2 = Lossfuction(o2, label)
            loss3 = Lossfuction(o3, label)
            loss4 = Lossfuction(o4, label)
            loss5 = Lossfuction(o5, label)
            loss6 = Lossfuction(o6, label)
            loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
        torch.save(net.state_dict(), 'netpig.pt')
