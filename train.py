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
import torchvision
import random
import numpy as np
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class MydataSet(Dataset):
    def __init__(self, transform=None, loader=default_loader, start=0, end=20000):
        super(MydataSet, self).__init__()
        self.ground_truth_path = '/home/gaotiegang01/liuhao/dataset/CASIAv2/CASIA2.0_Groundtruth'
        self.imgs_name = []
        # Au:0 Tp:1
        self.img_tag = []

        # self.au_path = '/home/gaotiegang01/liuhao/dataset/CASIAv2/CASIA2.0_revised/Au'
        # au_path_list = '/home/gaotiegang01/liuhao/dataset/CASIAv2/CASIA2.0_revised/au_list.txt'
        # f = open(au_path_list, "r")
        # lines = f.readlines()
        # count = 0
        # for line in lines:
        #     line = line.replace('\n', '')
        #     # 天知道这个txt从哪冒出来的
        #     if line.endswith('.txt'):
        #         continue
        #     count += 1
        #     if count > end or count < start:
        #         continue
        #     self.imgs_name.append(line)
        #     self.img_tag.append(0)

        self.tp_path = '/home/gaotiegang01/liuhao/dataset/CASIAv2/CASIA2.0_revised/Tp'
        tp_path_list = '/home/gaotiegang01/liuhao/dataset/CASIAv2/CASIA2.0_revised/tp_list.txt'
        f = open(tp_path_list, "r")
        lines = f.readlines()
        count = 0
        for line in lines:
            line = line.replace('\n', '')
            if line.endswith('.txt'):
                continue
            count += 1
            if count > end or count < start:
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

        return img_name, img, ground_truth

    def __len__(self):
        return len(self.imgs_name)

    def ground_truth_name(self, img_name):
        names = img_name.split('.')
        return names[0] + "_gt.png"


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    model = U2NET()
    model.cuda()

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = MydataSet(transform=transform_train, start=0, end=3300)
    train_dataloader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=4)

    test_dataset = MydataSet(transform=transform_train, start=4500, end=4520)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=0)

    Lossfuction = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    step_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100)

    for epoch in range(1000):
        start = time.time()
        loss_sum = 0
        iter_sum = 0
        model.train()
        for index, (imgs_name, data, ground_truth) in enumerate(train_dataloader):
            data = data.cuda()
            ground_truth = ground_truth.cuda()
            o0, o1, o2, o3, o4, o5, o6 = model(data)
            loss0 = Lossfuction(o0, ground_truth)
            loss1 = Lossfuction(o1, ground_truth)
            loss2 = Lossfuction(o2, ground_truth)
            loss3 = Lossfuction(o3, ground_truth)
            loss4 = Lossfuction(o4, ground_truth)
            loss5 = Lossfuction(o5, ground_truth)
            loss6 = Lossfuction(o6, ground_truth)
            loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            iter_sum += 1
            if index % 10 == 0:
                print(f'epoch={epoch} loss_sum={loss_sum} iter={iter_sum} loss={loss_sum / iter_sum}')
        step_lr.step()
        print(f'==>epoch={epoch} loss={loss_sum/iter_sum} cost_time={time.time()-start}\n')

        if epoch % 1 == 0:
            model.eval()
            for index, (imgs_name, data, ground_truth) in enumerate(test_dataloader):
                data = data.cuda()
                ground_truth = ground_truth.cuda()
                o0, o1, o2, o3, o4, o5, o6 = model(data)
                for i in range(o0.shape[0]):
                    ndarr = o0[i].mul(255).clamp_(0, 255).to('cpu', torch.uint8).numpy()
                    ndarr = ndarr[0]
                    im = Image.fromarray(ndarr)
                    im.save(os.path.join('./res', imgs_name[i])+'.png')

        torch.save(model.state_dict(), 'epoch='+str(epoch)+'.pth')

