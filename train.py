from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from u2net import U2NET
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch

data_dir = r'D:\MASKpicture\train_pig_body'
label_dir = r'D:\MASKpicture\label_pig_body'


class MydataSet(Dataset):
    def __init__(self,
                 data_dir,
                 label_dir):
        super(MydataSet, self).__init__()
        self.dataset = os.listdir(data_dir)
        self.dataset = self.dataset

    def __getitem__(self, index):
        try:
            image = Image.open(os.path.join(data_dir, self.dataset[index])).convert('RGB')
            label = Image.open(os.path.join(label_dir, self.dataset[index])).convert('L')
            pad = max(image.size)
            size = (pad, pad)
            transform = transforms.Compose([
                transforms.CenterCrop(size),
                transforms.Resize(490),
                transforms.ToTensor()
            ])
            imagedata = transform(image)
            labeldata = transform(label)

            return imagedata, labeldata
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.dataset)


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