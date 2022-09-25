import os
import torchvision as tv
import torch as t
import torch.nn as nn
from tqdm import tqdm

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 64
        self.main = nn.Sequential(
            # 3 x 96 x 96
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 32 x 32

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = self.main(x)
        x  = x.view(-1)
        return x
        
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        ngf = 64

        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 上(ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()  
            # 3 x 96 x 96
        )

    def forward(self, x):
        x = self.main(x)
        return x

transforms = tv.transforms.Compose([
    tv.transforms.Resize(96),
    tv.transforms.CenterCrop(96),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = tv.datasets.ImageFolder('holo', transform = transforms)
dataloader = t.utils.data.DataLoader(dataset,batch_size = 64, shuffle=True,num_workers = 0,drop_last=True)

model_G = Generator().cuda()
model_D = Discriminator().cuda()
optimizer_g = t.optim.Adam(model_G.parameters(), 1e-4)
optimizer_d = t.optim.Adam(model_D.parameters(), 1e-5)
criterion = t.nn.BCELoss().cuda()
true_labels = t.ones(64).cuda()
fake_labels = t.zeros(64).cuda()
test_noises = t.randn(64, 100, 1, 1).cuda()
train_noises = t.randn(64, 100, 1, 1).cuda()


for epoch in range(20000):
    all_loss_d = 0
    all_loss_g = 0
    tq = tqdm(dataloader)
    for cnt, (img, _) in enumerate(tq, 1):
        real_img = img.cuda()
        if cnt % 1 ==0:
            optimizer_d.zero_grad()
            output = model_D(real_img)
            r_loss_d = criterion(output, true_labels)
            r_loss_d.backward()

            
            fake_img = model_G(train_noises).detach()
            output = model_D(fake_img)
            f_loss_d = criterion(output, fake_labels)
            f_loss_d.backward()
            optimizer_d.step()
            all_loss_d+=f_loss_d.item()+r_loss_d.item()
            
        if cnt % 5 == 0:
            optimizer_g.zero_grad()
            fake_img = model_G(train_noises)
            output = model_D(fake_img)
            loss_g = criterion(output, true_labels)
            loss_g.backward()
            optimizer_g.step()
            all_loss_g+=loss_g.item()
            
        tq.set_description(f'Train Epoch {epoch}')
        tq.set_postfix({'D_Loss':float(all_loss_d/cnt),'G_loss':float(all_loss_g/cnt*5)})

    fix_fake_imgs = model_G(train_noises).detach()
    tv.utils.save_image(fix_fake_imgs,f'pic/{epoch}.jpg')
    if epoch %10==0:
        t.save(model_D.state_dict(), f'model/model_D_{epoch}.pth')
        t.save(model_G.state_dict(), f'model/model_G_{epoch}.pth')
            
    
            