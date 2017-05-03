
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import chairs_loader as chairs
from chairs_loader import ChairsDataset
import numpy as np
import cv2
import torch.optim as optim
import os
from tensorboard_logger import configure, log_value
import argparse

parser = argparse.ArgumentParser(description='vfm')
parser.add_argument('--model', type=str, default="runs/A", metavar='G',
                    help='model path')
args = parser.parse_args()

configure(args.model, flush_secs=5)

s=16

class VFM(nn.Module):

    def __init__(self):
        super(VFM, self).__init__()

        self.en = torch.nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ELU(),
        )

        self.en2 = torch.nn.Sequential(
            nn.Linear(s * s * 128, 256),
            torch.nn.ELU()
        )

        self.mean = nn.Linear(256, 64)
        self.logvar = nn.Linear(256, 64)

        self.de = torch.nn.Sequential(
            nn.Linear(64, 256),
            torch.nn.ELU(),
            nn.Linear(256, s * s * 128),
            torch.nn.ELU()
        )

        self.de2 = torch.nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1, padding=1),
            torch.nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1, padding=1),
            torch.nn.ELU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        h = self.en(x)
        #print(h, "en")
        h = self.en2(h.view(-1, s*s*128))

        mean = self.mean(h)
        logvar = self.logvar(h)

        z = self.reparameterize(mean, logvar)

        return z, mean, logvar

    def decode(self, z):
        h = self.de(z)
        out = self.de2(h.view(-1, 128, s, s))
        #print(out)

        return out

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_ = self.decode(z)

        return x_, mean, logvar

    def loss(self, x, recon, mean, logvar):
        bce = -(x * torch.log(recon + 1e-5) + (1.0 - x) * torch.log(1.0 - recon + 1e-5)).sum(1).sum(2).sum(3).mean()

        KLD_element = mean.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = KLD_element.mul_(-0.5).sum(1).mean()

        return bce + KLD

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        ChairsDataset(chairs.train_chairs),
        batch_size=8, shuffle=True,
        num_workers=4, pin_memory=False)

    model = VFM()
    epoch = 0

    if os.path.isfile('{}/checkpoint.tar'.format(args.model)):
        checkpoint = torch.load('{}/checkpoint.tar'.format(args.model))
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))

    # model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    while True:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict()
        }, '{}/checkpoint.tar'.format(args.model))


        for k in range(20):
            lerps = []
            start = np.random.uniform(-2, 2, (1, 64))
            end = np.random.uniform(-2, 2, (1, 64))

            for x in np.linspace(0, 1, 10):
                gen = model.decode(Variable(torch.FloatTensor(end*x+start*(1-x))))
                lerps.append(np.moveaxis(gen.data.numpy()[0], 0, -1)*255.0)

            cv2.imwrite("image-{}.png".format(k), np.hstack(lerps))


        total_loss = 0

        for i, (mat1, act, mat2) in enumerate(train_loader):
            x = Variable(mat1[0])

            optimizer.zero_grad()
            recon, mean, logvar = model.forward(x)

            loss = model.loss(x, recon, mean, logvar)
            total_loss += loss.data[0]
            loss.backward()

            optimizer.step()
            #print(recon.data.numpy()[0].shape)
            #print(np.amax(x.data.numpy()[0]))
            cv2.imshow("orig", np.moveaxis(x.data.numpy()[0], 0, -1))
            cv2.imshow("img", np.moveaxis(recon.data.numpy()[0], 0, -1))
            cv2.waitKey(1)

            log_value('loss', loss.data[0], i + epoch * len(train_loader))

            print("epoch {}, step {}/{}: {}".format(epoch, i, len(train_loader), loss.data[0]))

        epoch_loss = total_loss / len(train_loader)
        log_value('epoch loss', epoch_loss, epoch)
        print("AVG LOSS: {}".format(epoch_loss))

        epoch += 1
