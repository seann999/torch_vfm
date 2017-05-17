
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
img_size = 64
rnn_size = 512
rnn_input_size = 256
layers = 1
batch_size = 32
z_size = 128
action_size = 2

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
            nn.Linear(s * s * 128 + action_size, rnn_input_size),
            torch.nn.ELU()
        )

        self.rnn = nn.LSTM(rnn_input_size, rnn_size, layers, batch_first=True)

        self.mean = nn.Linear(rnn_size, z_size)
        self.logvar = nn.Linear(rnn_size, z_size)

        self.de = torch.nn.Sequential(
            nn.Linear(z_size, 256),
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
        eps = Variable(eps).cuda()
        return eps.mul(std).add_(mu)

    def encode(self, x, acts):
        x = x.view(batch_size*chairs.steps, 3, img_size, img_size)
        h = self.en(x)

        img_enc = h.view(-1, s*s*128)
        acts = acts.view(-1, 2)

        h = self.en2(torch.cat([img_enc, acts], dim=1))

        rnn_h = Variable(torch.zeros(layers, batch_size, rnn_size)).cuda()
        rnn_c = Variable(torch.zeros(layers, batch_size, rnn_size)).cuda()
        h = h.view(batch_size, chairs.steps, -1)

        h, (_, _) = self.rnn(h, (rnn_h, rnn_c))

        h = h.contiguous().view(batch_size*chairs.steps, rnn_size)

        mean = self.mean(h)
        logvar = self.logvar(h)

        z = self.reparameterize(mean, logvar)

        mean = mean.view(batch_size, chairs.steps, -1)
        logvar = logvar.view(batch_size, chairs.steps, -1)

        return z, mean, logvar

    def decode(self, z):
        z = z.view(batch_size*chairs.steps, -1)
        #print(z.size())
        h = self.de(z)
        out = self.de2(h.view(-1, 128, s, s))
        out = out.view(batch_size, chairs.steps, 3, img_size, img_size)

        return out

    def forward(self, x, acts):
        z, mean, logvar = self.encode(x, acts)
        x_ = self.decode(z)

        return x_, mean, logvar

    def loss(self, x, recon, mean, logvar):
        bce = -(x * torch.log(recon + 1e-5) + (1.0 - x) * torch.log(1.0 - recon + 1e-5)).sum(1).sum(2).sum(3).sum(4).mean()

        KLD_element = mean.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = KLD_element.mul_(-0.5).sum(1).sum(2).mean()

        return bce + KLD

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def train():
    train_loader = torch.utils.data.DataLoader(
        ChairsDataset(chairs.train_chairs),
        batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=False, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        ChairsDataset(chairs.test_chairs),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=False, drop_last=True)

    model = VFM().cuda()
    epoch = 0

    if os.path.isfile('{}/checkpoint.tar'.format(args.model)):
        checkpoint = torch.load('{}/checkpoint.tar'.format(args.model))
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))

    # model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # hidden = model.init_hidden(args.batch_size)

    while True:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict()
        }, '{}/checkpoint.tar'.format(args.model))

        def run(train):
            total_loss = 0

            if train:
                title = "train"
            else:
                title = "test"

            if train:
                loader = train_loader
            else:
                loader = test_loader

            for i, (mat1, act, mat2) in enumerate(loader):
                x = Variable(mat1).cuda()
                y = Variable(mat2).cuda()

                acts = Variable(act).cuda()

                # hidden = repackage_hidden(hidden)
                optimizer.zero_grad()
                recon, mean, logvar = model.forward(x, acts)

                #if train:
                loss = model.loss(y, recon, mean, logvar)
                total_loss += loss.data[0]
                loss.backward()

                if train:
                    optimizer.step()

                x_img = np.hstack([np.moveaxis(x.data.cpu().numpy()[0, m, ...], 0, -1) for m in range(chairs.steps)])
                y_img = np.hstack([np.moveaxis(y.data.cpu().numpy()[0, m, ...], 0, -1) for m in range(chairs.steps)])
                pred_img = np.hstack([np.moveaxis(recon.data.cpu().numpy()[0, m, ...], 0, -1) for m in range(chairs.steps)])

                cv2.imshow(title, np.vstack([x_img, y_img, pred_img]))
                cv2.waitKey(1)

                if train:
                    log_value("{} loss".format(title), loss.data[0], i + epoch * len(train_loader))

                    print("epoch {}, step {}/{}: {}".format(epoch, i, len(train_loader), loss.data[0]))

                print(act.numpy()[0])


            epoch_loss = total_loss / len(loader)
            log_value('epoch {} loss'.format(title), epoch_loss, epoch)
            print("AVG LOSS: {}".format(epoch_loss))

        run(True)
        run(False)
        epoch += 1

def lerp():
    global batch_size
    batch_size = 1
    chairs.steps = 1

    model = VFM().cuda()

    if os.path.isfile('{}/checkpoint.tar'.format(args.model)):
        checkpoint = torch.load('{}/checkpoint.tar'.format(args.model))
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))


    while True:
        print("generating")
        lerps = []
        start = np.random.uniform(-2, 2, (1, z_size))
        end = np.random.uniform(-2, 2, (1, z_size))

        for x in np.linspace(0, 1, 10):
            gen = model.decode(Variable(torch.FloatTensor(end*x+start*(1-x))).cuda())
            lerps.append(np.moveaxis(gen.data.cpu().numpy()[0, 0, ...], 0, -1))

        cv2.imshow("lerp", np.hstack(lerps))
        cv2.waitKey(0)
        #cv2.imwrite("image-{}.png".format(k), np.hstack(lerps))
        print("generated ")

if __name__ == "__main__":
    #train()
    lerp()
