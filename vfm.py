import torch
import torch.nn as nn
from torch.autograd import Variable

s=16
rnn_layers = 1

class VFM(nn.Module):

    def __init__(self, img_size=64, action_size=2, z_size=32, rnn_input_size=256, rnn_size=512, batch_size=32,
                 in_len=10, out_len=10):
        super(VFM, self).__init__()

        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.img_size = img_size
        self.in_len = in_len
        self.out_len = out_len

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

        self.rnn = nn.LSTM(rnn_input_size, rnn_size, rnn_layers, batch_first=True)

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
        x = x.view(self.batch_size*self.in_len, 3, self.img_size, self.img_size)
        h = self.en(x)

        img_enc = h.view(-1, s*s*128)
        acts = acts.view(-1, 2)

        h = self.en2(torch.cat([img_enc, acts], dim=1))

        rnn_h = Variable(torch.zeros(rnn_layers, self.batch_size, self.rnn_size)).cuda()
        rnn_c = Variable(torch.zeros(rnn_layers, self.batch_size, self.rnn_size)).cuda()
        h = h.view(self.batch_size, self.seq_len, -1)

        h, (_, _) = self.rnn(h, (rnn_h, rnn_c))

        h = h.contiguous().view(self.batch_size*self.seq_len, self.rnn_size)

        mean = self.mean(h)
        logvar = self.logvar(h)

        z = self.reparameterize(mean, logvar)

        mean = mean.view(self.batch_size, self.seq_len, -1)
        logvar = logvar.view(self.batch_size, self.seq_len, -1)

        return z, mean, logvar

    def decode(self, z):
        z = z.view(self.batch_size*self.seq_len, -1)
        #print(z.size())
        h = self.de(z)
        out = self.de2(h.view(-1, 128, s, s))
        out = out.view(self.batch_size, self.seq_len, 3, self.img_size, self.img_size)

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


