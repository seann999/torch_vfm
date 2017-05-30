from vfm import VFM
import os
import argparse
import torch
import torch.optim as optim

parser = argparse.ArgumentParser(description='vfm')
parser.add_argument('--model', type=str, default="runs/A", metavar='G',
                    help='model path')
args = parser.parse_args()

def train_doom():
    model = VFM().cuda()
    iters = 0

    if os.path.isfile('{}/checkpoint.tar'.format(args.model)):
        checkpoint = torch.load('{}/checkpoint.tar'.format(args.model))
        iters = checkpoint['iters']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    while True:
        break

if __name__ == "__main__":
    train_doom()