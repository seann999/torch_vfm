
import cv2
import torch
import torch.utils.data
import os
import numpy as np

root = "/Users/sean/data/rendered_chairs"
dirs = []

for d in os.listdir(root):
    if os.path.isdir(os.path.join(root, d)):
        dirs.append(d)

train_chairs = dirs[:1000]
test_chairs = dirs[1000:]
margin = 100

class ChairsDataset(torch.utils.data.Dataset):
    def __init__(self, chairs):
        self.chairs = chairs

    def __len__(self):
        return len(self.chairs)

    def __getitem__(self, idx):
        renders = os.path.join(root, self.chairs[idx], "renders")
        chair_imgs = os.listdir(renders)
        seq = []
        mats = []
        offset = np.random.randint(len(chair_imgs) - 1)

        for chair_name in chair_imgs[offset:offset+2]:
            img = cv2.imread(os.path.join(renders, chair_name))[margin:600-margin, margin:600-margin, [2, 1, 0]] / 255.0
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)

            mats.append(np.moveaxis(img.astype(np.float32), -1, 0))

        offset = 0

        for i in range(len(chair_imgs)):
            def get_data(i):
                chair_idx = (i + offset) % len(chair_imgs)
                chair_name = chair_imgs[chair_idx]
                mat = mats[chair_idx]

                tokens = chair_name.split("_")
                rotx = int(tokens[2][1:])
                roty = int(tokens[3][1:])

                return mat, roty, rotx

            mat1, roty1, rotx1 = get_data(i)
            mat2, roty2, rotx2 = get_data(i+1)

            diffy = (roty2 - roty1 + 360) % 360
            diffx = rotx2 - rotx1

            seq.append((mat1, (diffy, diffx), mat2))
            break

        mat1, act, mat2 = zip(*seq)

        return mat1, act, mat2

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        ChairsDataset(train_chairs),
        batch_size=16, shuffle=True,
        num_workers=32, pin_memory=False)

    for i, (mat1, act, mat2) in enumerate(train_loader):
        cv2.imshow("seq", np.hstack([mat.numpy()[0, ...] for mat in mat1]))
        cv2.waitKey(0)
