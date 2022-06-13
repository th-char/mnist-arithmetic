import glob
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os

mnist_ma_dataset_folder = "data"

class MNIST_MA(torch.utils.data.Dataset):
    def __init__(self, mnist_path, dataset_type="train", download=True):
        assert dataset_type in ["train", "test", "val"]
        train = (dataset_type != "test")
        self.dataset_type = dataset_type

        self.mnist_dataset = torchvision.datasets.MNIST(mnist_path, train=train, download=download,
                             transform=transforms.ToTensor())
        
        mnist_ma_dataset_fp = f"{mnist_ma_dataset_folder}/mnist_ma_{dataset_type}.csv"
        if os.path.isfile(mnist_ma_dataset_fp):
          self.mnist_ma_dataset = np.loadtxt(mnist_ma_dataset_fp, delimiter=",")
        else:
          self.mnist_ma_dataset = torch.randint(0, len(self.mnist_dataset), (self.__len__(), 3))
          np.savetxt(mnist_ma_dataset_fp, self.mnist_ma_dataset, delimiter=",")

    def __len__(self):
        if self.dataset_type == "train":
          return 25_000
        elif self.dataset_type == "val":
          return 5_000
        else:
          return 5_000
    
    def __getitem__(self, idx):
        elem_idxs = self.mnist_ma_dataset[idx]

        im1, lab1 = self.mnist_dataset.__getitem__(int(elem_idxs[0]))
        im2, lab2 = self.mnist_dataset.__getitem__(int(elem_idxs[1]))
        im3, lab3 = self.mnist_dataset.__getitem__(int(elem_idxs[2]))

        return idx, (im1, im2, im3), (lab1, lab2, lab3), lab1 * lab2 + lab3