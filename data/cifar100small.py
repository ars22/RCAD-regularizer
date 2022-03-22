import torch
from torchvision import datasets
import numpy as np
from PIL import Image

SEED = 43
ROOT = 'data'
TRAIN_SZ = 500

class CIFAR100Small(torch.utils.data.Dataset):

  def __init__(self, root, transform=None, target_transform=None, train=True, train_sz=10000):
    if train:
      pkl_data = torch.load(open(root + f'/cifar100small{train_sz}_train.pkl', 'rb'))
      self.data = pkl_data['data']
      self.targets = pkl_data['targets']
    else:
      pkl_data = torch.load(open(root + f'/cifar100small_test.pkl', 'rb'))
      self.data = pkl_data['data']
      self.targets = pkl_data['targets']
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index: int):
      """
      Args:
          index (int): Index
      Returns:
          tuple: (image, target) where target is index of the target class.
      """
      img, target = self.data[index], self.targets[index]
      # doing this so that it is consistent with all other datasets
      # to return a PIL Image
      img = Image.fromarray(img)
      if self.transform is not None:
          img = self.transform(img)
      if self.target_transform is not None:
          target = self.target_transform(target)
      return img, target

  def __len__(self):
    return len(self.data)




def main():

    train_dataset = datasets.CIFAR100(root='data/',
                    train=True,
                    transform=None,
                    download=True)
    test_dataset = datasets.CIFAR100(root='data/',
                    train=False,
                    transform=None,
                    download=True)
    indices = np.random.permutation(len(train_dataset.data))[:TRAIN_SZ]
    torch.save(
      {'data': train_dataset.data[indices], 'targets': np.array(train_dataset.targets)[indices]},
      open(ROOT + f'/cifar100small{TRAIN_SZ}_train.pkl', 'wb'))
    torch.save(
      {'data': test_dataset.data, 'targets': test_dataset.targets},
      open(ROOT + f'/cifar100small_test.pkl', 'wb'))


if __name__  == '__main__':
  
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  main()
