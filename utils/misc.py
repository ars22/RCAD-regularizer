import csv
import numpy as np
from pyhessian import hessian # Hessian computation
import time
from torchvision import transforms
import torch



def calc_max_eigenval_hessian(train_loader, model, criterion):
    print("Computing max eigen val ...")
    st = time.time()
    train_iterator = iter(train_loader)
    hessian_dataloader = []
    for _ in range(10):
        hessian_dataloader.append(next(train_iterator))
    hessian_comp = hessian(model, criterion, dataloader=hessian_dataloader, cuda=True)
    max_eigenval = hessian_comp.eigenvalues(top_n=1)[0][0]
    print("Time taken : %.2f" % (time.time() - st))
    model.train()
    return max_eigenval
    


def calc_entropy(input_tensor, reduction='mean'):
    lsm = torch.nn.LogSoftmax(dim=-1)
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(axis=1)
    if reduction == 'mean':
      return entropy.mean()
    elif reduction == 'sum':
      return entropy.sum()
    elif reduction == 'none':
      return entropy
    else:
      raise NotImplementedError



def compute_grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm



def set_track_bn_stats(model, track_bn_stats):
    for name, mod in model.named_modules():
        if 'bn' in name:
            mod.track_running_stats = track_bn_stats
    return model



def str2bool(x):
    return x.lower() == "true"




class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv'):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()
