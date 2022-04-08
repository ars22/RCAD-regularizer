import pdb
import argparse
from tkinter import image_names
import numpy as np
from torch.optim import optimizer
from torch.utils.data import dataset
from tqdm import tqdm
from pyhessian import hessian # Hessian computation
import torch
import time
import torch.nn as nn
from collections import defaultdict
from torch.autograd import Variable, grad_mode
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from utils.misc import CSVLogger
from utils.cutout import Cutout
from models.resnet import ResNet18
from models.wide_resnet import WideResNet
from torchvision.utils import save_image
from data.cifar100small import CIFAR100Small
import random
import os


# seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# argparse
model_options = ['resnet18', 'wideresnet']
dataset_options = ['cifar10', 'cifar100', 'svhn', 'cifar100small2k']
parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar100small2k',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--label_smoothing_factor', type=float, default=0.,
                    help='label smoothing')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')
parser.add_argument('--lr_drops', type=str, default='100,150,180',
                    help='lr drop epochs')
parser.add_argument('--lr_drop_factor', type=float, default=0.1,
                    help='lr drop factor')
parser.add_argument('--save_freq', type=int, default=50,
                    help='save freq')
parser.add_argument('--checkpoint_path', type=str, default='',
                    help='checkpoint_path')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight_decay')
parser.add_argument('--run_id', type=str, default='',
                    help='run id')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start_epoch')
parser.add_argument('--maxent_hf', action='store_true', default=False,
                    help='apply maxent on hf component')
parser.add_argument('--cutoff_freq', type=int, default=0,
                    help='cutoff frequency')
parser.add_argument('--lambd', type=float, default=0.,
                    help='regularizer strength')
parser.add_argument('--alpha', type=float, default=0.,
                    help='adv lr')
parser.add_argument('--epsilon', type=float, default=0.,
                    help='adv eps norm')
parser.add_argument('--attack_iters', type=int, default=0,
                    help='adv attack iters')
parser.add_argument('--eta', type=float, default=10.,
                    help='magnitude of grad')
parser.add_argument('--grad_svd_K', type=int, default=100,
                    help='number of components in grad svd')
args = parser.parse_args()
args.lr_drops = [int(x) for x in args.lr_drops.split(',')]
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = False  
print(args)


# seed fix
set_seed(args.seed)


# test_id
test_id = args.dataset + '_' + args.model + f'_{args.run_id}'


# Image Preprocessing
if args.dataset == 'svhn':
    mean_arr=[x / 255.0 for x in[109.9, 109.7, 113.8]]
    std_arr=[x / 255.0 for x in [50.1, 50.6, 50.8]]
else:
    mean_arr=[x / 255.0 for x in [125.3, 123.0, 113.9]]
    std_arr=[x / 255.0 for x in [63.0, 62.1, 66.7]]
def normalize(X):
    Y = X.clone()
    return transforms.functional.normalize(Y, mean_arr, std_arr, inplace=False)
to_PIL = transforms.ToPILImage()
to_tensor = transforms.ToTensor()


# Augmentation
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
random_crop = transforms.RandomCrop(32, padding=4)
rh_flip = transforms.RandomHorizontalFlip()
cutout = Cutout(n_holes=args.n_holes, length=args.length)
def augment(X):
    Y = X.clone()
    Y = torch.stack([random_crop(y) for y in Y], dim=0)
    Y = torch.stack([rh_flip(y) for y in Y], dim=0)
    if args.cutout:
        Y = torch.stack([cutout(y) for y in Y], dim=0)    
    return Y    


# track BN
def set_track_bn_stats(model, track_bn_stats):
    for name, mod in model.named_modules():
        if hasattr(mod, 'track_running_stats'):
            mod.track_running_stats = track_bn_stats
    return model


# grad norm
def compute_grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


# datasets
if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(root='data/',
                                      train=True,
                                      transform=train_transform,
                                      download=True)

    test_dataset = datasets.CIFAR100(root='data/',
                                     train=False,
                                     transform=test_transform,
                                     download=True)
elif args.dataset == 'cifar100small2k':
    num_classes = 100
    train_dataset = CIFAR100Small(root='data/',
                                      train=True,
                                      transform=train_transform,
                                      train_sz=2000)

    test_dataset = CIFAR100Small(root='data/',
                                     train=False,
                                     transform=test_transform)
elif args.dataset == 'svhn':
    num_classes = 10
    train_dataset = datasets.SVHN(root='data/',
                                  split='train',
                                  transform=train_transform,
                                  download=True)

    extra_dataset = datasets.SVHN(root='data/',
                                  split='extra',
                                  transform=train_transform,
                                  download=True)
    # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
    data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
    labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
    train_dataset.data = data
    train_dataset.labels = labels
    test_dataset = datasets.SVHN(root='data/',
                                 split='test',
                                 transform=test_transform,
                                 download=True)


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)


# model, criterion and optimizer
if args.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
elif args.model == 'wideresnet':
    if args.dataset == 'svhn':
        cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                         dropRate=0.4)
    else:
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)
cnn = cnn.cuda()
cent_loss_with_smoothing = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_factor).cuda()
cent_loss = nn.CrossEntropyLoss().cuda()
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
print(cnn)


# lr schedule
if args.dataset == 'svhn':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
else:
    scheduler = MultiStepLR(cnn_optimizer, milestones=args.lr_drops, gamma=args.lr_drop_factor)
    

# entropy
def calc_entropy(x, from_logits=True):
    if from_logits:
        log_probs = x.log_softmax(dim=-1)
    else:
        log_probs = np.log(x)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    predictive_entropy = -p_log_p.sum(axis=1)
    return predictive_entropy
  

# logging
if not os.path.exists('logs/'):
    os.mkdir('logs/')
filename = 'logs/' + test_id + '.csv'
print("Saving at ", filename)
import os
if os.path.exists(filename):
    os.remove(filename)
    print("Deleting existing ", filename)
metric_keys = [ 
    'epoch', 
    'acc_train',
    'loss_train',
    'cent_loss_train',
    'pred_ent_train',
    'pred_ent_adv_train',
    'acc_test', 
    'cent_loss_test',
    'grad_norm',
    'reg',
    'l2_dist']
metrics = {k:[] for k in metric_keys}
csv_logger = CSVLogger(args=args, fieldnames=metric_keys, filename=filename)


# checkpoints
if args.checkpoint_path != '':
    for _ in range(args.start_epoch):
        scheduler.step()
    print('loading from ', args.checkpoint_path)
    chkpt = torch.load(open(args.checkpoint_path, 'rb'))
    cnn.load_state_dict(chkpt['model'])
    cnn_optimizer.load_state_dict(chkpt['optimizer'])
    cnn.train()


# testing
def test(model, loader, metrics):
    model.eval()    # change model to 'eval' mode (BN uses moving mean/var).
    total_img = 0
    for i, (images, labels) in enumerate(loader):
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            pred = model(normalize(images))
            metrics['cent_loss_test'][-1] += cent_loss(pred, labels).item()
        metrics['acc_test'][-1] += (torch.max(pred.data, 1)[1] == labels).sum().item()        
        total_img += len(labels)
    metrics['acc_test'][-1] /= total_img
    metrics['cent_loss_test'][-1] /= (i+1)
    model.train()


# pgd attack
def attack_pgd(model, X, y):
    delta = torch.zeros_like(X).cuda()
    delta.requires_grad = True
    output = model(normalize(X + delta))
    index = slice(None,None,None)
    loss = cent_loss(cnn(normalize(X+delta)), y) * len(X)
    loss.backward()
    grad = delta.grad.detach()
    grad = delta.grad
    g = grad[index, :, :, :]    
    g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
    scaled_g = g/(g_norm + 1e-10)
    return scaled_g


    
# training
max_ent = - np.log(1./num_classes)
for epoch in range(args.start_epoch, args.epochs):

    for key in metric_keys:
        metrics[key].append(0.)
    metrics['epoch'][-1] = epoch
    progress_bar = tqdm(train_loader)
    progress_bar.set_description('Epoch ' + str(epoch))
    tqdm.write('learning_rate: %.5f' % (cnn_optimizer.param_groups[0]['lr']))
        
    for i, (images, labels) in enumerate(progress_bar):
        
        images = images.cuda()
        labels = labels.cuda()
        aug_images = augment(images)
  
        cnn = set_track_bn_stats(cnn, False)
        z = images
        metrics['pred_ent_train'][-1] += cnn.pred_ent(normalize(z)).mean().item()
        d = attack_pgd(
            model=cnn, X=images, y=labels)    
        metrics['pred_ent_adv_train'][-1] += cnn.pred_ent(normalize(z + args.alpha * d)).mean().item()
        metrics['l2_dist'][-1] += args.alpha * torch.norm(d.view(len(z), -1), dim=1, p=2).mean().item()
        reg =  args.lambd * (max_ent - cnn.pred_ent(normalize(z + args.alpha * d)).mean())
        cnn = set_track_bn_stats(cnn, True)
          
        cnn_optimizer.zero_grad()
        # prediction on aug image
        pred_aug = cnn(normalize(aug_images))
        # prediction on actual image
        pred = cnn(normalize(images))

        xentropy_loss = cent_loss_with_smoothing(pred, labels)
        xentropy_loss_aug = cent_loss_with_smoothing(pred_aug, labels)
        total_xentropy_loss = 0.5 * (xentropy_loss + xentropy_loss_aug)
        total_loss = total_xentropy_loss + reg

        # backprop        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(cnn.parameters(), 1.)
        grad_norm = compute_grad_norm(cnn)
        cnn_optimizer.step()
        
        # metrics        
        metrics['cent_loss_train'][-1] += xentropy_loss.item()
        metrics['loss_train'][-1] += total_loss.item()
        metrics['reg'][-1] += reg.item()
        metrics['grad_norm'][-1] += grad_norm
        
        # Calculate running average of accuracy
        acc = (torch.max(pred.data, 1)[1] == labels.data).sum().item() / len(pred)
        metrics['acc_train'][-1] += acc
        progress_bar.set_postfix(
            loss='%.3f' % total_loss.item(),
            acc='%.3f' % acc,
            gn='%.3f' % grad_norm,
            reg='%.3f' % reg.item())

    # epoch over train
    metrics['cent_loss_train'][-1] /= (i+1)
    metrics['loss_train'][-1] /= (i+1)
    metrics['reg'][-1] /= (i+1)
    metrics['grad_norm'][-1] /= (i+1)
    metrics['pred_ent_train'][-1] /= (i+1)
    metrics['pred_ent_adv_train'][-1] /= (i+1)
    metrics['acc_train'][-1] /= (i+1) 
    metrics['l2_dist'][-1] /= (i+1)

    # run test
    test(cnn, test_loader, metrics)
    scheduler.step()
    
    # saving
    if epoch % args.save_freq == 0:
        if not os.path.exists('checkpoints/'):
            os.mkdir('checkpoints/')
        cnn.eval()
        torch.save({
            'model': cnn.state_dict(),
            'optimizer': cnn_optimizer.state_dict()
        }, 'checkpoints/' + test_id + f'_{epoch}.pt')
        cnn.train()
    
    # logs
    row = {k:('%d' if type(metrics[k][-1]) == int else '%.4f') % metrics[k][-1] for k in metric_keys}
    csv_logger.writerow(row)
    for k in row:
        tqdm.write(f'{k}: {row[k]}')
    tqdm.write('\n')
    
    
torch.save({'model': cnn.state_dict(), 'optimizer': cnn_optimizer.state_dict()}, 'checkpoints/' + test_id + '_last.pt')
csv_logger.close()