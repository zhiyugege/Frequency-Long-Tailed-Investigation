# import numpy as np

# clean = []
# robust = []
# for i in range(5):
#     file = np.load("./log/resnet_log_exp"+str(i)+"_train_stats.npy")
#     clean.append(file[0][-1])
#     robust.append(file[1][-1])

# clean = np.array(clean)
# robust = np.array(robust)

# print(np.mean(clean),np.std(clean))
# print(np.mean(robust),np.std(robust))


import argparse
import copy
import logging
from resnet import *
from wideresnet import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from BaSS import get_CIFARB_dataset
from autoattack import AutoAttack
# installing AutoAttack by: pip install git+https://github.com/fra31/auto-attack

from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

upper_limit, lower_limit = 1, 0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def get_loaders(dir_, batch_size, DATASET='cifar10', _type=True):

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if DATASET == 'cifar10':
        test_dataset = datasets.CIFAR10(
            dir_, train=False, transform=test_transform, download=True)
    elif DATASET == 'cifar100':
        test_dataset = datasets.CIFAR100(
            dir_, train=False, transform=test_transform, download=True)

    if _type:
        test_dataset.data = get_CIFARB_dataset(test_dataset.data.copy())

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
    )
    return test_loader

def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, use_CWloss=False, normalize=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            if use_CWloss:
                loss = CW_loss(output, y)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = torch.clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(normalize(X + delta)), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts, eps=8, step=2, use_CWloss=False, normalize=None):
    epsilon = eps / 255.
    alpha = step / 255.
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, use_CWloss=use_CWloss, normalize=normalize)
        with torch.no_grad():
            output = model(normalize(X + pgd_delta))
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n


def evaluate_standard(test_loader, model, normalize=None):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(normalize(X))
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--data-dir', default='/data/cifar', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--model',default='resnet', type=str) 
    parser.add_argument('--dataset',default='cifar10', type=str) 
    parser.add_argument('--f-test', action='store_true')
    parser.add_argument('--exp',default='epoch110.pt', type=str) 
    return parser.parse_args()


def main():
    args = get_args()

    DATASET = args.dataset
    MODEL_TYPE = args.model
    FTEST = args.f_test
    DATA_ROOT = "/data/cifar/"

    if DATASET=='cifar10':
        NUM_CLASS=10
    elif DATASET=='cifar100':
        NUM_CLASS=100

    if FTEST:
        SIGN = '_BaSS'
    else:
        SIGN = ''

    if args.exp:
        SIGN = SIGN + '_' + args.exp

# settings
    model_dir = MODEL_TYPE+'_'+DATASET+SIGN

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler()
        ])

    logger.info(args)

    test_loader = get_loaders(DATA_ROOT, args.batch_size, _type=FTEST)
    best_state_dict = torch.load('./'+model_dir+'/model-res-best.pt')
    

    if MODEL_TYPE=='resnet':
        model_test =ResNet18(num_classes=NUM_CLASS).cuda()
    else:
        model_test = WideResNet(num_classes=NUM_CLASS).cuda()

    if 'state_dict' in best_state_dict.keys():
        model_test.load_state_dict(best_state_dict['state_dict'])
    else:
        model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()


    ### Evaluate clean acc ###
    _, test_acc = evaluate_standard(test_loader, model_test)
    print('Clean acc: ', test_acc)

    ### Evaluate PGD (CE loss) acc ###
    _, pgd_acc_CE = evaluate_pgd(test_loader, model_test, attack_iters=20, restarts=1, eps=8, step=2, use_CWloss=False)
    print('PGD-10 (10 restarts, step 2, CE loss) acc: ', pgd_acc_CE)

    ### Evaluate PGD (CW loss) acc ###
    _, pgd_acc_CW = evaluate_pgd(test_loader, model_test, attack_iters=20, restarts=1, eps=8, step=2, use_CWloss=True)
    print('PGD-10 (10 restarts, step 2, CW loss) acc: ', pgd_acc_CW)

    ### Evaluate AutoAttack ###
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    class normalize_model():
        def __init__(self, model):
            self.model_test = model
        def __call__(self, x):
            return self.model_test(x)
    new_model = normalize_model(model_test)
    epsilon = 8 / 255.
    adversary = AutoAttack(new_model, norm='Linf', eps=epsilon, version='standard')
    X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)

if __name__ == "__main__":
    main()