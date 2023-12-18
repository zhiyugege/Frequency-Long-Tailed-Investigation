import os

script1 = 'CUDA_VISIBLE_DEVICES=0 python train.py --model resnet --dataset cifar10'
script2 = 'CUDA_VISIBLE_DEVICES=1 python train.py --model resnet --dataset cifar10 --f-train'

script3 = 'CUDA_VISIBLE_DEVICES=2 python train.py --model resnet --dataset cifar100'
script4 = 'CUDA_VISIBLE_DEVICES=3 python train.py --model resnet --dataset cifar100 --f-train'
    # script = 'CUDA_VISIBLE_DEVICES=5 python train_resnet.py --f-train --f-test --exp c100_'+name
    
    
os.system(script1)