import os 

script1 = "CUDA_VISIBLE_DEVICES=0 python train_cifar.py\
        --model ResNet18 \
        --attack pgd \
        --lr-schedule piecewise \
        --norm l_inf \
        --epsilon 8 \
        --epochs 110 --attack-iters 10 --pgd-alpha 2 \
        --fname cifar10_resnet18_advTrain_inf \
		--optimizer 'momentum' \
	    --weight_decay 5e-4 \
        --batch-size 128 \
		--BNeval"

script2 = "CUDA_VISIBLE_DEVICES=1 python train_cifar.py\
        --model ResNet18 \
        --attack pgd \
        --lr-schedule piecewise \
        --norm l_inf \
        --epsilon 8 \
        --epochs 110 --attack-iters 10 --pgd-alpha 2 \
        --fname cifar10B_resnet18_advTrain_inf \
		--optimizer 'momentum' \
	    --weight_decay 5e-4 \
        --batch-size 128 \
		--BNeval \
       --f-train"

script3 = "CUDA_VISIBLE_DEVICES=2 python train_cifar.py\
        --model WideResNet \
        --attack pgd \
        --lr-schedule piecewise \
        --norm l_inf \
        --epsilon 8 \
        --epochs 110 --attack-iters 10 --pgd-alpha 2 \
        --fname cifar10_WRN34_advTrain_inf \
		--optimizer 'momentum' \
	    --weight_decay 5e-4 \
        --batch-size 128 \
		--BNeval"

script4 = "CUDA_VISIBLE_DEVICES=3 python train_cifar.py\
        --model WideResNet \
        --attack pgd \
        --lr-schedule piecewise \
        --norm l_inf \
        --epsilon 8 \
        --epochs 110 --attack-iters 10 --pgd-alpha 2 \
        --fname cifar10B_WRN34_advTrain_inf \
		--optimizer 'momentum' \
	    --weight_decay 5e-4 \
        --batch-size 128 \
		--BNeval \
        --f-train"

os.system(script1)

