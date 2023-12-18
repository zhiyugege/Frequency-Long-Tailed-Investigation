# [NeurIPS-23] Pytorch Implementation of [Revisiting Visual Model Robustness: A Frequency Long-Tailed Distribution View](https://github.com/zhiyugege/Frequency-Long-Tailed-Investigation/edit/master/README.md)

- [Paper link](https://openreview.net/pdf?id=eE5L1RkxW0)

- [Supplementary material](https://openreview.net/attachment?id=eE5L1RkxW0&name=supplementary_material)

- If you have any questions about the paper or code, please [contact us](zyllin@bjtu.edu.cn)

## Environments
- pytorch
- numpy
- autoattack

## Visualization of analytical experiments

- The `Visualize.ipynb` file contains the model's under-fitting Behavior visualization and sensitivity analysis experimental code for high-frequency components.
- We provide the data files required for visualization in the link, including input gradients under standard training and sampling results of the model loss space.

## How to use Balance Spectrum Sampling（BaSS）strategy
We provide two methods of using BaSS in the data preprocessing stage:
- for *CIFAR10* or *CIFAR100* dataset
```
train_set = torchvision.datasets.CIFAR10(root='/data/cifar', train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='/data/cifar', train=False, download=True)

print("CIFAR-B dataset :)")

train_set.data = get_BaSS_dataset(train_set.data.copy())
test_set.data = get_BaSS_dataset(test_set.data.copy())
```
We provide the algorithm details of the **get_BaSS_dataset()** function in the `BaSS_cifar.py` file

- for *ImageNet* dataset
```
TRAIN_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.Resize((128,128)),

        ## applying BaSS 
        FreqLog(),
        ##

        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        Lighting(0.05, IMAGENET_PCA['eigval'], 
                      IMAGENET_PCA['eigvec'])
    ])

TEST_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.Resize((128,128)),

        ## applying BaSS
        FreqLog(),
        ##

        transforms.ToTensor(),
    ])
```
We provide the algorithm details of the **FreqLog()** function in the `BaSS_imagegnet.py` file

## Examples of BaSS Working in Conjunction with Adversarial Training

Our adversarial training framework is based on [Bag of Tricks for Adversarial Training](https://openreview.net/forum?id=Xb8xvrtB8Ce) (ICLR 2021, [code](https://github.com/P2333/Bag-of-Tricks-for-AT)) and [MART](https://openreview.net/forum?id=rklOg6EFwS)(ICLR 2020, [code](https://github.com/YisenWang/MART)).

### Training
*We only modified the data processing part based on the original code framework.* For detailed code description, please refer to the readme file in the original repo. In order to train quickly, we edited the `easy_train.py` file to include the training script.
```
cd Bag-of-Tricks-for-AT
python easy_train.py

cd MART
python easy_train.py
```
We set the `--f-train` switch in the training script whether to use BaSS, for example
```
## standard training
script1 = 'CUDA_VISIBLE_DEVICES=0 python train.py --model resnet --dataset cifar10'

## training with BaSS
script2 = 'CUDA_VISIBLE_DEVICES=1 python train.py --model resnet --dataset cifar10 --f-train'
```

### Evaluation

Our evaluation results are based on **Clean acc**, **PGD(CE loss,CW loss)**,**Autoattack**.
```
cd Bag-of-Tricks-for-AT
python eval_cifar.py --out-dir 'path_to_the_model' --ATmethods 'TRADES' --f-test

cd MART
python test.py --dataset cifar10 --model resnet --f-test
```

