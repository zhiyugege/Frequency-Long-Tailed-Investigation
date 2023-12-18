"""
Module responsible for data augmentation constants and configuration.
"""

import torch as ch
from torchvision import transforms
import numpy as np
from PIL import Image

# lighting transform
# https://git.io/fhBOc

class FreqLog(ch.nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self,img):
        
        img = np.array(img)
        img = ch.tensor(img)
        spec = ch.fft.fftshift(ch.fft.fft2(img,dim=(0,1)),dim=(0,1))
        epsilon = 1e-8
        tmp = ch.zeros_like(spec)
        for i in range(3):
           
            fshift = spec[:,:,i].numpy()
            f_cal = fshift + epsilon
            magnitude_spectrum = np.abs(f_cal)
            max_index = (magnitude_spectrum==np.max(magnitude_spectrum))
            magnitude_spectrum[max_index] = 0
            psd1D, r_matrix = azimuthalAverage(magnitude_spectrum)
            weight = log_func(psd1D, r_matrix)
            weight[max_index] = 1
            fshift *= weight
            tmp[:,:,i] = ch.tensor(fshift)

        img = ch.real(ch.fft.ifft2(ch.fft.ifftshift(tmp,dim=(0,1)),dim=(0,1)))
        img = img.numpy()
        img = np.uint8(np.clip(img,0,255))
        return Image.fromarray(img)

def azimuthalAverage(image):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    center_size = image.shape[0]
    # if not center:
    #     center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    center = np.array([int(center_size/2),int(center_size/2)])
    
    r = np.hypot(x - center[0], y - center[1])
    save_matrix = r.astype(int)
    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]
    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)
    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0] # location of changed radius

    nr = rind[1:] - rind[:-1]        # number of radius bin
    # Cumulative sum to figure out sums for each radius bin
   
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]
    radial_prof = tbin / nr
    
    return radial_prof,save_matrix


def log_func(img2_psd, r_matrix):
    old_energy = img2_psd
    new_energy = np.log(img2_psd)
    old_ratio = old_energy/np.sum(old_energy)
    new_ratio = new_energy/np.sum(new_energy)
    update_ratio = new_ratio/old_ratio

    weight = np.ones_like(r_matrix).astype(float)
    for i in range(len(update_ratio)):
        weight[r_matrix==i+1] = update_ratio[i]

    return weight


IMAGENET_PCA = {
    'eigval':ch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec':ch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}
class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


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
