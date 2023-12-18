

import numpy as np

def get_CIFARB_dataset(self):
        
    data = []
    for index in range(len(self.data)):
        data.append(modify_psd(self.data[index]))
    return np.array(data)


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    center = np.array([16,16])
    
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

def modify_psd(image):
   
    epsilon = 1e-8
    tmp = np.zeros_like(image)
    for i in range(3):
        f1 = np.fft.fft2(image[:,:,i])
        fshift = np.fft.fftshift(f1)
        
        f_cal = fshift + epsilon
        magnitude_spectrum = np.abs(f_cal)
        max_index = (magnitude_spectrum==np.max(magnitude_spectrum))
        magnitude_spectrum[max_index] = 0
        psd1D, r_matrix = azimuthalAverage(magnitude_spectrum)
        
        weight = log_func(psd1D, r_matrix)

        weight[max_index] = 1
        fshift *= weight

        new_f1 =  np.fft.ifftshift(fshift)
        new_f1 = np.fft.ifft2(new_f1)

        tmp[:,:,i] = np.uint8(np.clip(np.real(new_f1),0,255))

    return tmp