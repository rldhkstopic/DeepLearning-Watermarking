import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from models.HidingRes import HidingRes
from skimage.transform import resize
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

def crop(image, types):
    h, w, _ = image.shape

    if types == 'random':
        crop_size = np.random.randint(64, min(h, w))
        start_x = np.random.randint(0, w - crop_size)
        start_y = np.random.randint(0, h - crop_size)
        return image[start_y:start_y+crop_size, start_x:start_x+crop_size]

    elif types == 'center':
        center_x, center_y = w // 2, h // 2
        crop_size = min(h, w) // 2
        return image[center_y-crop_size//2:center_y+crop_size//2, center_x-crop_size//2:center_x+crop_size//2]

    elif types == 'corner':
        corner = np.random.choice(['top_left', 'top_right', 'bottom_left', 'bottom_right'])
        crop_size = min(h, w) // 2
        if corner == 'top_left':
            return image[:crop_size, :crop_size]
        elif corner == 'top_right':
            return image[:crop_size, -crop_size:]
        elif corner == 'bottom_left':
            return image[-crop_size:, :crop_size]
        else:  # bottom_right
            return image[-crop_size:, -crop_size:]

    elif types == 'variable':
        crop_size = np.random.randint(64, min(h, w))
        start_x = np.random.randint(0, w - crop_size)
        start_y = np.random.randint(0, h - crop_size)
        return image[start_y:start_y+crop_size, start_x:start_x+crop_size]

    else:
        raise ValueError("Unknown crop type specified")
    
    
def netR(img, cropped=False): 
    if cropped:
        img = crop(img, cropped)
    img_tentsor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).cuda()
    rev_secret = Rnet(img_tentsor).squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    
    return resize(rev_secret, (256, 256))
    
def getImage(image, row_index, col_index=0):
    size = 256
    rstart = size * row_index + row_index +1
    rend = size * (row_index + 1) + row_index + 1
    
    cstart = size * col_index + col_index +1
    cend = size * (col_index + 1) + col_index + 1
    return image[rstart:rend, cstart:cend]

def ImageLoad(image_path, col_index):
    image = imread(image_path)

    secret = getImage(image, 0, col_index)
    cover_img = getImage(image, 1, col_index)
    clean_img = getImage(image, 2, col_index)
    diff_img = getImage(image, 3, col_index)
    container_img = getImage(image, 4, col_index)
    rev_secret = getImage(image, 5, col_index)
    
    cover_img = resize(cover_img, (256, 256))
    container_img = resize(container_img, (256, 256))
    secret = resize(secret, (256, 256))
    rev_secret = resize(rev_secret, (256, 256))
    
    return cover_img, container_img, clean_img, diff_img, secret, rev_secret

def process_data(image_path, col_index, netR, crop=False):
    cover_img, container_img, clean_img, _, secret, rev_secret = ImageLoad(image_path, col_index)
    rev_container_secret = netR(container_img, crop)
    img_metrics = calculate_metrics(cover_img, container_img, secret, rev_secret, rev_container_secret)
    return cover_img, container_img, clean_img, secret, rev_secret, rev_container_secret, img_metrics

def calculate_metrics(cover_img, container_img, secret, rev_secret, rev_container_secret):
    # 윈도우 크기 설정 (SSIM, PSNR)
    min_side = min(cover_img.shape[0], container_img.shape[1])
    win_size = min_side if min_side % 2 == 1 else min_side - 1
    
    # SSIM, PSNR 계산
    psnr_img = PSNR(cover_img, container_img)
    ssim_img = SSIM(cover_img, container_img, win_size=win_size, multichannel=True, channel_axis=2, data_range=1)
    psnr_sec = PSNR(secret, rev_secret)
    ssim_sec = SSIM(secret, rev_secret, win_size=win_size, multichannel=True, channel_axis=2, data_range=1)
    psnr_rnet = PSNR(secret, rev_container_secret)
    ssim_rnet = SSIM(secret, rev_container_secret, win_size=win_size, multichannel=True, channel_axis=2, data_range=1)

    return psnr_img, ssim_img, psnr_sec, ssim_sec, psnr_rnet, ssim_rnet


hostname = "gold"
runDate = 1204
runTime = "19_24_51"
dirName = f"{hostname}__{runDate}-{runTime}"       #"gold__1203-10_29/"

train_path = "./outputs/trainPics/" + dirName
valid_path = "./outputs/validationPics/" + dirName
Rnet_path = './outputs/checkPoints/gold__1203-10_29/netR_epoch_21,sumloss=24.817436,Rloss=22.630165.pth'

idx = 1
path_index = [train_path, valid_path]
target_path = path_index[idx]

epochs = []
batches = []
image_names = []
for filename in os.listdir(target_path):
    if filename.endswith(".png"):
        image_names.append(filename)
for name in image_names:
    epoch, batch = name.split("_")[1:3]
    epochs.append(epoch)
    batches.append(batch)
image_names = sorted([filename for filename in os.listdir(target_path) if filename.endswith(".png")])
unique_epochs = sorted(list(set(epochs)))
unique_batches = ["0000", "0500", "1100", "1700"] if target_path == train_path else ["0000", "0050", "0150", "0200"]

Rnet = HidingRes(in_c=3, out_c=3).cuda() 
Rnet.load_state_dict(torch.load(Rnet_path))
Rnet = torch.nn.DataParallel(Rnet)
Rnet.eval()

crop_types = ['random', 'center', 'corner', 'variable']

fig = plt.figure(figsize=(15, 3 * len(unique_epochs)))
# fig.suptitle(f"{target_path.split('/')[2]}", y=1.0)
for epoch_idx, epoch in enumerate(unique_epochs):
    for batch_idx, batch in enumerate(unique_batches):
        image_name = f"ResultPics_{epoch}_batch{batch}.png"
        image_path = os.path.join(target_path, image_name)
        if os.path.exists(image_path):                        
            crop_set = 3
            concat_list = []
            for col_index in range(4):
                cover_img, container_img, \
                    clean_img, secret, \
                        rev_secret, rev_container_secret, img_metrics\
                            = process_data(image_path, col_index, netR, crop_types[crop_set])
            
                secret = resize(crop(secret, crop_types[crop_set]), (256, 256))
                concat = np.concatenate((secret, rev_container_secret), axis=0)
                concat_list.append(concat)
            
            concat_image = np.concatenate((concat_list[0], concat_list[1]), axis=1)
            concat_image = np.concatenate((concat_image, concat_list[2]), axis=1)
            concat_image = np.concatenate((concat_image, concat_list[3]), axis=1)
            
            concat_image = (concat_image - np.min(concat_image)) / (np.max(concat_image) - np.min(concat_image))
            
            image_metrics = [('I-PSNR', img_metrics[0]), ('I-SSIM', img_metrics[1]), 
                                ('S-PSNR', img_metrics[2]), ('S-SSIM', img_metrics[3]),
                                ('R-PSNR', img_metrics[4]), ('R-SSIM', img_metrics[5])]
            
            ax = fig.add_subplot(len(unique_epochs), len(unique_batches), epoch_idx * len(unique_batches) + batch_idx + 1)
            ax.imshow(concat_image)
            ax.axis('off')
            ax.set_title(f"{epoch} : batch{batch}")
