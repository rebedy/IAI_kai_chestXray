##############################
# 0. Python module importing #
##############################

from os.path import join
import datetime
import random
from tqdm import tqdm
import pickle
import numpy as np
# =======================================#
import cv2
from PIL import Image
from sklearn.model_selection import StratifiedKFold

import albumentations as A
### Albumation Default
''' Compose p=1.0, Brightness p=0.5, CLAHE p=0.5 
    A.Compose(transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1.0)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,
                               brightness_by_max=True, always_apply=False, p=0.5)
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5)

    https://albumentations.readthedocs.io/en/latest/api/augmentations.html
'''
# -------------------------------------------------------------- #

from step0_config_setting import *
from data_io.misc_DY import *
from data_io.data_io_CXR_DataEntry import *


#########################
# 1. Parameters Setting #
#########################

GPU_NUMBER = str(GPU_NUMBER)
TODAY = str(datetime.date.today().strftime('%y%m%d'))
NOW = str(datetime.datetime.now().strftime('_%Hh%Mm'))
# -------------------------------------------------------------- #

KFOLD_DIR = makedirs_ifnot(join(DATASET_DIR, SUBJECT_NAME + '-' + TODAY + '_pickled'))
PICKLE_LOG_FILE = (SUBJECT_NAME + '_pickling_' + TODAY
                   + str(datetime.datetime.now().strftime('_%Hh%Mm')) + '.txt')


###############
### 2. MAIN ###
###############

log_text = open(join(KFOLD_DIR, PICKLE_LOG_FILE), 'a')

print("\n=====================================")
print("= Getting and Matching Data Info!!! =")
print("=====================================\n")
img_path_list = get_img_path_list(DATASET_DIR, log_text)
file_info_list = get_data_entry_excel_info(LABEL_FILE, log_text)
img_name_list, img_dir_list, label_list = match_pathNinfo(file_info_list, img_path_list, log_text)
print("======================================================================\r")
print('\n')

time_start = datetime.datetime.now()
print("     @ Pickling starts at " + str(time_start))
print("     @ Pickling starts at " + str(time_start) + '\r', file=log_text)

print("\n==============================")
print("= Stratified K-Fold start!!! =")
print("==============================\r")
print('\n\n\r', file=log_text)
print('[ Stratified K-Fold ]\r\n', file=log_text)

data_fold_splitter = StratifiedKFold(n_splits=NUMBER_OF_FOLD, shuffle=True)  # , random_state=10)
print('     Stratified splitting into %d Fold!\n' % NUMBER_OF_FOLD)
print('     Stratified splitting into %d Fold!\r\n' % NUMBER_OF_FOLD, file=log_text)
print('\n\n\r', file=log_text)

count_kfold = 1

for _, idx_list in data_fold_splitter.split(X=img_dir_list, y=label_list):
    print('\r')
    print("===================== %s th FOLD START! =====================\r" % str(count_kfold))
    print("===================== %s th FOLD START! =====================\r" % str(count_kfold), file=log_text)
    fold_start = datetime.datetime.now()
    print("     @ FOLD Start at " + str(fold_start) + '\r')
    print("     @ FOLD Start at " + str(fold_start) + '\r', file=log_text)
    print('\n\r')
    print('\n\r', file=log_text)

    fold_x = [img_dir_list[j] for j in idx_list]
    fold_y = [label_list[i] for i in idx_list]
    fold_name = [img_name_list[i][:-4] for i in idx_list]  # = [img_dir_list[i].split('/')[-1][:-4] for i in idx_list]
    print('   >>> FOLD length : ', len(fold_x))
    print('   >>> FOLD length : ', len(fold_x), '\r', file=log_text)
    for i in range(8):
        print('        # of class %d in fold_x : ' % i, fold_y.count(i))
        print('        # of class %d in fold_x : ' % i, fold_y.count(i), '\r', file=log_text)

    fold_y = np.array(fold_y)

    print('\r')
    print('\r\n\r', file=log_text)
    print("   >>> Image.open AND Resize Image")
    print("   >>> Image.open AND Resize Image\r", file=log_text)
    # TODO | DY: A method to read as PIL and receive as np
    ''' When resize, to make it smaller with interpolation use cv2.INTER_AREA
    and to make it bigger use cv2.INTER_CUBIC, cv2.INTER_LINEAR '''
    fold_array, tmp_ori_mean = [], 0
    for idx, each_img_path in enumerate(fold_x):
        tmp = np.array(Image.open(each_img_path))
        if len(np.array(tmp).shape) == 2:  # (1024, 1024)
            tmp_ = np.array(cv2.resize(tmp, RESHAPE, interpolation=cv2.INTER_AREA))
        elif np.array(tmp).shape[-1] == 4:
            tmp_ = np.array(cv2.resize(tmp[:, :, 0], RESHAPE, interpolation=cv2.INTER_AREA))
        else:  # len(np.array(tmp).shape) == 3:
            tmp_ = None
            print(" ShapeError DY : %s has different channel size. Please check." % fold_name[idx])

        fold_array.append(tmp_)
        tmp_ori_mean += tmp_

    fold_array = np.array(fold_array)
    print('        fold_array.shape after resizing (N,H,W,C) : ', fold_array.shape)
    print('        fold_array.shape after resizing (N,H,W,C) : ', fold_array.shape, '\r', file=log_text)

    ##### generated folders related to KFOLD #####
    fold_dir = mkdir_ifnot(join(KFOLD_DIR, '%sfold' % str(count_kfold)))

    '''
    ===============================================
     ##### Augmentation Off-Line Pickling!!! #####
    ===============================================
        ㄴ Augmenting Dataset followed by augmentation_option
        ㄴ To execute actual augmentation, set AUGMENTATION = True
    '''
    if AUGMENTATION:
        print('\r')
        print('\n\r', file=log_text)
        print("================================================================\r")
        print("================================================================\r", file=log_text)
        print('   >>> Dumping Augmented fold_array and label to Pickle file. \r')
        print('   >>> Dumping Augmented fold_array and label to Pickle file. \r', file=log_text)

        multiply_by_aug = 2
        print("        multiply_by_aug : ", multiply_by_aug, "\r")
        print("        multiply_by_aug : ", multiply_by_aug, "\r", file=log_text)

        ##### generate folders related to fold_aug! #####
        fold_aug_dir = mkdir_ifnot(join(fold_dir, '%s_times_aug' % str(multiply_by_aug)))

        ##### placeholder to store augmented data ####
        aug_array, aug_y, aug_name = [], [], []
        tmp_aug_mean = 0

        ##### Augmentation instance lists for Randomizing! #####
        p_list, CLAHE_clip_list = [1.0, 0.5], [2, 4, 6]
        brightness_list, contrast_list = [0.2, 0.3, 0.4], [0.2, 0.3, 0.4]

        ##### Augmentation! #####
        for idx, img in enumerate(tqdm(fold_array)):
            # aug prob list random select
            p_random, CLAHE_clip = random.choice(p_list), random.choice(CLAHE_clip_list)
            brightness, contrast = random.choice(brightness_list), random.choice(contrast_list)

            # aug method
            light = A.Compose([A.RandomBrightnessContrast(brightness_limit=brightness, contrast_limit=contrast, p=1.0),
                               A.CLAHE(clip_limit=CLAHE_clip, p=1.0)])  # , p = p_random)
            augmented = light(image=img, mask=None)['image']

            # aug mean image store
            tmp_aug_mean += img
            tmp_aug_mean += augmented

            # ori and aug data store
            aug_array.append(img)
            aug_y.append(fold_y[idx])
            aug_name.append('%s_aug1_MS_%s' % (fold_name[idx], str(fold_y[idx])))
            aug_array.append(augmented)
            aug_y.append(fold_y[idx])
            aug_name.append('%s_aug2_MS_%s' % (fold_name[idx], str(fold_y[idx])))

        ##### Generating Augmented Mean Image #####
        aug_mean_image = tmp_aug_mean.astype('float32')
        aug_mean_image /= len(aug_array)

        ##### Saving Augmented Mean Image (just to check) #####
        np.save(join(fold_dir, 'fold%d_aug_mean_image' % count_kfold), aug_mean_image)
        pil_aug_mean = Image.fromarray(aug_mean_image)#, 'I')
        # TODO | DY : fromarray 해도 mode가 RGBA이지만 저장에 문제 X
        pil_aug_mean.save(join(fold_dir, 'fold%d_aug_mean_image.tiff' % count_kfold), dpi=(300, 300))

        ##### Augmented Pixel-Wise Mean Subtract #####
        # TODO | DY : Didn't do `.astype(np.uint8))` and adjusted with mode. subtracted PIL Image.
        aug_MS_array = [tmp_aug - aug_mean_image for i, tmp_aug in enumerate(aug_array)]

        print('\r')
        print('\n\r', file=log_text)
        print("==============================================================\r")
        print("==============================================================\r", file=log_text)
        print('   >>> Saving aug_array to .png image file with names. \r')
        print('   >>> Saving aug_array to .png image file with names. \r', file=log_text)
        for idx, aug_MS in enumerate(aug_MS_array):
            pil_aug_MS_im = Image.fromarray(aug_MS)  # , 'I')
            pil_aug_MS_im.save(join(fold_aug_dir, aug_name[idx]+'.tiff'), dpi=(300, 300))

        print('   >>> Dumping Shuffled Augmented aug_array and label to Pickle file. \r')
        print('   >>> Dumping Shuffled Augmented aug_array and label to Pickle file. \r', file=log_text)
        c = list(zip(aug_array, aug_y))
        random.shuffle(c)
        aug_array, aug_y = zip(*c)

        with open(join(fold_dir, 'aug_MS_fold.pickle'), 'wb') as f:
            pickle.dump([aug_MS_array, aug_y], f)

    '''
    ==================================================
    ##### Original MS FOLD Off-Line Pickling!!! #####
    ==================================================
    '''
    if MEAN_SUBTRACTION:
        fold_ori_MS_dir = mkdir_ifnot(join(fold_dir, 'ori_MS'))

        ##### Generating Original Mean Image #####
        ori_mean_image = tmp_ori_mean.astype('float32')
        ori_mean_image /= len(fold_array)

        ##### Saving Original Mean Image #####
        np.save(join(fold_dir, 'fold%d_ori_mean_image' % count_kfold), ori_mean_image)
        pil_ori_mean = Image.fromarray(ori_mean_image)#, 'I')
        # TODO | DY : even though you do `fromarray`, mode is RGBA but no problem with saving.
        pil_ori_mean.save(join(fold_dir, 'fold%d_ori_mean_image.tiff' % count_kfold), dpi=(300, 300))

        ##### Original Pixel-Wise Mean Subtract #####
        # TODO | DY : .astype(np.uint8)) 여기서 이거 안하고 아래서 mode로 맞춰줌, 여기서 PIL 이미지 빼줌.
        ori_MS_array = [tmp_img - ori_mean_image for i, tmp_img in enumerate(fold_array)]

        print('\r')
        print('\n\r', file=log_text)
        print("==============================================================\r")
        print("==============================================================\r", file=log_text)
        print('   >>> Saving ori_MS_array to .png image file with names. \r')
        print('   >>> Saving ori_MS_array to .png image file with names. \r', file=log_text)
        for idx, ori_MS in enumerate(ori_MS_array):
            pil_ori_ms = Image.fromarray(ori_MS)  # , 'I')
            pil_ori_ms.save(join(fold_ori_MS_dir, '%s_ori_MS_%s.tiff' % (fold_name[idx], fold_y[idx])), dpi=(300, 300))

        print('   >>> Dumping Original Mean Subtraction fold_array and label to Pickle file. \r')
        print('   >>> Dumping Original Mean Subtraction fold_array and label to Pickle file. \r', file=log_text)
        with open(join(fold_dir, 'ori_MS_fold.pickle'), 'wb') as f:
                pickle.dump([ori_MS_array, fold_y], f)

        # ===================================================================================== #


    '''
    ===============================================
    ##### Original FOLD Off-Line Pickling!!! #####
    ===============================================
    '''
    if not MEAN_SUBTRACTION and not AUGMENTATION:

        fold_ori_dir = mkdir_ifnot(join(fold_dir, 'ori'))

        print('\r')
        print('\n\r', file=log_text)
        print("==============================================================\r")
        print("==============================================================\r", file=log_text)
        print('   >>> Saving aug_array to .png image file with names. \r')
        print('   >>> Saving aug_array to .png image file with names. \r', file=log_text)
        for idx, ori_fold in enumerate(fold_array):
            pil_im = Image.fromarray(fold_array[idx])  # , 'I')
            # scipy.misc.imsave has been deprecated since SciPy 1.0, and was removed in SciPy 1.2.
            pil_im.save(join(fold_ori_dir, '%s_%s.tiff' % (fold_name[idx], fold_y[idx])), dpi=(300, 300))

        print('   >>> Dumping Original fold_array and label to Pickle file. \r')
        print('   >>> Dumping original fold_array and label to Pickle file. \r', file=log_text)
        with open(join(fold_dir, 'ori_fold.pickle'), 'wb') as f:
            pickle.dump([fold_array, fold_y], f)

        # ===================================================================================== #

    else: pass

    count_kfold += 1
