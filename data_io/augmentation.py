import os
import cv2

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def augment_and_show(aug, image, mask=None, categories=[], label=[], bboxes=[]):

    augmented = aug(image=image, mask=mask, bboxes=bboxes, category_id=categories)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_aug = cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB)

    cv2.imwrite('/home/ubuntu/DATASET_STORAGE/NIH_chestXray_data/chestXray-200429_pickled' + '/' + i + '_'+label, image_aug)
    return augmented['image'], augmented['mask'], augmented['bboxes']


def find_in_dir(dirname):
    return [os.path.join(dirname, fname) for fname in sorted(os.listdir(dirname))]