import sys
import os.path as osp
from torchvision.datasets import CIFAR100

from dassl.utils import mkdir_if_missing
from PIL import Image
#import pylibjpeg

# def save_lossless(image, filename):
#     image.save(filename, format='JPEG', subsampling=pylibjpeg.SUSP_NONE, quality=100)

def extract_and_save_image(dataset, save_dir, session):
    if osp.exists(save_dir):
        print('Folder "{}" already exists'.format(save_dir))
        return

    print('Extracting images to "{}" ...'.format(save_dir))
    mkdir_if_missing(save_dir)

    # session
    txt_path_list = []
    txt_path = "./index_list/cifar100/session_" + str(session + 1) + '.txt'
    txt_path_list.append(txt_path)
    class_index = open(txt_path).read().splitlines()
    int_list = list(map(int, class_index))

    for i in range(len(dataset)):
        if i not in int_list:
            continue
        img, label = dataset[i]

        # #---test-----
        # import numpy as np
        # img1 = np.array(img)

        class_dir = osp.join(save_dir, str(label).zfill(3))
        mkdir_if_missing(class_dir)
        impath = osp.join(class_dir, str(i + 1).zfill(5) + ".png")
        img.save(impath, format='PNG')
        #save_lossless(img, impath)

        # #--test--
        # kkk = Image.open(impath)
        # img2 = np.array(kkk)
        # a = 0


def download_and_prepare(name, root, session):
    print("Dataset: {}".format(name))
    print("Root: {}".format(root))

   
    if name == "cifar100":
        train = CIFAR100(root, train=True, download=True)
        test = CIFAR100(root, train=False)
    else:
        raise ValueError

    train_dir = osp.join(root, name, "train-session" + str(session))
    test_dir = osp.join(root, name, "test")

    extract_and_save_image(train, train_dir, session)
    #extract_and_save_image(test, test_dir)


if __name__ == "__main__":
    download_and_prepare("cifar100", ".", session=8)
