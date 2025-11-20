import sys
import os.path as osp
from torchvision.datasets import CIFAR100

from dassl.utils import mkdir_if_missing
import shutil
from PIL import Image
#import pylibjpeg

# def save_lossless(image, filename):
#     image.save(filename, format='JPEG', subsampling=pylibjpeg.SUSP_NONE, quality=100)

def extract_and_save_image(save_dir, session):
    if osp.exists(save_dir):
        print('Folder "{}" already exists'.format(save_dir))
        return

    print('Extracting images to "{}" ...'.format(save_dir))
    mkdir_if_missing(save_dir)

    # session
    txt_path_list = []
    txt_path = "./index_list/cub200/session_" + str(session + 1) + '.txt'
    txt_path_list.append(txt_path)
    class_index = open(txt_path).read().splitlines()
    #int_list = list(map(int, class_index))

    for name in class_index:
        a = 0
        label = name.split("/")[2]
        file_name = name.split("/")[3]
        class_dir = osp.join(save_dir, label)
        mkdir_if_missing(class_dir)
        new_file_name = osp.join(save_dir, label, file_name)
        shutil.copy2(name, new_file_name)
        # impath = osp.join(class_dir, str(i + 1).zfill(5) + ".jpg")
        # img.save(impath, format='PNG')
        #save_lossless(img, impath)

        # #--test--
        # kkk = Image.open(impath)
        # img2 = np.array(kkk)
        # a = 0


def download_and_prepare(name, root, session):
    print("Dataset: {}".format(name))
    print("Root: {}".format(root))

   
    # if name == "cifar100":
    #     train = CIFAR100(root, train=True, download=True)
    #     test = CIFAR100(root, train=False)
    # else:
    #     raise ValueError

    train_dir = osp.join(root, name, "train-session" + str(session))
    test_dir = osp.join(root, name, "test")

    extract_and_save_image(train_dir, session)
    #extract_and_save_image(test, test_dir)


if __name__ == "__main__":
    download_and_prepare("miniimagenet", "/home/qihangran/git_project/constrained-FSCIL-main-IT/src/data", session=0)
