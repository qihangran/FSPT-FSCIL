import math
import random
import os.path as osp

from dassl.utils import listdir_nohidden

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing

CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly',
    'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
    'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
    'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',

    'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew',
    'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe',
    'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


@DATASET_REGISTRY.register()
class CIFAR10(DatasetBase):
    """CIFAR10 for SSL.

    Reference:
        - Krizhevsky. Learning Multiple Layers of Features
        from Tiny Images. Tech report.
    """

    dataset_dir = "cifar10"

    def __init__(self, cfg, base_sess=None):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train_dir = osp.join(self.dataset_dir, "train")
        test_dir = osp.join(self.dataset_dir, "test")

        # assert cfg.DATASET.NUM_LABELED > 0

        train_x, train_u, val = self._read_data_train(
            train_dir, cfg.DATASET.NUM_LABELED, cfg.DATASET.VAL_PERCENT, cfg.BASE_SESSION
        )
        test = self._read_data_test(test_dir)

        if cfg.DATASET.ALL_AS_UNLABELED:
            train_u = train_u + train_x

        if len(val) == 0:
            val = None

        #-------------------------------------------------------------------------
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            train_x = self.generate_fewshot_dataset(train_x, num_shots=num_shots)
            val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
            data = {"train": train_x, "val": val}
        #--------------------------------------------------------------------------------------

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    def _read_data_train(self, data_dir, num_labeled, val_percent, base_session=None):
        class_names = listdir_nohidden(data_dir)
        class_names.sort()
        num_labeled_per_class = num_labeled / len(class_names)
        items_x, items_u, items_v = [], [], []

        if base_session:
            class_names = class_names[:60]

        for label, class_name in enumerate(class_names):
            class_dir = osp.join(data_dir, class_name)
            imnames = listdir_nohidden(class_dir)

            # Split into train and val following Oliver et al. 2018
            # Set cfg.DATASET.VAL_PERCENT to 0 to not use val data
            num_val = math.floor(len(imnames) * val_percent)
            #num_val = math.floor(len(imnames) * 0)
            imnames_train = imnames[num_val:]
            imnames_val = imnames[:num_val]

            # Note we do shuffle after split
            random.shuffle(imnames_train)

            for i, imname in enumerate(imnames_train):
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=CLASSES[label])

                # if (i + 1) <= num_labeled_per_class:
                #     items_x.append(item)
                #
                # else:
                #     items_u.append(item)
                items_x.append(item)

            for imname in imnames_val:
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label)
                items_v.append(item)

        return items_x, items_u, items_v

    def _read_data_test(self, data_dir):
        class_names = listdir_nohidden(data_dir)
        class_names.sort()
        items = []

        for label, class_name in enumerate(class_names):
            class_dir = osp.join(data_dir, class_name)
            imnames = listdir_nohidden(class_dir)

            for imname in imnames:
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=CLASSES[label])
                items.append(item)

        return items


@DATASET_REGISTRY.register()
class CIFAR100(CIFAR10):
    """CIFAR100 for SSL.

    Reference:
        - Krizhevsky. Learning Multiple Layers of Features
        from Tiny Images. Tech report.
    """

    dataset_dir = "cifar100"

    def __init__(self, cfg):
        super().__init__(cfg)
