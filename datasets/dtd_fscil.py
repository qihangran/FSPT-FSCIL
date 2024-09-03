import math
import os
import random
import os.path as osp

from dassl.utils import listdir_nohidden

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing

CLASSES  = ['banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crosshatched', 'crystalline',
                  'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced', 'knitted',
                  'lacelike', 'lined', 'marbled', 'matted', 'meshed', 'paisley', 'perforated', 'pitted', 'pleated', 'polka_dotted', 'porous',
                  'potholed', 'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified', 'striped', 'studded', 'swirly', 'veined',
                  'waffled', 'woven', 'wrinkled', 'zigzagged']



@DATASET_REGISTRY.register()
class DescribableTextures_FSCIL(DatasetBase):


    dataset_dir = "dtd_fscil"

    def __init__(self, cfg, base_sess=None):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train_dir = osp.join(self.dataset_dir, "train-session"+str(cfg.SESSION))
        test_dir = osp.join(self.dataset_dir, "test-session"+str(cfg.SESSION))

        # assert cfg.DATASET.NUM_LABELED > 0

        train_x, train_u, val = self._read_data_train(
            train_dir, cfg.DATASET.NUM_LABELED, cfg.DATASET.VAL_PERCENT, cfg.SESSION
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

    def _read_data_train(self, data_dir, num_labeled, val_percent, session=0):
        class_names = listdir_nohidden(data_dir)
        class_names.sort()
        num_labeled_per_class = num_labeled / len(class_names)
        items_x, items_u, items_v = [], [], []

        base_classes = 32
        nways = 3

        for label, class_name in enumerate(class_names):
            #label_r = int(class_name)
            #label_r = int(class_name.split(".")[0])-1
            # if session == 0:
            #     label_r = label
            # else:
            #     label_r = base_classes + session*nways + label

            label_r = ((session - 1) >= 0) * base_classes + max(0, (session-1))*nways + label

            class_dir = osp.join(data_dir, class_name)
            imnames = listdir_nohidden(class_dir)

            # Split into train and val following Oliver et al. 2018
            # Set cfg.DATASET.VAL_PERCENT to 0 to not use val data
            num_val = math.floor(len(imnames) * val_percent)
            imnames_train = imnames[num_val:]
            imnames_val = imnames[:num_val]

            # Note we do shuffle after split
            random.shuffle(imnames_train)

            for i, imname in enumerate(imnames_train):
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label_r, classname=CLASSES[label_r])

                # if (i + 1) <= num_labeled_per_class:
                #     items_x.append(item)
                #
                # else:
                #     items_u.append(item)
                items_x.append(item)

            for imname in imnames_val:
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label_r)
                items_v.append(item)

        return items_x, items_u, items_v

    def _read_data_test(self, data_dir, session=0):
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




if __name__ == '__main__':
    dir = '/home/qihangran/git_project/constrained-FSCIL-main-IT/src/data/CUB_200_2011/test'
    names = os.listdir(dir)
    names.sort()
    names_new = [i[4:] for i in names]
    print(names_new)
    a = 0