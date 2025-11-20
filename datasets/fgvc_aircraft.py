import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class FGVCAircraft(DatasetBase):

    dataset_dir = "fgvc_aircraft"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        classnames = []
        with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, "images_variant_train.txt")
        val = self.read_data(cname2lab, "images_variant_val.txt")
        test = self.read_data(cname2lab, "images_variant_test.txt")

        # 根据session进行val和train的选择
        items_train = []
        items_test = []
        nums_train = [0 for i in range(100)]

        base_classes = [i for i in range(60)]
        classes1 = [i for i in range(60, 65)]
        classes2 = [i for i in range(65, 70)]
        classes3 = [i for i in range(70, 75)]
        classes4 = [i for i in range(75, 80)]
        classes5 = [i for i in range(80, 85)]
        classes6 = [i for i in range(85, 90)]
        classes7 = [i for i in range(90, 95)]
        classes8 = [i for i in range(95, 100)]

        for i in range(len(train)):
            data = train[i]
            label = data.label
            # base classes for all images
            if cfg.SESSION == 0:
                if label in base_classes:
                    items_train.append(data)
            elif cfg.SESSION == 1:
                if label in classes1 and nums_train[label] < 5:
                    items_train.append(data)
                    nums_train[label] += 1
            elif cfg.SESSION == 2:
                if label in classes2 and nums_train[label] < 5:
                    items_train.append(data)
                    nums_train[label] += 1
            elif cfg.SESSION == 3:
                if label in classes3 and nums_train[label] < 5:
                    items_train.append(data)
                    nums_train[label] += 1
            elif cfg.SESSION == 4:
                if label in classes4 and nums_train[label] < 5:
                    items_train.append(data)
                    nums_train[label] += 1
            elif cfg.SESSION == 5:
                if label in classes5 and nums_train[label] < 5:
                    items_train.append(data)
                    nums_train[label] += 1
            elif cfg.SESSION == 6:
                if label in classes6 and nums_train[label] < 5:
                    items_train.append(data)
                    nums_train[label] += 1
            elif cfg.SESSION == 7:
                if label in classes7 and nums_train[label] < 5:
                    items_train.append(data)
                    nums_train[label] += 1
            elif cfg.SESSION == 8:
                if label in classes8 and nums_train[label] < 5:
                    items_train.append(data)
                    nums_train[label] += 1

        for i in range(len(test)):
            data = test[i]
            label = data.label

            if cfg.SESSION == 0:
                if label in base_classes:
                    items_test.append(data)
            elif cfg.SESSION == 1:
                if label in base_classes+classes1:
                    items_test.append(data)
            elif cfg.SESSION == 2:
                if label in base_classes+classes1+classes2:
                    items_test.append(data)
            elif cfg.SESSION == 3:
                if label in base_classes+classes1+classes2+classes3:
                    items_test.append(data)
            elif cfg.SESSION == 4:
                if label in base_classes+classes1+classes2+classes3+classes4:
                    items_test.append(data)
            elif cfg.SESSION == 5:
                if label in base_classes+classes1+classes2+classes3+classes4+classes5:
                    items_test.append(data)
            elif cfg.SESSION == 6:
                if label in base_classes+classes1+classes2+classes3+classes4+classes5+classes6:
                    items_test.append(data)
            elif cfg.SESSION == 7:
                if label in base_classes+classes1+classes2+classes3+classes4+classes5+classes6+classes7:
                    items_test.append(data)
            elif cfg.SESSION == 8:
                if label in base_classes+classes1+classes2+classes3+classes4+classes5+classes6+classes7+classes8:
                    items_test.append(data)



        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        # super().__init__(train_x=train, val=val, test=test)
        super().__init__(train_x=items_train, val=val, test=items_test)

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imname = line[0] + ".jpg"
                classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
