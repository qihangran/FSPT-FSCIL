import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
import csv

miniImageNet_classnames = (['house_finch', 'robin', 'triceratops', 'green_mamba', 'harvestman', 'toucan', 'goose', 'jellyfish', 'nematode',
                           'king_crab', 'dugong', 'Walker_hound', 'Ibizan_hound', 'Saluki', 'golden_retriever', 'Gordon_setter', 'komondor',
                           'boxer', 'Tibetan_mastiff', 'French_bulldog', 'malamute', 'dalmatian', 'Newfoundland', 'miniature_poodle', 'white_wolf',
                           'African_hunting_dog', 'Arctic_fox', 'lion', 'meerkat', 'ladybug', 'rhinoceros_beetle', 'ant', 'black-footed_ferret',
                           'three-toed_sloth', 'rock_beauty', 'aircraft_carrier', 'ashcan', 'barrel', 'beer_bottle', 'bookshop', 'cannon', 'carousel',
                           'carton', 'catamaran', 'chime', 'clog', 'cocktail_shaker', 'combination_lock', 'crate', 'cuirass', 'dishrag', 'dome',
                           'electric_guitar', 'file', 'fire_screen', 'frying_pan', 'garbage_truck', 'hair_slide', 'holster', 'horizontal_bar'] + ['hourglass', 'iPod', 'lipstick', 'miniskirt', 'missile']
                            + ['mixing_bowl', 'oboe', 'organ', 'parallel_bars', 'pencil_box'] + ['photocopier', 'poncho', 'prayer_rug', 'reel', 'school_bus']
                            + ['scoreboard', 'slot', 'snorkel', 'solar_dish', 'spider_web'] + ['stage', 'tank', 'theater_curtain', 'tile_roof', 'tobacco_shop']
                            +['unicycle', 'upright', 'vase', 'wok', 'worm_fence'] + ['yawl', 'street_sign', 'consomme', 'trifle', 'hotdog']
                            +['orange', 'cliff', 'coral_reef', 'bolete', 'ear'])

id_names = ['n01532829', 'n01558993', 'n01704323', 'n01749939', 'n01770081', 'n01843383', 'n01855672', 'n01910747', 'n01930112',
            'n01981276', 'n02074367', 'n02089867', 'n02091244', 'n02091831', 'n02099601', 'n02101006', 'n02105505', 'n02108089',
            'n02108551', 'n02108915', 'n02110063', 'n02110341', 'n02111277', 'n02113712', 'n02114548', 'n02116738', 'n02120079',
            'n02129165', 'n02138441', 'n02165456', 'n02174001', 'n02219486', 'n02443484', 'n02457408', 'n02606052', 'n02687172',
            'n02747177', 'n02795169', 'n02823428', 'n02871525', 'n02950826', 'n02966193', 'n02971356', 'n02981792', 'n03017168',
            'n03047690', 'n03062245', 'n03075370', 'n03127925', 'n03146219', 'n03207743', 'n03220513', 'n03272010', 'n03337140',
            'n03347037', 'n03400231', 'n03417042', 'n03476684', 'n03527444', 'n03535780', 'n03544143', 'n03584254', 'n03676483',
            'n03770439', 'n03773504', 'n03775546', 'n03838899', 'n03854065', 'n03888605', 'n03908618', 'n03924679', 'n03980874',
            'n03998194', 'n04067472', 'n04146614', 'n04149813', 'n04243546', 'n04251144', 'n04258138', 'n04275548', 'n04296562',
            'n04389033', 'n04418357', 'n04435653', 'n04443257', 'n04509417', 'n04515003', 'n04522168', 'n04596742', 'n04604644',
            'n04612504', 'n06794110', 'n07584110', 'n07613480', 'n07697537', 'n07747607', 'n09246464', 'n09256479', 'n13054560', 'n13133613']


@DATASET_REGISTRY.register()
class miniImageNet(DatasetBase):

    dataset_dir = "miniimagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # classnames = []
        # with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(id_names)}

        #test = self.read_data(cname2lab, "images_variant_test.txt")
        #if cfg.SESSION == 0:

        txt_file = 'session_' + str(cfg.SESSION + 1) + '.txt'
        train = self.read_data(cname2lab, txt_file)
            #val = self.read_data(cname2lab, "images_variant_val.txt")

        # test from test.csv file
        cvs_file = 'test.csv'
        test = self.read_data_test(cname2lab, cvs_file, session=cfg.SESSION)




        # num_shots = cfg.DATASET.NUM_SHOTS
        # if num_shots >= 1:
        #     seed = cfg.SEED
        #     preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
        #
        #     if os.path.exists(preprocessed):
        #         print(f"Loading preprocessed few-shot data from {preprocessed}")
        #         with open(preprocessed, "rb") as file:
        #             data = pickle.load(file)
        #             train, val = data["train"], data["val"]
        #     else:
        #         train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        #         val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
        #         data = {"train": train, "val": val}
        #         print(f"Saving preprocessed few-shot data to {preprocessed}")
        #         with open(preprocessed, "wb") as file:
        #             pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        # train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        # super().__init__(train_x=train, val=val, test=test)
        super().__init__(train_x=train, val=[], test=test)

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                #line = line.strip().split(" ")
                parts = line.strip().split('/')
                id_name = parts[2]
                imname = parts[3]
                # content_after_second_slash = '/'.join(parts[2:])
                #imname = line[0] + ".jpg"
                #classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[id_name]
                item = Datum(impath=impath, label=label, classname=miniImageNet_classnames[label])
                items.append(item)

        return items

    def read_data_test(self, cname2lab, test_file, session = 0):
        filepath = os.path.join(self.dataset_dir, test_file)
        items = []
        base_classes = 60
        nways = 5
        range = base_classes + session * nways
        with open(filepath, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            # # 初始化存储列数据的列表
            # column1_data = []
            # column2_data = []
            # # 遍历每一行并获取数据
            for row in csv_reader:
                if row[1] in id_names[:range]:
                    impath = os.path.join(self.image_dir, row[0])
                    label = cname2lab[row[1]]
                    item = Datum(impath=impath, label=label, classname=miniImageNet_classnames[label])
                    items.append(item)
        return items
