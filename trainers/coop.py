import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from models import prompters
from utils import cosine_lr

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "CUB200": "a photo of a {}, a type of bird",
}
cifar100_classnames = [
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

cub200_classnames = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet',
           'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird',
           'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting',
           'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat',
           'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant',
           'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo',
           'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker',
           'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher',
           'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird',
           'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle',
           'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak',
           'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull',
           'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull',
           'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear',
           'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco',
           'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher',
           'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon',
           'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk',
           'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole',
           'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis',
           'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven',
           'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow',
           'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow',
           'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow',
           'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow',
           'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow',
           'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow',
           'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager',
           'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern',
           'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo',
           'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo',
           'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler',
           'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler',
           'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler',
           'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler',
           'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler',
           'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush',
           'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker',
           'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker',
           'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren',
           'Winter_Wren', 'Common_Yellowthroat']

dtd_classnames = ['banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crosshatched', 'crystalline',
                  'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced', 'knitted',
                  'lacelike', 'lined', 'marbled', 'matted', 'meshed', 'paisley', 'perforated', 'pitted', 'pleated', 'polka_dotted', 'porous',
                  'potholed', 'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified', 'striped', 'studded', 'swirly', 'veined',
                  'waffled', 'woven', 'wrinkled', 'zigzagged']

fgvc_aircraft_classnames = ['707-320', '727-200', '737-200', '737-300', '737-400', '737-500', '737-600', '737-700', '737-800', '737-900', '747-100', '747-200',
         '747-300', '747-400', '757-200', '757-300', '767-200', '767-300', '767-400', '777-200', '777-300', 'A300B4', 'A310', 'A318', 'A319',
         'A320', 'A321', 'A330-200', 'A330-300', 'A340-200', 'A340-300', 'A340-500', 'A340-600', 'A380', 'ATR-42', 'ATR-72', 'An-12', 'BAE 146-200',
         'BAE 146-300', 'BAE-125', 'Beechcraft 1900', 'Boeing 717', 'C-130', 'C-47', 'CRJ-200', 'CRJ-700', 'CRJ-900', 'Cessna 172', 'Cessna 208',
         'Cessna 525', 'Cessna 560', 'Challenger 600', 'DC-10', 'DC-3', 'DC-6', 'DC-8', 'DC-9-30', 'DH-82', 'DHC-1', 'DHC-6', 'DHC-8-100', 'DHC-8-300',
         'DR-400', 'Dornier 328', 'E-170', 'E-190', 'E-195', 'EMB-120', 'ERJ 135', 'ERJ 145', 'Embraer Legacy 600', 'Eurofighter Typhoon', 'F-16A/B',
         'F/A-18', 'Falcon 2000', 'Falcon 900', 'Fokker 100', 'Fokker 50', 'Fokker 70', 'Global Express', 'Gulfstream IV', 'Gulfstream V', 'Hawk T1',
         'Il-76', 'L-1011', 'MD-11', 'MD-80', 'MD-87', 'MD-90', 'Metroliner', 'Model B200', 'PA-28', 'SR-20', 'Saab 2000', 'Saab 340', 'Spitfire', 'Tornado', 'Tu-134', 'Tu-154', 'Yak-42']

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

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # n_cls = len(classnames)
        # n_cls = len(cifar100_classnames)
        data_classnames = classnames
        if cfg.DATASET.NAME == 'DescribableTextures_FSCIL':
            data_classnames = dtd_classnames
        elif cfg.DATASET.NAME == 'CUB200':
            data_classnames = cub200_classnames
        elif cfg.DATASET.NAME =='CIFAR100_FSCIL':
            data_classnames = cifar100_classnames
        elif cfg.DATASET.NAME == 'FGVCAircraft':
            data_classnames = fgvc_aircraft_classnames
        elif cfg.DATASET.NAME == 'miniImageNet':
            data_classnames = miniImageNet_classnames
        n_cls = len(data_classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        n_fast_ctx = cfg.TRAINER.COOP.F_N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

            # fast_ctx = torch.empty(4, ctx_dim, dtype=dtype)
            fast_ctx = torch.empty(cfg.TRAINER.COOP.F_N_CTX, ctx_dim, dtype=dtype)
            nn.init.normal_(fast_ctx, std=0.02)


            prompt_prefix = " ".join(["X"] * (n_ctx+n_fast_ctx))

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx+n_fast_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.fast_ctx = nn.Parameter(fast_ctx)
        #self.fast_ctx = fast_ctx

        # self.ctx.meta, self.fast_ctx.meta = True, True
        # self.ctx.adaptation, self.fast_ctx.adaptation = False, True

        # # key
        # key_vectors = torch.empty(10, ctx_dim, dtype=dtype)
        # nn.init.normal_(key_vectors, std=0.02)
        # self.keys = nn.Parameter(key_vectors)
        #
        # #prompts
        # prompts_vectors = torch.empty(10, n_ctx, ctx_dim, dtype=dtype)
        # nn.init.normal_(prompts_vectors, std=0.02)
        # self.prompts = nn.Parameter(prompts_vectors)


        # classnames = [name.replace("_", " ") for name in cifar100_classnames]
        classnames = [name.replace("_", " ") for name in data_classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + n_fast_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = torch.cat([self.ctx, self.fast_ctx], dim=0)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            # half_n_ctx = self.n_ctx // 2
            half_n_ctx = self.n_ctx

            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def get_adaptation_parameters(self, vars=None):
        """
        :return: adaptation parameters i.e. parameters changed in the inner loop
        """
        if vars is None:
            vars = self.vars
        return list(filter(lambda x: x.adaptation, list(vars)))

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            # if "prompt_learner.ctx" not in name:
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        self.metalearning = self.cfg.TRAIN.META
        self.iterators = {}

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        # input = batch[0]
        # label = batch[1]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def meta_learning(self, x_traj, y_traj, x_qry, y_qry):
        # inner parameters reset
        #fast_ctx = self.model.prompt_learner.fast_ctx
        #nn.init.normal_(self.model.prompt_learner.fast_ctx, std=0.02)
        #torch.nn.init.kaiming_normal_(self.model.prompt_learner.fast_ctx)
        #torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))

        meta_losses = [0 for _ in range(len(x_traj) + 1)]  # losses_q[i] is the loss on step i
        accuracy_meta_set = [0 for _ in range(len(x_traj) + 1)]
        # Doing a single inner update to get updated weights
        self.inner_update(x_traj[0], y_traj[0])

        # with torch.no_grad():
        #     # # Meta loss before any inner updates
        #     # meta_loss, last_layer_logits = self.meta_loss(x_qry[0], y_qry[0])
        #     # meta_losses[0] += meta_loss
        #     #
        #     # classification_accuracy = self.eval_accuracy(last_layer_logits, y_qry[0])
        #     # accuracy_meta_set[0] = accuracy_meta_set[0] + classification_accuracy
        #
        #     # Meta loss after a single inner update
        #     meta_loss, last_layer_logits = self.meta_loss(x_qry[0], y_qry[0])
        #     meta_losses[1] += meta_loss
        #
        #     classification_accuracy = self.eval_accuracy(last_layer_logits, y_qry[0])
        #     accuracy_meta_set[1] = accuracy_meta_set[1] + classification_accuracy

        # print(len(x_traj))
        for k in range(1, len(x_traj)):
            # Doing inner updates using fast weights
            self.inner_update(x_traj[k], y_traj[k])

        # Computing meta-loss with respect to latest weights

        meta_loss, logits = self.meta_loss(x_qry[0], y_qry[0])
        meta_losses[k + 1] += meta_loss

        # Computing accuracy on the meta and traj set for understanding the learning
        with torch.no_grad():
            pred_q = F.softmax(logits, dim=1).argmax(dim=1)
            classification_accuracy = torch.eq(pred_q, y_qry[0]).sum().item()  # convert to numpy
            accuracy_meta_set[k + 1] = accuracy_meta_set[k + 1] + classification_accuracy

        # Taking the meta gradient step
        self.optim.zero_grad()
        meta_loss = meta_losses[-1]
        meta_loss.backward()

        self.optim.step()
        accuracies = accuracy_meta_set[-1] / len(x_qry[0])

        return accuracies, meta_loss

    def meta_learning_inner(self, x_traj, y_traj):
        for k in range(0, len(x_traj)):
            self.inner_update(x_traj[k], y_traj[k])



    def inner_update(self, x, y):
        adaptation_weight_counter = 0

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(x)
                loss = F.cross_entropy(output, y)
        else:
            output = self.model(x)
            loss = F.cross_entropy(output, y)


        fast_weights = self.model.prompt_learner.fast_ctx
        #with torch.no_grad():
        grad = torch.autograd.grad(loss, fast_weights, create_graph=False)

        #---------------------------just update fast_ctx , i.e, prompts-----------
        g = grad[0]
        self.model.prompt_learner.fast_ctx.data = self.model.prompt_learner.fast_ctx.data - 0.03 * g
        # self.model.prompt_learner.fast_ctx = self.model.prompt_learner.fast_ctx - 0.03 * g
        #self.model.prompt_learner.fast_ctx = self.model.prompt_learner.fast_ctx - 0.1 * g


        #torch.cuda.empty_cache()

        # new_weights = []
        # for p in fast_weights:
        #     if p.adaptation:
        #         g = grad[adaptation_weight_counter]
        #         temp_weight = p - self.update_lr * g
        #         temp_weight.adaptation = p.adaptation
        #         temp_weight.meta = p.meta
        #         new_weights.append(temp_weight)
        #         adaptation_weight_counter += 1
        #     else:
        #         new_weights.append(p)
        #
        # return new_weights

    def reset_layer(self):
        # torch.nn.init.kaiming_normal_(self.model.prompt_learner.fast_ctx)
        torch.nn.init.normal_(self.model.prompt_learner.fast_ctx, std=0.02)

    def meta_loss(self, x, y):
        logits = self.model(x)
        loss_q = F.cross_entropy(logits, y)
        #torch.cuda.empty_cache()
        return loss_q, logits

    def eval_accuracy(self, logits, y):
        pred_q = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item()
        return correct


if __name__ == "__main__":
    a = 0
    import os
    from torchvision.datasets import CIFAR100
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess =clip. load('ViT-B/16', device)

    # Download the dataset
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

    aaa = clip.tokenize(f"a photo of a apple")

    # for c in cifar100.classes:
    #     c = c.replace("_", " ")

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c.replace('_', ' ')}") for c in cifar100.classes]).to(device)
    # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    corr = 0
    for i in range(10000):
        # Prepare the inputs
        # image, class_id = cifar100[3637]
        image, class_id = cifar100[i]
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        #values, indices = similarity[0].topk(5)
        values, indices = similarity[0].topk(1)
        if indices[0] == class_id:
            corr += 1

    print(corr)


    

