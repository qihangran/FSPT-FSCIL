import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter

from dassl.data import DataManager, data_manager, DatasetWrapper
from dassl.data.transforms import INTERPOLATION_MODES, build_transform
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator
import copy

def sample_iterators(cfg, dataset, t, batch_size=1):
    trainset = copy.deepcopy(dataset)
    class_labels = np.array([x.label for x in trainset])
    indices = np.zeros_like(class_labels)
    for a in [t]:
        indices = indices + (class_labels == a).astype(int)
    indices = np.nonzero(indices)
    trainset = [trainset[i] for i in indices[0]]
    tfm_train = build_transform(cfg, is_train=True)
    dataset_wrapper = DatasetWrapper
    dataloader_x = data_manager.build_data_loader(
        cfg,
        sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
        data_source=trainset,
        batch_size=batch_size,
        n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
        n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
        tfm=tfm_train,
        is_train=True,
        dataset_wrapper=dataset_wrapper
    )
    return dataloader_x

def complete_iterators(cfg, dataset, batch_size=15, num_workers = 1):
    trainset = copy.deepcopy(dataset)
    # class_labels = np.array([x.label for x in trainset])
    tfm_train = build_transform(cfg, is_train=True)
    dataset_wrapper = DatasetWrapper
    dataloader_x = data_manager.build_data_loader(
        cfg,
        sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
        data_source=trainset,
        batch_size=batch_size,
        n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
        n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
        tfm=tfm_train,
        is_train=True,
        dataset_wrapper=dataset_wrapper
    )
    return dataloader_x

def sample_data(iterators, it2, steps=2, reset=True):

    # Sample data for inner and meta updates
    x_traj, y_traj, x_rand, y_rand, x_rand_temp, y_rand_temp = [], [], [], [], [], []

    assert(steps < 16)
    counter = 0
    #
    x_rand_temp = []
    y_rand_temp = []

    class_counter = 0
    for it1 in iterators:
        # assert (len(iterators) == 1)
        steps_inner = 0
        rand_counter = 0
        for img, data in it1:
            class_to_reset = data[0].item()
            if reset:
                # # Resetting weights corresponding to classes in the inner updates; this prevents
                # # the learner from memorizing the data (which would kill the gradients due to inner updates)
                # self.reset_classifer(class_to_reset)
                #fast_vec rese
                a = 0

            counter += 1
            if steps_inner < steps:
                x_traj.append(img)
                y_traj.append(data)
                steps_inner += 1

            else:
                x_rand_temp.append(img)
                y_rand_temp.append(data)
                rand_counter += 1
                if rand_counter == 5:
                    break
        class_counter += 1

    # Sampling the random batch of data
    counter = 0
    for img, data in it2:
        if counter == 1:
            break
        x_rand.append(img)
        y_rand.append(data)
        counter += 1



    x_traj, y_traj, x_rand, y_rand = torch.stack(x_traj), torch.stack(y_traj), torch.stack(x_rand), torch.stack(
        y_rand)

    if len(y_rand_temp) > 0:
        y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
        x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)

        x_rand = torch.cat([x_rand, x_rand_temp], 1)
        y_rand = torch.cat([y_rand, y_rand_temp], 1)

    # print(y_traj)
    # print(y_rand)
    #
    # quit()
    return x_traj, y_traj, x_rand, y_rand

def sample_data_incremental(iterators, steps=2, reset=True):

    # Sample data for inner and meta updates
    x_traj, y_traj, x_rand, y_rand, x_rand_temp, y_rand_temp = [], [], [], [], [], []

    assert(steps < 16)
    counter = 0
    # #
    # x_rand_temp = []
    # y_rand_temp = []

    class_counter = 0
    for it1 in iterators:
        # assert (len(iterators) == 1)
        steps_inner = 0
        rand_counter = 0
        for img, data in it1:
            class_to_reset = data[0].item()
            if reset:
                # # Resetting weights corresponding to classes in the inner updates; this prevents
                # # the learner from memorizing the data (which would kill the gradients due to inner updates)
                # self.reset_classifer(class_to_reset)
                #fast_vec rese
                a = 0

            counter += 1
            if steps_inner < steps:
                x_traj.append(img)
                y_traj.append(data)
                steps_inner += 1

            else:
                x_rand_temp.append(img)
                y_rand_temp.append(data)
                rand_counter += 1
                if rand_counter == 5:
                    break
        class_counter += 1

    # # Sampling the random batch of data
    # counter = 0
    # for img, data in it2:
    #     if counter == 1:
    #         break
    #     x_rand.append(img)
    #     y_rand.append(data)
    #     counter += 1
    #
    # y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
    # x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)


    # x_traj, y_traj, x_rand, y_rand = torch.stack(x_traj), torch.stack(y_traj), torch.stack(x_rand), torch.stack(
    #     y_rand)
    x_traj, y_traj = torch.stack(x_traj), torch.stack(y_traj)

    return x_traj, y_traj


class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            val_result = checkpoint["val_result"]
            print(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            # self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch, metalearning):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        d_rand_iterator = self.complete_iterator
        #--session=0 random choise---------------------
        #--session=1....n, whole samples---------------
        cub200_baseclasses = 100
        cifar100_baseclasses = 60
        eurosat_baseclasses = 5
        dtd_baseclasses = 32
        aircraft_baseclasses = 60
        baseclasses = 60
        if self.cfg.DATASET.NAME == 'DescribableTextures_FSCIL':
            baseclasses = dtd_baseclasses
        elif self.cfg.DATASET.NAME == 'CUB200':
            baseclasses = cub200_baseclasses
        elif self.cfg.DATASET.NAME == 'CIFAR100_FSCIL':
            baseclasses = cifar100_baseclasses
        elif self.cfg.DATASET.NAME == 'FGVCAircraft':
            baseclasses = aircraft_baseclasses
        else:
            print('unknown dataset')

        if metalearning:
            if 0 == self.cfg.SESSION:
                #sampler = ts.SamplerFactory.get_sampler(args['dataset'], args['classes'], dataset, dataset_test)
                traj_classes = list(range(baseclasses))
                tasks = self.cfg.TRAIN.WAYS
                acc_best = 0.
                for step in range(self.cfg.TRAIN.META_STEP): #args['steps']
                    t1 = np.random.choice(traj_classes, tasks, replace=False)
                    d_traj_iterators = []
                    for t in t1:
                        if t in self.iterators:
                            train_iterator = self.iterators[t]
                        else:
                            train_iterator = sample_iterators(self.cfg, self.dm.dataset.train_x, t)
                            self.iterators[t] = train_iterator
                        d_traj_iterators.append(train_iterator)

                    x_spt, y_spt, x_qry, y_qry = sample_data(d_traj_iterators, d_rand_iterator,steps=self.cfg.TRAIN.SHOTS,reset=True)
                    if torch.cuda.is_available():
                        x_spt, y_spt, x_qry, y_qry = x_spt.to(self.device), y_spt.to(self.device), x_qry.to(self.device), y_qry.to(self.device)


                    accuracies, meta_loss = self.meta_learning(x_spt, y_spt, x_qry, y_qry)

                    # Evaluation during training for sanity checks
                    if step % 1 == 0:
                        #writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)
                        print('step: {} \t training acc {:.2f}'.format(step, accuracies))
                    if step > 0 and step % self.cfg.TRAIN.SAVE_STEP == 0:
                        acc = self.test()
                        if acc_best < acc:
                            self.save_model(step, self.output_dir,is_best=True)
                            acc_best = acc
                            print('Best Acc:',acc_best)
                        # self.after_step(step)
                        # self.test()
            else:
                # sampler = ts.SamplerFactory.get_sampler(args['dataset'], args['classes'], dataset, dataset_test)
                base_classes = baseclasses
                acc_best = 0.
                for step in range(self.cfg.TRAIN.META_STEP):  # args['steps']
                    t1 = np.arange(base_classes + self.cfg.TRAIN.WAYS*(self.cfg.SESSION-1), base_classes + self.cfg.TRAIN.WAYS*self.cfg.SESSION)
                    #t1 = np.random.choice(range(base_classes + self.cfg.TRAIN.WAYS*(self.cfg.SESSION-1), base_classes + self.cfg.TRAIN.WAYS*self.cfg.SESSION), size=5, replace=False)
                    d_traj_iterators = []
                    for t in t1:
                        if t in self.iterators:
                            train_iterator = self.iterators[t]
                        else:
                            train_iterator = sample_iterators(self.cfg, self.dm.dataset.train_x, t)
                            self.iterators[t] = train_iterator
                        d_traj_iterators.append(train_iterator)

                    x_spt, y_spt, x_qry, y_qry = sample_data(d_traj_iterators, d_rand_iterator,steps=self.cfg.TRAIN.SHOTS, reset=True)
                    if torch.cuda.is_available():
                        x_spt, y_spt, x_qry, y_qry = x_spt.to(self.device), y_spt.to(self.device), x_qry.to(
                            self.device), y_qry.to(self.device)

                    # x_spt, y_spt = sample_data_incremental(d_traj_iterators, steps=self.cfg.TRAIN.SHOTS, reset=True)
                    # if torch.cuda.is_available():
                    #     x_spt, y_spt = x_spt.to(self.device), y_spt.to(self.device)

                    #self.meta_learning_inner(x_spt, y_spt)
                    accuracies, meta_loss = self.meta_learning(x_spt, y_spt, x_qry, y_qry)

                    # Evaluation during training for sanity checks
                    if step % 1 == 0:
                        # writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)
                        print('step: {}'.format(step))
                    if step >=0 and step % self.cfg.TRAIN.SAVE_STEP == 0:
                        acc = self.test()
                        if acc_best < acc:
                            self.save_model(step, self.output_dir,is_best=True)
                            acc_best = acc
                            print('Best Acc:',acc_best)
        else:
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.run_epoch()
                self.after_epoch()
            self.after_train()

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)

    def meta_learning(self, x_traj, y_traj, x_qry, y_qry):
        raise NotImplementedError

    def meta_learning_inner(self, x_traj, y_traj):
        raise NotImplementedError

    def after_step(self, step):
        pass


class SimpleTrainer(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm
        #
        # ctrain_iterator = torch.utils.data.DataLoader(dm.dataset.train_x, batch_size=15, shuffle=True, num_workers=0)
        # self.complete_iterator = ctrain_iterator

        self.complete_iterator = complete_iterators(self.cfg, self.dm.dataset.train_x, batch_size=self.cfg.TRAIN.META_OUTER_BATCH)


    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building model")
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        #self.metalearning = True

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)

    def train(self):
        super().train(self.start_epoch, self.max_epoch, self.metalearning)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def after_step(self, step):
        self.save_model(step, self.output_dir)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            # output,_,_ = self.model_inference(input)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference(self, input):
        return self.model(input)

    def parse_batch_test(self, batch):
        # input = batch["img"]
        # label = batch["label"]
        input = batch[0]
        label = batch[1]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]


class TrainerXU(SimpleTrainer):
    """A base trainer using both labeled and unlabeled data.

    In the context of domain adaptation, labeled and unlabeled data
    come from source and target domains respectively.

    When it comes to semi-supervised learning, all data comes from the
    same domain.
    """

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u


class TrainerX(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain
