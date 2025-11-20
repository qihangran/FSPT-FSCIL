#--session 1-------
python train.py \
--root ./DATA \
--seed 1 \
--trainer CoOp \
--dataset-config-file configs/datasets/cifar100.yaml \
--config-file configs/trainers/CoOp/vit_l14_ep50.yaml \
--output-dir output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s1/seed1 \
--session 1 \
--meta-train \
--nways 5 \
--kshots 1 \
--meta-outer-batch 50 \
TRAINER.COOP.N_CTX 12 \
TRAINER.COOP.CSC False \
TRAINER.COOP.CLASS_TOKEN_POSITION end \
DATASET.NUM_SHOTS 0 \
DATALOADER.NUM_WORKERS 0 \
DATALOADER.TEST.BATCH_SIZE 100 \
TRAIN.META_STEP 5 \
MODEL.INIT_WEIGHTS output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s0/seed1/prompt_learner/model-best.pth.tar \
TRAIN.SAVE_STEP 1

sleep 15

#--session 2-------
python train.py \
--root ./DATA \
--seed 1 \
--trainer CoOp \
--dataset-config-file configs/datasets/cifar100.yaml \
--config-file configs/trainers/CoOp/vit_l14_ep50.yaml \
--output-dir output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s2/seed1 \
--session 2 \
--meta-train \
--nways 5 \
--kshots 1 \
--meta-outer-batch 50 \
TRAINER.COOP.N_CTX 12 \
TRAINER.COOP.CSC False \
TRAINER.COOP.CLASS_TOKEN_POSITION end \
DATASET.NUM_SHOTS 0 \
DATALOADER.NUM_WORKERS 0 \
DATALOADER.TEST.BATCH_SIZE 100 \
TRAIN.META_STEP 5 \
MODEL.INIT_WEIGHTS output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s1/seed1/prompt_learner/model-best.pth.tar \
TRAIN.SAVE_STEP 1

sleep 15

#--session 3-------
python train.py \
--root ./DATA \
--seed 1 \
--trainer CoOp \
--dataset-config-file configs/datasets/cifar100.yaml \
--config-file configs/trainers/CoOp/vit_l14_ep50.yaml \
--output-dir output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s3/seed1 \
--session 3 \
--meta-train \
--nways 5 \
--kshots 1 \
--meta-outer-batch 50 \
TRAINER.COOP.N_CTX 12 \
TRAINER.COOP.CSC False \
TRAINER.COOP.CLASS_TOKEN_POSITION end \
DATASET.NUM_SHOTS 0 \
DATALOADER.NUM_WORKERS 0 \
DATALOADER.TEST.BATCH_SIZE 100 \
TRAIN.META_STEP 5 \
MODEL.INIT_WEIGHTS output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s2/seed1/prompt_learner/model-best.pth.tar \
TRAIN.SAVE_STEP 1

#--session 4-------
python train.py \
--root ./DATA \
--seed 1 \
--trainer CoOp \
--dataset-config-file configs/datasets/cifar100.yaml \
--config-file configs/trainers/CoOp/vit_l14_ep50.yaml \
--output-dir output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s4/seed1 \
--session 4 \
--meta-train \
--nways 5 \
--kshots 1 \
--meta-outer-batch 50 \
TRAINER.COOP.N_CTX 12 \
TRAINER.COOP.CSC False \
TRAINER.COOP.CLASS_TOKEN_POSITION end \
DATASET.NUM_SHOTS 0 \
DATALOADER.NUM_WORKERS 0 \
DATALOADER.TEST.BATCH_SIZE 100 \
TRAIN.META_STEP 5 \
MODEL.INIT_WEIGHTS output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s3/seed1/prompt_learner/model-best.pth.tar \
TRAIN.SAVE_STEP 1

sleep 15

#--session 5-------
python train.py \
--root ./DATA \
--seed 1 \
--trainer CoOp \
--dataset-config-file configs/datasets/cifar100.yaml \
--config-file configs/trainers/CoOp/vit_l14_ep50.yaml \
--output-dir output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s5/seed1 \
--session 5 \
--meta-train \
--nways 5 \
--kshots 1 \
--meta-outer-batch 50 \
TRAINER.COOP.N_CTX 12 \
TRAINER.COOP.CSC False \
TRAINER.COOP.CLASS_TOKEN_POSITION end \
DATASET.NUM_SHOTS 0 \
DATALOADER.NUM_WORKERS 0 \
DATALOADER.TEST.BATCH_SIZE 100 \
TRAIN.META_STEP 5 \
MODEL.INIT_WEIGHTS output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s4/seed1/prompt_learner/model-best.pth.tar \
TRAIN.SAVE_STEP 1

sleep 15

#--session 6-------
python train.py \
--root ./DATA \
--seed 1 \
--trainer CoOp \
--dataset-config-file configs/datasets/cifar100.yaml \
--config-file configs/trainers/CoOp/vit_l14_ep50.yaml \
--output-dir output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s6/seed1 \
--session 6 \
--meta-train \
--nways 5 \
--kshots 1 \
--meta-outer-batch 50 \
TRAINER.COOP.N_CTX 12 \
TRAINER.COOP.CSC False \
TRAINER.COOP.CLASS_TOKEN_POSITION end \
DATASET.NUM_SHOTS 0 \
DATALOADER.NUM_WORKERS 0 \
DATALOADER.TEST.BATCH_SIZE 100 \
TRAIN.META_STEP 5 \
MODEL.INIT_WEIGHTS output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s5/seed1/prompt_learner/model-best.pth.tar \
TRAIN.SAVE_STEP 1

sleep 15

#--session 7-------
python train.py \
--root ./DATA \
--seed 1 \
--trainer CoOp \
--dataset-config-file configs/datasets/cifar100.yaml \
--config-file configs/trainers/CoOp/vit_l14_ep50.yaml \
--output-dir output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s7/seed1 \
--session 7 \
--meta-train \
--nways 5 \
--kshots 1 \
--meta-outer-batch 50 \
TRAINER.COOP.N_CTX 12 \
TRAINER.COOP.CSC False \
TRAINER.COOP.CLASS_TOKEN_POSITION end \
DATASET.NUM_SHOTS 0 \
DATALOADER.NUM_WORKERS 0 \
DATALOADER.TEST.BATCH_SIZE 100 \
TRAIN.META_STEP 5 \
MODEL.INIT_WEIGHTS output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s6/seed1/prompt_learner/model-best.pth.tar \
TRAIN.SAVE_STEP 1

sleep 15

#--session 8-------
python train.py \
--root ./DATA \
--seed 1 \
--trainer CoOp \
--dataset-config-file configs/datasets/cifar100.yaml \
--config-file configs/trainers/CoOp/vit_l14_ep50.yaml \
--output-dir output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s8/seed1 \
--session 8 \
--meta-train \
--nways 5 \
--kshots 1 \
--meta-outer-batch 50 \
TRAINER.COOP.N_CTX 12 \
TRAINER.COOP.CSC False \
TRAINER.COOP.CLASS_TOKEN_POSITION end \
DATASET.NUM_SHOTS 0 \
DATALOADER.NUM_WORKERS 0 \
DATALOADER.TEST.BATCH_SIZE 100 \
TRAIN.META_STEP 5 \
MODEL.INIT_WEIGHTS output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s7/seed1/prompt_learner/model-best.pth.tar \
TRAIN.SAVE_STEP 1