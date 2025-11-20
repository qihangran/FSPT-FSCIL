python train.py \
--root ./DATA \
--seed 1 \
--trainer CoOp \
--dataset-config-file configs/datasets/cifar100.yaml \
--config-file configs/trainers/CoOp/vit_l14_ep50.yaml \
--output-dir output/cifar100/CoOp/vit_l14_meta10_5/nctx16_scsFalse_s0/seed1 \
--session 0 \
--meta-train \
--nways 10 \
--kshots 5 \
--meta-outer-batch 50 \
TRAINER.COOP.N_CTX 12 \
TRAINER.COOP.CSC False \
TRAINER.COOP.CLASS_TOKEN_POSITION end \
DATASET.NUM_SHOTS 0 \
DATALOADER.NUM_WORKERS 0 \
DATALOADER.TEST.BATCH_SIZE 100 \
TRAIN.META_STEP 10001 \
TRAIN.SAVE_STEP 100