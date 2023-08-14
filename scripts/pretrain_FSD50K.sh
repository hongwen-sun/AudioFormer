# train 36796
# val 4170
# test 10231
CUDA_VISIBLE_DEVICES=2 python src/train.py \
    task_name=FSD50K_pretrain \
    experiment=pretrain \
    data=FSD50K_pretrain \
    +n_q=1 \
    +nq_rank=0 \
    +train_mode=mlm \
    model.lr=0.0001 \
    +mask_prob=0.25 \
    +audio_time=10.0 \
    +model_config=../model_config \
    data.batch_size=22 \
    +val_dataset_size=4170 \
    +train_dataset_size=51197 \
    logger=tensorboard \
    trainer=gpu \
    trainer.devices=1 \
    trainer.max_epochs=10 \
    trainer.precision=bf16 \
    callbacks.model_checkpoint.monitor=train/loss \
    callbacks.early_stopping.monitor=train/loss \
    callbacks.model_checkpoint.mode=min \
    paths.root_dir=/home/lizhaohui/ATT/AudioFormer \
    paths.data_dir=/media/data1/lizhaohui/ATT/data/FSD50K/webdataset \
    +trainer.val_check_interval=1.0 \
    callbacks.early_stopping.patience=10
