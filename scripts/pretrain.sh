#1932574 mix
#18886 test
#20550 balance
#1912024 unbalance or sampler
CUDA_VISIBLE_DEVICES=1,2,3 python src/train.py \
    task_name=pretrain \
    experiment=pretrain \
    data=mix \
    +n_q=1 \
    +nq_rank=0 \
    +train_mode=mlm \
    model.lr=0.0001 \
    +mask_prob=0.25 \
    +audio_time=10.0 \
    +model_config=../model_config \
    data.batch_size=22 \
    +val_dataset_size=18886 \
    +train_dataset_size=1932574 \
    logger=tensorboard \
    trainer=ddp \
    trainer.devices=3 \
    trainer.max_epochs=12 \
    trainer.precision=bf16 \
    callbacks.model_checkpoint.monitor=train/loss \
    callbacks.early_stopping.monitor=train/loss \
    callbacks.model_checkpoint.mode=min \
    paths.root_dir=/home/lizhaohui/ATT/AudioFormer \
    paths.data_dir=/media/data1/lizhaohui/ATT/data/webdataset_encodec \
    +trainer.val_check_interval=1.0 \
    +save_top_k=10 \
    callbacks.early_stopping.patience=10
