# train 36796
# val 4170
# test 10231
CUDA_VISIBLE_DEVICES=1 python src/train.py \
    task_name=finetune_method1_FSD50k \
    experiment=finetune_method1 \
    data=FSD50K \
    +n_q=1 \
    +nq_rank=0 \
    +mask_prob=0.1 \
    +is_mixup=False \
    +train_mode=mlc \
    +audio_time=10.0 \
    model.lr=0.0001 \
    +is_finetune=True \
    +model_config=../model_config \
    +pretrain_model_pt=/home/lizhaohui/ATT/AudioFormer/data/nq0_53.9.pt \
    data.batch_size=16 \
    +val_dataset_size=4710 \
    +train_dataset_size=36796 \
    +test_dataset_size=10231 \
    logger=tensorboard \
    callbacks.early_stopping.monitor=val/mAP \
    callbacks.model_checkpoint.monitor=val/mAP \
    trainer=gpu \
    trainer.devices=1 \
    trainer.precision=32 \
    trainer.max_epochs=4 \
    paths.root_dir=/home/lizhaohui/ATT/AudioFormer \
    paths.data_dir=/media/data1/lizhaohui/ATT/data/FSD50K/webdataset \
    +trainer.val_check_interval=3.0 \
    callbacks.early_stopping.patience=3 
