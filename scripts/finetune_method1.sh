#1932574 mix
#18886 test
#20550 balance
#1912024 unbalance or sampler
CUDA_VISIBLE_DEVICES=3 python src/train.py \
    task_name=finetune_method1 \
    experiment=finetune_method1 \
    data=balance \
    +n_q=1 \
    +nq_rank=0 \
    +mask_prob=0.0 \
    +is_mixup=False \
    +train_mode=mlc \
    +audio_time=10.0 \
    model.lr=0.0001 \
    +is_finetune=True \
    +model_config=../model_config \
    +pretrain_model_pt=/home/lizhaohui/ATT/AudioFormer/logs/pretrain/runs/2023-08-11_17-52-45/checkpoints/last_backbone.pt \
    data.batch_size=16 \
    +val_dataset_size=18886 \
    +train_dataset_size=20550 \
    +mode=max \
    +save_top_k=3 \
    logger=tensorboard \
    callbacks.early_stopping.monitor=val/mAP \
    callbacks.model_checkpoint.monitor=val/mAP \
    trainer=gpu \
    trainer.devices=1 \
    trainer.precision=32 \
    trainer.max_epochs=10 \
    paths.root_dir=/home/lizhaohui/ATT/AudioFormer \
    paths.data_dir=/media/data1/lizhaohui/ATT/data/webdataset_balance_encodec \
    +trainer.val_check_interval=1.0 \
    callbacks.early_stopping.patience=3 
