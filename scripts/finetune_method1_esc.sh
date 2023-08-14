#1932574 mix
#18886 test
#20550 balance
#1912024 unbalance or sampler
CUDA_VISIBLE_DEVICES=0 python src/train.py \
    task_name=finetune_method1_esc \
    experiment=finetune_method1_esc \
    data=esc \
    +n_q=1 \
    +nq_rank=0 \
    +mask_prob=0.0 \
    +is_mixup=False \
    +train_mode=mlc \
    +audio_time=5.0 \
    model.lr=0.00001 \
    +is_finetune=True \
    +model_config=../model_config \
    +pretrain_model_pt=/home/lizhaohui/ATT/AudioFormer/data/nq0_cpt_epoch_02.pt \
    data.batch_size=32 \
    +val_dataset_size=400 \
    +train_dataset_size=1600 \
    +mode=max \
    +save_top_k=3 \
    logger=tensorboard \
    callbacks.early_stopping.monitor=val/acc \
    callbacks.model_checkpoint.monitor=val/acc \
    trainer=gpu \
    trainer.devices=1 \
    trainer.precision=32 \
    trainer.max_epochs=20 \
    paths.root_dir=/home/lizhaohui/ATT/AudioFormer \
    paths.data_dir=/media/data1/lizhaohui/ATT/data/webdataset_esc_encodec \
    +trainer.val_check_interval=1.0 \
    callbacks.early_stopping.patience=3 
