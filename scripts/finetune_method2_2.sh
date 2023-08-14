#1932574 mix
#18886 test
#20550 balance
#1912024 unbalance or sampler
CUDA_VISIBLE_DEVICES=2 python src/train.py \
    task_name=finetune_method2_2 \
    experiment=finetune_method2_2 \
    data=balance \
    +n_q=3 \
    +nq_rank=0 \
    +alpha=0.97 \
    +mask_prob=0.0 \
    model.lr=0.00001 \
    +audio_time=10.0 \
    +temperature=0.03 \
    +is_finetune=True \
    +train_mode=mpc_mlc \
    +have_classifier_head=False \
    +model_config=../model_config \
    +pretrain_model_nq0=/home/lizhaohui/ATT/AudioFormer/logs/finetune_method2_2/runs/nq3_alpha095_sampler1_mAP50/checkpoints/nq3_mpc_sampler1_50mAP.pt \
    +pretrain_model_nq1=/home/lizhaohui/ATT/AudioFormer/logs/pretrain/runs/nq1_cpt_epoch6/nq1_cpt_epoch6.pt \
    +pretrain_model_nq2=/home/lizhaohui/ATT/AudioFormer/logs/pretrain/runs/nq2_cpt_epoch6/checkpoints/nq2_cpt_epoch6.pt \
    +pretrain_model_nq3=/home/lizhaohui/ATT/AudioFormer/logs/pretrain/runs/nq3_cpt_epoch6/checkpoints/nq3_cpt_epoch6.pt \
    logger=tensorboard \
    data.batch_size=16 \
    +val_dataset_size=18886 \
    +train_dataset_size=20550 \
    callbacks.early_stopping.monitor=val/mAP \
    callbacks.model_checkpoint.monitor=val/mAP \
    trainer=gpu \
    trainer.devices=1 \
    trainer.precision=32 \
    trainer.max_epochs=10 \
    paths.data_dir=/media/data1/lizhaohui/ATT/data/webdataset_balance_encodec \
    paths.root_dir=/home/lizhaohui/ATT/AudioFormer \
    +trainer.val_check_interval=1.0 \
    callbacks.early_stopping.patience=3 
