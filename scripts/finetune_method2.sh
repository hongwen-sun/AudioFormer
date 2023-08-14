#1932574 mix
#18886 test
#20550 balance
#1912024 unbalance or sampler
CUDA_VISIBLE_DEVICES=1 python src/train.py \
    task_name=finetune_method2 \
    experiment=finetune_method2 \
    data=balance \
    +n_q=3 \
    +nq_rank=0 \
    +alpha=0.8 \
    +mask_prob=0.0 \
    model.lr=0.0001 \
    +audio_time=10.0 \
    +temperature=0.05 \
    +is_finetune=True \
    +train_mode=mpc_mlc \
    +model_config=../model_config \
    +pretrain_model_nq0=/home/lizhaohui/workspace/ATT/AudioFormer/logs/pretrain/runs/continue-pre-training_epoch_02/checkpoints/epoch_02.pt \
    +pretrain_model_nq1=/home/lizhaohui/workspace/ATT/AT/logs/pretrain/pretrain_roformer_for_mlm_mix_nq_is_1/runs/2023-07-04_09-49-04/checkpoints/last_backbone.pt \
    +pretrain_model_nq2=/home/lizhaohui/workspace/ATT/AT/logs/pretrain/pretrain_roformer_for_mlm_mix_nq_is_2/step_087844.pt \
    +pretrain_model_nq3=/home/lizhaohui/workspace/ATT/AT/logs/pretrain/pretrain_roformer_for_mlm_mix_nq_is_3/runs/2023-07-07_20-35-29/checkpoints/last_backbone.pt \
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
    paths.data_dir=/home/lizhaohui/workspace/ATT/audioset/nq_4/balance \
    paths.root_dir=/home/lizhaohui/workspace/ATT/AudioFormer \
    +trainer.val_check_interval=1.0 \
    callbacks.early_stopping.patience=3 
