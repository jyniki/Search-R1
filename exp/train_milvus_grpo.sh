export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR='exp/dataset'

WAND_PROJECT='Search-R1'
export WANDB_API_KEY=local-cc3446e3021d25de41e2aab6a2ad846ad77e5170
export BASE_MODEL='/rt-vepfs/public_model/Qwen/Qwen2.5-7B'
export EXPERIMENT_NAME=search-r1-grpo-qwen2.5-7b-em-a800
export HYDRA_FULL_ERROR=1

# max_prompt_length = max_start_length + max_response_length * (max_turns - 1) + max_obs_length * max_turns
# = 1024 + 512 * (3 - 1) + 2048 * 3 = 8192

# set -x
# 先运行下面这行这个再启动 ray !!!!
export VLLM_ATTENTION_BACKEND=XFORMERS 

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
    data.max_start_length=1024 \
    data.max_obs_length=2048 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.project_name=$WAND_PROJECT \
    trainer.logger=["wandb"] \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=100 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=outputs/verl_checkpoints/$EXPERIMENT_NAME \
    trainer.auto_resume=True \
    trainer.resume_from_checkpoint=null \
    trainer.checkpoint_cleanup=True \
    trainer.max_checkpoints_to_keep=5 \
    max_turns=3 \
    retriever.search_engine=milvus \
    retriever.topk=3 \
    2>&1 | tee outputs/logs/$EXPERIMENT_NAME.log
