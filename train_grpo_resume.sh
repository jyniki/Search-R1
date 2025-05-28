export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR='/rt-vepfs/jy/dataset/nq_data/nq_hotpotqa'
# export DATA_DIR='/rt-vepfs/jy/dataset/nq_data/nq_search'

WAND_PROJECT='Search-R1'

export BASE_MODEL='/rt-vepfs/public_model/Qwen/Qwen2.5-7B'
export EXPERIMENT_NAME=search-r1-grpo-qwen2.5-7b-em-resume

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
export SEARCH_URL="http://127.0.0.1:7863/retrieve"

echo "Testing search API..."
curl -X POST "$SEARCH_URL" \
  -H "Content-Type: application/json" \
  -d '{"queries": ["test query"], "topk": 3, "return_scores": true}' \
  --connect-timeout 5 \
  --max-time 10 \
  || { echo "Search API not available at $SEARCH_URL"; exit 1; }
echo "Search API test passed!"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=1024 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=1024 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=128 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=outputs/verl_checkpoints/$EXPERIMENT_NAME \
    trainer.auto_resume=False \
    trainer.resume_from_checkpoint=outputs/verl_checkpoints/search-r1-grpo-qwen2.5-7b-em/global_step_10 \
    trainer.checkpoint_cleanup=True \
    trainer.max_checkpoints_to_keep=5 \
    max_turns=3 \
    retriever.url=$SEARCH_URL \
    retriever.topk=3 \
    2>&1 | tee outputs/logs/$EXPERIMENT_NAME.log