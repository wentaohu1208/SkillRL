#!/bin/bash
# =============================================================================
# G2: Aggressive Growth — 激进 skill bank 增长 (δ=0.95, max=8, 500 steps)
#
# 与 G1 的差异:
#   update_threshold: 0.4 → 0.95（几乎每次 validation 都触发）
#   max_new_skills:   3   → 8（每次生成更多）
#   test_freq:        5   → 3（更频繁触发）
#   total_epochs:     150 → 500（跑更久让 bank 膨胀）
#   max_prompt_length:4096→ 8192（防止 skill 过多导致 truncation error）
#
# 用法:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/run_aggressive_growth.sh
# =============================================================================
set -x

ENGINE=${1:-vllm}
shift || true

# ---------- 加载环境变量（API 密钥等） ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_BACKEND_LOG_LEVEL=debug
export VLLM_LOGGING_LEVEL=DEBUG
export HF_ENDPOINT=https://hf-mirror.com

# ---------- Wandb ----------
export WANDB_NAME="g2_aggressive_growth"

# ---------- 数据规模 ----------
train_data_size=16
val_data_size=64
group_size=8
num_cpus_per_env_worker=0.1

# ---------- 准备 dummy parquet ----------
python3 -m examples.data_preprocess.prepare_offline \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# ---------- 选择 API 后端 ----------
if [ -n "$SKILL_API_KEY" ]; then
    API_BACKEND="openai"
else
    API_BACKEND="azure"
fi

# ---------- 训练 ----------
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/alfworld/claude_style_skills.json \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.enable_dynamic_update=True \
    +env.skills_only_memory.update_threshold=0.95 \
    +env.skills_only_memory.max_new_skills=8 \
    +env.skills_only_memory.api_backend=$API_BACKEND \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='skillrl_scaling_experiment' \
    trainer.experiment_name='g2_aggressive_growth' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=3 \
    trainer.total_epochs=500 \
    trainer.val_before_train=False $@
