# 实验计划：Skill Bank 无限增长对 SkillRL 性能的影响

## 目标

验证假设：**Skill bank 无限增长是否会导致 SkillRL agent 性能下降？**

原论文（arXiv:2602.08234）在 150 个训练步中将 skill bank 从 55 增长到 ~100 个 skills，有三重自然刹车限制增长：
1. `update_threshold=0.4` — 仅在 success rate < 40% 时才触发更新
2. `max_new_skills=3` — 每次 evolution 最多加 3 个 skill
3. `total_epochs=150` — 训练步数有限

**本实验移除全部三重刹车**，推动 skill bank 增长至 500-1000+，观察是否存在性能拐点。

---

## 实验分组

| 组别 | 名称 | 关键配置 | 目的 |
|------|------|----------|------|
| **G1** | Baseline（基线） | δ=0.4, max=3, 150 步 | 复现原文（Figure 3） |
| **G2** | Aggressive Growth（激进增长） | δ=0.9, max=10, 500 步 | 核心实验：测试无限增长 |
| **G3** | No Evolution（无进化） | dynamic_update=False, 500 步 | 控制组：隔离"训练更久"和"更多 skills"的效果 |

---

## 阶段规划

### 阶段 1：环境与依赖安装
- **状态**：未开始
- **步骤**：
  1. 安装基础依赖（`requirements.txt`、vllm、flash-attn）
     ```bash
     cd /Users/wentaohu/project/SkillRL
     pip install -r requirements.txt
     pip install vllm==0.11.0
     pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
     pip install -e .
     ```
  2. 安装 `openai` 包（Azure API 调用需要）
     ```bash
     pip install openai
     ```
  3. 安装 ALFWorld 环境
     ```bash
     pip install alfworld
     pip install gymnasium==0.29.1
     pip install stable-baselines3==2.6.0
     alfworld-download -f  # 下载 PDDL & Game 文件 (~2GB)
     ```
  4. 设置 Azure OpenAI 环境变量（dynamic evolution 调用 o3 需要）
     ```bash
     export AZURE_OPENAI_API_KEY="你的key"
     export AZURE_OPENAI_ENDPOINT="你的endpoint"
     ```
  5. 确认 GPU 可用性（需要 4-8× H100 80GB）

### 阶段 2：数据与模型准备
- **状态**：未开始
- **步骤**：
  1. 准备 dummy parquet 数据文件（仅标记 modality，实际任务数据来自 ALFWorld 环境）
     ```bash
     python3 -m examples.data_preprocess.prepare --mode 'text' --train_data_size 16 --val_data_size 64
     ```
     输出：`$HOME/data/verl-agent/text/{train,test}.parquet`
  2. 下载 SFT checkpoint
     - 地址：https://huggingface.co/Jianwen/Alfworld-7B-SFT
     - 设置：`export MODEL_PATH=/path/to/Alfworld-7B-SFT`
  3. 做一次快速 dry-run 验证模型能正常加载

### 阶段 3：代码改动
- **状态**：未开始
- **需要修改的文件**：

  #### 3.1 `verl/trainer/ppo/ray_trainer.py` — 增加 skill bank size 监控
  - **改动 1（约第 835 行）**：在 `return metric_dict` 之前加入 skill bank 大小的 logging
    ```python
    # 在 return metric_dict 之前加入：
    if hasattr(self, 'envs') and hasattr(self.envs, 'retrieval_memory') and self.envs.retrieval_memory:
        skill_count = self.envs.retrieval_memory.get_skill_count()
        metric_dict['val/skill_bank_total'] = skill_count['total']
        metric_dict['val/skill_bank_general'] = skill_count['general']
        metric_dict['val/skill_bank_task_specific'] = skill_count['task_specific']
    ```
  - **改动 2（第 938 行）**：增大失败轨迹收集数量
    ```python
    # 原文：return failed[:10]
    # 改为：
    return failed[:20]
    ```

  #### 3.2 `agent_system/memory/skill_updater.py` — 增大失败分析样本数
  - **改动 1（第 144 行）**：允许更多失败样本进入 prompt
    ```python
    # 原文：for i, traj in enumerate(failed_trajectories[:5]):
    # 改为：
    for i, traj in enumerate(failed_trajectories[:10]):
    ```

  #### 3.3 创建实验脚本 `experiments/`
  - `experiments/run_baseline.sh` — G1 基线配置（与原文一致）
  - `experiments/run_aggressive_growth.sh` — G2 激进增长配置
  - `experiments/run_no_evolution.sh` — G3 无进化控制组

### 阶段 4：运行实验
- **状态**：未开始
- **执行顺序**：
  1. 先跑 G1（Baseline）验证环境无误（~24h）
  2. 再跑 G2（Aggressive Growth）核心实验（~80h）
  3. 最后跑 G3（No Evolution）控制组（~60h，无 o3 API 调用）
- **实时监控**：通过 Wandb dashboard 观察指标
- **Checkpoint 保存**：每 10 步保存一次（`trainer.save_freq=10`）

### 阶段 5：结果分析与可视化
- **状态**：未开始
- **产出物**：
  1. **核心图**：Success Rate vs Skill Bank Size（3 条曲线）
  2. **任务维度拆解**：每个 ALFWorld 任务类型的 success rate vs skill bank size
  3. **Context 效率**：平均 prompt 长度（tokens）vs skill bank size
  4. **Skill 质量分析**：检查不同规模阶段生成的 skills — 后期的 skill 是否质量更低 / 更冗余？
  5. **拐点识别**：如果存在拐点，在哪个规模出现？

---

## 关键监控指标（每次 validation step 记录）

| 指标 | 来源 | 用途 |
|------|------|------|
| `step` | trainer | 时间轴 |
| `val/skill_bank_total` | 新增代码 | 核心自变量 |
| `val/skill_bank_general` | 新增代码 | 分项统计 |
| `val/skill_bank_task_specific` | 新增代码 | 分项统计 |
| `val/{task}_success_rate` | 已有 | 每类任务因变量 |
| `val/overall_success_rate` | 已有 | 整体因变量 |
| 平均 prompt 长度（tokens） | vllm 日志 | context pollution 代理指标 |

---

## 三组实验的关键配置差异

### G1: Baseline（基线，复现原文）
```bash
+env.skills_only_memory.update_threshold=0.4
+env.skills_only_memory.max_new_skills=3
+env.skills_only_memory.enable_dynamic_update=True
trainer.total_epochs=150
trainer.experiment_name='baseline_reproduce'
```

### G2: Aggressive Growth（激进增长，核心实验）
```bash
+env.skills_only_memory.update_threshold=0.9     # 几乎每次 validation 都触发
+env.skills_only_memory.max_new_skills=10         # 每次最多加 10 个
+env.skills_only_memory.enable_dynamic_update=True
trainer.total_epochs=500                           # 跑更久让 bank 膨胀
trainer.experiment_name='aggressive_skill_growth'
```

### G3: No Evolution（无进化控制组）
```bash
+env.skills_only_memory.enable_dynamic_update=False  # 关闭 dynamic update
trainer.total_epochs=500                              # 同样跑 500 步
trainer.experiment_name='no_evolution_500steps'
```

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Prompt 超过 `max_prompt_length=4096` | 触发 truncation error 导致训练崩溃 | 预防性增大到 6000；监控 prompt 长度 |
| vLLM 长 prompt 导致 OOM | 训练崩溃 | 降低 `gpu_memory_utilization` 从 0.5 到 0.4；减小 `max_num_batched_tokens` |
| Azure o3 API 限速 / 费用 | Evolution 停滞或成本飙升 | 预算约 100 次 API 调用；设置消费告警 |
| Skill 语义重复 | Bank 数量增长但多样性不增 | 这是预期行为 — 正是我们要测试的"噪声 skill 对性能的影响" |
| 实验耗时过长（>80h） | GPU 资源争用 | 如果到 300 步时趋势已明确，可提前结束 |

---

## 决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-03-24 | 选择重跑 RL（方案 A）而非固定 ckpt 评估（方案 B） | 方案 B 无法区分 OOD 效应和 skill bank 规模效应；方案 A 让模型与 skills 共同演化 |
| 2026-03-24 | 设置 δ=0.9 作为激进增长阈值 | 确保即使 success rate 达到 89% 也继续触发 evolution |
| 2026-03-24 | 增加无进化控制组（G3） | 需要隔离"训练更多步"的效果和"更多 skills"的效果 |
| 2026-03-24 | 使用 template 检索模式（而非 embedding） | Template 模式注入所有相关 skills → 直接测试 context pollution；embedding 模式的 top-k 会掩盖效果 |
