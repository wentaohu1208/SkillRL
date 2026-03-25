# 实验计划：Skill Bank 无限增长对 SkillRL 性能的影响

## 目标

验证假设：**Skill bank 无限增长是否会导致 SkillRL agent 性能下降？**

原论文（arXiv:2602.08234）在 150 个训练步中将 skill bank 从 55 增长到 ~100 个 skills，有三重自然刹车限制增长：
1. `update_threshold=0.4` — 仅在 success rate < 40% 时才触发更新
2. `max_new_skills=3` — 每次 evolution 最多加 3 个 skill
3. `total_epochs=150` — 训练步数有限

**本实验移除全部三重刹车**，推动 skill bank 增长至 500-1000+，观察是否存在性能拐点。

---

## 实验分组（2x2 消融设计）

| 组别 | 名称 | RL 训练 | Skill Evolution | 关键配置 | 目的 |
|------|------|---------|-----------------|----------|------|
| **G1** | Baseline | Yes | 温和（原文） | δ=0.4, max=3, 150 步 | 复现原文（Figure 3） |
| **G2** | Aggressive Growth | Yes | 激进 | δ=0.7, max=8, 500 步, test_freq=3 | 核心实验：测试无限增长 |
| **G3** | No Evolution | Yes | 无 | dynamic_update=False, 500 步 | 控制组：隔离 RL 和 skill 的效果 |
| **G4** | Skill Only | No (lr=0) | 温和（原文） | lr=0, δ=0.4, max=3, 500 步 | 消融：纯 skill 增长无 RL 适应 |

### 2x2 消融矩阵

|  | Skill Evolution OFF | Skill Evolution ON |
|--|---------------------|---------------------|
| **RL OFF (lr=0)** | SFT 裸跑（基线参照） | **G4：只靠 skill 提升** |
| **RL ON** | **G3：只靠 RL 提升** | **G1/G2：两者结合** |

### G2 vs G1 激进在哪

| 参数 | G1 (Baseline) | G2 (Aggressive) | 效果 |
|------|--------------|-----------------|------|
| `update_threshold` | 0.4 | **0.7** | 更容易触发（SR<0.7 就更新） |
| `max_new_skills` | 3 | **8** | 每次生成更多 skill |
| `test_freq` | 5 | **3** | 更频繁 validation -> 更频繁触发更新 |
| `total_epochs` | 150 | **500** | 跑更久让 bank 膨胀 |

预计 G2 skill bank 增长速度约为 G1 的 5-8 倍。

---

## 阶段规划

### 阶段 1：环境与依赖安装
- **状态**：已完成
- **步骤**：
  1. 安装基础依赖
  2. 安装 ALFWorld 环境和数据
  3. 配置 OpenAI 兼容 API（DeepSeek 替代 Azure o3）
  4. 确认 GPU 可用性（需要 4x A800 80GB）

### 阶段 2：数据与模型准备
- **状态**：已完成
- **完成项**：
  - dummy parquet 数据文件已生成
  - SFT checkpoint 下载到 `/data/hwt/hf_data/Alfworld-7B-SFT/checkpoint-140`
  - ALFWorld 数据下载到 `/data/hwt/alfworld_data`（JSON + PDDL + logic）
  - `.env` 文件配置好 API 密钥和路径

### 阶段 3：代码改动
- **状态**：已完成（G1/G3），G2/G4 待创建脚本
- **已完成改动**：

  #### 3.1 `agent_system/memory/skill_updater.py`
  - 新增 `api_backend` 参数，支持 `"azure"` 和 `"openai"` 两种后端
  - `print()` 替换为 `logger`

  #### 3.2 `verl/trainer/ppo/ray_trainer.py`
  - 新增监控指标：`val/skill_bank_total`, `val/skill_bank_general`, `val/skill_bank_task_specific`
  - 新增 `val/overall_success_rate`（per-task 均值，与论文 Table 1 "All" 一致）
  - 新增 `val/avg_prompt_length`, `val/max_prompt_length`（token 级）
  - 传递 `api_backend` 配置给 SkillUpdater

  #### 3.3 实验脚本
  - `experiments/run_baseline.sh` — G1 基线配置（4 卡）
  - `experiments/run_baseline_2gpu.sh` — G1 双卡版
  - `experiments/run_baseline_single.sh` — G1 单卡版
  - `experiments/run_aggressive_growth.sh` — **待创建**
  - `experiments/run_no_evolution.sh` — **待创建**
  - `experiments/run_skill_only.sh` — **待创建**（G4）

### 阶段 4：运行实验
- **状态**：进行中（G1 环境验证中，遇到 OOM 问题）
- **硬件需求**：4x A800 80GB（2 卡不够，optimizer step OOM）
- **执行顺序**：
  1. G1（Baseline）验证环境 + 复现原文（~24h）← 当前在此
  2. G4（Skill Only, lr=0）快速消融（~40h）
  3. G2（Aggressive Growth）核心实验（~80h）
  4. G3（No Evolution）控制组（~60h）
- **监控**：TensorBoard（`tb_logs/`）
- **Checkpoint 保存**：每 10 步保存一次

### 阶段 5：结果分析与可视化
- **状态**：未开始
- **产出物**：
  1. **核心图**：Success Rate vs Training Steps（4 条曲线 G1-G4）
  2. **Skill Bank 增长曲线**：Skill Bank Size vs Training Steps
  3. **Context Pollution 图**：avg_prompt_length vs overall_success_rate
  4. **任务维度拆解**：每个 ALFWorld 任务类型的 success rate
  5. **拐点识别**：如果存在倒 U 型曲线，在哪个规模出现

---

## 关键监控指标（每次 validation step 记录）

| 指标 | 来源 | 用途 |
|------|------|------|
| `step` | trainer | 时间轴 |
| `val/skill_bank_total` | 新增代码 | 核心自变量 |
| `val/skill_bank_general` | 新增代码 | 分项统计 |
| `val/skill_bank_task_specific` | 新增代码 | 分项统计 |
| `val/{task}_success_rate` | 已有 | 每类任务因变量 |
| `val/overall_success_rate` | 新增代码 | 整体因变量（per-task 平均） |
| `val/avg_prompt_length` | 新增代码 | context pollution 代理指标 |
| `val/max_prompt_length` | 新增代码 | prompt 峰值监控 |

---

## 预期实验结果与结论映射

| 结果 | 说明 |
|------|------|
| G2 > G1 > G3 > G4 | 更多 skill + RL 始终有益，原文太保守 |
| G1 > G2 > G3 > G4 | **存在最优 skill bank 大小**，过多反而有害 |
| G1 ≈ G2 > G3 > G4 | RL 足够强，能 adapt 掉噪声 skill |
| G2 先涨后跌（倒 U 型） | 最有价值——存在 tipping point |
| G4 持续涨 | skill evolution 本身有价值，不依赖 RL |
| G4 不涨或跌 | 没有 RL adaptation，光堆 skill 没用 |

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Prompt 超过 `max_prompt_length=4096` | truncation error 导致训练崩溃 | G2 预防性增大到 8192 |
| vLLM 长 prompt OOM | 训练崩溃 | 降低 `gpu_memory_utilization`；减小 `max_num_batched_tokens` |
| 2 卡 / 单卡 OOM | optimizer step 爆显存 | **必须 4 卡**（已验证 2 卡不够） |
| DeepSeek API 限速 | skill evolution 停滞 | 预算约 200 次 API 调用 |
| Ray 临时目录撑满 `/tmp` | 训练崩溃 | 已设置 `RAY_TMPDIR=/data/hwt/ray_tmp` |
| 实验耗时过长（>80h） | GPU 资源争用 | 如果到 300 步趋势明确可提前结束 |

---

## 决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-03-24 | 选择重跑 RL（方案 A）而非固定 ckpt 评估 | 方案 B 无法区分 OOD 效应和 skill bank 规模效应 |
| 2026-03-24 | 设置 δ=0.9 → 0.7 作为激进增长阈值 | 0.7 更合理，0.9 可能导致高质量 skill 也被覆盖 |
| 2026-03-24 | 增加无进化控制组（G3） | 隔离"训练更久"和"更多 skills"的效果 |
| 2026-03-24 | 使用 template 检索模式 | Template 注入所有 skills → 直接测试 context pollution |
| 2026-03-25 | 用 DeepSeek 替代 Azure o3 | 无 Azure 访问权限；三组实验用同一 API，组间对比公平 |
| 2026-03-25 | 必须 4 卡训练 | 2 卡 / 单卡验证均 OOM（optimizer step 阶段） |
| 2026-03-25 | 使用 TensorBoard 替代 Wandb | 服务器无法访问外网 Wandb |
| 2026-03-25 | 新增 G4（Skill Only, lr=0） | 2x2 消融实验设计，解耦 RL adaptation 和 skill evolution 的各自贡献 |
