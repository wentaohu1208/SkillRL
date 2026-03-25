# 进度日志：SkillRL Skill Bank 规模化实验

## 第 1 次会话 -- 2026-03-24

### 已完成
- [x] 阅读并分析 SkillRL 论文（arXiv:2602.08234）-- 18 页含附录
- [x] 完整探索代码库结构，定位关键文件
- [x] 分析现有 skill bank：53 个 skills（12 通用，30 任务特定，11 常见错误）
- [x] 识别原始设计中的三重增长刹车（threshold、max_new_skills、total_epochs）
- [x] 确定采用方案 A（重跑 RL）而非方案 B（固定 ckpt 评估）-- 科学性更强
- [x] 设计 3 组实验：Baseline / Aggressive Growth / No Evolution
- [x] 确定具体代码改动（3 个文件，3 处修改）
- [x] 起草完整实验脚本（含全部超参数）
- [x] 创建规划文件（task_plan.md、findings.md、progress.md）

### 关键决策
1. **重跑 RL（非固定 ckpt 评估）**：避免混淆 OOD 效应和 skill bank 规模效应
2. **使用 template 模式（非 embedding）**：template 注入所有 skills -> 直接测试 context pollution
3. **三组实验设计**：Baseline + Aggressive Growth + No Evolution 控制组

---

## 第 2 次会话 -- 2026-03-25

### 已完成
- [x] `skill_updater.py`：新增 `api_backend="openai"` 支持 DeepSeek 等 OpenAI 兼容 API
- [x] `skill_updater.py`：`print()` 全部替换为 `logger`
- [x] `ray_trainer.py`：新增 `val/skill_bank_total/general/task_specific` 监控指标
- [x] `ray_trainer.py`：新增 `val/overall_success_rate`（per-task 均值）
- [x] `ray_trainer.py`：新增 `val/avg_prompt_length` 和 `val/max_prompt_length`
- [x] `ray_trainer.py`：传递 `api_backend` 配置给 SkillUpdater
- [x] code-review 通过，修复 `overall_success_rate` 计算 bug（排除总体 key）
- [x] 创建 `experiments/run_baseline.sh`（G1, 4 卡）
- [x] 创建 `experiments/run_baseline_2gpu.sh`（G1, 2 卡）
- [x] 创建 `experiments/run_baseline_single.sh`（G1, 单卡）
- [x] 创建 `.gitignore`
- [x] 创建 `.env`（含 API 密钥、模型路径）
- [x] 服务器环境配置：conda 环境 skillrl, pip 依赖安装
- [x] SFT checkpoint 下载到 `/data/hwt/hf_data/Alfworld-7B-SFT/checkpoint-140`
- [x] ALFWorld 数据下载到 `/data/hwt/alfworld_data`（JSON + PDDL + logic）
- [x] HuggingFace 镜像配置（`HF_ENDPOINT=https://hf-mirror.com`）
- [x] Ray 临时目录配置（`RAY_TMPDIR=/data/hwt/ray_tmp`）
- [x] TensorBoard 替代 Wandb（服务器外网受限）
- [x] G1 训练首次启动 -- 环境初始化成功，模型加载成功

### 遇到的问题
1. **单卡 OOM**：7B 模型 GRPO 训练，optimizer step 时 CUDA out of memory
2. **双卡 OOM**：同样在 optimizer step 阶段爆显存（`gpu_memory_utilization=0.3` 也不够）
3. **结论**：必须 4 卡才能跑 7B GRPO（原文配置）
4. `/tmp` 磁盘被 Ray session 撑满 -> 迁移到 `/data/hwt/ray_tmp`
5. `alfworld-download -f` PDDL 下载失败（416 错误）-> 手动从 GitHub release 下载合并
6. Hydra 参数覆盖语法冲突（命令行追加重复参数）-> 直接改脚本

### 新增实验设计
- **G4（Skill Only, lr=0）**：固定模型权重 + skill bank 自然增长
- 目的：解耦 RL adaptation 和 skill evolution 的各自贡献
- 实现：`actor_rollout_ref.actor.optim.lr=0`

### 待办事项
- [ ] 用 4 卡跑通 G1 baseline（等待 GPU 空闲）
- [ ] 创建 G2 激进增长脚本 (`experiments/run_aggressive_growth.sh`)
- [ ] 创建 G3 无进化脚本 (`experiments/run_no_evolution.sh`)
- [ ] 创建 G4 纯 skill 增长脚本 (`experiments/run_skill_only.sh`)
- [ ] 跑完 G1 后依次跑 G4 -> G2 -> G3
- [ ] 结果分析与可视化

### 开放问题
1. G2 的 `max_prompt_length` 是否需要从 4096 增大到 8192？（skill bank 500+ 时 prompt 可能超长）
2. G4 的 `lr=0` 是否会导致 verl 框架异常？（需要验证 optimizer 不会除零）
3. 是否需要追踪每个 skill 的实际使用频率？

### 阻塞项
- 需要 4 张空闲 A800 GPU 才能继续训练

---

## 会话模板

### 第 N 次会话 -- YYYY-MM-DD

#### 已完成
- [ ] ...

#### 遇到的问题
- ...

#### 下一步
- [ ] ...
