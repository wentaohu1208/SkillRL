# 进度日志：SkillRL Skill Bank 规模化实验

## 第 1 次会话 — 2026-03-24

### 已完成
- [x] 阅读并分析 SkillRL 论文（arXiv:2602.08234）— 18 页含附录
- [x] 完整探索代码库结构，定位关键文件
- [x] 分析现有 skill bank：53 个 skills（12 通用，30 任务特定，11 常见错误）
- [x] 识别原始设计中的三重增长刹车（threshold、max_new_skills、total_epochs）
- [x] 确定采用方案 A（重跑 RL）而非方案 B（固定 ckpt 评估）— 科学性更强
- [x] 设计 3 组实验：Baseline / Aggressive Growth / No Evolution
- [x] 确定具体代码改动（3 个文件，3 处修改）
- [x] 起草完整实验脚本（含全部超参数）
- [x] 创建规划文件（task_plan.md、findings.md、progress.md）

### 关键决策
1. **重跑 RL（非固定 ckpt 评估）**：避免混淆 OOD 效应和 skill bank 规模效应
2. **使用 template 模式（非 embedding）**：template 注入所有 skills → 直接测试 context pollution
3. **三组实验设计**：Baseline + Aggressive Growth + No Evolution 控制组

### 待办事项
- [ ] 将代码改动应用到 `ray_trainer.py` 和 `skill_updater.py`
- [ ] 创建实验 shell 脚本到 `experiments/` 目录
- [ ] 在 GPU 集群上配置环境
- [ ] 从 HuggingFace 下载 SFT checkpoint
- [ ] 先跑 baseline 实验验证环境
- [ ] 跑 aggressive growth 核心实验
- [ ] 跑 no evolution 控制组实验
- [ ] 分析结果并生成可视化

### 开放问题
1. 是否应预防性地将 `max_prompt_length` 从 4096 增大到 6000？还是等出问题再改？
2. 是否需要追踪每个 skill 的使用频率（模型在 `<think>` 标签中实际引用了哪些 skills）？
3. 是否应该比默认的 `save_freq=10` 更频繁地保存中间 skill bank 快照？

### 阻塞项
- 暂无。需确认 GPU 集群可用性。

---

## 会话模板

### 第 N 次会话 — YYYY-MM-DD

#### 已完成
- [ ] ...

#### 遇到的问题
- ...

#### 下一步
- [ ] ...
