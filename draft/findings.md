# 研究发现：SkillRL Skill Bank 规模化实验

## 论文分析（arXiv:2602.08234）

### SkillRL 流程（Algorithm 1）
1. **Skill 蒸馏**：基座模型收集轨迹 -> 教师模型（o3）蒸馏为结构化 skills
2. **SkillBank 构建**：分层结构 -- 通用技能（S_g）+ 任务特定技能（S_k）
3. **Cold-Start SFT**：用教师生成的 skill 增强轨迹微调基座模型
4. **RL + 递归进化**：GRPO 训练，在 validation 检查点动态更新 skill bank

### 原文 Skill Bank 增长数据（Figure 3）
- 初始：55 个 skills（12 通用，43 任务特定）
- 结束（step 150）：~100 个 skills（20 通用，80 任务特定）
- 增长速率：150 步内增加约 45 个（约 30 次 validation 机会）
- 增长主要来自任务特定技能（43 -> 80）
- 通用技能增长缓慢（12 -> 20）

### 关键性能数据（Table 1，ALFWorld）
| 方法 | Pick | Look | Clean | Heat | Cool | Pick2 | 总计 |
|------|------|------|-------|------|------|-------|------|
| SkillRL | 97.9 | 71.4 | 90.0 | 95.5 | 87.5 | 89.9 | **90.0** |
| GRPO*（无 skills） | 90.8 | 66.1 | 89.3 | 74.7 | 72.5 | 64.7 | 77.6 |
| 无动态进化 | -- | -- | -- | -- | -- | -- | **84.4** |

### 消融实验洞察（Table 3）
- 去掉分层结构（只保留任务特定 skills）：89.9 -> 76.8（-13.1%）
- 用原始轨迹替换 skills：89.9 -> 61.7（-28.2%）
- 去掉 cold-start SFT：89.9 -> 65.2（-24.7%）
- 去掉动态进化：89.9 -> 84.4（-5.5%）

### Context 效率（Figure 4）
- 原始记忆方案：平均 ~1,450 tokens/prompt（波动大）
- SkillRL：平均 <1,300 tokens/prompt（稳定，压缩 10.3%）

---

## 代码架构分析

### Skill Bank 存储
- **文件**：`memory_data/alfworld/claude_style_skills.json`
- **格式**：JSON，包含 `general_skills`、`task_specific_skills`（按类别分）、`common_mistakes`
- **当前规模**：12 通用 + 30 任务特定 + 11 常见错误 = 53 总计

### Skill 检索（Template 模式）
- **文件**：`agent_system/memory/skills_only_memory.py`
- **行为**：返回检测到的类别的**全部**任务特定 skills + 前 `top_k` 个通用 skills
- **动态 skills**：以 `dyn_NNN` 为前缀的 skills 始终被包含（绕过通用 skills 的 top_k 限制）
- **关键含义**：任务特定 skills 增长时，**全部**被注入 prompt -> prompt 线性增长

### 动态进化触发机制
- **文件**：`verl/trainer/ppo/ray_trainer.py`（第 860-942 行）
- **触发条件**：每次 validation 后调用 `_update_skills_from_validation()`
- **判断逻辑**：任意任务类别 `success_rate < update_threshold` 时触发
- **流程**：
  1. 收集失败轨迹（score <= 0），最多 10 条
  2. SkillUpdater 调用 LLM 分析失败，生成 1-3 个新 skill
  3. 新 skill 分配 `dyn_NNN` ID，仅加入训练环境
  4. 保存快照到 `updated_skills_step{N}.json`
- **防泄漏**：新 skills **不**加入 validation 环境

### SkillUpdater
- **文件**：`agent_system/memory/skill_updater.py`
- **API 后端**：支持 Azure OpenAI 和 OpenAI 兼容 API（DeepSeek 等）
- **去重**：已有 skill 标题传给 LLM prompt；dyn_ ID 在服务端重新分配
- **上限**：`max_new_skills_per_update`（默认 3）

### ALFWorld 任务类型
| 代码名称 | 论文名称 |
|----------|---------|
| `pick_and_place` | Pick |
| `look_at_obj_in_light` | Look |
| `pick_clean_then_place_in_recep` | Clean |
| `pick_heat_then_place_in_recep` | Heat |
| `pick_cool_then_place_in_recep` | Cool |
| `pick_two_obj_and_place` | Pick2 |

训练和验证**不分任务类型**，所有 6 类混在一个 batch 中。Success rate 通过解析 gamefile 路径名分类统计。

---

## 实验假设

### H1：Context Pollution（上下文污染）
随着 skill bank 增长，prompt 变长，越来越多边际/冗余的 skills 被注入。模型的注意力被稀释，难以在数百个 skills 中识别最相关的。**预期信号**：success rate 先升后降；prompt 长度与性能下降相关。

### H2：信噪比退化
后期生成的 skills 基于越来越罕见的失败模式。这些 skills 可能过于特定、与通用原则矛盾、或纯属噪声。与高质量初始 skills 一起注入时，会稀释整体 skill 质量。

### H3：检索质量退化（Template 模式特有）
Template 模式返回**所有**任务特定 skills。当每个任务类型有 120 个 skills 时，模型接收到的 skills 数量是原始 5-6 个的 ~20 倍，其中大部分是边际的。

### H4：RL Adaptation 能力上限
即使有 RL 去适应膨胀的 skill bank，策略网络的容量和 attention 机制可能存在处理超长 prompt 的瓶颈。G2 测试的正是：RL adaptation 能否无限 scale。

### 零假设
模型能有效忽略无关 skills，性能会 plateau 而非下降。SkillRL 框架对 skill bank 规模化是鲁棒的。

---

## 环境配置发现

### 硬件需求
- **4x A800 80GB 是最低要求**（7B 模型 GRPO 训练）
- 2 卡验证失败：optimizer step OOM（Adam 动量需要 2x 模型大小显存）
- 单卡验证失败：同样 OOM
- 原因：co-located 模式下 Actor/Rollout/Ref 共享 GPU，FSDP offload 仍不够

### Ray 临时目录
- 默认 `/tmp`，容易撑满
- 解决：`export RAY_TMPDIR=/data/hwt/ray_tmp`

### HuggingFace 访问
- 服务器无法直连 huggingface.co
- 解决：`export HF_ENDPOINT=https://hf-mirror.com`

### SFT Checkpoint 结构
- 下载的 `Jianwen/Alfworld-7B-SFT` 实际模型在 `checkpoint-140/` 子目录
- `MODEL_PATH` 需指向 `checkpoint-140/`

---

## 相关工作与对比

### 其他 Agent 系统中的记忆规模化
- **Mem0**（Chhikara et al., 2025）：扩展记忆但在复杂环境中表现不佳（ALFWorld 21.4%）
- **MemRL**（Zhang et al., 2026）：用 RL 更新记忆但只达到 22.2% -- 原始轨迹方案失效
- **EvolveR**（Wu et al., 2025）：43.8% -- 有进步但仍受轨迹存储开销限制
- **SimpleMem+GRPO**：60.0% -- 结合记忆和 RL，但无 skill 抽象

以上工作**均未研究**记忆/skill bank 本身的规模化行为。

### 长上下文研究
- 长上下文模型可以处理 100K+ tokens，但**有效利用率**超过一定长度后下降
- "Lost in the Middle"（Liu et al., 2024）：模型难以利用长上下文中间部分的信息
- 这与本实验直接相关：skill bank 增长后，关键 skills 可能被"淹没"在大量 prompt 内容中
