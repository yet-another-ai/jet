# Jet 开发计划

更新日期：2026-09-21

状态：开发前的实施基线；本文不表示相关功能已经实现或性能已经验证。

## 1. 项目定位与目标

Jet 是一个 **Judgement Engine**。输入为 `question` 和一组 `options`，复用现有自回归 Chat 模型，在不采样、不生成自由文本的情况下，计算每个候选答案的条件对数似然，并在该问题的候选集合内归一化。

核心工作负载是 **teacher-forced sequence scoring**：候选 token 全部已知，通过批量前向计算取得每个目标 token 的条件概率。优化重点是多问题/多候选批处理、共同前缀复用，以及 logits 的高效归约。

首版目标：

- 提供可嵌入的 Rust API 和用于运行、验证、压测的 CLI。
- 在 NVIDIA CUDA 和 Apple Silicon Metal 上运行同一套评分逻辑，保留 CPU 正确性基线。
- 使用本地 GGUF 模型，模型格式和硬件执行由 llama.cpp 负责。
- 返回可解释、可复现的候选评分；明确其与答案正确率之间的区别。
- 在有足够请求时提高持续评分吞吐，同时保持有界内存和可配置的等待时间。

本阶段不包含浏览器/WASM、TTS、聊天 UI、自由文本生成、Agent 工具循环、模型训练和云模型聚合。HTTP 服务、其他推理后端、多 GPU、跨机器执行可在核心引擎稳定后独立评估。

`xlai` 是原生构建和模型封装的参考，不是需要复制的产品架构。Jet 不依赖其聊天运行时；不修改同级 `xlai` 项目。

## 2. 已确定的方向与实施默认值

| 项目 | 首版约定 |
| --- | --- |
| 核心任务 | 对给定候选做条件似然评分，无 sampler |
| 推理底座 | Rust + llama.cpp C API/薄 C++ bridge |
| 模型范围 | 首先验证 llama.cpp 支持的 decoder-only causal Transformer Chat GGUF 模型 |
| 加速后端 | Linux/Windows NVIDIA CUDA；macOS Apple Silicon Metal；CPU 基线 |
| 原始分数 | 目标 token 的完整词表 log-softmax 之和 |
| 候选归一化 | 对每个问题内部的原始累计 log-probability 做 softmax |
| 长度处理 | 默认不除以 token 数，不加 length penalty |
| 温度与采样变换 | 使用原始模型分布，温度固定为 1，不应用 top-k/top-p、重复惩罚、grammar mask |
| 终止策略 | 默认 `continuation`；可显式选择经过验证的 `assistant_turn` |
| 调度方式 | 常驻模型、受控 context/sequence 生命周期、按 token 和内存预算组批 |
| 第一优先级 | 正确的原始分数，其次是批处理收益，再进行内核级优化 |

上表中的技术默认值是实施起点。若验证表明需要改变，应更新本文、接口版本或评分配置版本，不能隐式改变已有请求的含义。

混合注意力、滑动窗口、循环状态模型、MoE 和多模态模型不因文件格式为 GGUF 就自动视为已支持；逐类验证评分、序列隔离和状态共享能力后再加入支持矩阵。

## 3. 评分的数学定义

### 3.1 共同条件与目标 token

设 `x` 为该问题所有候选共同使用的 prompt token 序列；候选 `i` 的实际计分序列为 `y_i = [y_i,1, ..., y_i,L_i]`。

```text
token_logprob[i, t]
  = logits(x, y_i,<t)[y_i,t]
    - logsumexp(logits(x, y_i,<t)[全部词表 token])

log_probability[i]
  = sum_t(token_logprob[i, t])

normalizer
  = logsumexp(log_probability[0], ..., log_probability[K-1])

normalized_log_probability[i]
  = log_probability[i] - normalizer

normalized_probability[i]
  = exp(normalized_log_probability[i])
```

实现要求：

- 模型前向在 inference 模式执行，不构建梯度。
- 使用稳定的 log-softmax/logsumexp，不直接连乘 token 概率。
- 参考归约路径用较高精度累计，例如读取 f32 logits、使用 f64 做归约和序列求和。
- 不直接累计目标 token 的原始 logits；不同上下文的词表归一化项不能省略。
- 不在每个位置只对“候选中出现的 token”做 softmax；那会改变为另一种受约束分布。
- 默认不返回原始序列概率的 `exp(log_probability)`，避免其很小时下溢；保留 log 值。
- 极小的 `normalized_probability` 可能下溢为 0，仍保留对应有限的归一化 log 值。
- 明确定义非有限值处理：NaN、正无穷是错误；可解释的负无穷分数可对应 0 权重；若所有候选均为负无穷则返回评分错误，不伪造均匀分布。

### 3.2 概率的含义

输出是模型对给定候选 token 序列的**相对似然权重**，不是经过校准的“答案正确率”。即使所有候选都不合理，归一化后仍然总和为 1。

只有在候选事件互斥、无重复、使用同一条件上下文时，才能进一步把结果解释为“限定输出属于这些候选事件时的条件概率”。默认 `continuation` 遇到互为前缀的候选时，这些事件会重叠，因此统一使用 `normalized_probability` 名称，并报告重叠情况。

评分针对约定 tokenizer 得到的 token 序列，不承诺对所有能解码成同一文本的 tokenization 求概率总和。

### 3.3 长度、校准与判断能力

- 累计 log-probability 保留原始序列似然的含义，同时会受到答案长度和措辞的影响。
- `mean_token_log_probability` 可以作为诊断信息返回，不参与首版默认归一化。
- 后续若提供平均分、length penalty、先验校正或校准温度，应使用独立的 `ranking_score`/评分模式，不能覆盖原始分数或冒充原始概率。
- 使用已有 Chat 模型不保证每类判断任务都有足够准确率；应有与目标应用相匹配的标注集验收。
- 对会先输出思考过程的模型，固定支持的非思考/直接回答模式。直接答案似然不等于对所有可能推理路径边缘化后的答案概率。

## 4. Prompt、分词和终止契约

### 4.1 Prompt 构造

首版默认行为：

1. 将 `question` 放入 user 消息，可配置一个固定 system 指令。
2. 用经过验证的模型 chat template 生成 assistant 回答开始前的共同前缀。
3. 将各 option 作为该 assistant 消息的候选续写。
4. 默认不把整个 options 列表写入 user prompt。这样增删其他选项只改变归一化分母，不改变保留选项的原始条件分数。

需要“看完所有选项再选答案”的任务可后续增加显式 prompt 模式；这种模式会改变共同条件，应单独版本化和评估。

记录模板内容哈希、system 指令/配置标识、tokenizer/模型标识、评分契约版本及终止策略。禁止使用字符串拼接模拟所有模型的 chat template。

### 4.2 分词边界

- 不假定 `tokenize(prefix) + tokenize(option)` 等于 `tokenize(prefix + option)`。
- 使用模型正确的 assistant 边界生成稳定 prompt，验证候选内容开始位置；必要时借助模板的消息/内容边界信息。
- 为支持的模板明确 BOS、特殊 token、前导空格和结尾控制 token 的处理；候选 suffix 不重复添加 BOS。
- options 按普通答案文本处理，不允许字符串中的特殊标记未经约定就变成控制 token。
- 保留原始空格、换行和 Unicode 文本，不做隐式 trim、大小写转换或 Unicode 归一化。
- prepared request 中保存完整 token 序列、共同前缀范围、计分 token 范围和输出位置映射。
- 如果某模板存在无法可靠分离的跨边界 token，先明确其 token 级条件定义并验证；首版无法支持时应拒绝该模板，不静默使用错误的 target mask。
- 前缀复用以实际相同的 token 和模型状态配置为准，不能只比较可见字符串。

### 4.3 终止策略

| 策略 | 计分目标 | 解释 |
| --- | --- | --- |
| `continuation`（默认） | option 内容 token | 输出以该候选 token 序列开头的似然，不包含“此处停止” |
| `assistant_turn` | option 内容 + 该模板约定的结束 token 序列 | 输出该答案并以指定方式结束 assistant 回合的似然 |

`assistant_turn` 要求已验证的结束序列，不能默认认为 EOS 等于 end-of-turn。若模型存在多个合法结束形式，首版使用一个明确的规范结束序列，不声称对所有结束形式做了边缘化。模板无法提供可靠结束策略时返回不支持错误。

结果分别报告 `option_token_count` 和 `scored_token_count`，后者包含实际计分的结束 token。

### 4.4 输入边界与异常

- 空 question、空 options 列表、空字符串 option、tokenization 后无内容 token 的 option：首版明确拒绝。
- 单个有效 option 可以评分，归一化权重为 1；这不表达确信程度。
- 同一问题中，最终计分 token 序列完全相同的重复选项返回重复候选错误，不能重复计入分母。
- 互为前缀的候选可以按 `continuation` 评分，但返回 `candidate_events_overlap` 等明确诊断。
- prompt 加目标超出模型/context 限制时返回明确错误；允许分块计算，不允许通过分块绕过最大语义上下文或静默截断。
- 任一候选失败时该问题整体失败，不对剩余候选悄悄重新归一化；同一批的其他问题可独立完成。

## 5. 批量前向评分算法

### 5.1 先建立独立参考路径

对每个候选独立执行 teacher forcing，不共享前缀、不跨问题组批。它作为优化实现的 oracle，而不是最终吞吐方案。

```text
tokens = prompt_tokens + scored_target_tokens
inputs = tokens 去掉最后一个 token
forward(inputs, causal_attention=true)
从 prompt 最后一个位置开始，取预测各目标 token 的 logits
逐位置计算 log-softmax(target_token)，累计得到候选分数
```

参考实现必须覆盖单 token 候选、多 token 候选和结束 token 序列，且用可手算 logits 用例独立校验归约，避免优化路径和参考路径共享同一个错误。

### 5.2 首 token 与预测位置对齐

对于目标 `[a, b, c]`：

| 输出位置 | 对应目标 |
| --- | --- |
| prompt 最后一个 token 的输出 | `a` |
| 输入 `a` 后的输出 | `b` |
| 输入 `b` 后的输出 | `c` |

不计结束符时不必输入最后的 `c`；若还要计算结束 token 概率，则需输入 `c` 并继续按同一移位规则计分。

### 5.3 同问题共享前缀

1. prompt 只 prefill 一次，只请求必要的输出位置。
2. 从 prompt 末尾分布提取每个候选首 token 的 log-probability。
3. 将相同前缀状态关联到各候选 sequence，后续候选各自拥有独立状态。
4. 把各候选其余已知输入 token 组成多序列 batch，在容量允许时整段提交，否则分块提交。
5. 归约所需目标 log-probability，累计到对应候选。
6. 所有候选完成后才对该问题归一化；随后释放/清理候选状态。

注意事项：

- KV 状态和前缀末尾 logits 是两类资源。新一次 decode 可能覆盖 logits；先保存首 token 的必要标量或复制必要输出。
- sequence ID、position 和因果可见范围必须正确，候选分支不能相互注意。
- 多 sequence 共享是否为零拷贝取决于模型状态类型与锁定后端版本，必须实际验证；有正确性保障的复制回退优先于错误共享。
- 第一版只复用共同 prompt。复用 options 之间的更长共同前缀/trie 是后续优化；若复用候选 token，仍需把它们的分数计入每个候选。

### 5.4 多问题批处理

- 引擎接收 `judge_batch`，将问题展开为共享前缀组与候选分支。
- 按实际 token 工作量、输出 logits 行数、活跃 sequence 数和内存预算组批，不只按问题数限流。
- 支持不等长序列、最后不足一批、候选数不同和超大问题独立拆批。
- 维护 request、option、输入 token、原生 logits 输出行之间的显式映射。
- 若采用 padding，padding 不计分、不改变 position，也不能进入其他序列可见状态；优先使用后端支持的无 padding 多序列 batch。
- 同时提供离线批量模式和有界队列的在线微批模式；在线模式配置最大排队时间，不能无限等待凑满 batch。
- 取消和错误回收必须清理对应 sequence；复用 ID 前保证旧状态已移除。

## 6. 模块与代码组织

拟采用小型 Cargo workspace；按实际职责拆分，避免为未来功能提前增加抽象。

```text
jet/
  PLAN.md
  Cargo.toml
  Cargo.lock
  crates/
    jet-core/          # 请求/结果、评分配置、数学归约、错误类型
    jet-llama-sys/     # 锁定原生依赖的构建与底层 FFI
    jet-engine/        # 模板准备、llama 适配、评分与批调度、资源管理
    jet-cli/           # 本地文件输入、JSONL 输出、设备检查、基准入口
  vendor/
    llama.cpp/        # 固定 commit 的 submodule，使用其内含 ggml
  tests/
    fixtures/         # 小型请求、预期契约；不提交大型模型权重
  benches/
  docs/
    scoring.md
    model-support.md
    benchmarking.md
    development.md
```

依赖方向：`jet-core` 不依赖原生运行时；`jet-engine` 依赖 core/sys；CLI 调用 engine。unsafe、原生指针与版本差异集中在 FFI 边界，不能泄漏到公共评分接口。

若多个 Rust 类型共享原生 model/context，应明确所有权和析构顺序。未经验证不得为原生句柄添加 `Send`/`Sync`；优先由专用 worker 独占 context，模型生命周期覆盖其所有 context。

## 7. 最小后端能力

首版 llama 适配层需要：

- 模型加载、元数据、tokenizer、经过验证的 chat template 和 vocab 大小。
- context 创建及独立的 `n_batch`、`n_ubatch`、`n_seq_max` 等容量设置。
- 多序列 batch 构造：token、position、sequence IDs、是否请求 logits。
- 指定位置 logits 读取，或等价的目标 token log-probability 提取。
- 前缀状态共享/复制、sequence 移除和 context 清理。
- 设备选择、GPU offload、线程设置、内存/性能诊断。
- 明确的原生错误映射和 context 失效恢复策略。

`xlai` 的参考缺口：其现有安全封装主要暴露单序列 decode 和末位置 sampling，未提供 Jet 所需的完整多序列评分接口；其 llama 构建路径尚未接入 CUDA。应参考模型加载和构建经验，按上述能力建立 Jet 的封装，而不是沿用生成循环。

不要把不同版本的 llama.cpp 头文件、静态库、Rust bindings 混用。固定上游 commit，自动生成匹配 bindings，并用少量版本适配函数隔离易变 API。若有必要补丁，保持最小、记录原因与上游对应项。

## 8. 内存与吞吐策略

### 8.1 显式预算

预算至少包括：模型权重、活跃前缀和候选状态、计算缓冲区、原生输出 logits、CPU 侧结果缓冲，以及系统/其他进程预留空间。

- NVIDIA 使用独立显存预算；Apple Silicon 考虑系统共享的统一内存压力。
- 同一问题前缀共享后，仍需为所有活跃候选 suffix 预留状态容量。
- `n_batch`、`n_ubatch`、输出行数量上限和 sequence 数分开设置。
- 不能假设减小 microbatch 就限制了整次调用返回的 logits；必要时拆分逻辑 decode 调用并及时归约。
- 发生资源不足时有界拆批/降低活跃序列数；无法容纳单个有效请求时明确报错。
- 重试需要回滚分数和状态，防止 token 被重复计分；禁止静默更换量化模型、终止策略或上下文长度。

### 8.2 Logits 输出与归约

完整 logits 的主要体积为 `输出位置数 × vocab_size × 每元素字节数`。只请求预测目标 token 所必需的行，prompt 中间位置不产生无用输出。

首版路径：分块前向 → 读取 logits → 稳定 log-softmax → 提取目标标量 → 释放临时结果。

测量后再决定是否实现设备端融合归约：目标为在 CUDA/Metal 上计算全词表 logsumexp 和目标 logit 差值，只回传目标 token log-probability。此优化需要底层图或内核支持，不能仅调用原生 logits getter 就宣称避免了输出拷贝。

精确概率仍需要完整词表归一化。只计算目标 token 的输出投影而省略其余词表，不是本计划默认算法。

### 8.3 缓存与工作分配

- 第一阶段模型常驻、同一问题前缀复用；跨请求前缀缓存不是 MVP 必需项。
- 后续跨请求缓存需有容量和淘汰策略，key 至少区分模型权重/量化、实际 token 前缀及所有影响状态的推理配置。
- 缓存命中也必须保留/重建首目标 token 所需输出，不能仅保留 KV。
- 根据长度分桶、持续补批和小型 autotune 选择有收益的配置；不得仅为 GPU 利用率数字而牺牲有效吞吐。
- 多 worker/context 不等于高吞吐；先测量单设备单执行 worker 的批处理上限，再决定是否增加并发 context。

## 9. 对外接口草案

以下为接口形状，不作为已发布 ABI/API 承诺。

```text
Engine::load(model_config, execution_config)

Engine::judge(JudgementRequest) -> JudgementResult
Engine::judge_batch([JudgementRequest]) -> [Result<JudgementResult, JudgementError>]

JudgementRequest:
  request_id
  question
  options: [{ option_id, text }]
  scoring: { termination }

JudgementResult:
  request_id
  options: [{
    option_id,
    log_probability,
    normalized_log_probability,
    normalized_probability,
    option_token_count,
    scored_token_count
  }]
  best_option_id
  metadata: {
    model_identity,
    backend_identity,
    prompt_template_hash,
    scoring_contract_version,
    termination,
    candidate_events_overlap
  }
  diagnostics（按需）
```

补充约定：

- `judge_batch` 的返回顺序与输入一致；候选结果也保留输入顺序。
- `best_option_id` 按原始累计分数取最大值；精确同分时按输入顺序稳定返回，并保留全部分数供调用方判断。
- options 之间不共享采样状态，因为不存在 sampler。
- 调试模式可返回逐 token log-probability、耗时拆分与计分 token IDs；默认不保存完整 logits 或用户文本到日志。
- execution 配置包含设备、context 预算、token batch、microbatch、sequence 上限、队列容量与组批等待上限。
- 模型加载/预热是引擎生命周期操作，不在每个请求中重复执行。
- 使用同步 worker 包装异步提交时，明确取消、超时和背压语义；不使用无限增长的结果队列。

CLI 初步命令：

```text
jet devices
jet judge --model <model.gguf> --input <requests.jsonl> --output <results.jsonl>
jet bench --model <model.gguf> --workload <workload.json>
```

输入输出使用 UTF-8 JSONL，stdout/结果文件只写机器可读结果，诊断信息走 stderr。第一版不要求用户运行另一个聊天服务。

## 10. 实施阶段与完成条件

### P0：冻结评分契约与参考数据

- [ ] 将第 3–4 节细化为 `docs/scoring.md`，固定默认 prompt、终止和错误策略。
- [ ] 选择一个允许本地分发/下载的小型 Chat 模型作为 CPU/开发 fixture，再选择一个代表性较大模型用于 GPU 性能验证。
- [ ] 记录模型来源、revision、文件哈希、量化、tokenizer/template、许可和下载方式；权重不提交 Git。
- [ ] 固定 llama.cpp commit，核对所需 C API 与三种后端的可用性。
- [ ] 准备小型标注判断集及分词/边界 fixture，分开衡量评分实现正确性与判断任务准确率。

完成条件：任一 fixture 都能明确列出共同 prompt tokens、计分 tokens、结束策略和预期行为；不存在尚未定义的隐式长度处理或候选归一化方式。

### P1：CPU 单候选正确性基线

- [ ] 建立 Cargo workspace、模型加载、模板/分词准备和错误类型。
- [ ] 实现安全 logits 读取、单序列 teacher forcing、数值稳定归约。
- [ ] 实现逐候选参考执行、逐问题归一化和最小 CLI。
- [ ] 完成可手算归约、首 token 对齐、结束符、重复/重叠候选等测试。
- [ ] 在条件允许时用独立参考实现核对同模型同 token 序列；量化不同的结果只能做质量对照，不能当作严格相等的 oracle。

完成条件：本地 CPU 可完成端到端 judge；原始 token 分数和聚合分数有可追踪验证；无需 sampling 或自由文本生成。

### P2：多序列批量评分

- [ ] 实现原生 batch 的 position/sequence/logits 标志及输出行映射。
- [ ] 支持一个问题的多候选和多个问题的合批，先不依赖共享前缀优化。
- [ ] 独立配置 token batch、microbatch、sequence 数和输出内存上限。
- [ ] 覆盖变长、拆批、尾批、取消/错误清理和稳定结果顺序。
- [ ] 与 P1 对照原始分数，建立批大小变化的误差报告。

完成条件：不同 batch/分块策略下的分数在规定容差内一致，候选之间无状态泄漏，内存有界。

### P3：共同前缀复用与调度

- [ ] 实现每问题 prompt 一次 prefill，以及安全的候选状态分叉。
- [ ] 正确保存首目标 token 所需的前缀末尾信息。
- [ ] 增加有界队列、按预算组批、在线最大等待时间和离线连续处理。
- [ ] 实现资源不足时的可恢复拆批，候选失败时整题失败。
- [ ] 加入前缀处理计数、缓存/共享情况和阶段耗时等观测。

完成条件：与无共享 P2 的每个候选原始分数一致；共享前缀实际计算次数可验证；取消和请求复用无旧 KV 残留。

### P4：CUDA 与 Metal 跨平台验收

- [ ] 接入 CUDA/Metal 构建、设备枚举、GPU offload 与平台内存配置。
- [ ] 在真实 Linux NVIDIA、Windows NVIDIA、Apple Silicon 设备上验证实际执行后端。
- [ ] 在同模型同量化下对照 CPU，记录数值差异与判断排序差异。
- [ ] 完成短/长 prompt、候选数、候选长度、batch 大小和内存压力矩阵。
- [ ] 验证不可用设备、驱动/运行库缺失、模型不支持等错误；回退 CPU 必须明确可见，不能把 CPU 结果标成 GPU。

完成条件：每个宣称支持的平台都有实际推理结果和基准记录；仅编译通过不代表 GPU 推理验收通过。受设备条件限制的目标标为未验证，不提前宣称支持。

### P5：基于测量的吞吐优化

- [ ] 对 prefill、候选 forward、词表投影、logits 传输、归约、队列等待分别 profile。
- [ ] 调整 batch/microbatch、长度分桶、输出选择与计算/归约流水线。
- [ ] 评估 Flash Attention 与模型/设备支持情况，逐配置验证分数。
- [ ] 仅在确认收益后实现设备端目标 logprob 归约，并保留参考路径对照。
- [ ] 将不同量化下的速度、内存和评分/准确率偏移一起报告；不自动以更低精度替代用户模型。

完成条件：相对同平台、同模型的正确基线有可复现收益，且正确性/内存验收保持通过。未获收益的实验不进入默认路径。

### P6：首版交付与可维护性

- [ ] 完成公共 API、CLI 文档、评分含义说明、模型支持矩阵和可复现示例。
- [ ] 完成 CI、可控的模型 fixture 获取和各目标发行包。
- [ ] 建立原生依赖升级流程：固定版本 → 正确性矩阵 → 性能回归 → 更新锁定版本。
- [ ] 归档基准原始数据、硬件配置、编译参数和已知限制。
- [ ] 在目标判断数据集上报告准确率；需要置信度用途时另做校准评估。

完成条件：新环境能按文档得到可重复的评分，调用方能知道实际模型/后端/评分语义，所有发布宣称均有对应测试或测量证据。

## 11. 正确性验收矩阵

| 类别 | 必须覆盖的检查 |
| --- | --- |
| 数学归约 | 可手算 logits、给同一行加常数分数不变、极端负分数、非有限值、归一化和约等于 1 |
| Token 对齐 | 单 token、多 token、prompt 末位置、最后目标 token 不必输入、多 token 结束序列 |
| 分词 | 中文/英文/混合文本、前导空格、换行、Unicode、特殊标记文本、无重复 BOS |
| 候选语义 | 单候选、重复候选、互为前缀、空输入、长度差异、不同终止策略 |
| 批处理 | 独立执行/合批一致、不等长、不同候选数、不同 batch/ubatch、跨分块、尾批 |
| 状态隔离 | 候选顺序置换、问题顺序置换、请求取消、错误后继续、ID 重用、重复运行 |
| 前缀共享 | 开关共享结果一致、仅 prompt 计算一次、首 token 分数不因后续 decode 覆盖而丢失 |
| 资源边界 | context 超限、sequence 上限、输出内存上限、拆批重试、设备不足、禁止隐式截断 |
| 跨平台 | 同权重/量化/token 在 CPU、CUDA、Metal 的逐 token/序列/归一化误差 |

候选置换测试应按 option ID 对齐比较原始分数；在默认 prompt 模式下，增加不重复候选不应改变原有候选的原始分数。精确同分时 `best_option_id` 的稳定顺序规则单独测试。

数值验收：

- 数学单元测试给出明确误差阈值；例如 f64 归一化和的检查可从 `1e-12` 量级开始。
- 实际后端容差在 P1/P2 用固定 fixture 测量后写入测试配置，必须明确为逐 token 误差、累计误差和归一化误差，不能用“看起来相近”验收。
- CPU/CUDA/Metal 或不同量化不要求 bitwise 相同；量化变化与后端浮点差异分开报告。
- 累计误差要考虑目标长度；如果前两名分差落在误差区间内，报告为近似并列敏感案例，不把一次排名翻转简单归为实现错误或忽略。
- CPU CI 覆盖确定性逻辑与小模型；GPU 验收补充同设备重复运行和不同 batch 配置。

## 12. 性能基准与成功标准

### 12.1 对照组

1. P1：逐问题、逐候选、独立上下文的 teacher forcing。
2. P2：多序列 batch，重复处理各候选前缀。
3. P3：多序列 batch + 同问题前缀共享。
4. P5：上述路径 + 单独启用的进一步优化。

主要对照是正确的 teacher-forcing 实现。自由生成/逐 token sampling 不是公平的唯一性能基线；官方 perplexity/multiple-choice 工具可作为额外参考，需确保模板、计分范围和长度归一化一致。

### 12.2 工作负载

初始矩阵按模型上下文能力和设备容量裁剪，不强制所有组合都可运行：

- prompt 长度：约 128、1K、4K、16K tokens。
- 每题候选数：2、4、8、32。
- 候选长度：1、8、32、128 tokens；增加混合长度批次。
- 问题批量：从 1 递增至吞吐饱和或预算上限。
- 缓存状态：模型冷启动、模型热启动、同题前缀共享；跨请求缓存若引入则另列。
- 资源情形：长 prompt/短候选、短 prompt/多长候选、小词表/大词表、接近内存预算。

先选代表性子集做开发回归，再在发行前跑完整的可支持矩阵，避免无目的地做全部笛卡尔积。

### 12.3 记录指标

- 冷启动加载和实际预热耗时。
- questions/s、options/s、计分目标 tokens/s。
- 实际执行的 prompt/suffix token 数、前缀复用节省量；与逻辑计分 token 数分开。
- 端到端延迟 p50/p95/p99，在线排队与执行时间分别记录。
- 分词/模板、prefill、候选前向、logits 传输/归约阶段耗时；有异步重叠时说明计时口径。
- GPU 显存/统一内存和进程 RSS 峰值；设备利用率作为辅助指标。
- 拆批、重试、取消、失败率，以及同数据集的判断准确率和分数差异。

每份报告固定模型哈希、量化、模板、终止策略、scoring 版本、上下文配置、后端 commit、编译参数、驱动/系统、硬件、batch/ubatch/sequence 数、预热与重复次数。保存原始结果，不只保留平均值。

首轮测量后再为各代表性设备设定吞吐和延迟回归阈值，不预先承诺固定加速倍数或“100% GPU 利用率”。相对收益必须伴随正确性、准确率和内存数据。

## 13. 构建、CI 与发行

- 按平台提供 CPU、CUDA、Metal 构建配置；CUDA/Metal 使用明确 Cargo features 和原生 CMake 设置。
- CUDA 构建覆盖声明支持的 GPU 架构，区分构建期 CUDA toolkit、发行包运行库和用户驱动要求。
- Metal 包含运行所需资源，确认在干净 Apple Silicon 环境可执行。
- 发布包不能默认只针对构建机 CPU 指令集；明确最低架构要求或提供对应变体。
- 仅使用 llama.cpp 内含 ggml，避免无必要地引入另一套独立 ggml。
- CI 必需项：format、lint、数学/契约测试、小模型 CPU 集成测试和目标平台构建检查。
- 真实 GPU 测试使用专用或手动 runner，记录为发行门槛；GPU 不可用时不能把测试跳过解释为通过。
- fixture 权重使用固定 revision/hash 下载与缓存，校验许可；常规纯单元测试不依赖网络或模型下载。
- 依赖升级必须重新生成 bindings 并跑评分回归；保存补丁、第三方许可和构建说明。

## 14. 风险与应对

| 风险 | 应对 |
| --- | --- |
| 返回概率被误认为正确率 | 固定术语，保留原始分数，文档说明候选集合依赖，校准独立评估 |
| 长度/措辞影响判断 | 默认保留原始语义；使用真实标注集测量，其他排名策略独立版本化 |
| 模板/分词边界不一致 | 逐模板 fixture、记录 token 范围、支持矩阵、无法验证时明确拒绝 |
| Off-by-one 或 logits 行映射错误 | 可手算测试、逐候选 oracle、跨 batch/顺序置换对照 |
| 共享 KV 泄漏或新模型不支持共享 | sequence 生命周期检查、独立状态回退、逐模型架构验收 |
| Logits 吞吐/内存成为瓶颈 | 输出位置筛选、逻辑分块、及时归约，必要时设备端融合 |
| 批处理导致排队延迟过高 | 在线等待上限与离线吞吐模式分开，分别测量吞吐和延迟 |
| GPU OOM 或统一内存压力 | 预算、预留空间、有界拆批、清晰失败；不静默截断或换模型 |
| 量化改变临界选项排序 | 报告分数漂移和任务准确率，保留同权重参考，接近同分案例单列 |
| llama.cpp API 演进 | 固定 commit、薄适配边界、升级回归，不跟随未验证的最新分支 |

## 15. 首版完成定义

- [ ] `judge` 和 `judge_batch` 能对本地支持模型返回每个选项的原始与归一化分数。
- [ ] 全流程无 sampling，自由文本生成不是评分所需步骤。
- [ ] 默认概率定义、终止策略、长度处理和 prompt 构造都有文档与测试。
- [ ] 变长多问题/多候选批处理及同题前缀共享已通过独立参考对照。
- [ ] 内存、队列、sequence 生命周期、取消和失败行为有明确边界。
- [ ] CPU、CUDA、Metal 的支持状态与真实验证记录一致。
- [ ] 已报告相对 teacher-forcing 基线的性能与评分/判断质量。
- [ ] Rust API、CLI、构建说明、模型支持矩阵、依赖锁定和许可信息可交付。

## 16. 首版之后的候选工作

按实测需求排序，暂不作为 MVP 阻塞项：

- 跨请求共享 system/question 前缀缓存、候选 trie 共享。
- CUDA/Metal 融合目标 log-probability 归约。
- 平均 token 分数、先验校正、校准与拒答/全部不合适策略。
- 将 A/B/C/D 等标签作为候选的独立任务模式，先验证其 prompt 和单 token 假设。
- 更广模型架构、MLX 或其他适合评分的后端。
- HTTP/gRPC 服务、多 GPU 和分布式离线评分。

## 17. 参考资料

外部链接指向当前上游文档；实施时必须记录所采用的固定 commit，不能据此假设本地依赖版本已经具备全部能力。

- [llama.cpp C API：batch、logits、sequence memory](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h)
- [llama.cpp perplexity 与 multiple-choice 评分实现](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/perplexity.cpp)
- [llama.cpp 多序列 batch 示例](https://github.com/ggml-org/llama.cpp/blob/master/examples/batched/batched.cpp)
- [llama.cpp CUDA/Metal 构建说明](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [lm-evaluation-harness：loglikelihood、token 对齐与 chat template](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md)
- 同级 `../xlai/crates/sys/xlai-sys-llama/`：现有模型封装与原生构建参考。
- 同级 `../xlai/crates/backends/xlai-backend-llama-cpp/`：模板、tokenizer、模型配置参考；其生成循环不作为 Jet 的评分执行路径。
