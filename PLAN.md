# Jet 开发计划

更新日期：2026-09-22

状态：持续路线图。CPU Decisions Engine、批量候选评分和可选的有界 thinking 核心路径已实现并通过 Qwen3 本地验收；多模型真实权重矩阵、GPU、在线调度、性能优化和发行工程仍未完成，未完成项不视为已经验证。

## 0. 当前实现快照

截至 2026-09-22，仓库已经完成一个可运行的 CPU Decisions Engine 基线、Linux Vulkan 后端，以及可选的有界 thinking 核心路径。它是下文长期路线图的第一轮落地，不代表整个计划已经完成。

已完成：

- Rust 1.98 / Edition 2024 workspace：`jet-core`、`jet-llama-sys`、`jet-engine`、`jet-cli`。
- Decisions 风格的 `noul`、`choice`、`score` 请求/响应；请求明确不接受 `model` 字段。
- `Engine::load`、`Engine::decide`、`Engine::decide_batch` Rust API 和 `jet judge` JSONL CLI。
- continuation-only teacher-forced scoring、逐候选 reference path、原生多序列 batch、同题 prompt 前缀复用和按 sequence/output 预算分波。
- GGUF 内置 Jinja chat template；普通非 thinking 模型可直接评分，reasoning 模板支持 `Disabled`、`Auto`、`Required` 三种模式。
- 可选的有界 thinking：配置最大 token、temperature、top-k、top-p 和 seed；每题只生成一份 trace，经模板的 `reasoning_content` continuation 转入最终答案区，再把冻结前缀用于候选批量评分。
- 模板协议不硬编码 `<think>`：复用固定 llama.cpp 的 reasoning 元数据、结束标记和 content continuation，可覆盖 tag 与 channel 类型协议；已移除仅允许 `qwen3` 架构的加载限制。
- llama.cpp `v0.4.1` / commit `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`。
- 可选 `vulkan` Cargo feature、`Backend::Vulkan`/`--backend vulkan` 显式选择、全模型与 KV/offload 配置；已在 Linux Intel Arc A770 上通过固定 Qwen3 模型 CLI 与 smoke test 验收。
- 固定测试模型 `Qwen/Qwen3-0.6B-GGUF` revision `23749fefcc72300e3a2ad315e1317431b06b590a`、`Qwen3-0.6B-Q8_0.gguf`，SHA-256 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`。
- 与固定 llama.cpp commit 匹配的 bindings 已检入；普通构建不加载 libclang。`mise.toml` 通过 `[tools."http:llvm"]` 直接使用 LLVM GitHub release，仅在显式 `generate-bindings` 时需要。
- format、Clippy、纯单元测试、固定模型集成测试和 CLI golden test 已通过；Qwen3 真实模型测试覆盖 8-token thinking 预算、强制协议收尾、final-answer continuation 和后续共享前缀 batch scoring。

当前尚未完成：

- CUDA、Metal 和其他硬件后端，以及 Windows/macOS Vulkan 的构建与真实设备验收。
- DeepSeek、GPT-OSS、Kimi、Gemma 等 reasoning 模型及至少一个普通非 thinking 模型的真实权重/模板验收；当前除 Qwen3 外仅通过 llama.cpp 统一模板协议层接入，不能宣称已经逐模型验证。
- thinking 生成目前按问题串行执行；跨问题 generation batch、阶段耗时观测和 request/question 级预算覆盖尚未实现，现有配置作用于整个 Engine/CLI 运行。
- 在线有界队列、最大等待时间、取消、超时、背压和跨请求缓存。
- 设备端 logits 归约、候选 trie 复用和基于 profile 的吞吐优化。
- 完整 CI、发行包、跨平台支持矩阵、性能基准和任务准确率/校准评估。
- HTTP/gRPC 服务、多 GPU 和分布式执行。

当前已实现的 Decisions API 是首轮对外契约；下文部分较早的通用 `JudgementRequest`/`options` 草案保留为长期设计背景，不表示要覆盖或隐式改变现有接口。后续若统一两者，应通过单独的 API 设计和兼容性评审完成。

## 1. 项目定位与目标

Jet 是一个 **Judgement Engine**。输入为 `question` 和一组 `options`，复用现有自回归 Chat 模型，通过 teacher forcing 计算每个候选答案的条件对数似然，并在该问题的候选集合内归一化。候选评分本身不采样；模型和模板支持时，可先生成一份有界 thinking trace，再在该固定 trace 条件下评分所有候选。

核心工作负载是 **teacher-forced sequence scoring**：候选 token 全部已知，通过批量前向计算取得每个目标 token 的条件概率。优化重点是多问题/多候选批处理、共同前缀复用，以及 logits 的高效归约。

首版目标：

- 提供可嵌入的 Rust API 和用于运行、验证、压测的 CLI。
- 在 NVIDIA CUDA 和 Apple Silicon Metal 上运行同一套评分逻辑，保留 CPU 正确性基线。
- 使用本地 GGUF 模型，模型格式和硬件执行由 llama.cpp 负责。
- 返回可解释、可复现的候选评分；明确其与答案正确率之间的区别。
- 在有足够请求时提高持续评分吞吐，同时保持有界内存和可配置的等待时间。

本阶段不包含浏览器/WASM、TTS、聊天 UI、通用自由文本生成、Agent 工具循环、模型训练和云模型聚合。有界 thinking 是候选评分的可选准备阶段，不扩展为通用 completion API。HTTP 服务、其他推理后端、多 GPU、跨机器执行可在核心引擎稳定后独立评估。

`xlai` 是原生构建和模型封装的参考，不是需要复制的产品架构。Jet 不依赖其聊天运行时；不修改同级 `xlai` 项目。

## 2. 已确定的方向与实施默认值

| 项目 | 首版约定 |
| --- | --- |
| 核心任务 | 对给定候选做条件似然评分；候选 scoring 无 sampler，可选 thinking 使用有界 sampler |
| 推理底座 | Rust + llama.cpp C API/薄 C++ bridge |
| 模型范围 | 首先验证 llama.cpp 支持的 decoder-only causal Transformer Chat GGUF 模型 |
| 加速后端 | 已实现 Linux Vulkan 与 CPU 基线；后续 Linux/Windows NVIDIA CUDA、macOS Apple Silicon Metal |
| 原始分数 | 目标 token 的完整词表 log-softmax 之和 |
| 候选归一化 | 对每个问题内部的原始累计 log-probability 做 softmax |
| 长度处理 | 默认不除以 token 数，不加 length penalty |
| 温度与采样变换 | 候选评分使用原始 logits，不应用 temperature、top-k/top-p 或 grammar mask；thinking 单独配置采样参数和 token 预算 |
| 终止策略 | 默认 `continuation`；可显式选择经过验证的 `assistant_turn` |
| 调度方式 | 常驻模型、受控 context/sequence 生命周期、按 token 和内存预算组批 |
| 第一优先级 | 正确的原始分数，其次是批处理收益，再进行内核级优化 |

上表中的技术默认值是实施起点。若验证表明需要改变，应更新本文、接口版本或评分配置版本，不能隐式改变已有请求的含义。

混合注意力、滑动窗口、循环状态模型、MoE 和多模态模型不因文件格式为 GGUF 就自动视为已支持；逐类验证评分、序列隔离和状态共享能力后再加入支持矩阵。

## 3. 评分的数学定义

### 3.1 共同条件与目标 token

设 `x` 为该问题所有候选共同使用的 prompt token 序列；thinking 关闭时它是直接答案前缀，thinking 开启时它还包含本题唯一一次生成并冻结的 reasoning trace 及模板生成的 final-answer transition。候选 `i` 的实际计分序列为 `y_i = [y_i,1, ..., y_i,L_i]`。

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
- thinking 关闭时，使用模板提供且经过验证的非思考/直接回答路径；普通非 thinking 模型不因缺少开关而被拒绝。
- thinking 开启时，返回的是给定本次随机或固定 seed 生成的 trace 后的候选条件概率，不是对所有可能推理路径边缘化后的答案概率。

## 4. Prompt、分词和终止契约

### 4.1 Prompt 构造

当前 Decisions prompt 行为：

1. 固定 system instruction 要求按语义选择；完整响应必须是候选列表中一个原样复制的带引号答案，禁止解释、前后空白或额外文本。user prompt 末尾用一行重复同一输出契约，但不手写 `Answer:`；模型内置 chat template 生成唯一的 assistant 边界。
2. 外部 JSON 只作为 API；Tera 将 `state`、`question.instructions` 和候选语义展开为带 `Context`、`Instruction`、`Candidate answers` 分区的自然语言 user prompt。对象 key 排序，数组保留原顺序。
3. 用经过验证的模型 chat template 生成 assistant 回答开始前的共同前缀。
4. 将 criterion 语义文本（`noul` 缺省为 `No`/`Yes`）作为带引号的 assistant 候选续写，评分完成后反向映射为 `false`/`true`、`choice` key 或 `score` 零基索引。

启用 thinking 时，模板首先渲染 reasoning 起点；引擎在 token 预算内生成一份 reasoning，使用同一模板的 `reasoning_content` continuation 进入最终答案区，然后把所得完整 token 前缀作为所有候选的共同条件。不同模型的起止 tag、channel 和 final transition 必须来自模板协议，不能统一拼接 `<think>`/`</think>`。

由于当前 prompt 包含全部候选语义，增删或改写候选会改变共同条件，原有候选的原始分数也可能变化；这与仅改变 softmax 分母的 closed-set 后处理不同，必须在评分语义和测试中明确保留。

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

### 4.3 Thinking 模式与预算

| 模式 | 行为 |
| --- | --- |
| `Disabled`（默认） | 使用经过验证的直接答案路径；模板无法可靠关闭 thinking 时返回不支持错误 |
| `Auto` | 模板暴露完整 reasoning 结束标记与 final continuation 时启用 thinking，否则普通非 reasoning 模型回退到直接评分 |
| `Required` | 必须完成 thinking 协议；缺少结束标记或 final continuation 时返回不支持错误 |

当前 `ThinkingConfig` 属于 Engine 配置，CLI 对应 `--thinking`、`--thinking-tokens`、`--thinking-temperature`、`--thinking-top-k`、`--thinking-top-p` 和 `--thinking-seed`。`max_tokens` 限制 reasoning block 内生成 token；预算耗尽时 sampler 可能先补完整 UTF-8，再附加协议结束 token，因此 closure token 可使实际输出数略高于预算。

每题 thinking 只生成一次。分波、reference/batched 对照和候选顺序变化必须复用同一份冻结 trace，不能重新采样后把结果当成同一条件下的评分。候选 log-probability 始终从未经 thinking sampler 修改的原始 logits 计算。

### 4.4 终止策略

| 策略 | 计分目标 | 解释 |
| --- | --- | --- |
| `continuation`（默认） | option 内容 token | 输出以该候选 token 序列开头的似然，不包含“此处停止” |
| `assistant_turn` | option 内容 + 该模板约定的结束 token 序列 | 输出该答案并以指定方式结束 assistant 回合的似然 |

`assistant_turn` 要求已验证的结束序列，不能默认认为 EOS 等于 end-of-turn。若模型存在多个合法结束形式，首版使用一个明确的规范结束序列，不声称对所有结束形式做了边缘化。模板无法提供可靠结束策略时返回不支持错误。

结果分别报告 `option_token_count` 和 `scored_token_count`，后者包含实际计分的结束 token。

### 4.5 输入边界与异常

- 空 question、空 options 列表、空字符串 option、tokenization 后无内容 token 的 option：首版明确拒绝。
- 单个有效 option 可以评分，归一化权重为 1；这不表达确信程度。
- 同一问题中，最终计分 token 序列完全相同的重复选项返回重复候选错误，不能重复计入分母。
- 互为前缀的候选可以按 `continuation` 评分，但返回 `candidate_events_overlap` 等明确诊断。
- prompt 加目标超出模型/context 限制时返回明确错误；允许分块计算，不允许通过分块绕过最大语义上下文或静默截断。
- thinking prompt、预算、协议 closure、final transition 和最长候选共同受单序列 context 限制；不能为完成 closure 静默扩大上下文或截断 reasoning。
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

thinking 开启时，先完成并冻结该问题的 reasoning 和 final-answer transition；以下算法中的 prompt 指该完整公共前缀。当前实现会为 scoring 重新 prefill 该前缀，不复用 generation 阶段的临时 KV，但同一题的所有候选仍共享一次 scoring prefill。

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
- 纯 KV-cache 模型的候选 scoring 可跨问题合批；hybrid/recurrent 模型因 llama.cpp 多 sequence state 分叉不等价而自动逐候选组执行。thinking generation 仍按问题串行执行，后续应单独评估 generation continuous batching，不能把候选 batch 能力等同于 thinking 已合批。
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
- 模板 capability 探测：普通非 reasoning、可关闭 reasoning、必须 reasoning 和协议未知路径；返回 thinking start/end 与 final-answer continuation，而不是依赖单个 `supports_thinking` 布尔值。
- reasoning-budget sampler：监测自然结束、预算耗尽后强制协议 closure、避免未完成 reasoning 时接受 EOG，并确保候选 scoring 不继承 grammar/采样变换。
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

预算至少包括：模型权重、thinking generation 临时状态、冻结的 reasoning/final 前缀、活跃候选状态、计算缓冲区、原生输出 logits、CPU 侧结果缓冲，以及系统/其他进程预留空间。

- NVIDIA 使用独立显存预算；Apple Silicon 考虑系统共享的统一内存压力。
- 同一问题前缀共享后，仍需为所有活跃候选 suffix 预留状态容量。
- `n_batch`、`n_ubatch`、输出行数量上限和 sequence 数分开设置。
- 不能假设减小 microbatch 就限制了整次调用返回的 logits；必要时拆分逻辑 decode 调用并及时归约。
- 发生资源不足时有界拆批/降低活跃序列数；无法容纳单个有效请求时明确报错。
- 重试需要回滚分数和状态，防止 token 被重复计分；禁止静默更换量化模型、终止策略或上下文长度。
- thinking 的 `max_tokens` 不包含强制 closure 的全部开销；context 预算必须为 UTF-8 补全、结束标记和 final transition 留出空间。

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

当前已发布形状是 `EngineConfig` + Decisions `DecisionRequest`，thinking 通过 `EngineConfig::thinking` 和 CLI 参数配置。以下较早的通用 Judgement 接口仍是长期草案，不作为已发布 ABI/API 承诺。

当前 thinking 配置：

```text
ThinkingConfig {
  mode: Disabled | Auto | Required,
  max_tokens,
  temperature,
  top_k,
  top_p,
  seed
}
```

现阶段同一 Engine 实例内的所有问题共享该配置；若增加 request/question 级覆盖，需要定义 batch 中不同预算、seed 和采样配置的调度与复现语义。

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
- options 之间不存在采样状态；可选 sampler 只用于每题一次的 thinking，完成后所有 options 共享冻结 trace 并独立 teacher forcing。
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

### P0：冻结评分契约与参考数据（部分完成）

- [x] 将已实现的评分语义细化为 `docs/scoring.md`，固定 prompt、continuation、thinking 条件概率、usage 和错误策略。
- [x] 选择 Qwen3-0.6B Q8_0 作为 CPU/开发 fixture；较大 GPU 性能模型仍待选择。
- [x] 记录 CPU fixture 的来源、revision、文件哈希、量化和下载方式；权重不提交 Git。
- [x] 固定 llama.cpp commit，并核对首轮 CPU 路径所需 C API；CUDA/Metal 能力仍待验收。
- [ ] 准备小型标注判断集及分词/边界 fixture，分开衡量评分实现正确性与判断任务准确率。

完成条件：任一 fixture 都能明确列出共同 prompt tokens、计分 tokens、结束策略和预期行为；不存在尚未定义的隐式长度处理或候选归一化方式。

### P1：CPU 单候选正确性基线（核心路径完成）

- [x] 建立 Cargo workspace、模型加载、模板/分词准备和错误类型。
- [x] 实现安全 logits 读取、单序列 teacher forcing、数值稳定归约。
- [x] 实现逐候选参考执行、逐问题归一化和最小 CLI。
- [ ] 完成可手算归约、首 token 对齐、结束符、重复/重叠候选等测试。
- [ ] 在条件允许时用独立参考实现核对同模型同 token 序列；量化不同的结果只能做质量对照，不能当作严格相等的 oracle。

完成条件：本地 CPU 可完成端到端 judge；原始 token 分数和聚合分数有可追踪验证；候选评分无需 sampling 或自由文本生成。

### P2：多序列批量评分（核心路径完成）

- [x] 实现原生 batch 的 position/sequence/logits 标志及输出行映射。
- [x] 支持一个问题的多候选和多个问题的合批，并保留独立 reference path 对照。
- [x] 独立配置 token batch、microbatch、sequence 数和输出内存上限。
- [ ] 覆盖变长、拆批、尾批、取消/错误清理和稳定结果顺序。
- [x] 与 P1 reference path 对照原始分数，并固定累计分数与归一化概率容差。

完成条件：不同 batch/分块策略下的分数在规定容差内一致，候选之间无状态泄漏，内存有界。

### P3：共同前缀复用与调度（前缀复用完成，在线调度未开始）

- [x] 在候选可放入单组 sequence 预算时实现每问题 prompt 一次 prefill，以及安全的候选状态分叉。
- [x] thinking 开启时每题只生成一次 trace，并将 reasoning + final transition 组成的冻结前缀交给同一套共享/分波 scoring 路径。
- [x] 正确保存并立即归约首目标 token 所需的前缀末尾 logits。
- [ ] 增加有界队列、按预算组批、在线最大等待时间和离线连续处理。
- [x] 实现 sequence/output 预算下的可恢复分波，不截断输入；候选失败时所属请求失败。
- [ ] 加入前缀处理计数、缓存/共享情况和阶段耗时等观测。

完成条件：与无共享 P2 的每个候选原始分数一致；共享前缀实际计算次数可验证；取消和请求复用无旧 KV 残留。

### P3-T：有界 thinking 与多模板协议（核心路径完成，支持矩阵待验收）

- [x] 增加 `Disabled`、`Auto`、`Required` 模式及公开 `ThinkingConfig`，CLI 支持 token budget、temperature、top-k、top-p 和 seed。
- [x] 普通非 reasoning 模板可直接评分；不再因没有 `enable_thinking` 开关而一律拒绝模型。
- [x] 通过 llama.cpp 模板解析结果取得 reasoning 起止信息，并用 `reasoning_content` continuation 构造 final-answer 前缀；未硬编码单一 `<think>` 协议。
- [x] 接入 reasoning-budget sampler，支持自然结束、预算耗尽强制 closure、未完成 reasoning 时拦截 EOG，以及最终候选 raw-logits scoring。
- [x] 用固定 Qwen3-0.6B Q8_0 真实模型验证 8-token 预算、closure、final continuation、thinking token 计数和后续 batched prefix sharing。
- [x] 增加 tag/channel 协议状态判断单元测试，并验证 format、Clippy、常规测试和固定模型 ignored integration test。
- [ ] 用真实 GGUF 分别验收 DeepSeek、GPT-OSS、Kimi、Gemma 系列及至少一个普通非 thinking 模型；记录模板版本、协议 marker、开关行为和 context 开销。
- [ ] 增加固定 trace 注入或可观测 fixture，使 reference/batched、分波、候选置换和错误恢复能在完全相同 reasoning 条件下逐分数对照。
- [ ] 评估并实现跨问题 thinking generation batching；在此之前分别记录串行 generation 和 batched scoring 的耗时。
- [ ] 决定是否增加 request/question 级 thinking override；若增加，定义 seed、复现、usage 和混合预算调度语义。

完成条件：支持矩阵中的每种模板都由真实模型或可复现 tokenizer/template fixture 验证；`Auto` 不会误入 reasoning 区，`Disabled` 不会在无法关闭时静默评分，`Required` 能稳定完成 closure 和 final transition；同一冻结 trace 下 reference/batched 原始分数一致。

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

- [x] 当前 Decisions API、thinking 配置、conditional-on-trace 评分语义和 usage 口径已写入 README 与评分文档。
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
| Thinking 模式 | 普通非 reasoning fallback、可关闭/必须 reasoning、协议未知、自然结束、预算耗尽、EOG、UTF-8 补全、final continuation |
| Thinking 复现 | 固定 seed/固定 trace、同一 trace 下 reference/batched/分波一致、候选置换不重新生成 trace、usage token 口径 |
| 资源边界 | context 超限、sequence 上限、输出内存上限、拆批重试、设备不足、禁止隐式截断 |
| 跨平台 | 同权重/量化/token 在 CPU、Vulkan、CUDA、Metal 的逐 token/序列/归一化误差 |

当前 Tera prompt 包含完整候选语义，所以候选增删、改写或重排可能改变共同条件及原始分数。候选置换测试应分别验证 key/结果反向映射与稳定顺序；只有渲染后的完整 prompt 和冻结 thinking trace 都相同时，才能把原始分数相等作为 reference/batched 或分波一致性的断言。精确同分时 `best_option_id` 的稳定顺序规则单独测试。

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
4. P3-T：生成一次有界 thinking + 冻结 reasoning/final 前缀 + P3 候选评分。
5. P5：上述路径 + 单独启用的进一步优化。

主要对照是正确的 teacher-forcing 实现。自由生成/逐 token sampling 不是公平的唯一性能基线；官方 perplexity/multiple-choice 工具可作为额外参考，需确保模板、计分范围和长度归一化一致。

### 12.2 工作负载

初始矩阵按模型上下文能力和设备容量裁剪，不强制所有组合都可运行：

- prompt 长度：约 128、1K、4K、16K tokens。
- 每题候选数：2、4、8、32。
- 候选长度：1、8、32、128 tokens；增加混合长度批次。
- 问题批量：从 1 递增至吞吐饱和或预算上限。
- thinking：Disabled/Auto/Required，预算 0 边界、8/64/256/1K tokens，自然结束与强制 closure；支持与不支持 reasoning 的模板分开测量。
- 缓存状态：模型冷启动、模型热启动、同题前缀共享；跨请求缓存若引入则另列。
- 资源情形：长 prompt/短候选、短 prompt/多长候选、小词表/大词表、接近内存预算。

先选代表性子集做开发回归，再在发行前跑完整的可支持矩阵，避免无目的地做全部笛卡尔积。

### 12.3 记录指标

- 冷启动加载和实际预热耗时。
- questions/s、options/s、计分目标 tokens/s。
- 实际执行的 prompt/suffix token 数、前缀复用节省量；与逻辑计分 token 数分开。
- 端到端延迟 p50/p95/p99，在线排队与执行时间分别记录。
- 分词/模板、prefill、候选前向、logits 传输/归约阶段耗时；有异步重叠时说明计时口径。
- thinking template render、逐 token generation、closure/final transition、scoring 重新 prefill 分别计时；串行 generation 与 batched scoring 不合并成一个误导性 tokens/s。
- GPU 显存/统一内存和进程 RSS 峰值；设备利用率作为辅助指标。
- 拆批、重试、取消、失败率，以及同数据集的判断准确率和分数差异。

每份报告固定模型哈希、量化、模板、终止策略、scoring 版本、上下文配置、后端 commit、编译参数、驱动/系统、硬件、batch/ubatch/sequence 数、预热与重复次数。保存原始结果，不只保留平均值。

首轮测量后再为各代表性设备设定吞吐和延迟回归阈值，不预先承诺固定加速倍数或“100% GPU 利用率”。相对收益必须伴随正确性、准确率和内存数据。

## 13. 构建、CI 与发行

- 按平台提供 CPU、Vulkan、CUDA、Metal 构建配置；加速后端使用明确 Cargo features 和原生 CMake 设置。
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
| 模板声称支持 thinking 但开关或 marker 不完整 | 比较开启/关闭渲染结果，要求可验证的 end marker 和 final continuation；`Required` 报错，`Auto` 仅对确认的普通模板回退 |
| 有界 thinking 改变概率复现性 | 每题只生成一次并冻结 trace，记录 seed/采样配置；正确性对照使用相同 trace，不把不同采样结果直接比较 |
| thinking closure 超出 token/context 预算 | 区分 reasoning credits 与 UTF-8/协议 closure 开销，预留 final transition 和候选空间，超限明确失败 |
| Off-by-one 或 logits 行映射错误 | 可手算测试、逐候选 oracle、跨 batch/顺序置换对照 |
| 共享 KV 泄漏或新模型不支持共享 | sequence 生命周期检查、独立状态回退、逐模型架构验收 |
| Logits 吞吐/内存成为瓶颈 | 输出位置筛选、逻辑分块、及时归约，必要时设备端融合 |
| 批处理导致排队延迟过高 | 在线等待上限与离线吞吐模式分开，分别测量吞吐和延迟 |
| GPU OOM 或统一内存压力 | 预算、预留空间、有界拆批、清晰失败；不静默截断或换模型 |
| 量化改变临界选项排序 | 报告分数漂移和任务准确率，保留同权重参考，接近同分案例单列 |
| llama.cpp API 演进 | 固定 commit、薄适配边界、升级回归，不跟随未验证的最新分支 |

## 15. 首版完成定义

- [ ] `judge` 和 `judge_batch` 能对本地支持模型返回每个选项的原始与归一化分数。
- [x] 候选评分全流程无 sampling；可选 thinking sampling 有明确 token 预算、协议 closure、模式开关和冻结 trace。
- [ ] 默认概率定义、终止策略、长度处理和 prompt 构造都有文档与测试。
- [x] 变长多问题/多候选批处理及同题前缀共享已通过固定 Qwen3 模型的独立 reference 对照；Qwen3.5 hybrid/recurrent 路径已验证必须隔离候选组，并加入 reference/batched 回归测试。
- [ ] thinking 的跨模板真实模型矩阵和相同冻结 trace 下的 reference/batched 对照完成。
- [ ] 内存、队列、sequence 生命周期、取消和失败行为有明确边界。
- [ ] CPU、Vulkan、CUDA、Metal 的支持状态与真实验证记录一致。
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
