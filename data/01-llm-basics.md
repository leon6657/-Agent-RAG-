# 大语言模型（LLM）基础

## 什么是 LLM

大语言模型（Large Language Model, LLM）是一种基于深度学习的自然语言处理模型，通常基于 Transformer 架构，通过海量文本数据预训练得到。LLM 的核心能力是理解和生成人类语言，代表性模型包括 GPT 系列、Claude、Gemini、DeepSeek 等。

## Transformer 架构核心

Transformer 是 LLM 的基石，由 Vaswani 等人在 2017 年提出。其核心创新是**自注意力机制（Self-Attention）**，允许模型在生成每个 token 时关注输入序列中的所有 token，从而捕捉长距离依赖关系。

关键组件：
- **多头注意力（Multi-Head Attention）**：并行计算多个注意力头，捕捉不同语义子空间的信息
- **前馈神经网络（FFN）**：每个 token 独立通过两层线性变换 + 激活函数
- **位置编码（Positional Encoding）**：为序列中的每个位置注入位置信息
- **层归一化（Layer Normalization）**：稳定训练过程

## 预训练与微调

LLM 的生命周期通常经历两个阶段：

1. **预训练（Pre-training）**：在大规模无标注语料上通过自监督学习训练。常见的训练目标包括下一 token 预测（Next Token Prediction）和掩码语言建模（Masked Language Modeling）。这一阶段模型获得通用的语言理解能力。

2. **微调（Fine-tuning）**：在预训练基础上，使用特定任务的有标注数据继续训练。包括指令微调（Instruction Tuning）和人类反馈强化学习（RLHF），使模型对齐人类偏好。

## 参数规模的含义

模型参数数量从数十亿到数千亿不等。更大的参数通常意味着更强的表达能力和更多知识，但同时也带来更高的推理成本和部署难度。在实际工程中，需要根据应用场景在模型能力、延迟、成本之间做权衡。

## Prompt Engineering

Prompt（提示词）是用户输入给 LLM 的指令或问题。良好的 Prompt 设计可以显著提升模型输出的质量。常见技巧包括：少样本学习（Few-shot）、思维链（Chain-of-Thought）、角色设定等。
