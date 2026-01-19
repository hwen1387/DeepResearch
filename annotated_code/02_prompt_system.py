"""
【核心文件 2】系统提示词 - 详细注释版

原文件: inference/prompt.py

这个文件定义了 Agent 的"操作手册"：
- 它是谁
- 它能做什么
- 如何调用工具
- 如何给出答案
"""

# ============================================================================
# 系统提示词: SYSTEM_PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
{"type": "function", "function": {"name": "PythonInterpreter", "description": "Executes Python code in a sandboxed environment. To use this tool, you must follow this format:
1. The 'arguments' JSON object must be empty: {}.
2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.

IMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.

Example of a correct call:
<tool_call>
{\"name\": \"PythonInterpreter\", \"arguments\": {}}
<code>
import numpy as np
# Your code here
print(f\"The result is: {np.mean([1,2,3])}\")
</code>
</tool_call>", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "google_scholar", "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries. This tool will also return results from google search", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries for Google Scholar."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "parse_file", "description": "This is a tool that can be used to parse multiple user uploaded local files such as PDF, DOCX, PPTX, TXT, CSV, XLSX, DOC, ZIP, MP4, MP3.", "parameters": {"type": "object", "properties": {"files": {"type": "array", "items": {"type": "string"}, "description": "The file name of the user uploaded local files to be parsed."}}, "required": ["files"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Current date: """


# ============================================================================
# 详细解析: 系统提示词的结构
# ============================================================================

"""
系统提示词分为三个主要部分:

┌─────────────────────────────────────────────────────────────┐
│                   第一部分: 角色定义                         │
├─────────────────────────────────────────────────────────────┤
│ "You are a deep research assistant..."                      │
│                                                              │
│ 作用:                                                        │
│ 1. 定义 Agent 的身份: 深度研究助手                         │
│ 2. 定义核心功能: 多源调查、信息综合                        │
│ 3. 定义输出要求: 全面、准确、客观                          │
│ 4. 定义答案格式: <answer></answer> 标签                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   第二部分: 工具定义                         │
├─────────────────────────────────────────────────────────────┤
│ <tools>                                                      │
│ {"type": "function", "function": {...}}                     │
│ </tools>                                                     │
│                                                              │
│ 包含 5 个工具:                                              │
│ 1. search - 网络搜索                                        │
│ 2. visit - 网页访问                                         │
│ 3. PythonInterpreter - 代码执行                            │
│ 4. google_scholar - 学术搜索                               │
│ 5. parse_file - 文件解析                                   │
│                                                              │
│ 每个工具包含:                                               │
│ - name: 工具名称                                            │
│ - description: 工具功能描述                                 │
│ - parameters: 参数定义 (JSON Schema 格式)                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   第三部分: 调用格式                         │
├─────────────────────────────────────────────────────────────┤
│ "For each function call, return a json object..."           │
│                                                              │
│ <tool_call>                                                  │
│ {"name": <function-name>, "arguments": <args-json-object>}  │
│ </tool_call>                                                 │
│                                                              │
│ 作用:                                                        │
│ 1. 告诉 LLM 如何格式化工具调用                             │
│ 2. 使用 XML 标签包裹 JSON                                   │
│ 3. 标准化输出格式，便于解析                                │
└─────────────────────────────────────────────────────────────┘
"""


# ============================================================================
# 工具定义详解
# ============================================================================

"""
工具 1: search (网络搜索)
─────────────────────────

定义:
{
  "name": "search",
  "description": "执行 Google 搜索，返回 Top 搜索结果。支持多查询。",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "array",
        "items": {"type": "string"},
        "description": "搜索查询列表"
      }
    },
    "required": ["query"]
  }
}

使用示例:
<tool_call>
{"name": "search", "arguments": {"query": ["量子计算", "量子纠缠"]}}
</tool_call>

返回格式:
# 搜索查询: "量子计算"

## 量子计算 - 维基百科
URL: https://zh.wikipedia.org/wiki/量子计算
量子计算是利用量子力学原理进行计算...
Date: 2024-01-15

## ...

───────────────────────────────────────────────────────────────

工具 2: visit (网页访问)
─────────────────────────

定义:
{
  "name": "visit",
  "description": "访问网页并返回内容摘要",
  "parameters": {
    "type": "object",
    "properties": {
      "url": {
        "type": "array",
        "items": {"type": "string"},
        "description": "要访问的 URL 列表"
      },
      "goal": {
        "type": "string",
        "description": "访问网页的信息目标"
      }
    },
    "required": ["url", "goal"]
  }
}

使用示例:
<tool_call>
{
  "name": "visit",
  "arguments": {
    "url": ["https://example.com/article"],
    "goal": "找到关于量子计算的最新进展"
  }
}
</tool_call>

返回格式:
# URL: https://example.com/article

**理由 (Rationale):**
该网页包含 2024 年量子计算的研究进展...

**证据 (Evidence):**
[完整的原始段落内容]

**总结 (Summary):**
量子计算在 2024 年取得三大突破...

───────────────────────────────────────────────────────────────

工具 3: PythonInterpreter (Python 解释器)
────────────────────────────────────────

定义:
{
  "name": "PythonInterpreter",
  "description": "在沙箱环境中执行 Python 代码",
  "parameters": {
    "type": "object",
    "properties": {},
    "required": []
  }
}

特殊格式:
- arguments 必须为空: {}
- 代码放在 <code></code> 标签中

使用示例:
<tool_call>
{"name": "PythonInterpreter", "arguments": {}}
<code>
import numpy as np
data = [1, 2, 3, 4, 5]
mean = np.mean(data)
print(f"平均值: {mean}")
</code>
</tool_call>

返回格式:
标准输出:
平均值: 3.0

执行时间: 0.15秒

───────────────────────────────────────────────────────────────

工具 4: google_scholar (学术搜索)
──────────────────────────────────

定义:
{
  "name": "google_scholar",
  "description": "通过 Google Scholar 搜索学术文献",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "array",
        "items": {"type": "string"},
        "description": "学术搜索查询列表"
      }
    },
    "required": ["query"]
  }
}

使用示例:
<tool_call>
{"name": "google_scholar", "arguments": {"query": ["Transformer architecture"]}}
</tool_call>

返回格式:
# 学术搜索查询: "Transformer architecture"

## Attention is All You Need
链接: https://arxiv.org/abs/1706.03762
发表信息: NeurIPS 2017
年份: 2017
引用次数: 50000+
摘要: We propose a new simple network architecture...

## ...

───────────────────────────────────────────────────────────────

工具 5: parse_file (文件解析)
─────────────────────────────

定义:
{
  "name": "parse_file",
  "description": "解析用户上传的本地文件",
  "parameters": {
    "type": "object",
    "properties": {
      "files": {
        "type": "array",
        "items": {"type": "string"},
        "description": "要解析的文件名列表"
      }
    },
    "required": ["files"]
  }
}

支持格式:
PDF, DOCX, PPTX, TXT, CSV, XLSX, DOC, ZIP, MP4, MP3

使用示例:
<tool_call>
{"name": "parse_file", "arguments": {"files": ["report.pdf", "data.xlsx"]}}
</tool_call>

返回格式:
# 文件: report.pdf

## 第 1 页
标题: 2024 年度报告
内容: ...

## 第 2 页
...

# 文件: data.xlsx

## Sheet: 销售数据
| 月份 | 销售额 |
|------|--------|
| 1月  | 100万  |
...
"""


# ============================================================================
# EXTRACTOR_PROMPT: 网页内容提取提示词
# ============================================================================

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""

"""
解析 EXTRACTOR_PROMPT
────────────────────

用途:
这个提示词用于 visit 工具的第二阶段（网页内容摘要）

工作流程:
1. Jina AI 抓取网页 → 原始 HTML/Markdown
2. LLM + EXTRACTOR_PROMPT → 结构化摘要
3. 返回 JSON: {rational, evidence, summary}

占位符:
- {webpage_content}: 网页的完整内容（最多 95K tokens）
- {goal}: 用户的信息目标

输出要求:
- rational: 为什么这部分内容相关
- evidence: 完整的原始文本（尽可能详细）
- summary: 简洁的总结

为什么需要三个字段?
- rational: 帮助 Agent 理解信息的相关性
- evidence: 保留完整上下文，防止信息丢失
- summary: 方便快速理解核心内容
"""


# ============================================================================
# 关键设计原则
# ============================================================================

"""
好的提示词设计原则:

1. **明确的角色定义**
   ✅ "You are a deep research assistant"
   ❌ "You are an AI"

2. **结构化的工具描述**
   ✅ JSON Schema 格式
   ❌ 自由文本描述

3. **清晰的输出格式**
   ✅ <tool_call>{JSON}</tool_call>
   ❌ "调用工具时使用 JSON 格式"

4. **上下文信息**
   ✅ Current date: 2024-01-15
   ❌ 没有时间信息

5. **示例驱动** (可选)
   可以添加 Few-Shot 示例来提高工具使用准确率

如何修改提示词:
1. 修改角色定义 → 改变 Agent 行为
2. 添加新工具 → 扩展 Agent 能力
3. 调整输出格式 → 改变解析逻辑
4. 添加约束条件 → 限制 Agent 行为

最佳实践:
- 保持提示词清晰简洁
- 使用标准格式（XML + JSON）
- 提供足够的上下文
- 定期测试和迭代
"""


# ============================================================================
# 实际使用示例
# ============================================================================

"""
完整的工具调用示例:

用户问题:
"搜索并总结 2024 年量子计算的最新进展"

Agent 执行过程:
┌─────────────────────────────────────────────────────────────┐
│ 第 1 轮: 搜索                                                │
├─────────────────────────────────────────────────────────────┤
│ LLM 输出:                                                    │
│ <think>我需要先搜索相关信息</think>                         │
│ <tool_call>                                                  │
│ {"name": "search", "arguments": {                            │
│   "query": ["2024 量子计算进展", "quantum computing 2024"]  │
│ }}                                                           │
│ </tool_call>                                                 │
│                                                              │
│ 工具返回:                                                    │
│ <tool_response>                                              │
│ # 搜索结果...                                               │
│ </tool_response>                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 第 2 轮: 访问网页                                            │
├─────────────────────────────────────────────────────────────┤
│ LLM 输出:                                                    │
│ <think>我需要访问搜索结果中的网页获取详细信息</think>       │
│ <tool_call>                                                  │
│ {"name": "visit", "arguments": {                             │
│   "url": ["https://example.com/quantum-2024"],              │
│   "goal": "提取 2024 年量子计算的主要进展"                  │
│ }}                                                           │
│ </tool_call>                                                 │
│                                                              │
│ 工具返回:                                                    │
│ <tool_response>                                              │
│ # URL: https://example.com/quantum-2024                     │
│ **证据**: [详细内容]                                        │
│ **总结**: 2024 年量子计算取得三大突破...                   │
│ </tool_response>                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 第 3 轮: 给出答案                                            │
├─────────────────────────────────────────────────────────────┤
│ LLM 输出:                                                    │
│ <think>现在我有足够的信息来回答了</think>                   │
│ <answer>                                                     │
│ 2024 年量子计算领域取得了以下主要进展：                    │
│ 1. 纠错码技术突破...                                        │
│ 2. 量子比特数量增加到 1000 个...                           │
│ 3. 首次实现商业化应用...                                    │
│ </answer>                                                    │
└─────────────────────────────────────────────────────────────┘
"""
