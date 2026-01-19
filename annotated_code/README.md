# Agent 核心代码 - 详细注释版

> 🎯 本目录包含 DeepResearch Agent 系统的核心代码，配有详细的中文注释，帮助您快速理解和构建 Agent。

---

## 📚 文件列表

### 1. `01_react_agent_core.py` ⭐⭐⭐⭐⭐

**最重要的文件** - ReAct Agent 主循环实现

```python
# 原文件: inference/react_agent.py (第 126-226 行)
# 阅读时间: 1-2 小时
# 难度: ⭐⭐⭐⭐
```

**包含内容**:
- ✅ 完整的 Agent 主循环逻辑
- ✅ 消息历史管理
- ✅ 工具调用机制
- ✅ 四大终止条件（超时、答案、次数、上下文）
- ✅ LLM API 调用与重试机制

**关键概念**:
- ReAct 循环：Think → Act → Observe → Repeat
- 消息格式：system/user/assistant 角色
- 工具调用格式：`<tool_call>` XML 标签
- 检查点机制：四个关键检查点

**学习重点**:
1. 理解 `while` 循环的整体结构
2. 掌握消息历史的更新逻辑
3. 理解工具调用的触发和执行
4. 掌握各种终止条件的处理

---

### 2. `02_prompt_system.py` ⭐⭐⭐⭐⭐

**系统提示词设计** - Agent 的"操作手册"

```python
# 原文件: inference/prompt.py
# 阅读时间: 30 分钟
# 难度: ⭐⭐⭐
```

**包含内容**:
- ✅ SYSTEM_PROMPT 完整定义
- ✅ 5 个工具的 JSON Schema 定义
- ✅ 工具调用格式说明
- ✅ EXTRACTOR_PROMPT（网页摘要提示词）

**关键概念**:
- 角色定义：定义 Agent 的身份和目标
- 工具定义：JSON Schema 格式
- 输出格式：XML 标签包裹 JSON
- 提示词结构：三部分组成

**学习重点**:
1. 理解提示词的三个组成部分
2. 掌握 JSON Schema 工具定义格式
3. 理解如何添加新工具定义
4. 学习提示词设计的最佳实践

---

### 3. `03_tool_search_implementation.py` ⭐⭐⭐⭐

**工具实现示例** - 学习如何创建 Agent 工具

```python
# 原文件: inference/tool_search.py
# 阅读时间: 1 小时
# 难度: ⭐⭐⭐
```

**包含内容**:
- ✅ Search 工具完整实现
- ✅ Serper API 集成
- ✅ 批量查询支持
- ✅ 中英文自动检测
- ✅ 重试机制和错误处理

**关键概念**:
- BaseTool 继承
- @register_tool 装饰器
- call() 方法实现
- 参数解析和验证
- 结构化输出

**学习重点**:
1. 理解工具类的基本结构
2. 掌握参数定义和解析
3. 学习 API 调用和错误处理
4. 了解如何创建自己的工具

---

### 4. `04_minimal_agent_example.py` ⭐⭐⭐⭐⭐

**完整可运行示例** - 100 行代码实现一个基础 Agent

```python
# 从零实现的最小 Agent
# 阅读时间: 1 小时
# 难度: ⭐⭐⭐
```

**包含内容**:
- ✅ 完整的 Agent 类实现（MinimalAgent）
- ✅ 两个示例工具（Search, Calculator）
- ✅ 主循环逻辑
- ✅ 工具调用机制
- ✅ 三个测试用例

**关键概念**:
- Agent 初始化
- 工具注册表
- 消息历史管理
- 工具执行流程
- 终止条件判断

**学习重点**:
1. 理解完整的 Agent 实现流程
2. 掌握工具注册和调用
3. 学习如何处理边界情况
4. 能够独立运行和调试

**运行方法**:
```bash
# 1. 启动 vLLM 服务器
vllm serve /path/to/model --port 6001

# 2. 运行示例
python 04_minimal_agent_example.py
```

---

## 🎓 学习路径

### 快速入门（1 天）

```
上午 (2-3 小时):
├─ 第 1 步: 阅读 02_prompt_system.py
│  └─ 理解系统提示词的结构
│
├─ 第 2 步: 阅读 04_minimal_agent_example.py
│  └─ 运行示例，观察输出
│
└─ 第 3 步: 修改 04_minimal_agent_example.py
   └─ 添加一个新的工具（例如：天气查询）

下午 (2-3 小时):
├─ 第 4 步: 阅读 01_react_agent_core.py (第 1-3 节)
│  └─ 重点理解主循环逻辑
│
├─ 第 5 步: 阅读 03_tool_search_implementation.py
│  └─ 理解工具实现模式
│
└─ 第 6 步: 动手实践
   └─ 基于模板创建一个完整的工具
```

### 深入学习（1 周）

```
第 1-2 天: 核心概念
├─ 完整阅读 01_react_agent_core.py
├─ 理解四大检查点
├─ 掌握消息历史管理
└─ 实践：修改终止条件

第 3-4 天: 工具系统
├─ 深入学习 03_tool_search_implementation.py
├─ 阅读其他工具实现（visit, file, python）
├─ 实践：创建 2-3 个自定义工具
└─ 实践：实现批量工具调用

第 5-6 天: 提示词工程
├─ 研究 02_prompt_system.py
├─ 学习 Few-Shot 提示词
├─ 实践：优化系统提示词
└─ 实践：添加新工具定义

第 7 天: 综合项目
├─ 构建一个专业 Agent（例如：研究助手）
├─ 集成 3-5 个工具
├─ 实现完整的错误处理
└─ 编写文档和测试
```

---

## 🔑 关键代码片段

### 1. Agent 主循环框架

```python
messages = [system_msg, user_msg]

for turn in range(MAX_TURNS):
    # 调用 LLM
    response = call_llm(messages)
    messages.append({"role": "assistant", "content": response})

    # 检查答案
    if "<answer>" in response:
        return extract_answer(response)

    # 执行工具
    if "<tool_call>" in response:
        tool_result = execute_tool(response)
        messages.append({"role": "user", "content": tool_result})
```

### 2. 工具定义模板

```python
@register_tool("your_tool")
class YourTool(BaseTool):
    name = "your_tool"
    description = "工具描述"
    parameters = {
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "参数说明"}
        },
        "required": ["param"]
    }

    def call(self, params):
        # 实现工具逻辑
        return result
```

### 3. 工具调用格式

```xml
<!-- LLM 输出 -->
<tool_call>
{"name": "search", "arguments": {"query": ["量子计算"]}}
</tool_call>

<!-- 工具返回 -->
<tool_response>
搜索结果...
</tool_response>
```

---

## 📖 配套文档

在阅读这些带注释的代码时，建议结合以下文档：

| 文档 | 用途 |
|------|------|
| `CLAUDE.md` | 技术参考和命令速查 |
| `ARCHITECTURE_CN.md` | 深度架构分析 |
| `PROJECT_GUIDE_CN.md` | 完整项目指南 |
| `PROJECT_GUIDE_APPENDIX.md` | 开发指南和进阶主题 |

---

## 💡 学习建议

### ✅ 推荐的学习方式

1. **先运行，再理解**
   - 先运行 `04_minimal_agent_example.py`
   - 观察输出，建立直观认识
   - 然后阅读代码，理解细节

2. **边读边改**
   - 阅读代码时，尝试修改参数
   - 添加 print 语句观察中间结果
   - 实验不同的工具组合

3. **循序渐进**
   - 不要一开始就读完整实现
   - 先理解核心概念
   - 逐步深入细节

4. **动手实践**
   - 每读完一个文件，做一个小练习
   - 创建自己的工具
   - 修改提示词

### ❌ 应避免的学习方式

1. ❌ 跳过运行直接看代码
2. ❌ 试图一次理解所有细节
3. ❌ 只看不动手
4. ❌ 忽略错误处理部分

---

## 🛠️ 实践练习

### 练习 1: 添加新工具（简单）

```python
# 在 04_minimal_agent_example.py 中添加一个"翻译"工具

class TranslatorTool:
    def __init__(self):
        self.name = "translator"

    def call(self, arguments: dict) -> str:
        text = arguments.get("text", "")
        target_lang = arguments.get("target_lang", "en")

        # TODO: 实现翻译逻辑
        # 提示: 可以调用翻译 API 或使用 LLM
        return f"翻译结果: {text} -> {target_lang}"
```

### 练习 2: 优化提示词（中等）

```python
# 修改 02_prompt_system.py 中的系统提示词
# 添加 Few-Shot 示例来提高工具使用准确率

SYSTEM_PROMPT = """
...

# 工具使用示例

## 示例 1: 搜索
用户: "2024 年诺贝尔物理学奖得主是谁？"
助手: <tool_call>{"name": "search", "arguments": {"query": ["2024 Nobel Prize Physics"]}}</tool_call>

## 示例 2: 计算
用户: "123 乘以 456 等于多少？"
助手: <tool_call>{"name": "calculator", "arguments": {"expression": "123 * 456"}}</tool_call>
"""
```

### 练习 3: 实现异步 Agent（高级）

```python
# 参考 NestBrowse 的实现
# 将 04_minimal_agent_example.py 改造为异步版本

import asyncio

class AsyncMinimalAgent:
    async def run(self, question: str):
        # 使用 asyncio.gather 并发执行多个工具
        pass
```

---

## 🐛 常见问题

### Q1: 如何调试 Agent？

```python
# 在循环中添加详细日志
for turn in range(max_turns):
    print(f"\n{'='*60}")
    print(f"第 {turn+1} 轮")
    print(f"消息数量: {len(messages)}")
    print(f"最后一条消息: {messages[-1]['content'][:100]}...")
    print(f"{'='*60}\n")
```

### Q2: Agent 不调用工具怎么办？

检查：
1. 提示词中的工具定义是否清晰
2. 工具参数格式是否正确
3. 添加 Few-Shot 示例
4. 调整 temperature（降低可能提高准确性）

### Q3: 如何限制 Agent 的执行时间？

```python
import time

start_time = time.time()
timeout = 300  # 5 分钟

for turn in range(max_turns):
    if time.time() - start_time > timeout:
        print("超时！")
        break

    # ... 循环逻辑
```

---

## 📞 获取帮助

如果您在学习过程中遇到问题：

1. 📖 查看配套文档：`PROJECT_GUIDE_CN.md`
2. 💬 查看常见问题：`FAQ.md`
3. 🐛 提交 Issue：[GitHub Issues](https://github.com/Alibaba-NLP/DeepResearch/issues)

---

## 🎯 下一步

完成这些带注释代码的学习后，您可以：

1. ✅ 理解 Agent 的核心原理
2. ✅ 创建自己的工具
3. ✅ 修改和优化提示词
4. ✅ 构建简单的 Agent 系统

**推荐下一步**:
1. 阅读完整的 `inference/react_agent.py`
2. 学习 `WebAgent/NestBrowse/` 的异步实现
3. 研究 `WebAgent/ParallelMuse/` 的多轨迹聚合
4. 构建您自己的专业 Agent 项目

祝学习愉快！🚀
