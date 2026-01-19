"""
【完整示例】最小化 Agent 实现 - 100 行代码

这是一个完全可运行的最小 Agent 示例。
包含完整的注释和错误处理。

学习目标:
1. 理解 Agent 的核心循环
2. 掌握工具调用机制
3. 学会处理消息历史
4. 了解终止条件

前置条件:
1. 已启动 vLLM 服务器（或配置 OpenRouter）
2. 已配置环境变量
"""

import os
import re
import json
from openai import OpenAI


# ============================================================================
# 步骤 1: 定义工具
# ============================================================================

class SearchTool:
    """
    简单的搜索工具（模拟）

    实际使用时，可以替换为真实的 API 调用
    """

    def __init__(self):
        self.name = "search"

    def call(self, arguments: dict) -> str:
        """执行搜索"""
        query = arguments.get("query", [""])[0]

        # 模拟搜索结果
        return f"""
# 搜索结果: "{query}"

1. [示例文章 1](https://example.com/1)
这是关于 {query} 的详细介绍...

2. [示例文章 2](https://example.com/2)
更多关于 {query} 的信息...
        """


class CalculatorTool:
    """
    简单的计算器工具
    """

    def __init__(self):
        self.name = "calculator"

    def call(self, arguments: dict) -> str:
        """执行计算"""
        expression = arguments.get("expression", "")

        try:
            # 安全的计算（仅支持基本运算）
            result = eval(expression, {"__builtins__": {}}, {})
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"


# ============================================================================
# 步骤 2: 定义 Agent
# ============================================================================

class MinimalAgent:
    """
    最小化 Agent 实现

    特点:
    - 完整的 ReAct 循环
    - 工具注册和调用
    - 消息历史管理
    - 终止条件检查
    """

    def __init__(self, api_key: str, api_base: str, model: str = "deepresearch"):
        """
        初始化 Agent

        参数:
            api_key: OpenAI API Key
            api_base: API 基础 URL
            model: 模型名称
        """

        # ====================================================================
        # 配置 OpenAI 客户端
        # ====================================================================

        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        self.model = model

        # ====================================================================
        # 注册工具
        # ====================================================================

        # 工具注册表
        self.tools = {
            "search": SearchTool(),
            "calculator": CalculatorTool()
        }

        # ====================================================================
        # 定义系统提示词
        # ====================================================================

        self.system_prompt = """你是一个智能助手。你可以使用以下工具：

<tools>
1. search - 搜索信息
   参数: {"query": ["搜索关键词"]}

2. calculator - 执行计算
   参数: {"expression": "数学表达式"}
</tools>

调用工具的格式:
<tool_call>
{"name": "工具名", "arguments": {参数}}
</tool_call>

完成任务后，使用以下格式给出答案:
<answer>你的答案</answer>
        """

    def run(self, question: str, max_turns: int = 10) -> dict:
        """
        运行 Agent

        参数:
            question: 用户问题
            max_turns: 最大轮次

        返回:
            结果字典: {answer, messages, termination}
        """

        # ====================================================================
        # 初始化消息历史
        # ====================================================================

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]

        print(f"🤔 问题: {question}\n")

        # ====================================================================
        # 主循环
        # ====================================================================

        for turn in range(1, max_turns + 1):
            print(f"{'='*60}")
            print(f"第 {turn} 轮")
            print(f"{'='*60}\n")

            # ================================================================
            # 步骤 A: 调用 LLM
            # ================================================================

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )

                content = response.choices[0].message.content

                # 添加到消息历史
                messages.append({
                    "role": "assistant",
                    "content": content
                })

                print(f"🤖 Agent 响应:\n{content}\n")

            except Exception as e:
                print(f"❌ LLM 调用失败: {e}")
                return {
                    "answer": None,
                    "messages": messages,
                    "termination": "llm_error"
                }

            # ================================================================
            # 步骤 B: 检查是否完成
            # ================================================================

            if "<answer>" in content and "</answer>" in content:
                # 提取答案
                answer = self._extract_answer(content)

                print(f"✅ 找到答案!\n")
                print(f"📝 最终答案: {answer}\n")

                return {
                    "answer": answer,
                    "messages": messages,
                    "termination": "answer_found"
                }

            # ================================================================
            # 步骤 C: 检查是否有工具调用
            # ================================================================

            if "<tool_call>" in content and "</tool_call>" in content:
                # 执行工具
                tool_result = self._execute_tools(content)

                # 添加工具结果到消息历史
                messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{tool_result}\n</tool_response>"
                })

                print(f"🔧 工具返回:\n{tool_result}\n")

        # ====================================================================
        # 达到最大轮次
        # ====================================================================

        print(f"⚠️ 达到最大轮次 ({max_turns})，未找到答案\n")

        return {
            "answer": None,
            "messages": messages,
            "termination": "max_turns_exceeded"
        }

    def _execute_tools(self, content: str) -> str:
        """
        执行工具调用

        参数:
            content: LLM 响应内容

        返回:
            工具执行结果
        """

        # 提取 <tool_call> 标签内容
        match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)

        if not match:
            return "错误: 无法解析工具调用"

        try:
            # 解析 JSON
            tool_call = json.loads(match.group(1).strip())

            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})

            print(f"🔧 调用工具: {tool_name}")
            print(f"📌 参数: {json.dumps(tool_args, ensure_ascii=False)}\n")

            # 获取工具
            tool = self.tools.get(tool_name)

            if not tool:
                return f"错误: 未知工具 '{tool_name}'"

            # 执行工具
            result = tool.call(tool_args)

            return result

        except json.JSONDecodeError:
            return "错误: 工具调用不是有效的 JSON"

        except Exception as e:
            return f"错误: 工具执行失败 - {str(e)}"

    def _extract_answer(self, content: str) -> str:
        """
        提取 <answer> 标签中的内容

        参数:
            content: LLM 响应内容

        返回:
            答案文本
        """

        match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)

        if match:
            return match.group(1).strip()
        else:
            return ""


# ============================================================================
# 步骤 3: 使用示例
# ============================================================================

def main():
    """
    主函数：演示如何使用 MinimalAgent
    """

    # ========================================================================
    # 配置
    # ========================================================================

    # 方式 1: 使用本地 vLLM 服务器
    api_key = "EMPTY"
    api_base = "http://127.0.0.1:6001/v1"
    model = "deepresearch"

    # 方式 2: 使用 OpenRouter (取消注释以使用)
    # api_key = os.environ.get("OPENROUTER_API_KEY")
    # api_base = "https://openrouter.ai/api/v1"
    # model = "alibaba/tongyi-deepresearch-30b-a3b"

    # ========================================================================
    # 创建 Agent
    # ========================================================================

    agent = MinimalAgent(
        api_key=api_key,
        api_base=api_base,
        model=model
    )

    # ========================================================================
    # 测试用例
    # ========================================================================

    # 测试 1: 搜索任务
    print("\n" + "="*70)
    print("测试 1: 搜索任务")
    print("="*70 + "\n")

    result1 = agent.run(
        question="搜索量子计算的最新进展",
        max_turns=5
    )

    # 测试 2: 计算任务
    print("\n" + "="*70)
    print("测试 2: 计算任务")
    print("="*70 + "\n")

    result2 = agent.run(
        question="计算 123 * 456 的结果",
        max_turns=5
    )

    # 测试 3: 组合任务
    print("\n" + "="*70)
    print("测试 3: 组合任务")
    print("="*70 + "\n")

    result3 = agent.run(
        question="搜索 Python 的创建者，然后计算 2024 - 1991 的结果",
        max_turns=10
    )

    # ========================================================================
    # 输出总结
    # ========================================================================

    print("\n" + "="*70)
    print("执行总结")
    print("="*70 + "\n")

    for i, result in enumerate([result1, result2, result3], 1):
        print(f"测试 {i}:")
        print(f"  终止原因: {result['termination']}")
        print(f"  答案: {result['answer']}")
        print(f"  总轮次: {len([m for m in result['messages'] if m['role'] == 'assistant'])}")
        print()


# ============================================================================
# 运行示例
# ============================================================================

if __name__ == "__main__":
    """
    运行方式:

    1. 确保 vLLM 服务器已启动:
       vllm serve /path/to/model --port 6001

    2. 运行此脚本:
       python 04_minimal_agent_example.py

    3. 观察输出:
       - 每一轮的 Agent 响应
       - 工具调用过程
       - 最终答案
    """

    main()


# ============================================================================
# 学习要点总结
# ============================================================================

"""
通过这个最小 Agent 示例，您应该理解:

1. **Agent 的核心是循环**
   while not finished:
       LLM 响应 → 解析 → 执行工具 → 更新消息 → 继续

2. **消息历史是关键**
   messages = [
       system_msg,
       user_msg,
       assistant_msg,
       tool_response,
       ...
   ]

3. **工具调用有标准格式**
   <tool_call>
   {"name": "tool_name", "arguments": {...}}
   </tool_call>

4. **终止条件很重要**
   - 找到 <answer> 标签
   - 达到最大轮次
   - 发生错误

5. **错误处理不可少**
   - LLM 调用失败
   - JSON 解析错误
   - 工具执行异常

扩展方向:

1. **添加更多工具**
   - 数据库查询
   - API 调用
   - 文件操作

2. **优化循环逻辑**
   - 上下文截断
   - 工具结果摘要
   - 并发工具调用

3. **增强提示词**
   - Few-shot 示例
   - 思维链提示
   - 工具使用指南

4. **改进终止条件**
   - Token 数量限制
   - 时间限制
   - 置信度阈值

5. **添加调试功能**
   - 日志记录
   - 可视化
   - 性能分析

下一步学习:
1. 阅读完整的 react_agent.py
2. 学习异步 Agent (NestBrowse)
3. 理解多轨迹聚合 (ParallelMuse)
4. 尝试构建自己的专业 Agent
"""
