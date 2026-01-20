"""
【核心文件 1】ReAct Agent 主循环 - 详细注释版

原文件: inference/react_agent.py
核心行数: 126-226

这是整个 Agent 系统的心脏，理解这个文件就理解了 80% 的 Agent 逻辑。
"""

import time
import json
from typing import Dict, List

# ============================================================================
# 核心类: MultiTurnReactAgent
# ============================================================================

class MultiTurnReactAgent:
    """
    多轮 ReAct 代理

    ReAct = Reasoning (推理) + Acting (行动)
    这个类实现了一个可以：
    1. 思考问题
    2. 调用工具
    3. 观察结果
    4. 继续思考
    的循环系统
    """

    def _run(self, data: Dict, model: str, planning_port: int) -> Dict:
        """
        Agent 的主执行函数

        参数:
            data: 包含 question（问题）和 answer（参考答案）的字典
            model: 模型名称
            planning_port: vLLM 服务器端口号

        返回:
            包含完整执行信息的字典：
            - question: 原始问题
            - answer: 参考答案
            - messages: 完整的对话历史
            - prediction: 模型生成的答案
            - termination: 终止原因
        """

        # ====================================================================
        # 步骤 1: 初始化
        # ====================================================================

        # 提取用户问题
        question = data['item']['question']

        # 记录开始时间（用于超时检查）
        start_time = time.time()

        # 获取参考答案（用于评估，不会给 Agent 看到）
        answer = data['item']['answer']

        # ====================================================================
        # 步骤 2: 构建初始消息列表
        # ====================================================================

        # 构建系统提示词（包含当前日期）
        system_prompt = SYSTEM_PROMPT + today_date()

        # 初始化消息历史
        # 这是 Agent 的"记忆"，包含所有对话内容
        messages = [
            {
                "role": "system",      # 系统消息：定义 Agent 的能力和行为
                "content": system_prompt
            },
            {
                "role": "user",        # 用户消息：初始问题
                "content": question
            }
        ]

        # ====================================================================
        # 步骤 3: 设置循环控制参数
        # ====================================================================

        # 最大 LLM 调用次数（防止无限循环）
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN  # 通常是 100

        # 当前轮次计数器
        round = 0

        # ====================================================================
        # 步骤 4: 主循环 - Agent 的核心逻辑
        # ====================================================================

        while num_llm_calls_available > 0:

            # ================================================================
            # 检查点 1: 超时检查
            # ================================================================

            # 如果运行时间超过 150 分钟，终止执行
            if time.time() - start_time > 150 * 60:
                prediction = 'No answer found after 2h30mins'
                termination = 'timeout'
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }

            # ================================================================
            # 执行步骤 A: 调用 LLM 生成响应
            # ================================================================

            round += 1
            num_llm_calls_available -= 1

            # 调用 LLM API（通过 vLLM 服务器）
            # 输入: 完整的消息历史
            # 输出: LLM 生成的文本
            content = self.call_server(messages, planning_port)

            print(f'Round {round}: {content}')

            # 清理响应内容（移除可能的错误标签）
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]

            # ================================================================
            # 执行步骤 B: 将 LLM 响应添加到消息历史
            # ================================================================

            messages.append({
                "role": "assistant",   # 助手消息：LLM 的响应
                "content": content.strip()
            })

            # ================================================================
            # 执行步骤 C: 检查是否有工具调用
            # ================================================================

            if '<tool_call>' in content and '</tool_call>' in content:
                """
                工具调用格式示例:
                <tool_call>
                {"name": "search", "arguments": {"query": ["量子计算"]}}
                </tool_call>
                """

                # 提取工具调用的 JSON 部分
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]

                try:
                    # ----------------------------------------------------
                    # 特殊处理: Python 解释器工具
                    # ----------------------------------------------------
                    if "python" in tool_call.lower():
                        # Python 代码在 <code> 标签中
                        try:
                            code_raw = content.split('<tool_call>')[1] \
                                              .split('</tool_call>')[0] \
                                              .split('<code>')[1] \
                                              .split('</code>')[0].strip()

                            # 调用 Python 解释器工具
                            result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                        except:
                            result = "[Python Interpreter Error]: Formatting error."

                    # ----------------------------------------------------
                    # 通用工具调用处理
                    # ----------------------------------------------------
                    else:
                        # 解析 JSON
                        tool_call = json5.loads(tool_call)

                        # 提取工具名称和参数
                        tool_name = tool_call.get('name', '')      # 例如: "search"
                        tool_args = tool_call.get('arguments', {})  # 例如: {"query": [...]}

                        # 调用工具
                        result = self.custom_call_tool(tool_name, tool_args)

                except:
                    # 工具调用格式错误
                    result = 'Error: Tool call is not a valid JSON.'

                # ----------------------------------------------------
                # 包装工具结果为 <tool_response> 格式
                # ----------------------------------------------------
                result = "<tool_response>\n" + result + "\n</tool_response>"

                # 将工具结果添加到消息历史（作为 user 消息）
                messages.append({
                    "role": "user",
                    "content": result
                })

            # ================================================================
            # 检查点 2: 是否找到答案
            # ================================================================

            if '<answer>' in content and '</answer>' in content:
                """
                答案格式示例:
                <answer>
                量子计算是利用量子力学原理进行计算的新型计算模式...
                </answer>
                """
                termination = 'answer'
                break  # 找到答案，退出循环

            # ================================================================
            # 检查点 3: LLM 调用次数是否用完
            # ================================================================

            if num_llm_calls_available <= 0 and '<answer>' not in content:
                # 达到最大调用次数但还没找到答案
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            # ================================================================
            # 检查点 4: 上下文长度检查
            # ================================================================

            max_tokens = 110 * 1024  # 110K tokens
            token_count = self.count_tokens(messages)

            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                """
                上下文溢出处理:
                1. 告诉 Agent 上下文已满
                2. 要求 Agent 基于现有信息给出最佳答案
                3. 强制终止
                """
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")

                # 更新最后一条消息，提示 Agent 上下文已满
                messages[-1]['content'] = """
                You have now reached the maximum context length you can handle.
                You should stop making tool calls and, based on all the information above,
                think again and provide what you consider the most likely answer in the following format:
                <think>your final thinking</think>
                <answer>your answer</answer>
                """

                # 最后一次调用 LLM
                content = self.call_server(messages, planning_port)
                messages.append({"role": "assistant", "content": content.strip()})

                # 提取答案
                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'] \
                                    .split('<answer>')[1] \
                                    .split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'

                # 返回结果
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }

        # ====================================================================
        # 步骤 5: 循环结束后的处理
        # ====================================================================

        # 提取最终答案
        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'] \
                            .split('<answer>')[1] \
                            .split('</answer>')[0]
            termination = 'answer'
        else:
            # 没有找到答案
            prediction = 'No answer found.'
            termination = 'answer not found'

            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'

        # ====================================================================
        # 步骤 6: 返回完整结果
        # ====================================================================

        result = {
            "question": question,         # 原始问题
            "answer": answer,             # 参考答案（用于评估）
            "messages": messages,         # 完整对话历史
            "prediction": prediction,     # 模型预测的答案
            "termination": termination    # 终止原因
        }

        return result


# ============================================================================
# 辅助函数: call_server
# ============================================================================

def call_server(self, messages: List[Dict], planning_port: int, max_tries: int = 10) -> str:
    """
    调用 vLLM 服务器生成响应

    参数:
        messages: 消息历史
        planning_port: vLLM 服务器端口
        max_tries: 最大重试次数

    返回:
        LLM 生成的文本
    """

    from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

    # 配置 OpenAI 客户端（vLLM 兼容 OpenAI API）
    openai_api_key = "EMPTY"  # vLLM 不需要真实 API key
    openai_api_base = f"http://127.0.0.1:{planning_port}/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=600.0  # 10 分钟超时
    )

    # 重试机制（指数退避）
    base_sleep_time = 1
    for attempt in range(max_tries):
        try:
            print(f"--- 尝试调用 LLM (第 {attempt + 1}/{max_tries} 次) ---")

            # 调用 API
            chat_response = client.chat.completions.create(
                model=self.model,
                messages=messages,

                # --------------------------------------------------------
                # 停止标记 (stop)
                # --------------------------------------------------------
                # 作用: 当 LLM 生成这些字符串时立即停止
                # 目的: 防止 LLM 自己编造 <tool_response> 内容（幻觉）
                # 两个版本: 带/不带换行符，覆盖更多情况
                stop=["\n<tool_response>", "<tool_response>"],

                # --------------------------------------------------------
                # 采样温度 (temperature)
                # --------------------------------------------------------
                # 作用: 控制生成的随机性和创造性
                # 范围: 0.0 (完全确定) → 2.0 (高度随机)
                # 0.6: 平衡准确性和多样性，适合 Agent 任务
                #   - 工具调用格式准确
                #   - 搜索策略有一定多样性
                # 调优: 事实查询用 0.3，创意任务用 1.0+
                temperature=self.llm_generate_cfg.get('temperature', 0.6),

                # --------------------------------------------------------
                # 核采样概率 (top_p / nucleus sampling)
                # --------------------------------------------------------
                # 作用: 限制候选词范围，只从累计概率达到 p 的词中采样
                # 范围: 0.0 → 1.0
                # 0.95: 考虑概率前 95% 的词
                #   - 排除低概率的"奇怪"词
                #   - 保持生成质量
                # 与 temperature 配合: temperature 调整概率，top_p 限制范围
                top_p=self.llm_generate_cfg.get('top_p', 0.95),

                # --------------------------------------------------------
                # 最大生成长度 (max_tokens)
                # --------------------------------------------------------
                # 作用: 限制单次生成的最大 token 数量
                # 10000 tokens ≈ 7500 英文单词 或 5000-10000 汉字
                # 设计考虑:
                #   - 工具调用: 100-200 tokens
                #   - 思考内容: 500-1000 tokens
                #   - 最终答案: 2000-5000 tokens
                #   - 安全边界: 防止无限生成
                # 注意: 这是上限，正常响应遇到 stop 标记会提前结束
                max_tokens=10000,

                # --------------------------------------------------------
                # 存在惩罚 (presence_penalty)
                # --------------------------------------------------------
                # 作用: 惩罚已出现过的 token，鼓励生成新内容
                # 范围: -2.0 → 2.0
                # 1.1: 较高的惩罚值
                # 目的:
                #   - 避免重复调用同一工具
                #   - 鼓励多样化的搜索策略
                #   - 减少答案中的重复表达
                # 示例: 避免 "量子计算...量子计算...量子计算..."
                #       而是 "量子计算...这项技术...它能够..."
                presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1)
            )

            # 提取生成的内容
            content = chat_response.choices[0].message.content

            # 检查内容是否有效
            if content and content.strip():
                print("--- LLM 调用成功 ---")
                return content.strip()
            else:
                print(f"警告: 第 {attempt + 1} 次调用返回空响应")

        except (APIError, APIConnectionError, APITimeoutError) as e:
            print(f"错误: 第 {attempt + 1} 次调用失败，API 错误: {e}")

        except Exception as e:
            print(f"错误: 第 {attempt + 1} 次调用失败，未知错误: {e}")

        # 如果不是最后一次尝试，等待后重试
        if attempt < max_tries - 1:
            sleep_time = base_sleep_time * (2 ** attempt)  # 指数退避
            print(f"等待 {sleep_time} 秒后重试...")
            time.sleep(sleep_time)

    # 所有尝试都失败
    raise Exception(f"LLM 调用失败，已尝试 {max_tries} 次")


# ============================================================================
# 辅助函数: custom_call_tool
# ============================================================================

def custom_call_tool(self, tool_name: str, tool_args: Dict) -> str:
    """
    调用指定的工具

    参数:
        tool_name: 工具名称，例如 "search"
        tool_args: 工具参数，例如 {"query": ["量子计算"]}

    返回:
        工具执行结果（字符串）
    """

    # 从工具注册表获取工具
    tool = TOOL_MAP.get(tool_name)

    if not tool:
        return f"错误: 未知工具 '{tool_name}'"

    try:
        # 将参数转换为 JSON 字符串
        params_json = json.dumps(tool_args)

        # 调用工具的 call 方法
        result = tool.call(params_json)

        return result

    except Exception as e:
        return f"工具调用失败: {str(e)}"


# ============================================================================
# 关键概念总结
# ============================================================================

"""
理解这个文件的关键点:

1. **消息历史是核心**
   messages = [
       {"role": "system", "content": "系统提示"},
       {"role": "user", "content": "用户问题"},
       {"role": "assistant", "content": "LLM 响应"},
       {"role": "user", "content": "工具结果"},
       ...
   ]

2. **循环是灵魂**
   while 条件:
       LLM 响应 → 解析 → 执行工具 → 添加结果 → 继续

3. **四个检查点**
   - 超时检查（150 分钟）
   - 答案检查（<answer> 标签）
   - 调用次数检查（100 次）
   - 上下文检查（110K tokens）

4. **工具调用格式**
   <tool_call>
   {"name": "工具名", "arguments": {参数}}
   </tool_call>

   <tool_response>
   工具返回结果
   </tool_response>

5. **终止方式**
   - 正常: 找到 <answer>
   - 超时: 150 分钟
   - 次数: 100 次 LLM 调用
   - 上下文: 110K tokens

建议学习路径:
1. 先理解 while 循环的整体结构
2. 然后看四个检查点的逻辑
3. 最后看工具调用的处理细节
"""
