# ============================================================================
# Python 代码沙箱执行工具
# ============================================================================
#
# 功能说明：
#   本模块实现了在沙箱环境中安全执行 Python 代码的功能
#
# 核心特性：
#   - 使用 SandboxFusion 服务提供隔离的执行环境
#   - 支持多端点配置（负载均衡 + 容错）
#   - 智能重试机制（最多 8 次，随机选择端点）
#   - 超时控制（默认 50 秒）
#   - 分离 stdout 和 stderr 输出
#
# 安全性：
#   - 代码在隔离的容器中执行，不会影响主系统
#   - 网络访问受限
#   - 文件系统访问受限
#   - 执行时间受限（防止无限循环）
#
# SandboxFusion 服务：
#   - 项目地址: https://github.com/bytedance/SandboxFusion
#   - 需要单独部署 SandboxFusion 服务
#   - 通过 HTTP API 提交代码并获取执行结果
#
# 环境变量配置：
#   SANDBOX_FUSION_ENDPOINT: 多个端点用逗号分隔
#   示例: "http://endpoint1:8080,http://endpoint2:8080,http://endpoint3:8080"
#
# ============================================================================

import re
from typing import Dict, List, Optional, Union, Any
import json5
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.utils.utils import extract_code
from sandbox_fusion import run_code, RunCodeRequest, RunStatus
from requests.exceptions import Timeout
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====================================================================
# 中文字符检测正则表达式
# ====================================================================
# Unicode 范围 U+4E00 到 U+9FFF 覆盖常用中文字符
CHINESE_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')


def has_chinese_chars(data: Any) -> bool:
    """检测数据中是否包含中文字符"""
    text = f'{data}'
    return bool(CHINESE_CHAR_RE.search(text))


# ====================================================================
# SandboxFusion 端点配置
# ====================================================================
# 多端点配置的优势：
#   1. 负载均衡: 将请求分散到多个服务器
#   2. 容错性: 某个端点失败时可以尝试其他端点
#   3. 高可用性: 单个服务器故障不影响整体服务
#
# 配置方式：
#   - 从环境变量 SANDBOX_FUSION_ENDPOINT 读取
#   - 多个端点用逗号分隔
#   - 示例: "http://10.0.0.1:8080,http://10.0.0.2:8080,http://10.0.0.3:8080"
SANDBOX_FUSION_ENDPOINTS = []

# 从环境变量加载端点列表
if 'SANDBOX_FUSION_ENDPOINT' in os.environ:
    SANDBOX_FUSION_ENDPOINTS = os.environ['SANDBOX_FUSION_ENDPOINT'].split(',')


# ====================================================================
# PythonInterpreter 工具类
# ====================================================================
# 继承自 BaseToolWithFileAccess（支持文件访问的工具基类）
# 注册为 qwen_agent 工具
@register_tool('PythonInterpreter', allow_overwrite=True)
class PythonInterpreter(BaseToolWithFileAccess):
    name = "PythonInterpreter"

    # ----------------------------------------------------------------
    # 工具描述（传递给 LLM）
    # ----------------------------------------------------------------
    # 重要说明：
    #   1. 强调使用 print() 语句
    #      - 原因：只有 print 的内容会出现在 stdout
    #      - LLM 需要明确知道这一点，否则可能写出没有输出的代码
    #
    #   2. XML 标签格式要求
    #      - 代码必须在 <code></code> 标签内
    #      - arguments 字段为空字典 {}
    #      - 这是为了避免 JSON 转义问题（代码中的引号、换行等）
    #
    # 示例格式：
    #   <tool_call>
    #   {"name": "PythonInterpreter", "arguments": {}}
    #   <code>
    #   import numpy as np
    #   result = np.mean([1, 2, 3, 4, 5])
    #   print(f"平均值: {result}")
    #   </code>
    #   </tool_call>
    description = 'Execute Python code in a sandboxed environment. Use this to run Python code and get the execution results.\n**Make sure to use print() for any output you want to see in the results.**\nFor code parameters, use placeholders first, and then put the code within <code></code> XML tags, such as:\n<tool_call>\n{"purpose": <detailed-purpose-of-this-tool-call>, "name": <tool-name>, "arguments": {"code": ""}}\n<code>\nHere is the code.\n</code>\n</tool_call>\n'

    # ----------------------------------------------------------------
    # 参数定义（JSON Schema 格式）
    # ----------------------------------------------------------------
    # code: 要执行的 Python 代码字符串
    # 注意：实际使用中，代码会从 <code></code> 标签中提取
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. Must be provided within <code></code> XML tags. Remember to use print() statements for any output you want to see.",
            }
        },
        "required": ["code"],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        # self.summary_mapping = SummaryMapping()  # 保留，未来可能用于结果摘要


    # ====================================================================
    # args_format 属性：根据语言返回参数格式说明
    # ====================================================================
    # 作用：
    #   - 告诉 LLM 如何格式化代码参数
    #   - 中文环境使用 "Markdown code block" 提示
    #   - 英文环境使用 "triple backticks" 提示
    #
    # 原理：
    #   - 检测工具名称、描述、参数中是否包含中文
    #   - 根据语言环境返回对应的格式说明
    # ====================================================================
    @property
    def args_format(self) -> str:
        fmt = self.cfg.get('args_format')
        if fmt is None:
            if has_chinese_chars([self.name_for_human, self.name, self.description, self.parameters]):
                # 中文环境
                fmt = 'The input for this tool should be a Markdown code block.'
            else:
                # 英文环境
                fmt = 'Enclose the code within triple backticks (`) at the beginning and end of the code.'
        return fmt

    # ====================================================================
    # observation 方法：处理工具执行结果
    # ====================================================================
    # 作用：
    #   - 在 qwen_agent 框架中，此方法用于格式化工具输出
    #   - 对于 PythonInterpreter，直接返回结果字符串
    #
    # 参数：
    #   tool: 工具调用信息
    #   tool_dict: 工具字典
    #   tool_results: 工具执行结果（应为字符串）
    #   empty_mode: 是否为空模式
    #   readpage: 是否为页面读取模式
    #   max_observation_length: 最大观察长度
    #   tokenizer: 分词器
    # ====================================================================
    def observation(self, tool: dict, tool_dict: dict, tool_results, empty_mode: bool=False, readpage: bool=False, max_observation_length: int=None, tokenizer=None):
        print('test')
        # 断言结果必须是字符串类型
        assert isinstance(tool_results, str), f"result of python code should be str, instead of {type(tool_results)}. {tool_results}"
        return tool_results

    # ====================================================================
    # function 属性：返回工具的函数定义
    # ====================================================================
    # 用于 OpenAI-style function calling 格式
    @property
    def function(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
        }

    # ====================================================================
    # call 方法：执行 Python 代码（工具的核心方法）
    # ====================================================================
    # 参数：
    #   params: 代码字符串（从 <code></code> 标签中提取）
    #   files: 文件列表（可选，通常不使用）
    #   timeout: 超时时间（秒），默认 50 秒
    #   **kwargs: 额外参数
    #
    # 返回值：
    #   执行结果字符串，格式：
    #     stdout:
    #     [标准输出内容]
    #     stderr:
    #     [标准错误内容]
    #
    # 工作流程：
    #   1. 随机选择一个 SandboxFusion 端点
    #   2. 提交代码执行请求
    #   3. 如果失败（超时或错误），选择另一个端点重试
    #   4. 最多重试 8 次
    #   5. 返回执行结果或错误信息
    # ====================================================================
    def call(self, params, files= None, timeout = 50, **kwargs) -> str:
        try:
            code=params
            last_error = None

            # ================================================================
            # 多端点重试循环（最多 8 次）
            # ================================================================
            # 为什么重试 8 次？
            #   - SandboxFusion 服务可能偶尔超时或繁忙
            #   - 8 次足够覆盖多个端点的多轮尝试
            #   - 假设有 3 个端点，8 次尝试可以让每个端点至少被尝试 2-3 次
            #
            # 随机选择端点的原因：
            #   - 负载均衡：避免所有请求都打到同一个端点
            #   - 容错性：某个端点故障时自动切换到其他端点
            #   - random.choice() 确保每次尝试可能选择不同的端点
            for attempt in range(8):
                try:
                    # ------------------------------------------------------------
                    # 步骤 1: 随机选择一个端点
                    # ------------------------------------------------------------
                    endpoint = random.choice(SANDBOX_FUSION_ENDPOINTS)
                    print(f"Attempt {attempt + 1}/5 using endpoint: {endpoint}")

                    # ------------------------------------------------------------
                    # 步骤 2: 提交代码执行请求
                    # ------------------------------------------------------------
                    # RunCodeRequest 参数：
                    #   - code: Python 代码字符串
                    #   - language: 'python'（支持其他语言如 'bash', 'javascript'）
                    #   - run_timeout: 代码执行超时时间（秒）
                    #
                    # run_code 参数：
                    #   - max_attempts: 1（在此层不重试，重试逻辑在外层）
                    #   - client_timeout: HTTP 请求超时时间
                    #   - endpoint: SandboxFusion 服务的 URL
                    code_result = run_code(
                        RunCodeRequest(code=code, language='python', run_timeout=timeout),
                        max_attempts=1,
                        client_timeout=timeout,
                        endpoint=endpoint
                    )
                    print("[Python] Code Result", code_result)

                    # ------------------------------------------------------------
                    # 步骤 3: 解析执行结果
                    # ------------------------------------------------------------
                    result = []

                    # stdout: 标准输出（print() 语句的内容）
                    if code_result.run_result.stdout:
                        result.append(f"stdout:\n{code_result.run_result.stdout}")

                    # stderr: 标准错误（异常、警告等）
                    if code_result.run_result.stderr:
                        result.append(f"stderr:\n{code_result.run_result.stderr}")

                    # 检查是否接近超时（execution_time >= timeout-1）
                    # 即使代码完成了，如果接近超时也提示用户
                    if code_result.run_result.execution_time >= timeout-1:
                        result.append(f"[PythonInterpreter Error] TimeoutError: Execution timed out.")

                    result = '\n'.join(result)
                    print('SUCCESS RUNNING TOOL')

                    # 如果结果为空（没有 stdout 也没有 stderr），返回成功消息
                    return result if result.strip() else 'Finished execution.'

                # ============================================================
                # 异常处理：Timeout（超时）
                # ============================================================
                # Timeout 异常：HTTP 请求超时或代码执行超时
                # 处理策略：记录错误，继续尝试其他端点
                except Timeout as e:
                    last_error = f'[Python Interpreter Error] TimeoutError: Execution timed out on endpoint {endpoint}.'
                    print(f"Timeout on attempt {attempt + 1}: {last_error}")

                    # 注意：这里检查 attempt == 4 但循环是 range(8)
                    # 说明最初设计是 5 次重试，后来改为 8 次但忘记更新此处
                    # 实际会重试 8 次（因为外层循环是 range(8)）
                    if attempt == 4:  # 第 5 次失败（索引 4）
                        return last_error
                    continue  # 继续下一次重试

                # ============================================================
                # 异常处理：其他异常
                # ============================================================
                # 可能的异常：
                #   - 网络连接失败
                #   - SandboxFusion 服务返回错误
                #   - 代码执行失败（如语法错误、运行时错误）
                except Exception as e:
                    last_error = f'[Python Interpreter Error]: {str(e)} on endpoint {endpoint}'
                    print(f"Error on attempt {attempt + 1}: {last_error}")
                    if attempt == 4:  # 第 5 次失败（索引 4）
                        return last_error
                    continue  # 继续下一次重试

            # ================================================================
            # 所有尝试都失败
            # ================================================================
            # 返回最后一次错误信息或通用失败消息
            return last_error if last_error else '[Python Interpreter Error]: All attempts failed.'

        # ====================================================================
        # 外层异常处理：捕获意外错误
        # ====================================================================
        # 例如：params 格式错误、SANDBOX_FUSION_ENDPOINTS 为空等
        except Exception as e:
            return f"[Python Interpreter Error]: {str(e)}"

    # ====================================================================
    # call_specific_endpoint 方法：测试特定端点
    # ====================================================================
    # 作用：
    #   - 用于测试和调试特定的 SandboxFusion 端点
    #   - 不使用重试机制，直接测试单个端点
    #   - 返回成功/失败状态和执行时间
    #
    # 参数：
    #   params: 代码字符串或包含代码的字典
    #   endpoint: 要测试的端点 URL
    #   timeout: 超时时间（秒），默认 30 秒
    #   **kwargs: 额外参数
    #
    # 返回值：
    #   元组 (成功标志, 结果/错误信息, 执行时间)
    #   - 成功: (True, "stdout:\n...", 1.23)
    #   - 失败: (False, "错误信息", None)
    #
    # 使用场景：
    #   - 部署新端点后验证其可用性
    #   - 诊断特定端点的问题
    #   - 性能测试和基准测试
    # ====================================================================
    def call_specific_endpoint(self, params: Union[str, dict], endpoint: str, timeout: Optional[int] = 30, **kwargs) -> tuple:
        """Test a specific endpoint directly"""

        # ----------------------------------------------------------------
        # 步骤 1: 提取代码
        # ----------------------------------------------------------------
        try:
            # 如果 params 是字符串，解析为 JSON
            if type(params) is str:
                params = json5.loads(params)

            # 尝试从 'code' 或 'raw' 字段获取代码
            code = params.get('code', '')
            if not code:
                code = params.get('raw', '')

            # --------------------------------------------------------
            # 处理三引号代码块
            # --------------------------------------------------------
            # 匹配格式: ```python\n代码内容\n```
            # 正则说明：
            #   - ```[^\n]*: 三引号 + 可选语言标识符
            #   - \n(.+?): 捕获代码内容
            #   - ```: 结束的三引号
            #   - re.DOTALL: . 匹配换行符
            triple_match = re.search(r'```[^\n]*\n(.+?)```', code, re.DOTALL)
            if triple_match:
                code = triple_match.group(1)  # 提取代码块内容
        except Exception:
            # 如果解析失败，使用 qwen_agent 的 extract_code 工具
            code = extract_code(params)

        # ----------------------------------------------------------------
        # 步骤 2: 验证代码不为空
        # ----------------------------------------------------------------
        if not code.strip():
            return False, '[Python Interpreter Error]: Empty code.', None

        # ----------------------------------------------------------------
        # 步骤 3: 执行代码并计时
        # ----------------------------------------------------------------
        try:
            start_time = time.time()

            # 调用 SandboxFusion 执行代码
            code_result = run_code(
                RunCodeRequest(code=code, language='python', run_timeout=timeout),
                max_attempts=1,  # 不重试
                client_timeout=timeout,
                endpoint=endpoint
            )

            end_time = time.time()

            # --------------------------------------------------------
            # 提取执行结果
            # --------------------------------------------------------
            result = []
            if code_result.run_result.stdout:
                result.append(f"stdout:\n{code_result.run_result.stdout}")
            if code_result.run_result.stderr:
                result.append(f"stderr:\n{code_result.run_result.stderr}")

            result = '\n'.join(result)
            execution_time = end_time - start_time

            # 返回：(成功, 结果, 执行时间)
            return True, result if result.strip() else 'Finished execution.', execution_time

        # ----------------------------------------------------------------
        # 异常处理
        # ----------------------------------------------------------------
        except Timeout as e:
            return False, f'[Python Interpreter Error] TimeoutError: Execution timed out.', None
        except Exception as e:
            return False, f'[Python Interpreter Error]: {str(e)}', None
