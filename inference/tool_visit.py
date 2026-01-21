# ============================================================================
# 网页访问和摘要工具
# ============================================================================
#
# 功能说明：
#   本模块实现了访问网页并使用 LLM 生成内容摘要的功能
#
# 核心特性：
#   - 使用 Jina Reader API 将网页转换为 Markdown 格式
#   - 使用 LLM 根据用户目标提取相关信息
#   - 支持批量 URL 处理
#   - Token 数限制（最多 95000 tokens）
#   - 多层重试机制（网页读取重试、摘要生成重试、JSON 解析重试）
#   - 超时控制（默认 200 秒）
#
# 工作流程：
#   1. 使用 Jina Reader API 读取网页内容（转为 Markdown）
#   2. 截断内容到 95000 tokens（避免超过 LLM 上下文限制）
#   3. 使用 EXTRACTOR_PROMPT 让 LLM 提取相关信息
#   4. 解析 LLM 返回的 JSON 结果（包含 evidence 和 summary）
#   5. 如果失败，逐步截断内容并重试
#
# 服务依赖：
#   - Jina Reader API: https://r.jina.ai/{url}
#   - OpenAI-compatible LLM API: 用于内容摘要
#
# 环境变量配置：
#   - VISIT_SERVER_TIMEOUT: LLM 调用超时时间（默认 200 秒）
#   - WEBCONTENT_MAXLENGTH: 网页内容最大长度（默认 150000 字符）
#   - JINA_API_KEYS: Jina API 密钥
#   - API_KEY: LLM API 密钥
#   - API_BASE: LLM API 基础 URL
#   - SUMMARY_MODEL_NAME: 用于摘要的模型名称
#
# ============================================================================

import json
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from prompt import EXTRACTOR_PROMPT  # 导入摘要提示词
from openai import OpenAI
import random
from urllib.parse import urlparse, unquote
import time
from transformers import AutoTokenizer
import tiktoken

# ====================================================================
# 环境变量配置
# ====================================================================
# VISIT_SERVER_TIMEOUT: LLM 摘要服务的超时时间（秒）
#   - 200 秒足够处理大多数网页摘要任务
#   - 过短可能导致复杂网页摘要失败
VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))

# WEBCONTENT_MAXLENGTH: 网页内容最大长度（字符数）
#   - 150000 字符约等于 37500 tokens（中文）
#   - 超过此长度会截断
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))

# JINA_API_KEYS: Jina Reader API 密钥
#   - 从 https://jina.ai/ 获取
#   - 用于将网页转换为 Markdown 格式
JINA_API_KEYS = os.getenv("JINA_API_KEYS", "")


# ====================================================================
# truncate_to_tokens: Token 级别的文本截断函数
# ====================================================================
# 作用：
#   - 将文本截断到指定的 token 数量
#   - 使用 tiktoken (OpenAI 的 tokenizer) 确保准确性
#
# 参数：
#   - text: 要截断的文本
#   - max_tokens: 最大 token 数（默认 95000）
#
# 为什么是 95000 tokens？
#   - 大多数 LLM 的上下文限制在 100K-128K tokens
#   - 95000 留出空间给：
#     * 系统提示词（EXTRACTOR_PROMPT）
#     * 用户目标描述
#     * LLM 的输出
#   - 避免超过模型上下文限制导致错误
#
# 为什么用 tiktoken？
#   - tiktoken 是 OpenAI 的官方 tokenizer
#   - 使用 cl100k_base 编码（用于 GPT-3.5/GPT-4）
#   - 比简单的字符数或词数更准确
#
# 返回值：
#   - 如果文本 token 数 ≤ max_tokens: 返回原文本
#   - 否则: 返回截断后的文本（保证 token 数 = max_tokens）
@staticmethod
def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    # 使用 cl100k_base 编码（GPT-3.5/GPT-4 的编码）
    encoding = tiktoken.get_encoding("cl100k_base")

    # 将文本编码为 token ID 列表
    tokens = encoding.encode(text)

    # 检查是否需要截断
    if len(tokens) <= max_tokens:
        return text  # 不需要截断

    # 截断 token 列表
    truncated_tokens = tokens[:max_tokens]

    # 将截断后的 token 解码回文本
    return encoding.decode(truncated_tokens)

# ====================================================================
# OSS_JSON_FORMAT: 定义 LLM 输出的 JSON 格式
# ====================================================================
# 说明：
#   - 此格式定义可能用于提示 LLM 返回结构化数据
#   - 包含三个字段：
#     1. rational: 定位相关部分
#     2. evidence: 提取原始内容
#     3. summary: 简洁总结
#
# 注意：
#   - 目前代码中未直接使用此变量
#   - EXTRACTOR_PROMPT (在 prompt.py) 定义了实际使用的提示词
OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""


# ====================================================================
# Visit 工具类定义
# ====================================================================
@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    name = 'visit'

    # ----------------------------------------------------------------
    # 工具描述（传递给 LLM）
    # ----------------------------------------------------------------
    # 说明：
    #   - 访问网页并返回摘要
    #   - 支持单个 URL 或多个 URL
    #   - 根据用户目标提取相关信息
    description = 'Visit webpage(s) and return the summary of the content.'

    # ----------------------------------------------------------------
    # 参数定义（JSON Schema 格式）
    # ----------------------------------------------------------------
    # 重要特性：
    #   1. url 支持两种类型：
    #      - string: 单个 URL
    #      - array: 多个 URL
    #
    #   2. goal 参数：
    #      - 用户的访问目标
    #      - 指导 LLM 提取哪些信息
    #      - 示例: "找到公司的联系方式" 或 "总结文章的主要观点"
    #
    # 示例调用：
    #   单个 URL: {"url": "https://example.com", "goal": "找到产品价格"}
    #   多个 URL: {"url": ["https://site1.com", "https://site2.com"], "goal": "比较两家公司的服务"}
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],  # 支持字符串或数组
                "items": {
                    "type": "string"
                    },
                "minItems": 1,  # 数组至少包含 1 个 URL
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
        },
        "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s)."
        }
        },
        "required": ["url", "goal"]  # 两个参数都是必需的
    }

    # ====================================================================
    # call 方法：工具的入口方法
    # ====================================================================
    # 参数：
    #   params: 包含 'url' 和 'goal' 的字典
    #   **kwargs: 额外参数
    #
    # 返回值：
    #   格式化的摘要字符串
    #
    # 执行逻辑：
    #   - 模式 1（单个 URL）: 直接调用 readpage_jina
    #   - 模式 2（批量 URL）: 遍历所有 URL，合并结果
    # ====================================================================
    def call(self, params: Union[str, dict], **kwargs) -> str:
        # ----------------------------------------------------------------
        # 步骤 1: 参数验证和解析
        # ----------------------------------------------------------------
        try:
            url = params["url"]
            goal = params["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        start_time = time.time()

        # ----------------------------------------------------------------
        # 步骤 2: 创建日志文件夹
        # ----------------------------------------------------------------
        # 注意：代码中创建了 log_folder 但未使用
        # 保留作为未来可能的日志记录功能
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)

        # ================================================================
        # 步骤 3: 根据 URL 类型选择执行模式
        # ================================================================

        # ========== 模式 1: 单个 URL ==========
        if isinstance(url, str):
            response = self.readpage_jina(url, goal)

        # ========== 模式 2: 批量 URL ==========
        else:
            response = []
            assert isinstance(url, List)
            start_time = time.time()

            # --------------------------------------------------------
            # 遍历所有 URL
            # --------------------------------------------------------
            for u in url:
                # ------------------------------------------------
                # 超时检查：最多 900 秒（15 分钟）
                # ------------------------------------------------
                # 为什么是 900 秒？
                #   - 批量 URL 访问可能很耗时
                #   - 15 分钟是合理的总时间限制
                #   - 超时后返回默认错误消息，避免无限等待
                if time.time() - start_time > 900:
                    # 超时，返回默认错误消息
                    cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    cur_response += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    cur_response += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                else:
                    try:
                        # 调用 readpage_jina 处理单个 URL
                        cur_response = self.readpage_jina(u, goal)
                    except Exception as e:
                        # 捕获异常，记录错误信息
                        cur_response = f"Error fetching {u}: {str(e)}"

                response.append(cur_response)

            # --------------------------------------------------------
            # 合并多个 URL 的结果
            # --------------------------------------------------------
            # 使用 =======  作为分隔符
            response = "\n=======\n".join(response)

        # ----------------------------------------------------------------
        # 步骤 4: 打印和返回结果
        # ----------------------------------------------------------------
        print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip()


    # ====================================================================
    # call_server 方法：调用 LLM 生成网页摘要
    # ====================================================================
    # 作用：
    #   - 将网页内容和用户目标发送给 LLM
    #   - LLM 根据 EXTRACTOR_PROMPT 提取相关信息
    #   - 返回 JSON 格式的摘要结果
    #
    # 参数：
    #   - msgs: OpenAI 格式的消息列表
    #           示例: [{"role": "user", "content": "..."}]
    #   - max_retries: 最大重试次数（默认 2）
    #
    # 返回值：
    #   - 成功: JSON 字符串（包含 evidence 和 summary）
    #   - 失败: 空字符串 ""
    #
    # 环境变量：
    #   - API_KEY: LLM API 密钥
    #   - API_BASE: LLM API 基础 URL
    #   - SUMMARY_MODEL_NAME: 用于摘要的模型名称
    # ====================================================================
    def call_server(self, msgs, max_retries=2):
        # ----------------------------------------------------------------
        # 步骤 1: 从环境变量读取配置
        # ----------------------------------------------------------------
        api_key = os.environ.get("API_KEY")
        url_llm = os.environ.get("API_BASE")
        model_name = os.environ.get("SUMMARY_MODEL_NAME", "")

        # ----------------------------------------------------------------
        # 步骤 2: 创建 OpenAI 客户端
        # ----------------------------------------------------------------
        # 注意：使用 OpenAI-compatible API
        #   - 可以是 OpenAI 官方 API
        #   - 也可以是自部署的兼容服务（如 vLLM, Text Generation Inference）
        client = OpenAI(
            api_key=api_key,
            base_url=url_llm,
        )

        # ----------------------------------------------------------------
        # 步骤 3: 带重试的 LLM 调用
        # ----------------------------------------------------------------
        # 为什么重试 2 次？
        #   - LLM API 可能偶尔超时或返回错误
        #   - 2 次足够处理大部分临时问题
        #   - 过多重试会浪费时间和资源
        for attempt in range(max_retries):
            try:
                # --------------------------------------------------------
                # 调用 LLM API
                # --------------------------------------------------------
                # temperature=0.7: 摘要任务需要一定创造性
                #   - 0.0: 过于确定性，可能生成重复内容
                #   - 0.7: 平衡准确性和多样性
                #   - 1.0+: 过于随机，可能产生不相关内容
                chat_response = client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    temperature=0.7  # 摘要任务推荐温度
                )

                # 提取 LLM 返回的内容
                content = chat_response.choices[0].message.content

                # --------------------------------------------------------
                # 处理 LLM 返回的内容
                # --------------------------------------------------------
                if content:
                    try:
                        # 尝试直接解析为 JSON
                        json.loads(content)
                    except:
                        # ------------------------------------------------
                        # JSON 提取：处理 LLM 可能返回 Markdown 格式
                        # ------------------------------------------------
                        # 问题：
                        #   - LLM 有时会在 JSON 外添加文本说明
                        #   - 示例: "这是结果：\n{"evidence": "...", "summary": "..."}"
                        #   - 或使用 Markdown 代码块: ```json\n{...}\n```
                        #
                        # 解决方案：
                        #   - 查找第一个 '{' 和最后一个 '}'
                        #   - 提取中间的 JSON 部分
                        left = content.find('{')
                        right = content.rfind('}')
                        if left != -1 and right != -1 and left <= right:
                            content = content[left:right+1]  # 提取 JSON

                    return content

            except Exception as e:
                # --------------------------------------------------------
                # 异常处理
                # --------------------------------------------------------
                # print(e)  # 注释掉，避免过多日志
                if attempt == (max_retries - 1):  # 最后一次尝试失败
                    return ""  # 返回空字符串表示失败
                continue  # 继续下一次重试


    # ====================================================================
    # jina_readpage 方法：使用 Jina Reader API 读取网页内容
    # ====================================================================
    # 作用：
    #   - 将 URL 传递给 Jina Reader 服务
    #   - Jina Reader 会：
    #     1. 爬取网页内容
    #     2. 去除广告、导航栏等无关元素
    #     3. 转换为干净的 Markdown 格式
    #
    # Jina Reader 服务：
    #   - 服务地址: https://r.jina.ai/{url}
    #   - 示例: https://r.jina.ai/https://example.com
    #   - 返回 Markdown 格式的网页内容
    #
    # 参数：
    #   - url: 要读取的网页 URL
    #
    # 返回值：
    #   - 成功: Markdown 格式的网页内容
    #   - 失败: "[visit] Failed to read page."
    #
    # 重试策略：
    #   - 最多 3 次重试
    #   - 每次失败后等待 0.5 秒
    #   - 超时时间 50 秒
    # ====================================================================
    def jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina service.

        Args:
            url: The URL to read

        Returns:
            str: The webpage content or error message
        """
        max_retries = 3  # 最多 3 次重试
        timeout = 50     # 每次请求超时 50 秒

        # ----------------------------------------------------------------
        # 重试循环
        # ----------------------------------------------------------------
        for attempt in range(max_retries):
            # --------------------------------------------------------
            # 构造请求头（包含 API 密钥）
            # --------------------------------------------------------
            headers = {
                "Authorization": f"Bearer {JINA_API_KEYS}",
            }

            try:
                # ------------------------------------------------
                # 调用 Jina Reader API
                # ------------------------------------------------
                # URL 格式: https://r.jina.ai/{原始URL}
                # 示例: https://r.jina.ai/https://example.com
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout  # 超时控制
                )

                # ------------------------------------------------
                # 检查响应状态
                # ------------------------------------------------
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content  # 成功读取
                else:
                    # 非 200 状态码，打印错误信息
                    print(response.text)
                    raise ValueError("jina readpage error")

            except Exception as e:
                # ------------------------------------------------
                # 异常处理和重试
                # ------------------------------------------------
                time.sleep(0.5)  # 等待 0.5 秒后重试

                # 最后一次尝试失败，返回错误消息
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."

        # 所有尝试都失败
        return "[visit] Failed to read page."

    # ====================================================================
    # html_readpage_jina 方法：带多次重试的网页读取
    # ====================================================================
    # 作用：
    #   - 调用 jina_readpage 读取网页
    #   - 最多尝试 8 次
    #   - 过滤无效响应（错误消息、空内容等）
    #
    # 参数：
    #   - url: 要读取的网页 URL
    #
    # 返回值：
    #   - 成功: 有效的网页内容
    #   - 失败: "[visit] Failed to read page."
    #
    # 为什么需要 8 次尝试？
    #   - jina_readpage 内部已有 3 次重试
    #   - html_readpage_jina 再重试 8 次
    #   - 总共最多 3 * 8 = 24 次底层请求
    #   - 这是为了处理 Jina 服务偶尔返回空内容或错误的情况
    # ====================================================================
    def html_readpage_jina(self, url: str) -> str:
        max_attempts = 8  # 最多 8 次尝试

        for attempt in range(max_attempts):
            # --------------------------------------------------------
            # 调用 jina_readpage 读取网页
            # --------------------------------------------------------
            content = self.jina_readpage(url)
            service = "jina"
            print(service)

            # --------------------------------------------------------
            # 验证返回内容是否有效
            # --------------------------------------------------------
            # 无效内容包括：
            #   1. "[visit] Failed to read page.": 读取失败
            #   2. "[visit] Empty content.": 内容为空
            #   3. "[document_parser]": 文档解析错误
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                return content  # 返回有效内容

        # 所有尝试都失败或返回无效内容
        return "[visit] Failed to read page."

    # ====================================================================
    # readpage_jina 方法：核心网页摘要生成流程
    # ====================================================================
    # 作用：
    #   - 读取网页内容
    #   - 截断到 95000 tokens
    #   - 使用 LLM 根据用户目标提取相关信息
    #   - 处理各种失败情况（网页读取失败、摘要生成失败、JSON 解析失败）
    #
    # 参数：
    #   - url: 要读取的网页 URL
    #   - goal: 用户访问目标（指导 LLM 提取哪些信息）
    #
    # 返回值：
    #   格式化的摘要字符串，包含：
    #     - Evidence: 原始网页中的相关内容
    #     - Summary: 简洁总结
    #
    # 多层重试机制：
    #   1. 网页读取重试（html_readpage_jina: 最多 8 次）
    #   2. 摘要生成重试（summary_retries: 最多 3 次，逐步截断内容）
    #   3. JSON 解析重试（parse_retry_times: 最多 3 次）
    # ====================================================================
    def readpage_jina(self, url: str, goal: str) -> str:
        """
        Attempt to read webpage content and generate summary.

        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page

        Returns:
            str: The formatted summary or error message
        """

        # ----------------------------------------------------------------
        # 配置
        # ----------------------------------------------------------------
        summary_page_func = self.call_server  # LLM 调用函数
        max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 1))  # LLM 调用重试次数

        # ================================================================
        # 步骤 1: 读取网页内容
        # ================================================================
        content = self.html_readpage_jina(url)

        # ================================================================
        # 步骤 2: 验证网页内容是否有效
        # ================================================================
        if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
            # ============================================================
            # 步骤 3: 截断内容到 95000 tokens
            # ============================================================
            # 原因：
            #   - 避免超过 LLM 上下文限制
            #   - 95000 tokens 留出空间给提示词和输出
            content = truncate_to_tokens(content, max_tokens=95000)

            # ============================================================
            # 步骤 4: 构造 LLM 提示
            # ============================================================
            # EXTRACTOR_PROMPT: 在 prompt.py 中定义
            # 包含：
            #   - 网页内容
            #   - 用户目标
            #   - 输出格式要求（JSON 格式，包含 evidence 和 summary）
            messages = [{"role":"user","content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]

            # ============================================================
            # 步骤 5: 调用 LLM 生成摘要
            # ============================================================
            parse_retry_times = 0
            raw = summary_page_func(messages, max_retries=max_retries)

            # ============================================================
            # 步骤 6: 摘要生成重试机制（如果返回内容太短）
            # ============================================================
            # 问题：
            #   - LLM 有时会返回很短或空的摘要
            #   - 可能是因为内容太长，LLM 无法有效处理
            #
            # 解决方案：
            #   - 逐步截断内容长度
            #   - 重新生成摘要
            #
            # 重试策略：
            #   - 第 1 次重试: 截断到 70% 长度（0.7 * len(content)）
            #   - 第 2 次重试: 截断到 70% 长度（0.7 * 0.7 * len(content) = 49%）
            #   - 第 3 次重试: 截断到 70% 长度（0.7 * 0.7 * 0.7 * len(content) = 34%）
            #   - 第 4 次尝试: 截断到 25000 字符（最后的保底尝试）
            summary_retries = 3

            while len(raw) < 10 and summary_retries >= 0:
                # --------------------------------------------------------
                # 计算截断长度
                # --------------------------------------------------------
                # summary_retries > 0: 截断到 70% 长度
                # summary_retries == 0: 最后尝试，截断到 25000 字符
                truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000

                # 打印状态信息
                status_msg = (
                    f"[visit] Summary url[{url}] "
                    f"attempt {3 - summary_retries + 1}/3, "
                    f"content length: {len(content)}, "
                    f"truncating to {truncate_length} chars"
                ) if summary_retries > 0 else (
                    f"[visit] Summary url[{url}] failed after 3 attempts, "
                    f"final truncation to 25000 chars"
                )
                print(status_msg)

                # --------------------------------------------------------
                # 截断内容并重新生成摘要
                # --------------------------------------------------------
                content = content[:truncate_length]
                extraction_prompt = EXTRACTOR_PROMPT.format(
                    webpage_content=content,
                    goal=goal
                )
                messages = [{"role": "user", "content": extraction_prompt}]
                raw = summary_page_func(messages, max_retries=max_retries)
                summary_retries -= 1

            # ============================================================
            # 步骤 7: 解析 LLM 返回的 JSON
            # ============================================================
            parse_retry_times = 0

            # --------------------------------------------------------
            # 预处理：去除 Markdown 代码块标记
            # --------------------------------------------------------
            # LLM 可能返回: ```json\n{...}\n```
            # 需要去除 ```json 和 ```
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()

            # --------------------------------------------------------
            # JSON 解析重试循环（最多 3 次）
            # --------------------------------------------------------
            while parse_retry_times < 3:
                try:
                    raw = json.loads(raw)  # 尝试解析 JSON
                    break  # 成功则跳出循环
                except:
                    # JSON 解析失败，重新调用 LLM
                    raw = summary_page_func(messages, max_retries=max_retries)
                    parse_retry_times += 1

            # ============================================================
            # 步骤 8: 格式化输出结果
            # ============================================================

            # --------------------------------------------------------
            # 情况 1: JSON 解析失败（重试 3 次仍失败）
            # --------------------------------------------------------
            if parse_retry_times >= 3:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"

            # --------------------------------------------------------
            # 情况 2: JSON 解析成功
            # --------------------------------------------------------
            else:
                # 格式：
                #   The useful information in {url} for user goal {goal} as follows:
                #
                #   Evidence in page:
                #   [LLM 提取的原始内容]
                #
                #   Summary:
                #   [LLM 生成的简洁总结]
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

            # --------------------------------------------------------
            # 最终验证：检查摘要长度
            # --------------------------------------------------------
            # 如果摘要太短（<10 字符）且所有重试都用完了
            # 返回失败消息
            if len(useful_information) < 10 and summary_retries < 0:
                print("[visit] Could not generate valid summary after maximum retries")
                useful_information = "[visit] Failed to read page"

            return useful_information

        # ================================================================
        # 步骤 9: 网页读取失败的处理
        # ================================================================
        # 如果 html_readpage_jina 返回无效内容
        # 返回默认错误消息
        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            return useful_information

