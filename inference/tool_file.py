# ============================================================================
# 文件解析工具
# ============================================================================
#
# 功能说明：
#   本模块实现了多种文件格式的解析功能，支持文档、表格、媒体等 15+ 种格式
#
# 核心特性：
#   - 支持文档格式: PDF, DOCX, PPTX, TXT, HTML, DOC
#   - 支持表格格式: CSV, TSV, XLSX, XLS
#   - 支持压缩格式: ZIP
#   - 支持视频格式: MP4, MOV, AVI, MKV, WEBM
#   - 支持音频格式: MP3, WAV, AAC, OGG, FLAC
#   - 自动路径解析（相对路径 → 绝对路径）
#   - 支持 HTTP/HTTPS URL 的文件
#   - 异步处理（使用 async/await）
#   - 内容压缩（如果超过 token 限制）
#
# 工作原理：
#   1. 解析文件路径（处理相对路径、URL）
#   2. 调用 SingleFileParser 解析文件内容
#   3. 检查 token 数，如果超过限制则压缩
#   4. 返回格式化的文件内容
#
# 输入格式：
#   - files: 文件路径列表（可以是本地路径或 URL）
#   示例: {"files": ["report.pdf", "https://example.com/data.xlsx"]}
#
# 输出格式：
#   - 格式化的文件内容字符串列表
#   示例: ["# File: report.pdf\n内容...", "# File: data.xlsx\n内容..."]
#
# ============================================================================
"""
输入:
    - query/goal: str (用户目标，通常不直接传递给此工具)
    - Docs: List[file]/List[url] (文件路径或 URL 列表)
    - file type: 'pdf', 'docx', 'pptx', 'txt', 'html', 'csv', 'tsv', 'xlsx', 'xls', 'doc', 'zip', '.mp4', '.mov', '.avi', '.mkv', '.webm', '.mp3', '.wav', '.aac', '.ogg', '.flac'
输出:
    - answer: str (解析后的文件内容)
    - useful_information: str (有用信息摘要)
"""
import sys
import os
import re
import time
import copy
import json
from typing import Dict, Iterator, List, Literal, Tuple, Union, Any, Optional
import json5
import asyncio
from openai import OpenAI, AsyncOpenAI
import pdb
import bdb

# ====================================================================
# qwen_agent 框架导入
# ====================================================================
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.agents import Assistant
from qwen_agent.llm import BaseChatModel
from qwen_agent.settings import DEFAULT_WORKSPACE, DEFAULT_MAX_INPUT_TOKENS
from qwen_agent.llm.schema import ASSISTANT, USER, FUNCTION, Message, DEFAULT_SYSTEM_MESSAGE, SYSTEM, ROLE
from qwen_agent.tools import BaseTool
from qwen_agent.log import logger
from qwen_agent.utils.tokenization_qwen import count_tokens, tokenizer
from qwen_agent.settings import DEFAULT_WORKSPACE, DEFAULT_MAX_INPUT_TOKENS

# ====================================================================
# 路径配置：添加 file_tools 模块到 Python 路径
# ====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append('../../')

# ====================================================================
# 导入文件解析器
# ====================================================================
# SingleFileParser: 核心解析器，支持 15+ 种文件格式
# compress: 内容压缩工具，当文件内容超过 token 限制时使用
# VideoAgent: 专门处理 MP3 等音频文件的 Agent
from file_tools.file_parser import SingleFileParser, compress
from file_tools.video_agent import VideoAgent

# ====================================================================
# 文件摘要生成提示词
# ====================================================================
# 说明：
#   当文件内容被解析后，可以使用 LLM 根据用户目标提取关键信息
#   （目前此提示词在代码中未使用，保留作为未来扩展）
#
# 输出格式：
#   1. Rational: 定位与用户目标相关的文件部分
#   2. Evidence: 提取最相关的原始内容（可能很长）
#   3. Summary: 简洁总结，评估信息对目标的贡献
FILE_SUMMARY_PROMPT = """
Please process the following file content and user goal to extract relevant information:

## **File Content**
{file_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the file content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.
""".strip()


# ============================================================================
# file_parser 异步函数：解析多个文件的核心逻辑
# ============================================================================
# 参数：
#   params: 包含 'files' 字段的字典，值为文件路径列表
#   **kwargs: 额外参数（传递给 SingleFileParser）
#
# 返回值：
#   文件内容列表（如果总 token 数超过限制，返回压缩后的内容）
#
# 工作流程：
#   1. 解析文件路径（处理相对路径、绝对路径、URL）
#   2. 遍历所有文件，调用 SingleFileParser 解析
#   3. 检查总 token 数，如果超过 DEFAULT_MAX_INPUT_TOKENS 则压缩
#   4. 返回解析结果
# ============================================================================
async def file_parser(params, **kwargs):
    """Parse files with automatic path resolution"""

    # ====================================================================
    # 步骤 1: 获取并规范化文件路径列表
    # ====================================================================
    urls = params.get('files', [])
    if isinstance(urls, str):
        urls = [urls]  # 如果传入单个字符串，转换为列表

    # ====================================================================
    # 步骤 2: 解析文件路径（相对路径 → 绝对路径）
    # ====================================================================
    # 路径类型处理：
    #   - HTTP/HTTPS URL: 直接使用（会传递给 IDP 服务解析）
    #   - 相对路径: 转换为绝对路径
    #   - 绝对路径: 直接使用
    #
    # 嵌套列表处理：
    #   某些情况下 urls 可能是嵌套列表 [["file1.pdf"], ["file2.docx"]]
    #   需要展平处理
    resolved_urls = []
    for url in urls:
        if isinstance(url, list):
            # 处理嵌套列表
            for sub_url in url:
                if sub_url.startswith(("http://", "https://")):
                    resolved_urls.append(sub_url)  # HTTP/HTTPS URL
                else:
                    # 相对路径或绝对路径
                    abs_path = os.path.abspath(sub_url)
                    if os.path.exists(abs_path):
                        resolved_urls.append(abs_path)  # 存在的本地文件
                    else:
                        resolved_urls.append(sub_url)  # 可能是不存在的路径，保留原值
        else:
            if url.startswith(("http://", "https://")):
                resolved_urls.append(url)  # HTTP/HTTPS URL
            else:
                # 相对路径或绝对路径
                abs_path = os.path.abspath(url)
                if os.path.exists(abs_path):
                    resolved_urls.append(abs_path)  # 存在的本地文件
                else:
                    resolved_urls.append(url)  # 可能是不存在的路径，保留原值

    # ====================================================================
    # 步骤 3: 遍历文件并解析
    # ====================================================================
    results = []  # 格式化的结果（包含 "# File: xxx" 标题）
    file_results = []  # 纯文件内容（用于可能的压缩）

    for url in resolved_urls:
        try:
            # ------------------------------------------------------------
            # 调用 SingleFileParser 解析文件
            # ------------------------------------------------------------
            # SingleFileParser 支持：
            #   - 本地文件路径
            #   - HTTP/HTTPS URL
            #   - 15+ 种文件格式
            result = SingleFileParser().call(json.dumps({'url': url}), **kwargs)
            results.append(f"# File: {os.path.basename(url)}\n{result}")
            file_results.append(result)
        except Exception as e:
            # 解析失败时添加错误信息
            results.append(f"# Error processing {os.path.basename(url)}: {str(e)}")

    # ====================================================================
    # 步骤 4: Token 数检查和内容压缩
    # ====================================================================
    # DEFAULT_MAX_INPUT_TOKENS: Qwen 模型的默认最大输入 token 数
    # 如果所有文件的总 token 数超过限制，调用 compress() 压缩
    #
    # compress() 的作用：
    #   - 使用 LLM 提取关键信息
    #   - 去除冗余内容
    #   - 保留最重要的部分
    if count_tokens(json.dumps(results)) < DEFAULT_MAX_INPUT_TOKENS:
        return results  # 不需要压缩
    else:
        return compress(file_results)  # 压缩后返回

# ====================================================================
# FileParser 工具类
# ====================================================================
# 说明：
#   - 未注册为工具（@register_tool 被注释）
#   - 实际在 DeepResearch 系统中，文件解析通过其他方式集成
#   - 此类保留作为参考实现
#
# 与 file_parser() 函数的关系：
#   - file_parser(): 纯函数实现，更灵活
#   - FileParser: 类封装，符合 qwen_agent 工具规范
# ====================================================================
# @register_tool("file_parser")
class FileParser(BaseTool):
    name = "parse_file"

    # ----------------------------------------------------------------
    # 工具描述
    # ----------------------------------------------------------------
    # 说明：支持 15+ 种文件格式
    # 包括文档（PDF, DOCX）、表格（XLSX, CSV）、媒体（MP4, MP3）等
    description = "This is a tool that can be used to parse multiple user uploaded local files such as PDF, DOCX, PPTX, TXT, CSV, XLSX, DOC, ZIP, MP4, MP3."

    # ----------------------------------------------------------------
    # 参数定义
    # ----------------------------------------------------------------
    # files: 文件名列表（数组类型）
    # array_type: 'string' 表示数组元素是字符串
    parameters = [
        {
            'name': 'files',
            'type': 'array',
            'array_type': 'string',  # 数组元素类型
            'description': 'The file name of the user uploaded local files to be parsed.',
            'required': True
        }
    ]

    # ====================================================================
    # call 方法：异步执行文件解析
    # ====================================================================
    # 参数：
    #   params: 包含 'files' 字段的字典
    #   file_root_path: 文件根目录路径
    #
    # 返回值：
    #   解析结果列表
    #
    # 工作流程：
    #   1. 分离 MP3 文件和其他文件（MP3 使用 VideoAgent 处理）
    #   2. 使用 file_parser() 处理普通文件
    #   3. 使用 VideoAgent 处理 MP3 文件
    #   4. 合并结果并返回
    # ====================================================================
    async def call(self, params, file_root_path):
        file_name = params["files"]
        outputs = []

        # ----------------------------------------------------------------
        # 步骤 1: 分离文件类型
        # ----------------------------------------------------------------
        # 为什么 MP3 需要单独处理？
        #   - MP3 是音频文件，需要语音识别（ASR）
        #   - VideoAgent 集成了语音转文字功能
        #   - 其他文件使用标准 IDP（Intelligent Document Processing）
        file_path = []  # 非 MP3 文件
        omnifile_path = []  # MP3 文件
        for f_name in file_name:
            if '.mp3' not in f_name:
                file_path.append(os.path.join(file_root_path, f_name))
            else:
                omnifile_path.append(os.path.join(file_root_path, f_name))

        # ----------------------------------------------------------------
        # 步骤 2: 处理普通文件（PDF, DOCX, XLSX 等）
        # ----------------------------------------------------------------
        if len(file_path):
            params = {'files': file_path}
            response = await file_parser(params)  # 调用异步 file_parser 函数

            # --------------------------------------------------------
            # 内容截断：[:30000]
            # --------------------------------------------------------
            # 为什么截断到 30000？
            #   - 防止单个工具调用返回过多内容
            #   - 30000 个字符约等于 7500 个 token（中文）
            #   - 保持结果在合理长度范围内
            response = response[:30000]

            parsed_file_content = ' '.join(response)
            # 添加 token 数提示和文件内容
            outputs.extend([f'File token number: {len(parsed_file_content.split())}\nFile content:\n']+response)


        # ----------------------------------------------------------------
        # 步骤 3: 处理 MP3 音频文件
        # ----------------------------------------------------------------
        if len(omnifile_path):
            params['files'] = omnifile_path
            agent = VideoAgent()  # 创建 VideoAgent 实例
            res = await agent.call(params)  # 异步调用 VideoAgent

            # VideoAgent 返回 JSON 字符串，需要解析
            res = json.loads(res)
            outputs += res

        return outputs
