# ============================================================================
# Google 网络搜索工具
# ============================================================================
#
# 功能说明：
#   本模块实现了基于 Google 的通用网络搜索功能，用于检索网页、新闻、博客等内容
#
# 核心特性：
#   - 使用 Serper API 访问 Google Search
#   - 支持批量查询（一次调用搜索多个关键词）
#   - 自动中英文检测，配置对应的搜索地区和语言
#   - 返回前 10 个搜索结果，包含标题、链接、摘要、发布日期等
#   - 自动重试机制（最多 5 次，处理网络波动）
#
# 与 Scholar 工具的区别：
#   - Search: 通用网页搜索（本模块）
#   - Scholar: 学术论文搜索（tool_scholar.py）
#
# 典型应用场景：
#   - 查找最新新闻和时事信息
#   - 搜索产品评测和用户体验
#   - 获取公司官网和联系方式
#   - 查找技术博客和教程
#
# ============================================================================

import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
import asyncio
from typing import Dict, List, Optional, Union
import uuid
import http.client
import json

import os

# ====================================================================
# 环境变量配置
# ====================================================================
# SERPER_KEY_ID: 从 https://serper.dev/ 获取的 API 密钥
# 用于访问 Google Search 服务
SERPER_KEY=os.environ.get('SERPER_KEY_ID')


# ====================================================================
# Search 工具类定义
# ====================================================================
# 注册为 qwen_agent 工具，allow_overwrite=True 允许覆盖同名工具
@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"

    # ----------------------------------------------------------------
    # 工具描述（会传递给 LLM，指导模型何时使用此工具）
    # ----------------------------------------------------------------
    # 强调 "batched" 特性：鼓励 LLM 一次传入多个相关查询
    # 强调 "top 10 results"：每个查询返回前 10 个结果
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."

    # ----------------------------------------------------------------
    # 参数定义（JSON Schema 格式）
    # ----------------------------------------------------------------
    # 说明：
    #   - query: 必须是数组类型（即使只有 1 个查询也要用数组）
    #   - 支持批量查询，提高效率
    #
    # 示例：
    #   单个查询: {"query": ["OpenAI GPT-4 发布时间"]}
    #   批量查询: {"query": ["OpenAI GPT-4 发布时间", "GPT-4 技术细节", "GPT-4 定价"]}
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",  # 必须是数组
                "items": {
                    "type": "string"  # 数组元素是字符串
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],  # query 是必需参数
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
    # ====================================================================
    # 核心方法：使用 Serper API 执行单次 Google 网络搜索
    # ====================================================================
    # 参数：
    #   query: 单个搜索关键词字符串
    #
    # 返回值：
    #   格式化的搜索结果字符串，包含标题、链接、摘要等信息
    #
    # 工作流程：
    #   1. 检测查询语言（中文 vs 英文）
    #   2. 根据语言配置搜索地区和语言参数
    #   3. 建立 HTTPS 连接到 Serper API
    #   4. 带重试机制地发送请求（最多 5 次）
    #   5. 解析返回的 JSON 结果
    #   6. 格式化为 Markdown 格式返回
    # ====================================================================
    def google_search_with_serp(self, query: str):
        # ----------------------------------------------------------------
        # 辅助函数：检测文本是否包含中文字符
        # ----------------------------------------------------------------
        # 原理：Unicode 中文字符范围是 U+4E00 到 U+9FFF
        # 只要有一个字符在此范围内，就判断为包含中文
        def contains_chinese_basic(text: str) -> bool:
            return any('\u4E00' <= char <= '\u9FFF' for char in text)

        # ----------------------------------------------------------------
        # 步骤 1: 建立 HTTPS 连接
        # ----------------------------------------------------------------
        conn = http.client.HTTPSConnection("google.serper.dev")

        # ----------------------------------------------------------------
        # 步骤 2: 根据查询语言配置搜索参数
        # ----------------------------------------------------------------
        # 为什么需要区分中英文？
        #   - 不同语言的用户希望看到本地化的搜索结果
        #   - 例如搜索 "苹果"，中文用户希望看到中国的新闻和网站
        #   - 而搜索 "apple"，英文用户希望看到美国的结果
        #
        # 参数说明：
        #   - location: 搜索地理位置（影响结果排序和本地化内容）
        #   - gl: Google 域名地区代码（cn=google.cn, us=google.com）
        #   - hl: 搜索结果语言（zh-cn=简体中文, en=英文）

        if contains_chinese_basic(query):
            # ========== 中文查询配置 ==========
            payload = json.dumps({
                "q": query,
                "location": "China",      # 搜索地点：中国
                "gl": "cn",              # 使用 google.cn
                "hl": "zh-cn"            # 返回简体中文结果
            })

        else:
            # ========== 英文查询配置 ==========
            payload = json.dumps({
                "q": query,
                "location": "United States",  # 搜索地点：美国
                "gl": "us",                   # 使用 google.com
                "hl": "en"                    # 返回英文结果
            })

        headers = {
                'X-API-KEY': SERPER_KEY,  # API 密钥
                'Content-Type': 'application/json'
            }


        # ----------------------------------------------------------------
        # 步骤 3: 带重试的请求发送（最多 5 次）
        # ----------------------------------------------------------------
        # 为什么重试 5 次？
        #   - Serper API 偶尔会出现超时或临时故障
        #   - 5 次足够处理大部分网络波动问题
        #   - 避免因偶发错误导致整个 Agent 失败
        for i in range(5):
            try:
                conn.request("POST", "/search", payload, headers)  # 注意端点是 /search（不是 /scholar）
                res = conn.getresponse()
                break  # 成功则跳出循环
            except Exception as e:
                print(e)
                if i == 4:  # 第 5 次（索引 4）还失败，则返回超时信息
                    return f"Google search Timeout, return None, Please try again later."
                continue  # 继续下一次重试

        # ----------------------------------------------------------------
        # 步骤 4: 解析响应数据
        # ----------------------------------------------------------------
        data = res.read()
        results = json.loads(data.decode("utf-8"))

        # ----------------------------------------------------------------
        # 步骤 5: 提取和格式化搜索结果
        # ----------------------------------------------------------------
        try:
            # 检查是否有有效结果
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0

            # --------------------------------------------------------
            # 遍历 organic 结果（Google 返回的主要搜索结果列表）
            # --------------------------------------------------------
            # organic 结果不包括广告、知识图谱等特殊结果
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1

                    # ------------------------------------------------
                    # 提取网页元数据字段
                    # ------------------------------------------------

                    # date: 内容发布日期（如果可用）
                    # 示例: "2023-10-15" 或 "3 days ago"
                    # 注意：不是所有网页都有发布日期
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    # source: 网站来源
                    # 示例: "Wikipedia" 或 "BBC News"
                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    # snippet: 网页摘要/片段
                    # Google 自动提取的与查询最相关的内容片段
                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    # ------------------------------------------------
                    # 格式化为 Markdown 格式
                    # ------------------------------------------------
                    # 格式: 序号. [标题](链接)
                    #       发布日期
                    #       来源
                    #       摘要片段
                    #
                    # 示例:
                    #   1. [OpenAI 发布 GPT-4](https://example.com)
                    #   Date published: 2023-03-14
                    #   Source: OpenAI Blog
                    #   GPT-4 是 OpenAI 最新的大语言模型...
                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"

                    # 移除无关内容（某些搜索结果可能包含视频提示）
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

            # --------------------------------------------------------
            # 组装最终返回内容
            # --------------------------------------------------------
            # 格式:
            #   A Google search for '量子计算' found 10 results:
            #
            #   ## Web Results
            #   1. [网页标题](链接)
            #      Date published: 2023-10-15
            #      Source: Wikipedia
            #      网页摘要...
            #
            #   2. [另一个网页](链接)
            #      ...
            content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            return content
        except:
            return f"No results found for '{query}'. Try with a more general query."



    # ====================================================================
    # search_with_serp 方法：包装 google_search_with_serp
    # ====================================================================
    # 这是一个简单的包装方法，未来可以在这里添加额外的逻辑
    # （如缓存、日志记录等）
    def search_with_serp(self, query: str):
        result = self.google_search_with_serp(query)
        return result

    # ====================================================================
    # call 方法：工具的入口方法（由 qwen_agent 框架调用）
    # ====================================================================
    # 参数：
    #   params: LLM 传递的参数（JSON 字符串或字典）
    #           格式: {"query": ["关键词1", "关键词2", ...]}
    #   **kwargs: 额外参数（通常不使用）
    #
    # 返回值：
    #   格式化的搜索结果字符串
    #
    # 执行逻辑：
    #   - 模式 1（单个查询）: 直接调用 search_with_serp
    #   - 模式 2（批量查询）: 顺序执行多个搜索并合并结果
    # ====================================================================
    def call(self, params: Union[str, dict], **kwargs) -> str:
        # ------------------------------------------------------------
        # 步骤 1: 参数验证和解析
        # ------------------------------------------------------------
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        # ------------------------------------------------------------
        # 步骤 2: 根据查询类型选择执行模式
        # ------------------------------------------------------------

        # ========== 模式 1: 单个查询 ==========
        if isinstance(query, str):
            response = self.search_with_serp(query)

        # ========== 模式 2: 批量查询 ==========
        else:
            assert isinstance(query, List)

            # ------------------------------------------------
            # 顺序执行多个搜索
            # ------------------------------------------------
            # 注意：这里使用顺序执行（for 循环），而不是并发执行
            # 原因：
            #   - 每个搜索请求本身已经有重试机制
            #   - 顺序执行更简单、更稳定
            #   - 与 Scholar 工具不同（Scholar 使用 ThreadPoolExecutor）
            #
            # 对比：
            #   - tool_scholar.py 使用 ThreadPoolExecutor(max_workers=3)
            #   - tool_search.py 使用顺序执行
            #   - 两者都能工作，选择取决于 API 稳定性和性能需求
            responses = []
            for q in query:
                responses.append(self.search_with_serp(q))

            # ------------------------------------------------
            # 合并多个搜索结果
            # ------------------------------------------------
            # 使用 =======  作为分隔符
            # 示例输出：
            #   A Google search for 'GPT-4' found 10 results:
            #   [结果1]
            #   =======
            #   A Google search for 'Claude' found 8 results:
            #   [结果2]
            response = "\n=======\n".join(responses)

        return response

