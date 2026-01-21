# ============================================================================
# Google Scholar 学术搜索工具
# ============================================================================
#
# 功能说明：
#   本模块实现了基于 Google Scholar 的学术文献搜索功能，用于检索论文、期刊等学术资源
#
# 核心特性：
#   - 使用 Serper API 访问 Google Scholar
#   - 支持批量查询（传入多个搜索关键词）
#   - 提供丰富的文献元数据（标题、PDF链接、引用数、发表年份等）
#   - 自动重试机制（最多 5 次，处理网络波动）
#
# 与 Search 工具的区别：
#   - Search: 通用网页搜索，返回新闻、博客、网站等
#   - Scholar: 学术搜索，返回论文、期刊文章、学术出版物
#
# ============================================================================

import os
import json
import requests
from typing import Union, List
from qwen_agent.tools.base import BaseTool, register_tool
from concurrent.futures import ThreadPoolExecutor
import http.client

# ====================================================================
# 环境变量配置
# ====================================================================
# SERPER_KEY_ID: 从 https://serper.dev/ 获取的 API 密钥
# 用于访问 Google Scholar 搜索服务
SERPER_KEY=os.environ.get('SERPER_KEY_ID')


# ====================================================================
# Scholar 工具类定义
# ====================================================================
# 注册为 qwen_agent 工具，allow_overwrite=True 允许覆盖同名工具
@register_tool("google_scholar", allow_overwrite=True)
class Scholar(BaseTool):
    name = "google_scholar"

    # ----------------------------------------------------------------
    # 工具描述（会传递给 LLM，指导模型何时使用此工具）
    # ----------------------------------------------------------------
    description = "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries."

    # ----------------------------------------------------------------
    # 参数定义（JSON Schema 格式）
    # ----------------------------------------------------------------
    # 说明：
    #   - query: 支持数组类型，可一次传入多个搜索关键词
    #   - minItems: 1 表示至少需要 1 个查询关键词
    #   - 批量查询特性：可以在一次工具调用中搜索多个不同的学术主题
    #
    # 示例：
    #   单个查询: {"query": ["量子计算"]}
    #   批量查询: {"query": ["量子计算", "量子纠缠", "量子退相干"]}
    parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string", "description": "The search query."},
                    "minItems": 1,  # 至少需要 1 个查询
                    "description": "The list of search queries for Google Scholar."
                },
            },
        "required": ["query"],  # query 是必需参数
    }

    # ====================================================================
    # 核心方法：使用 Serper API 执行单次 Google Scholar 搜索
    # ====================================================================
    # 参数：
    #   query: 单个搜索关键词字符串
    #
    # 返回值：
    #   格式化的搜索结果字符串，包含标题、PDF链接、引用数等信息
    #
    # 工作流程：
    #   1. 建立 HTTPS 连接到 Serper API
    #   2. 构造请求 payload (只包含查询关键词)
    #   3. 带重试机制地发送请求（最多 5 次）
    #   4. 解析返回的 JSON 结果
    #   5. 格式化为 Markdown 格式返回
    # ====================================================================
    def google_scholar_with_serp(self, query: str):
        # ------------------------------------------------------------
        # 步骤 1: 建立 HTTPS 连接
        # ------------------------------------------------------------
        conn = http.client.HTTPSConnection("google.serper.dev")

        # ------------------------------------------------------------
        # 步骤 2: 构造请求 payload
        # ------------------------------------------------------------
        # 注意：学术搜索不需要 location/gl/hl 参数
        # （与 tool_search.py 不同，Scholar 搜索不区分地区）
        payload = json.dumps({
        "q": query,  # 搜索关键词
        })
        headers = {
        'X-API-KEY': SERPER_KEY,  # API 密钥
        'Content-Type': 'application/json'
        }

        # ------------------------------------------------------------
        # 步骤 3: 带重试的请求发送（最多 5 次）
        # ------------------------------------------------------------
        # 为什么重试 5 次？
        #   - Serper API 偶尔会出现超时或临时故障
        #   - 5 次足够处理大部分网络波动问题
        #   - 避免因偶发错误导致整个 Agent 失败
        for i in range(5):
            try:
                conn.request("POST", "/scholar", payload, headers)  # 注意端点是 /scholar
                res = conn.getresponse()
                break  # 成功则跳出循环
            except Exception as e:
                print(e)
                if i == 4:  # 第 5 次（索引 4）还失败，则返回超时信息
                    return f"Google Scholar Timeout, return None, Please try again later."
                continue  # 继续下一次重试


        # ------------------------------------------------------------
        # 步骤 4: 解析响应数据
        # ------------------------------------------------------------
        data = res.read()

        results = json.loads(data.decode("utf-8"))
        # ------------------------------------------------------------
        # 步骤 5: 提取和格式化学术搜索结果
        # ------------------------------------------------------------
        try:
            # 检查是否有有效结果
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0

            # --------------------------------------------------------
            # 遍历 organic 结果（Google Scholar 返回的主要结果列表）
            # --------------------------------------------------------
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1

                    # ------------------------------------------------
                    # 提取学术特有的字段
                    # ------------------------------------------------

                    # year: 论文发表年份（如 "2023"）
                    date_published = ""
                    if "year" in page:
                        date_published = "\nDate published: " + str(page["year"])

                    # publicationInfo: 期刊/会议信息
                    # 示例: "Nature, 2023" 或 "NeurIPS 2023"
                    publicationInfo = ""
                    if "publicationInfo" in page:
                        publicationInfo = "\npublicationInfo: " + page["publicationInfo"]

                    # snippet: 论文摘要或片段
                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    # pdfUrl: PDF 下载链接（如果可用）
                    # 注意：不是所有论文都有免费 PDF
                    link_info = "no available link"
                    if "pdfUrl" in page:
                        link_info = "pdfUrl: " + page["pdfUrl"]

                    # citedBy: 引用次数（衡量论文影响力的重要指标）
                    # 示例: "citedBy: 1234" 表示被引用 1234 次
                    citedBy = ""
                    if "citedBy" in page:
                        citedBy = "\ncitedBy: " + str(page["citedBy"])

                    # ------------------------------------------------
                    # 格式化为 Markdown 格式
                    # ------------------------------------------------
                    # 格式: 序号. [标题](链接)
                    #       发表信息
                    #       发表年份
                    #       引用次数
                    #       摘要片段
                    redacted_version = f"{idx}. [{page['title']}]({link_info}){publicationInfo}{date_published}{citedBy}\n{snippet}"

                    # 移除无关内容（某些搜索结果可能包含视频提示）
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

            # --------------------------------------------------------
            # 组装最终返回内容
            # --------------------------------------------------------
            # 格式:
            #   A Google scholar for '量子计算' found 10 results:
            #
            #   ## Scholar Results
            #   1. [论文标题](PDF链接)
            #      publicationInfo: Nature, 2023
            #      Date published: 2023
            #      citedBy: 1234
            #      论文摘要...
            #
            #   2. [另一篇论文](链接)
            #      ...
            content = f"A Google scholar for '{query}' found {len(web_snippets)} results:\n\n## Scholar Results\n" + "\n\n".join(web_snippets)
            return content
        except:
            return f"No results found for '{query}'. Try with a more general query."


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
    #   - 模式 1（单个查询）: 直接调用 google_scholar_with_serp
    #   - 模式 2（批量查询）: 使用线程池并发执行多个搜索
    # ====================================================================
    def call(self, params: Union[str, dict], **kwargs) -> str:
        # ------------------------------------------------------------
        # 步骤 1: 参数验证和解析
        # ------------------------------------------------------------
        try:
            params = self._verify_json_format_args(params)  # 验证 JSON 格式
            query = params["query"]
        except:
            return "[google_scholar] Invalid request format: Input must be a JSON object containing 'query' field"

        # ------------------------------------------------------------
        # 步骤 2: 根据查询类型选择执行模式
        # ------------------------------------------------------------

        # ========== 模式 1: 单个查询 ==========
        if isinstance(query, str):
            response = self.google_scholar_with_serp(query)

        # ========== 模式 2: 批量查询 ==========
        else:
            assert isinstance(query, List)

            # ------------------------------------------------
            # 使用线程池并发执行多个搜索
            # ------------------------------------------------
            # max_workers=3: 最多 3 个并发线程
            #
            # 为什么是 3 而不是更多？
            #   - Serper API 有速率限制
            #   - 3 个并发足够加速批量查询，同时避免触发 API 限流
            #   - 过多并发可能导致请求被拒绝
            #
            # executor.map() 的作用：
            #   - 自动将 query 列表中的每个元素传递给 google_scholar_with_serp
            #   - 并发执行所有搜索
            #   - 按原始顺序返回结果
            with ThreadPoolExecutor(max_workers=3) as executor:
                response = list(executor.map(self.google_scholar_with_serp, query))

            # ------------------------------------------------
            # 合并多个搜索结果
            # ------------------------------------------------
            # 使用 =======  作为分隔符
            # 示例输出：
            #   A Google scholar for '量子计算' found 10 results:
            #   [结果1]
            #   =======
            #   A Google scholar for '量子纠缠' found 8 results:
            #   [结果2]
            response = "\n=======\n".join(response)

        return response
