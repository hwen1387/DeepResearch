"""
【核心文件 3】Search 工具实现 - 详细注释版

原文件: inference/tool_search.py

这个文件展示了如何实现一个标准的 Agent 工具。
学习这个文件后，您可以轻松创建自己的工具。
"""

import json
import http.client
from typing import List, Union, Optional
from qwen_agent.tools.base import BaseTool, register_tool
import os

# ============================================================================
# 配置: 从环境变量获取 API Key
# ============================================================================

# Serper API Key (从 https://serper.dev/ 获取)
SERPER_KEY = os.environ.get('SERPER_KEY_ID')


# ============================================================================
# 工具类: Search
# ============================================================================

@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    """
    网络搜索工具

    功能: 使用 Google Serper API 执行网络搜索
    特点:
    - 支持批量查询（一次调用可以搜索多个问题）
    - 自动检测中英文，使用相应的地区设置
    - 返回 Top 10 搜索结果
    - 包含标题、链接、摘要、日期、来源
    """

    # ========================================================================
    # 工具元数据定义
    # ========================================================================

    name = "search"

    description = """
    执行批量网络搜索：提供一个查询数组 'query'；
    工具将为每个查询检索 Top 10 结果，一次调用返回所有结果。
    """

    # 参数定义（JSON Schema 格式）
    # LLM 会根据这个定义来构造工具调用
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",              # 数组类型
                "items": {
                    "type": "string"           # 数组元素是字符串
                },
                "description": "查询字符串数组。在一次调用中包含多个互补的搜索查询。"
            },
        },
        "required": ["query"],  # query 参数是必需的
    }

    # ========================================================================
    # 初始化方法
    # ========================================================================

    def __init__(self, cfg: Optional[dict] = None):
        """
        初始化工具

        参数:
            cfg: 配置字典（可选）
        """
        super().__init__(cfg)

    # ========================================================================
    # 核心方法: google_search_with_serp
    # ========================================================================

    def google_search_with_serp(self, query: str) -> str:
        """
        使用 Serper API 执行单个 Google 搜索

        参数:
            query: 搜索查询字符串

        返回:
            格式化的搜索结果（Markdown 格式）

        工作流程:
            1. 检测查询语言（中文/英文）
            2. 构建 API 请求
            3. 发送请求（带重试机制）
            4. 解析响应
            5. 格式化结果
        """

        # ====================================================================
        # 步骤 1: 语言检测
        # ====================================================================

        def contains_chinese_basic(text: str) -> bool:
            """
            检查文本是否包含中文字符

            使用 Unicode 范围检测：
            - \u4E00 到 \u9FFF 是常用汉字的 Unicode 范围
            """
            return any('\u4E00' <= char <= '\u9FFF' for char in text)

        # ====================================================================
        # 步骤 2: 建立连接
        # ====================================================================

        # 创建 HTTPS 连接到 Serper API
        conn = http.client.HTTPSConnection("google.serper.dev")

        # ====================================================================
        # 步骤 3: 构建请求参数
        # ====================================================================

        if contains_chinese_basic(query):
            # 中文查询：使用中国地区设置
            payload = json.dumps({
                "q": query,                    # 查询字符串
                "location": "China",           # 搜索地区
                "gl": "cn",                    # 国家代码
                "hl": "zh-cn"                  # 语言设置
            })
        else:
            # 英文查询：使用美国地区设置
            payload = json.dumps({
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en"
            })

        # ====================================================================
        # 步骤 4: 设置请求头
        # ====================================================================

        headers = {
            'X-API-KEY': SERPER_KEY,          # API 认证密钥
            'Content-Type': 'application/json'
        }

        # ====================================================================
        # 步骤 5: 发送请求（带重试机制）
        # ====================================================================

        # 最多重试 5 次
        for i in range(5):
            try:
                # 发送 POST 请求
                conn.request("POST", "/search", payload, headers)

                # 获取响应
                res = conn.getresponse()

                # 成功，跳出循环
                break

            except Exception as e:
                # 请求失败
                print(f"搜索请求失败 (尝试 {i+1}/5): {e}")

                # 如果是最后一次尝试，返回错误
                if i == 4:
                    return f"Google 搜索超时，请稍后重试。"

                # 继续重试
                continue

        # ====================================================================
        # 步骤 6: 读取和解析响应
        # ====================================================================

        # 读取响应数据
        data = res.read()

        # 解析 JSON
        results = json.loads(data.decode("utf-8"))

        # ====================================================================
        # 步骤 7: 处理搜索结果
        # ====================================================================

        try:
            # 检查是否有有机搜索结果
            if "organic" not in results:
                raise Exception(f"未找到查询 '{query}' 的结果。请使用更宽泛的查询。")

            # 存储格式化的结果
            web_snippets = list()
            idx = 0

            # ================================================================
            # 遍历每个搜索结果
            # ================================================================

            if "organic" in results:
                for page in results["organic"]:
                    idx += 1

                    # ----------------------------------------------------
                    # 提取可选字段
                    # ----------------------------------------------------

                    # 发布日期（如果有）
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    # 来源（如果有）
                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    # 摘要（如果有）
                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    # ----------------------------------------------------
                    # 格式化单个结果
                    # ----------------------------------------------------

                    """
                    格式示例:
                    1. [页面标题](https://example.com)
                    Date published: 2024-01-15
                    Source: Wikipedia
                    页面摘要内容...
                    """

                    redacted_version = (
                        f"{idx}. [{page['title']}]({page['link']})"
                        f"{date_published}{source}\n{snippet}"
                    )

                    # 清理无用内容
                    redacted_version = redacted_version.replace(
                        "Your browser can't play this video.", ""
                    )

                    # 添加到结果列表
                    web_snippets.append(redacted_version)

            # ================================================================
            # 步骤 8: 组合最终输出
            # ================================================================

            """
            最终输出格式:

            A Google search for '量子计算' found 10 results:

            ## Web Results

            1. [量子计算 - 维基百科](https://zh.wikipedia.org/wiki/量子计算)
            Date published: 2024-01-15
            Source: Wikipedia
            量子计算是利用量子力学原理进行计算...

            2. [量子计算的最新进展](https://example.com/article)
            ...
            """

            content = (
                f"A Google search for '{query}' found {len(web_snippets)} results:\n\n"
                f"## Web Results\n" +
                "\n\n".join(web_snippets)
            )

            return content

        except Exception as e:
            # 搜索失败或无结果
            return f"未找到 '{query}' 的结果。请尝试更通用的查询。"

    # ========================================================================
    # 包装方法: search_with_serp
    # ========================================================================

    def search_with_serp(self, query: str) -> str:
        """
        搜索包装器

        这个方法只是简单地调用 google_search_with_serp
        保留这个方法是为了代码的扩展性和一致性
        """
        result = self.google_search_with_serp(query)
        return result

    # ========================================================================
    # 工具调用入口: call
    # ========================================================================

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        工具调用的统一入口

        这是所有 Agent 工具必须实现的方法。
        Agent 会调用这个方法来执行工具功能。

        参数:
            params: 工具参数
                - 类型: 字符串（JSON）或字典
                - 格式: {"query": ["查询1", "查询2", ...]}

        返回:
            工具执行结果（字符串）

        支持两种调用模式:
            1. 单查询: {"query": "量子计算"}
            2. 多查询: {"query": ["量子计算", "量子纠缠"]}
        """

        # ====================================================================
        # 步骤 1: 解析参数
        # ====================================================================

        try:
            # 提取 query 参数
            query = params["query"]

        except Exception as e:
            # 参数格式错误
            return "[Search] 无效的请求格式：输入必须是包含 'query' 字段的 JSON 对象"

        # ====================================================================
        # 步骤 2: 处理单查询 vs 多查询
        # ====================================================================

        if isinstance(query, str):
            # ------------------------------------------------------------
            # 模式 1: 单个查询
            # ------------------------------------------------------------

            """
            输入: {"query": "量子计算"}
            处理: 直接搜索
            """

            response = self.search_with_serp(query)

        else:
            # ------------------------------------------------------------
            # 模式 2: 多个查询（批量）
            # ------------------------------------------------------------

            """
            输入: {"query": ["量子计算", "量子纠缠"]}
            处理: 遍历每个查询，分别搜索，然后合并结果
            """

            assert isinstance(query, List)

            responses = []

            # 对每个查询执行搜索
            for q in query:
                responses.append(self.search_with_serp(q))

            # 用分隔符合并多个结果
            response = "\n=======\n".join(responses)

        # ====================================================================
        # 步骤 3: 返回结果
        # ====================================================================

        return response


# ============================================================================
# 使用示例
# ============================================================================

"""
示例 1: 单查询
─────────────

Agent 输出:
<tool_call>
{"name": "search", "arguments": {"query": ["量子计算"]}}
</tool_call>

工具执行:
search_tool = Search()
result = search_tool.call({"query": ["量子计算"]})

返回:
A Google search for '量子计算' found 10 results:

## Web Results

1. [量子计算 - 维基百科](https://zh.wikipedia.org/wiki/量子计算)
Date published: 2024-01-15
Source: Wikipedia
量子计算是利用量子力学原理...

2. ...


示例 2: 多查询（批量）
─────────────────────

Agent 输出:
<tool_call>
{
  "name": "search",
  "arguments": {
    "query": ["量子计算", "量子纠缠", "量子比特"]
  }
}
</tool_call>

工具执行:
result = search_tool.call({
    "query": ["量子计算", "量子纠缠", "量子比特"]
})

返回:
A Google search for '量子计算' found 10 results:
...

=======

A Google search for '量子纠缠' found 10 results:
...

=======

A Google search for '量子比特' found 10 results:
...


示例 3: 错误处理
───────────────

无效输入:
<tool_call>
{"name": "search", "arguments": {"invalid_param": "test"}}
</tool_call>

返回:
[Search] 无效的请求格式：输入必须是包含 'query' 字段的 JSON 对象


网络错误:
如果 Serper API 无法访问，重试 5 次后返回:
Google 搜索超时，请稍后重试。


无结果:
如果查询太具体，返回:
未找到 'XXX' 的结果。请尝试更通用的查询。
"""


# ============================================================================
# 创建自定义工具的模板
# ============================================================================

"""
基于 Search 工具，创建自己的工具：

from qwen_agent.tools.base import BaseTool, register_tool
import json

@register_tool("your_tool_name")
class YourTool(BaseTool):
    # 1. 定义元数据
    name = "your_tool_name"

    description = "你的工具描述"

    parameters = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "参数1的描述"
            },
            "param2": {
                "type": "number",
                "description": "参数2的描述"
            }
        },
        "required": ["param1"]
    }

    # 2. 实现 __init__ 方法
    def __init__(self, cfg=None):
        super().__init__(cfg)

    # 3. 实现 call 方法（必需）
    def call(self, params, **kwargs):
        try:
            # 解析参数
            param1 = params["param1"]
            param2 = params.get("param2", default_value)

            # 执行工具逻辑
            result = your_logic(param1, param2)

            # 返回字符串结果
            return result

        except Exception as e:
            return f"工具执行失败: {str(e)}"

关键点:
1. 继承 BaseTool
2. 使用 @register_tool 装饰器
3. 定义 name, description, parameters
4. 实现 call(params) 方法
5. 返回字符串结果
6. 处理异常
"""


# ============================================================================
# 工具设计最佳实践
# ============================================================================

"""
✅ 好的工具设计:

1. **清晰的参数定义**
   - 使用 JSON Schema 格式
   - 提供详细的 description
   - 明确 required 字段

2. **健壮的错误处理**
   - 参数验证
   - 重试机制
   - 友好的错误消息

3. **结构化的输出**
   - 使用 Markdown 格式
   - 清晰的层次结构
   - 易于解析

4. **性能优化**
   - 支持批量操作
   - 合理的超时设置
   - 适当的缓存

5. **文档完整**
   - 详细的 docstring
   - 使用示例
   - 错误处理说明

❌ 应避免的做法:

1. 返回非字符串类型
2. 没有错误处理
3. 参数定义不清晰
4. 无限等待/阻塞
5. 输出格式混乱

测试工具的方法:

# 创建工具实例
tool = Search()

# 测试单查询
result1 = tool.call({"query": ["test query"]})
print(result1)

# 测试多查询
result2 = tool.call({"query": ["query1", "query2"]})
print(result2)

# 测试错误情况
result3 = tool.call({"invalid": "params"})
print(result3)
"""
