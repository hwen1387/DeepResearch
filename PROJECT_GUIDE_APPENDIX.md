# Tongyi DeepResearch 项目指南 - 附录

> 本文档是 PROJECT_GUIDE_CN.md 的补充，包含后续章节

---

## 7. 开发指南

### 7.1 项目结构导航

```
DeepResearch/
│
├── 📂 inference/                        # 【核心】主推理引擎
│   ├── react_agent.py                   # ReAct 代理核心 (248 行)
│   ├── run_multi_react.py               # 多线程编排器 (228 行)
│   ├── prompt.py                        # 系统提示词
│   ├── tool_search.py                   # 搜索工具
│   ├── tool_visit.py                    # 网页访问工具
│   ├── tool_file.py                     # 文件解析工具
│   ├── tool_python.py                   # Python 解释器
│   ├── tool_scholar.py                  # 学术搜索工具
│   ├── file_tools/                      # 文件解析子系统
│   │   ├── file_parser.py               # 文档解析器
│   │   ├── idp.py                       # Alibaba IDP 集成
│   │   ├── video_agent.py               # 音视频处理
│   │   └── utils.py                     # 工具函数
│   ├── eval_data/                       # 评估数据
│   │   ├── file_corpus/                 # 引用文件存放处
│   │   └── *.jsonl                      # 查询数据集
│   └── run_react_infer.sh               # 启动脚本
│
├── 📂 WebAgent/                         # 【扩展】专业代理家族
│   ├── NestBrowse/                      # 浏览器代理 (异步 + MCP)
│   │   ├── infer_async_nestbrowse.py    # 主执行文件
│   │   ├── toolkit/                     # 工具集
│   │   │   ├── browser.py               # Visit/Click/Fill
│   │   │   ├── mcp_client.py            # MCP 协议客户端
│   │   │   └── tool_search.py           # 搜索工具
│   │   └── prompts.py                   # 提示词
│   │
│   ├── ParallelMuse/                    # 并行轨迹聚合
│   │   ├── compressed_reasoning_aggregation.py
│   │   └── prompts.py
│   │
│   ├── WebDancer/                       # 原生搜索代理 (NeurIPS 2025)
│   ├── WebSailor/                       # 超人推理 + DUPO RL
│   ├── WebWatcher/                      # 视觉语言代理
│   ├── WebShaper/                       # 数据合成
│   ├── WebWeaver/                       # 证据结构化
│   ├── WebResearcher/                   # 长视野推理
│   ├── WebResummer/                     # 上下文摘要
│   └── WebLeaper/                       # 信息跳跃
│
├── 📂 Agent/                            # 【训练】代理训练基础设施
│   ├── AgentScaler/                     # 持续预训练框架
│   └── AgentFounder/                    # 训练基础
│
├── 📂 evaluation/                       # 【评估】基准测试
│   ├── evaluate_deepsearch_official.py  # DeepSearch 基准
│   ├── evaluate_hle_official.py         # HLE 基准
│   └── prompt.py                        # 评判提示词
│
├── 📄 .env.example                      # 环境配置模板
├── 📄 requirements.txt                  # Python 依赖
├── 📄 README.md                         # 项目说明
├── 📄 FAQ.md                            # 常见问题
├── 📄 CLAUDE.md                         # Claude Code 指南
├── 📄 ARCHITECTURE_CN.md                # 架构分析文档
└── 📄 PROJECT_GUIDE_CN.md               # 本项目指南
```

### 7.2 关键文件说明

| 文件 | 行数 | 核心功能 | 何时修改 |
|-----|------|---------|---------|
| `inference/react_agent.py` | 248 | ReAct 主循环 | 修改代理行为、工具调用逻辑 |
| `inference/run_multi_react.py` | 228 | 多线程编排 | 修改并发策略、检查点逻辑 |
| `inference/prompt.py` | 52 | 系统提示词 | 修改代理指令、工具定义 |
| `inference/tool_*.py` | 100-250 | 各工具实现 | 添加/修改工具功能 |
| `inference/run_react_infer.sh` | 118 | 启动脚本 | 修改服务器配置、端口 |
| `evaluation/evaluate_*.py` | 200+ | 评估脚本 | 添加新基准、修改评判逻辑 |

### 7.3 开发工作流

#### 本地开发环境

```bash
# 1. 克隆仓库
git clone https://github.com/Alibaba-NLP/DeepResearch.git
cd DeepResearch

# 2. 创建开发分支
git checkout -b feature/my-new-feature

# 3. 安装依赖 (开发模式)
pip install -e .
pip install -r requirements-dev.txt  # 如果有开发依赖

# 4. 配置环境
cp .env.example .env
vim .env  # 填写 API keys

# 5. 运行测试
pytest tests/  # 如果有测试
```

#### 代码修改示例

**场景: 修改搜索工具，添加日期过滤**

```bash
# 1. 编辑工具文件
vim inference/tool_search.py

# 修改 Search.call() 方法
def call(self, params: str, **kwargs) -> str:
    params_dict = json.loads(params)
    query_list = params_dict['query']
    date_filter = params_dict.get('date_filter', None)  # 新增参数

    results = []
    for query in query_list:
        # 添加日期过滤逻辑
        if date_filter:
            query += f" after:{date_filter}"
        result = self.google_search_with_serp([query])
        results.append(result)

    return '\n\n'.join(results)

# 2. 更新工具描述 (prompt.py)
vim inference/prompt.py

# 在 SYSTEM_PROMPT 中添加 date_filter 参数说明
{"type": "function", "function": {
    "name": "search",
    "parameters": {
        ...
        "date_filter": {
            "type": "string",
            "description": "日期过滤，格式: YYYY-MM-DD"
        }
    }
}}

# 3. 测试修改
cat > eval_data/test_date_filter.jsonl << EOF
{"question": "搜索 2024 年之后关于量子计算的新闻", "answer": ""}
EOF

python inference/run_multi_react.py \
    --model $MODEL_PATH \
    --dataset eval_data/test_date_filter.jsonl \
    --output outputs/test \
    --max_workers 1

# 4. 查看结果
cat outputs/test/*.jsonl | jq '.messages[] | select(.role == "user" and (.content | contains("<tool_response>")))'

# 5. 提交更改
git add inference/tool_search.py inference/prompt.py
git commit -m "feat: 为搜索工具添加日期过滤功能"
git push origin feature/my-new-feature
```

### 7.4 调试技巧

#### 打印调试信息

在 `react_agent.py` 中添加调试输出：

```python
# 在 _run() 方法中
def _run(self, data, model, planning_port):
    print(f"[DEBUG] 处理问题: {data['question'][:50]}...")

    for turn in range(MAX_LLM_CALL_PER_RUN):
        print(f"[DEBUG] 第 {turn+1} 轮开始")

        response = self.call_server(messages, planning_port)
        print(f"[DEBUG] LLM 响应长度: {len(response)} 字符")

        if "<tool_call>" in response:
            tool_calls = extract_tool_calls(response)
            print(f"[DEBUG] 提取到 {len(tool_calls)} 个工具调用")

            for tc in tool_calls:
                print(f"[DEBUG] 调用工具: {tc['name']}")
                result = self.custom_call_tool(tc['name'], tc['arguments'])
                print(f"[DEBUG] 工具返回长度: {len(result)} 字符")
```

#### 保存中间结果

```python
# 在主循环中保存每轮的消息
import json
import os

debug_dir = "debug_output"
os.makedirs(debug_dir, exist_ok=True)

for turn in range(MAX_LLM_CALL_PER_RUN):
    # ... 执行逻辑 ...

    # 保存当前状态
    with open(f"{debug_dir}/turn_{turn}.json", 'w') as f:
        json.dump({
            'turn': turn,
            'messages': messages,
            'token_count': get_token_count(messages)
        }, f, indent=2, ensure_ascii=False)
```

#### 使用 Python 调试器

```python
# 在关键位置设置断点
import pdb

def custom_call_tool(self, tool_name, tool_args):
    if tool_name == "visit":  # 只在 visit 工具时断点
        pdb.set_trace()

    tool = TOOL_MAP.get(tool_name)
    ...
```

#### 日志记录

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepresearch.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 在代码中使用
logger.info(f"开始处理查询: {query}")
logger.debug(f"工具参数: {params}")
logger.warning(f"上下文长度接近上限: {token_count}/110000")
logger.error(f"工具调用失败: {e}")
```

---

## 8. 常见问题

### 8.1 安装和配置问题

#### Q: pip install 出现依赖冲突

**A:** 使用 Python 3.10.0 并创建干净的虚拟环境

```bash
# 确认 Python 版本
python --version  # 必须是 3.10.0

# 删除旧环境
conda env remove -n react_infer_env

# 重新创建
conda create -n react_infer_env python=3.10.0
conda activate react_infer_env

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

#### Q: vLLM 启动失败，报 CUDA 错误

**A:** 检查 CUDA 版本和 GPU 驱动

```bash
# 检查 CUDA 版本
nvcc --version

# 检查 GPU 状态
nvidia-smi

# 确保 CUDA 12.0+
# 如果版本过低，需要升级 CUDA toolkit

# 重新安装 vLLM
pip uninstall vllm
pip install vllm --no-cache-dir
```

#### Q: API Key 配置后仍然报错

**A:** 检查 .env 文件是否正确加载

```bash
# 确认 .env 文件存在
ls -la .env

# 检查环境变量是否加载
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('SERPER_KEY_ID'))"

# 如果返回 None，手动加载
export $(cat .env | xargs)

# 或在代码中明确加载
# react_agent.py 开头添加:
from dotenv import load_dotenv
load_dotenv()
```

### 8.2 运行时问题

#### Q: 推理过程中出现 "context_length_exceeded"

**A:** 优化上下文管理

```python
# 方案 1: 增加上下文截断阈值 (谨慎)
# react_agent.py:195
if get_token_count(messages) > 120000:  # 从 110000 增加到 120000

# 方案 2: 实现上下文压缩 (推荐)
def compress_context(messages, max_length=110000):
    """压缩消息历史"""
    current_length = get_token_count(messages)

    if current_length <= max_length:
        return messages

    # 保留 system 消息和最近的 N 轮对话
    system_msg = messages[0]
    recent_messages = messages[-20:]  # 保留最近 20 条

    # 摘要中间部分
    middle_messages = messages[1:-20]
    summary = summarize_messages(middle_messages)

    return [system_msg, {"role": "user", "content": f"[之前的对话摘要]\n{summary}"}] + recent_messages

# 在主循环中使用
messages = compress_context(messages)
```

#### Q: 工具调用失败，返回错误

**A:** 添加重试和错误处理

```python
# 在 custom_call_tool 中
def custom_call_tool(self, tool_name, tool_args):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            tool = TOOL_MAP.get(tool_name)
            if not tool:
                return f"错误: 未知工具 '{tool_name}'"

            result = tool.call(tool_args)

            # 检查结果是否有效
            if result and len(result) > 0:
                return result

        except Exception as e:
            logger.error(f"工具 {tool_name} 调用失败 (尝试 {attempt+1}/{max_retries}): {e}")

            if attempt == max_retries - 1:
                return f"工具调用失败: {str(e)}"

            time.sleep(2 ** attempt)  # 指数退避
```

#### Q: 推理速度很慢

**A:** 多方面优化

```bash
# 1. 增加并发度
--max_workers 50  # 根据 GPU 数量调整

# 2. 使用更快的模型进行摘要
# .env 中:
SUMMARY_MODEL_NAME=gpt-3.5-turbo  # 替代 gpt-4

# 3. 启用缓存
pip install diskcache

# 在 tool_search.py 中:
from diskcache import Cache
cache = Cache('.cache/search_results')

@cache.memoize(expire=3600)  # 1 小时缓存
def google_search_with_serp(query_list, api_key):
    ...

# 4. 减少 vLLM 推理延迟
vllm serve $MODEL_PATH \
    --max-num-seqs 8 \  # 增加批处理大小
    --enable-prefix-caching \  # 启用前缀缓存
    --gpu-memory-utilization 0.95  # 提高 GPU 利用率
```

### 8.3 结果质量问题

#### Q: 模型回答不准确或不完整

**A:** 调整提示词和参数

```python
# 1. 增强系统提示词
SYSTEM_PROMPT = """
You are a deep research assistant. Your core function is...

# Important Guidelines
1. ALWAYS use multiple tools to verify information
2. Cross-reference facts from at least 2-3 sources
3. Explicitly state when information is uncertain
4. Provide detailed reasoning in <think> tags
5. Only give final answer when you have high confidence

...
"""

# 2. 调整采样参数
# 更确定性的生成
--temperature 0.3 --presence_penalty 0.8

# 或使用 multiple rollouts + voting
--roll_out_count 5

# 然后选择 majority vote 答案
```

#### Q: 模型不使用工具或使用错误的工具

**A:** 优化工具描述和示例

```python
# 在 prompt.py 中添加详细描述
{"type": "function", "function": {
    "name": "search",
    "description": """
    执行 Google 搜索。

    **何时使用**:
    - 查找最新信息 (新闻、事件、数据)
    - 获取多个信息源
    - 探索性研究

    **输入**: 查询字符串列表
    **输出**: Top 10 搜索结果 (标题、URL、摘要)

    **示例**:
    <tool_call>
    {"name": "search", "arguments": {"query": ["2024 Nobel Prize Physics"]}}
    </tool_call>
    """,
    ...
}}

# 添加 Few-Shot 示例
SYSTEM_PROMPT += """
# Tool Usage Examples

**Example 1**: Factual Query
User: "Who won the 2024 Nobel Prize in Physics?"
Assistant: <think>I need to search for recent information</think>
<tool_call>{"name": "search", "arguments": {"query": ["2024 Nobel Prize Physics winner"]}}</tool_call>

**Example 2**: Deep Research
User: "Compare GPT-4 and Claude 3"
Assistant: <think>I should search for both and visit official sources</think>
<tool_call>{"name": "search", "arguments": {"query": ["GPT-4 specifications", "Claude 3 capabilities"]}}</tool_call>
...
"""
```

#### Q: 生成的答案包含幻觉 (hallucination)

**A:** 强化证据验证

```python
# 修改 SYSTEM_PROMPT
SYSTEM_PROMPT = """
...

# Anti-Hallucination Guidelines
1. **NEVER** make up facts or sources
2. **ALWAYS** cite tool outputs when stating facts
3. Use phrases like "According to [source]..." or "Based on [search result]..."
4. If information cannot be found, explicitly state: "I could not find reliable information about..."
5. Mark uncertain information with qualifiers: "possibly", "likely", "approximately"

# Answer Format
<answer>
[Your answer]

**Sources**:
- [Search] "query" → found at URL
- [Visit] URL → key information extracted
- [Scholar] "paper title" → citation
</answer>
"""
```

### 8.4 评估问题

#### Q: 评估脚本报错

**A:** 检查数据格式

```bash
# 确认输出文件格式正确
head -1 outputs/results_rollout_0.jsonl | jq .

# 应包含字段: question, answer, prediction, termination, messages

# 如果字段缺失，检查 run_multi_react.py 的输出逻辑
```

#### Q: LLM 评判器结果不一致

**A:** 使用多个评判器投票

```python
# evaluation/ensemble_judge.py
def ensemble_judge(question, prediction, reference):
    judges = [
        ('gpt-4', judge_with_gpt4),
        ('claude-3.5', judge_with_claude),
        ('qwen2.5-72b', judge_with_qwen)
    ]

    votes = []
    for name, judge_fn in judges:
        result = judge_fn(question, prediction, reference)
        votes.append(result['correct'])

    # Majority voting
    final_verdict = sum(votes) > len(votes) / 2

    return {
        'correct': final_verdict,
        'individual_votes': dict(zip([j[0] for j in judges], votes))
    }
```

---

## 9. 进阶主题

### 9.1 自定义 WebAgent

创建您自己的专业代理：

```bash
# 1. 创建项目目录
mkdir -p WebAgent/MyCustomAgent
cd WebAgent/MyCustomAgent

# 2. 创建文件结构
touch __init__.py
touch infer_my_agent.py
touch prompts.py
mkdir toolkit
touch toolkit/__init__.py
touch toolkit/tool_custom.py

# 3. 实现主循环 (基于 NestBrowse 模板)
# infer_my_agent.py
```

**模板代码**:

```python
# infer_my_agent.py
import asyncio
from openai import OpenAI
from prompts import SYSTEM_PROMPT

async def agentic_loop(question, max_turns=100):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    client = OpenAI(api_key="YOUR_KEY", base_url="http://localhost:6001/v1")

    for turn in range(max_turns):
        # 调用 LLM
        response = client.chat.completions.create(
            model="model_name",
            messages=messages
        )

        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})

        # 检查终止
        if "<answer>" in content:
            return extract_answer(content), messages

        # 执行工具
        if "<tool_call>" in content:
            tool_results = await execute_tools(content)
            messages.append({"role": "user", "content": tool_results})

    return None, messages

async def main():
    question = "Your research question"
    answer, messages = await agentic_loop(question)
    print(f"Answer: {answer}")

if __name__ == '__main__':
    asyncio.run(main())
```

### 9.2 强化学习训练

基于 WebSailor 的 DUPO 算法进行训练：

```python
# Agent/AgentScaler/dupo_trainer.py
import torch
import torch.nn.functional as F

class DUPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model):
        self.policy = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model

    def train_step(self, batch_questions):
        """
        DUPO: Duplicating Sampling Policy Optimization

        对每个问题采样多个轨迹，计算优势函数，更新策略
        """
        all_trajectories = []

        # 1. 采样多个轨迹
        for question in batch_questions:
            trajectories = []
            for _ in range(self.n_samples):
                trajectory = self.rollout(question)
                trajectories.append(trajectory)

            all_trajectories.append(trajectories)

        # 2. 计算奖励
        rewards = []
        for trajs in all_trajectories:
            traj_rewards = [self.reward_model(t) for t in trajs]
            rewards.append(traj_rewards)

        # 3. 计算优势 (leave-one-out baseline)
        advantages = []
        for traj_rewards in rewards:
            baseline = (sum(traj_rewards) - traj_rewards[i]) / (len(traj_rewards) - 1)
            adv = [r - baseline for r in traj_rewards]
            advantages.append(adv)

        # 4. 策略梯度更新
        policy_loss = 0
        for trajs, advs in zip(all_trajectories, advantages):
            for traj, adv in zip(trajs, advs):
                # 计算 log 概率
                log_probs = self.policy.compute_log_probs(traj)

                # 加权损失
                policy_loss += -adv * log_probs.sum()

        # 5. 反向传播
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.item()
```

### 9.3 生产部署

部署到生产环境的最佳实践：

```yaml
# docker-compose.yml
version: '3.8'

services:
  vllm-server-1:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: >
      --model /models/deepresearch
      --host 0.0.0.0
      --port 6001
      --max-num-seqs 16
      --gpu-memory-utilization 0.95
    volumes:
      - ./models:/models
    ports:
      - "6001:6001"

  vllm-server-2:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=1
    command: >
      --model /models/deepresearch
      --host 0.0.0.0
      --port 6002
    volumes:
      - ./models:/models
    ports:
      - "6002:6002"

  agent-service:
    build: .
    depends_on:
      - vllm-server-1
      - vllm-server-2
    environment:
      - SERPER_KEY_ID=${SERPER_KEY_ID}
      - JINA_API_KEYS=${JINA_API_KEYS}
      - API_KEY=${API_KEY}
    volumes:
      - ./inference:/app/inference
      - ./outputs:/app/outputs
    command: python run_api_server.py

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - agent-service
```

**API 服务器示例** (`run_api_server.py`):

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from inference.react_agent import MultiTurnReactAgent

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    max_turns: int = 100

class QueryResponse(BaseModel):
    task_id: str
    status: str

# 任务队列
task_queue = {}

@app.post("/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest, background_tasks: BackgroundTasks):
    task_id = generate_task_id()

    task_queue[task_id] = {"status": "pending"}

    # 后台执行
    background_tasks.add_task(run_agent, task_id, request.question, request.max_turns)

    return QueryResponse(task_id=task_id, status="submitted")

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in task_queue:
        return {"error": "Task not found"}

    task = task_queue[task_id]

    if task["status"] == "pending":
        return {"status": "processing"}
    elif task["status"] == "completed":
        return {"status": "completed", "answer": task["answer"], "messages": task["messages"]}
    else:
        return {"status": "failed", "error": task["error"]}

def run_agent(task_id, question, max_turns):
    try:
        agent = MultiTurnReactAgent(...)
        messages, prediction, termination = agent._run(
            {"question": question},
            model="deepresearch",
            planning_port=6001
        )

        task_queue[task_id] = {
            "status": "completed",
            "answer": prediction,
            "messages": messages
        }
    except Exception as e:
        task_queue[task_id] = {
            "status": "failed",
            "error": str(e)
        }
```

---

## 10. 总结

### 10.1 Tongyi DeepResearch 的优势

✅ **长视野推理**: 最多 100 轮交互，150 分钟推理时间
✅ **多源信息整合**: 网络、学术、文档、代码执行
✅ **SOTA 性能**: 在多个基准测试中领先
✅ **可扩展性**: 丰富的 WebAgent 家族，易于定制
✅ **开源**: 完整的代码、数据合成方法、训练流程

### 10.2 最佳使用场景

🎯 **学术研究**: 文献综述、论文调研
🎯 **市场分析**: 竞品分析、行业报告
🎯 **数据分析**: 文件解析、统计计算、可视化
🎯 **信息验证**: 多源交叉验证、事实核查
🎯 **复杂查询**: 多跳推理、长尾问题

### 10.3 学习资源

📚 **论文**: [Tongyi DeepResearch Technical Report](https://arxiv.org/pdf/2510.24701)
📚 **博客**: [官方技术博客](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)
📚 **模型**: [HuggingFace](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B) | [ModelScope](https://modelscope.cn/models/iic/Tongyi-DeepResearch-30B-A3B)
📚 **WebAgent 系列**: 13+ 专业论文 (见 README.md)

### 10.4 社区支持

💬 **GitHub Issues**: [提交问题](https://github.com/Alibaba-NLP/DeepResearch/issues)
💬 **WeChat 群**: 见 README.md 中的二维码
💬 **联系邮箱**: yongjiang.jy@alibaba-inc.com

---

## 附录

### A. 配置参数速查表

| 参数 | 默认值 | 说明 | 调优建议 |
|-----|--------|------|---------|
| `MAX_LLM_CALL_PER_RUN` | 100 | 最大 LLM 调用次数 | 复杂任务增加到 150 |
| `TIMEOUT` | 9000 秒 | 单次查询超时 | 简单任务减少到 1800 |
| `MAX_CONTEXT_LENGTH` | 110000 tokens | 最大上下文 | 不建议增加 |
| `TEMPERATURE` | 0.85 | 采样温度 | 事实查询降低到 0.3 |
| `PRESENCE_PENALTY` | 1.1 | 重复惩罚 | 增加到 1.5 提高多样性 |
| `MAX_WORKERS` | 20-30 | 并发线程数 | 根据 GPU 数量调整 |
| `ROLLOUT_COUNT` | 3 | Rollout 次数 | 评估时增加到 5 |

### B. API 成本估算

| 服务 | 每次成本 | 典型使用量 | 单查询成本 |
|-----|---------|-----------|-----------|
| Serper Search | $0.002 | 3-5 次 | $0.006-0.01 |
| Jina Reader | $0.005 | 5-10 页 | $0.025-0.05 |
| OpenAI摘要 (GPT-4) | $0.03/1K tokens | 50K tokens | $1.50 |
| Dashscope IDP | $0.05/文档 | 1-2 文档 | $0.05-0.10 |
| SandboxFusion | $0.001/执行 | 2-3 次 | $0.002-0.003 |
| **总计** | - | - | **$1.58-1.66** |

**降低成本的方法**:
- 使用 GPT-3.5 替代 GPT-4 做摘要 (成本降低 95%)
- 启用缓存机制
- 限制 visit 工具的使用频率

### C. 术语表

| 术语 | 英文 | 解释 |
|-----|------|------|
| **ReAct** | Reasoning + Acting | 推理与行动交替的代理范式 |
| **Rollout** | - | 一次完整的推理执行过程 |
| **Tool Call** | - | 代理调用外部工具的行为 |
| **Context Length** | - | 对话历史的 token 总数 |
| **Termination** | - | 推理结束的原因 |
| **DUPO** | Duplicating Sampling Policy Optimization | WebSailor 的 RL 算法 |
| **MCP** | Model Context Protocol | 模型-浏览器通信协议 |
| **IDP** | Intelligent Document Processing | 智能文档处理服务 |
| **vLLM** | - | 高性能 LLM 推理引擎 |
| **MoE** | Mixture of Experts | 混合专家模型架构 |

---

*本指南最后更新: 2026-01-19*
*版本: 1.0*
*作者: Claude Code with Sonnet 4.5*
