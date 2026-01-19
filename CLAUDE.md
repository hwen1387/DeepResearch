# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Tongyi DeepResearch** is an agentic large language model (30.5B parameters, 3.3B activated per token) designed for long-horizon, deep information-seeking tasks. The repository contains:
- Main inference framework using ReAct (Reasoning + Acting) pattern
- WebAgent family of specialized search agents
- Tool ecosystem for web search, webpage reading, file parsing, and code execution
- Evaluation harness for benchmark testing

## Environment Setup

**Required Python version: 3.10.0** (other versions may cause dependency issues)

```bash
# Create environment
conda create -n react_infer_env python=3.10.0
conda activate react_infer_env

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys (see Configuration section)
```

## Configuration

Edit `.env` file with required API credentials:
- `SERPER_KEY_ID`: Web search and Google Scholar (from https://serper.dev/)
- `JINA_API_KEYS`: Webpage content reading (from https://jina.ai/)
- `API_KEY` / `API_BASE`: OpenAI-compatible API for page summarization
- `DASHSCOPE_API_KEY`: File parsing service (from https://dashscope.aliyun.com/)
- `SANDBOX_FUSION_ENDPOINT`: Python code execution sandbox (see https://github.com/bytedance/SandboxFusion)
- `MODEL_PATH`: Path to model weights
- `DATASET`: Path to evaluation dataset (JSON or JSONL format)
- `OUTPUT_PATH`: Directory for output results

## Running Inference

### Main DeepResearch Model Inference

```bash
# Start vLLM servers and run inference
bash inference/run_react_infer.sh
```

This script:
1. Starts 8 vLLM servers on ports 6001-6008 (modify CUDA_VISIBLE_DEVICES as needed)
2. Waits for servers to be ready (polls `/v1/models` endpoint)
3. Runs parallel inference with `run_multi_react.py`

**Key parameters** (configured in `.env` or `run_react_infer.sh`):
- `MAX_WORKERS`: Parallel execution threads (default: 30)
- `ROLLOUT_COUNT`: Number of rollouts per question (default: 3)
- `TEMPERATURE`: LLM sampling temperature (default: 0.85)
- `PRESENCE_PENALTY`: Token repetition penalty (default: 1.1)

### Direct Python Inference

```bash
cd inference
python run_multi_react.py \
  --model $MODEL_PATH \
  --dataset eval_data/questions.jsonl \
  --output ./outputs \
  --max_workers 20 \
  --roll_out_count 3 \
  --temperature 0.85 \
  --presence_penalty 1.1
```

### Using OpenRouter API

To use OpenRouter instead of local deployment:
1. Modify `inference/react_agent.py` line 62: Set `openai_api_base` to OpenRouter URL
2. Set API key in line 61
3. Change model name to `alibaba/tongyi-deepresearch-30b-a3b`
4. Uncomment lines 87-88 to extract reasoning content

## Dataset Format

The system accepts two formats:

**JSONL (recommended):**
```json
{"question": "What is the capital of France?", "answer": "Paris"}
{"question": "Explain quantum computing", "answer": "..."}
```

**JSON:**
```json
[
  {"question": "What is the capital of France?", "answer": "Paris"},
  {"question": "Explain quantum computing", "answer": "..."}
]
```

**File references:** For questions requiring file parsing, prepend filename to question:
```json
{"question": "(Uploaded 1 file: ['report.pdf'])\n\nWhat are the key findings?", "answer": "..."}
```
Place files in `eval_data/file_corpus/` directory.

## Running Evaluation

```bash
cd evaluation

# Evaluate on DeepSearch benchmarks
python evaluate_deepsearch_official.py \
  --input_fp ../inference/outputs \
  --dataset <benchmark_name>

# Evaluate on Humanity's Last Exam
python evaluate_hle_official.py \
  --input_fp ../inference/outputs \
  --dataset hle
```

## Architecture Overview

### Core Components

```
DeepResearch/
├── inference/              # Main ReAct agent implementation
│   ├── react_agent.py      # MultiTurnReactAgent: main agentic loop
│   ├── run_multi_react.py  # Parallel inference orchestrator
│   ├── prompt.py           # System prompts and tool definitions
│   ├── tool_search.py      # Web search via Serper API
│   ├── tool_visit.py       # Webpage extraction via Jina + LLM summarization
│   ├── tool_file.py        # File parser (PDF, DOCX, XLSX, MP4, etc.)
│   ├── tool_python.py      # Python code execution via SandboxFusion
│   ├── tool_scholar.py     # Google Scholar academic search
│   └── file_tools/         # File parsing utilities and IDP integration
├── WebAgent/               # Family of specialized web agents
│   ├── NestBrowse/         # Browser-based agent with MCP client
│   ├── ParallelMuse/       # Multi-trajectory reasoning aggregation
│   ├── WebDancer/          # Native agentic search (NeurIPS 2025)
│   ├── WebSailor/          # Extended thinking + RL training
│   ├── WebWatcher/         # Vision-language deep research
│   └── ...                 # Other specialized agents
├── Agent/                  # Agent scaling and training infrastructure
│   ├── AgentScaler/        # Continual pre-training framework
│   └── AgentFounder/       # Agent training foundations
└── evaluation/             # Benchmark evaluation scripts
```

### ReAct Agent Execution Flow

The main agent loop in `inference/react_agent.py` (`MultiTurnReactAgent._run()`):

1. **Initialize**: Build message list with system prompt + user query
2. **Agent Loop** (max 100 turns, 150 min timeout):
   - Call vLLM/OpenRouter API with conversation history
   - Parse response for `<tool_call>` XML tags
   - Extract JSON with `{"name": "tool_name", "arguments": {...}}`
   - Route to appropriate tool in `TOOL_MAP`
   - Wrap result in `<tool_response>` tags
   - Append to message history
3. **Termination**: When `<answer>...</answer>` tags found or timeout/context exceeded
4. **Output**: Full conversation trace with termination reason

### Tool System

Tools registered in `TOOL_MAP` (see `inference/react_agent.py` lines 31-38):

| Tool | Purpose | API/Service |
|------|---------|-------------|
| `search` | Google web search (batch queries) | Serper API |
| `visit` | Extract webpage content & summarize | Jina AI + OpenAI |
| `parse_file` | Parse documents/media files | Dashscope IDP |
| `google_scholar` | Academic paper search | Serper Scholar API |
| `PythonInterpreter` | Execute Python code in sandbox | SandboxFusion |

**Tool call format:**
```xml
<tool_call>
{"name": "search", "arguments": {"query": ["query1", "query2"]}}
</tool_call>
```

**Tool response format:**
```xml
<tool_response>
Result content here
</tool_response>
```

### WebAgent Family

The `WebAgent/` directory contains 13+ specialized agent implementations. Key examples:

- **NestBrowse**: Uses async/await pattern with MCP (Model Context Protocol) client for browser interaction (Visit, Click, Fill actions). Reference this for building async agents.
- **ParallelMuse**: Implements multi-trajectory aggregation - runs parallel rollouts, distills each into structured report, then integrates reports for final answer.
- **WebDancer**: Native agentic search model with 4-stage training (data construction, trajectory sampling, SFT, RL).

Each WebAgent project is self-contained with its own inference scripts, prompts, and evaluation code. They share tool implementations from the main `inference/` directory or have specialized variants.

## Key Implementation Details

### Agent Constraints

- **Max context length**: 128K tokens (Qwen tokenizer)
- **Max agent turns**: 100 (configurable via `MAX_LLM_CALL_PER_RUN` env var)
- **Timeout**: 150 minutes per query (9000 seconds)
- **Tool call parsing**: Extracts JSON between `<tool_call>` and `</tool_call>` tags
- **Answer extraction**: Content between `<answer>` and `</answer>` tags

### Tool Implementation Pattern

When adding new tools:
1. Create class inheriting from `qwen_agent.tools.BaseTool`
2. Define `name`, `description`, `parameters` (JSON Schema)
3. Implement `call(params, **kwargs) -> str` method
4. Add to `TOOL_CLASS` list in `inference/react_agent.py`

Example structure:
```python
from qwen_agent.tools import BaseTool, register_tool

@register_tool('my_tool')
class MyTool(BaseTool):
    description = 'Tool description for LLM'
    parameters = [{
        'name': 'param1',
        'type': 'string',
        'description': 'Parameter description',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        # Parse params (JSON string)
        # Execute tool logic
        # Return formatted string result
        return result
```

### Visit Tool Implementation

The `visit` tool (`inference/tool_visit.py`) has a three-stage pipeline:
1. **Fetch**: Uses Jina AI reader API (`r.jina.ai/{url}`) with retry logic
2. **Summarize**: Calls LLM with `EXTRACTOR_PROMPT` to extract relevant info
3. **Format**: Returns JSON with `{rational, evidence, summary}` fields

**Token management**: Truncates webpage content to 95K tokens before summarization to avoid context overflow.

### File Parser Tool

Supports 15+ formats via `inference/file_tools/`:
- Documents: PDF, DOCX, PPTX, TXT, HTML, CSV, XLSX, XLS, DOC, ZIP
- Media: MP4, MP3, WAV, AAC, OGG, FLAC, MOV, MKV, WEBM

Uses Alibaba IDP (Intelligent Document Processing) service for robust parsing. Falls back to native parsers if IDP unavailable.

### Python Interpreter

Uses SandboxFusion (https://github.com/bytedance/SandboxFusion) for secure code execution:
- Multiple endpoint support with round-robin load balancing
- 8 retry attempts with exponential backoff
- 50-second timeout per execution
- Returns stdout + stderr + execution time

**Special format**: Code must be in `<code>...</code>` tags after empty arguments JSON:
```xml
<tool_call>
{"name": "PythonInterpreter", "arguments": {}}
<code>
import numpy as np
print(f"Result: {np.mean([1,2,3])}")
</code>
</tool_call>
```

## Common Tasks

### Modify Agent Behavior

Edit `inference/react_agent.py`:
- Line 126-226: Main agent loop logic
- Line 59-108: LLM API call with retry logic
- Line 228-247: Tool call parsing and dispatch

### Add or Modify Tools

1. Edit tool file in `inference/tool_*.py`
2. Modify tool class `call()` method
3. Update tool description/parameters if needed
4. No need to restart inference servers (tools run in main process)

### Change Prompts

Edit `inference/prompt.py`:
- `SYSTEM_PROMPT`: Main agent instructions and tool definitions (lines 1-35)
- `EXTRACTOR_PROMPT`: Webpage content extraction format (lines 37-50)

### Adjust Inference Parameters

Edit `inference/run_react_infer.sh` or pass to `run_multi_react.py`:
- `--temperature`: Sampling temperature (0.0-2.0, default 0.85)
- `--presence_penalty`: Repetition penalty (0.0-2.0, default 1.1)
- `--max_workers`: Parallel threads (default 30)
- `--roll_out_count`: Rollouts per question (default 3)

### Debug Failed Inference

Check output JSONL for termination reason:
```json
{
  "termination": "max_turn_exceeded",  // or "timeout" or "context_length_exceeded"
  "messages": [...],  // Full conversation history
  "prediction": "..."  // Extracted answer or error
}
```

Common issues:
- `max_turn_exceeded`: Agent used 100 tool calls without answering (increase `MAX_LLM_CALL_PER_RUN`)
- `timeout`: Query took >150 minutes (increase timeout in `react_agent.py` line 141)
- `context_length_exceeded`: Conversation history >110K tokens (reduce webpage summarization verbosity)

### Work with WebAgent Projects

Each WebAgent project has its own setup:

**Example: WebDancer**
```bash
cd WebAgent/WebDancer
pip install -r requirements.txt

# Deploy model with sglang
cd scripts
bash deploy_model.sh /path/to/WebDancer-32B

# Run demo
bash run_demo.sh
```

**Example: NestBrowse**
```bash
cd WebAgent/NestBrowse
# Edit config in prompts.py
python infer_async_nestbrowse.py
```

Refer to individual `WebAgent/*/README.md` files for specific instructions.

## Performance Notes

**Typical inference times:**
- Simple factual queries: 2-5 minutes
- Complex multi-hop research: 10-30 minutes
- Deep research with code execution: 20-60 minutes

**Scaling:**
- Single GPU: Run single vLLM server, set `MAX_WORKERS=1`
- Multi-GPU: Default config uses 8 GPUs with 8 vLLM servers
- Distributed: Set `WORLD_SIZE` and `RANK` environment variables for multi-node inference

**Resource requirements:**
- Model: ~30GB VRAM (30B-A3B model)
- Inference: Additional ~10GB for KV cache per concurrent request
- Recommended: A100 40GB or H100 for optimal performance

## Important Notes

- The model was trained with specific prompts and tools. **Do not modify system prompts** without retraining or careful evaluation.
- Tool outputs are **not** sandboxed in the agent loop - malicious tool results could affect downstream reasoning.
- The `visit` tool uses an external LLM for summarization - costs scale with number of webpage visits.
- File parsing via Dashscope IDP requires an active Alibaba Cloud account.
- SandboxFusion endpoints must be deployed separately for Python code execution to work.

## Additional Resources

- Paper: https://arxiv.org/pdf/2510.24701
- Blog: https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/
- Model: https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B
- FAQ: See `FAQ.md` in repository root
- WebAgent papers: Links in `WebAgent/README.md`
