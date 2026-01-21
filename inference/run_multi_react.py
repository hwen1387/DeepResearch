# ============================================================================
# DeepResearch Agent 并行执行编排器
# ============================================================================
#
# 功能说明：
#   本模块实现了批量并行执行 DeepResearch Agent 任务的主程序
#
# 核心特性：
#   - 多 rollout 支持：每个问题可执行多次（用于聚合多条推理轨迹）
#   - 数据分片：支持将数据集分割到多个 worker 节点
#   - 断点续传：自动跳过已完成的任务
#   - 并行执行：使用线程池并发处理多个问题
#   - 端口分配：Sticky 策略确保同一问题使用相同的 vLLM 端口
#   - 错误处理：捕获超时、异常，记录到输出文件
#
# 使用场景：
#   - 批量评估 Agent 在 benchmark 上的性能
#   - 生成多条推理轨迹用于训练或分析
#   - 分布式执行大规模任务
#
# 典型命令：
#   python run_multi_react.py \
#     --model /path/to/model \
#     --dataset eval_data/questions.jsonl \
#     --output ./outputs \
#     --max_workers 30 \
#     --roll_out_count 3 \
#     --temperature 0.85 \
#     --presence_penalty 1.1
#
# ============================================================================

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import threading
from datetime import datetime
from react_agent import MultiTurnReactAgent
import time
import math

if __name__ == "__main__":
    # ====================================================================
    # 命令行参数解析
    # ====================================================================
    parser = argparse.ArgumentParser()

    # ----------------------------------------------------------------
    # 模型和数据路径参数
    # ----------------------------------------------------------------
    parser.add_argument("--model", type=str, default="",
                        help="模型路径或模型名称")
    parser.add_argument("--output", type=str, default="",
                        help="输出目录，结果将保存为 JSONL 文件")
    parser.add_argument("--dataset", type=str, default="gaia",
                        help="数据集路径（JSON 或 JSONL 格式）")

    # ----------------------------------------------------------------
    # LLM 采样参数
    # ----------------------------------------------------------------
    # temperature: 控制输出的随机性
    #   - 0.0: 确定性输出（每次生成相同）
    #   - 0.6-0.9: 平衡创造性和稳定性（推荐）
    #   - 1.0+: 高随机性（用于创意任务）
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="LLM 采样温度 (0.0-2.0)")

    # top_p: 核采样参数
    #   - 0.95: 从累积概率 95% 的 token 中采样
    #   - 值越小，输出越保守
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="核采样参数 (0.0-1.0)")

    # presence_penalty: 重复惩罚
    #   - 1.0: 无惩罚
    #   - >1.0: 惩罚重复 token（鼓励多样性）
    #   - 1.1: 轻微惩罚，避免无限循环和重复内容
    parser.add_argument("--presence_penalty", type=float, default=1.1,
                        help="重复惩罚系数 (0.0-2.0)")

    # ----------------------------------------------------------------
    # 并行执行参数
    # ----------------------------------------------------------------
    # max_workers: 并发线程数
    #   - 推荐值：20-50（取决于 GPU 数量和内存）
    #   - 过高可能导致 OOM 或排队等待
    parser.add_argument("--max_workers", type=int, default=20,
                        help="并发执行的线程数")

    # roll_out_count: 每个问题执行的次数
    #   - 1: 单次执行（用于基础评估）
    #   - 3-5: 多轨迹聚合（提高答案质量）
    #   - 用于训练数据生成或 Self-Consistency 投票
    parser.add_argument("--roll_out_count", type=int, default=3,
                        help="每个问题的 rollout 次数（多轨迹聚合）")

    # ----------------------------------------------------------------
    # 数据分片参数（用于分布式执行）
    # ----------------------------------------------------------------
    # total_splits: 将数据集分成几份
    # worker_split: 当前 worker 处理第几份（1-indexed）
    #
    # 示例：
    #   - total_splits=3, worker_split=1: 处理前 1/3
    #   - total_splits=3, worker_split=2: 处理中间 1/3
    #   - total_splits=3, worker_split=3: 处理后 1/3
    parser.add_argument("--total_splits", type=int, default=1,
                        help="数据集总分片数（用于分布式）")
    parser.add_argument("--worker_split", type=int, default=1,
                        help="当前 worker 处理的分片编号（1-indexed）")

    args = parser.parse_args()

    # ====================================================================
    # 步骤 1: 初始化配置和路径
    # ====================================================================
    model = args.model
    output_base = args.output
    roll_out_count = args.roll_out_count
    total_splits = args.total_splits
    worker_split = args.worker_split

    # ----------------------------------------------------------------
    # 验证 worker_split 参数
    # ----------------------------------------------------------------
    # worker_split 必须在 [1, total_splits] 范围内
    # 例如：total_splits=3 时，worker_split 可以是 1, 2, 或 3
    if worker_split < 1 or worker_split > total_splits:
        print(f"Error: worker_split ({worker_split}) must be between 1 and total_splits ({total_splits})")
        exit(1)

    # ----------------------------------------------------------------
    # 创建输出目录结构
    # ----------------------------------------------------------------
    # 目录结构：
    #   output_base/
    #     model_name_sglang/
    #       dataset_name/
    #         iter1.jsonl
    #         iter2.jsonl
    #         iter3.jsonl
    model_name = os.path.basename(model.rstrip('/'))  # 提取模型名称
    model_dir = os.path.join(output_base, f"{model_name}_sglang")
    dataset_dir = os.path.join(model_dir, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 打印配置信息
    # ----------------------------------------------------------------
    print(f"Model name: {model_name}")
    print(f"Data set path: {args.dataset}")
    print(f"Output directory: {dataset_dir}")
    print(f"Number of rollouts: {roll_out_count}")
    print(f"Data splitting: {worker_split}/{total_splits}")

    # ====================================================================
    # 步骤 2: 加载数据集
    # ====================================================================
    # 支持两种格式：
    #   1. JSON: 一个包含多个对象的数组
    #      示例: [{"question": "...", "answer": "..."}, ...]
    #
    #   2. JSONL: 每行一个 JSON 对象
    #      示例:
    #        {"question": "...", "answer": "..."}
    #        {"question": "...", "answer": "..."}
    data_filepath = f"{args.dataset}"
    try:
        if data_filepath.endswith(".json"):
            # --------------------------------------------------------
            # 加载 JSON 格式
            # --------------------------------------------------------
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = json.load(f)
            # 验证数据格式
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
            if items and not isinstance(items[0], dict):
                raise ValueError("Input JSON list items must be objects.")

        elif data_filepath.endswith(".jsonl"):
            # --------------------------------------------------------
            # 加载 JSONL 格式
            # --------------------------------------------------------
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")

        items = items
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_filepath}")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing input file {data_filepath}: {e}")
        exit(1)

    # ====================================================================
    # 步骤 3: 数据分片（用于分布式执行）
    # ====================================================================
    # 分片算法：
    #   1. 计算每份的大小：items_per_split = ceil(total / splits)
    #   2. 计算当前 worker 的索引范围：[start_idx, end_idx)
    #   3. 提取对应的数据切片
    #
    # 示例：
    #   total_items = 100, total_splits = 3
    #   items_per_split = ceil(100 / 3) = 34
    #
    #   worker_split=1: items[0:34]   (34 items)
    #   worker_split=2: items[34:68]  (34 items)
    #   worker_split=3: items[68:100] (32 items)
    #
    # 使用 math.ceil() 的原因：
    #   - 确保最后一个 worker 不会遗漏任何数据
    #   - 前面的 worker 可能多处理几个样本
    total_items = len(items)
    items_per_split = math.ceil(total_items / total_splits)
    start_idx = (worker_split - 1) * items_per_split
    end_idx = min(worker_split * items_per_split, total_items)

    # 提取当前 worker 的数据切片
    items = items[start_idx:end_idx]

    print(f"Total items in dataset: {total_items}")
    print(f"Processing items {start_idx} to {end_idx-1} ({len(items)} items)")

    # ====================================================================
    # 步骤 4: 配置输出文件路径
    # ====================================================================
    # 输出文件命名规则：
    #   - 无分片: iter1.jsonl, iter2.jsonl, iter3.jsonl
    #   - 有分片: iter1_split1of3.jsonl, iter2_split1of3.jsonl, ...
    #
    # 为什么区分分片？
    #   - 方便后续合并结果
    #   - 避免多个 worker 写入同一文件（文件冲突）
    if total_splits > 1:
        # 添加分片后缀
        output_files = {i: os.path.join(dataset_dir, f"iter{i}_split{worker_split}of{total_splits}.jsonl") for i in range(1, roll_out_count + 1)}
    else:
        # 无分片后缀
        output_files = {i: os.path.join(dataset_dir, f"iter{i}.jsonl") for i in range(1, roll_out_count + 1)}

    # ====================================================================
    # 步骤 5: 实现断点续传机制
    # ====================================================================
    # 原理：
    #   1. 读取已存在的输出文件
    #   2. 提取已完成的问题（检查 "question" 字段且无 "error"）
    #   3. 将已完成的问题添加到 processed_queries 集合
    #   4. 后续执行时跳过这些问题
    #
    # 优势：
    #   - 程序崩溃或中断后可以继续执行
    #   - 避免重复计算浪费资源
    #   - 支持增量添加新问题
    processed_queries_per_rollout = {}

    for rollout_idx in range(1, roll_out_count + 1):
        output_file = output_files[rollout_idx]
        processed_queries = set()  # 存储已完成的问题

        # ----------------------------------------------------------------
        # 读取现有输出文件
        # ----------------------------------------------------------------
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            # ------------------------------------------------
                            # 判断是否为成功完成的任务
                            # ------------------------------------------------
                            # 条件：
                            #   1. 包含 "question" 字段
                            #   2. 不包含 "error" 字段（失败的任务会有 error）
                            if "question" in data and "error" not in data:
                                processed_queries.add(data["question"].strip())
                        except json.JSONDecodeError:
                            # 跳过无效的 JSON 行
                            print(f"Warning: Skipping invalid line in output file: {line.strip()}")
            except FileNotFoundError:
                pass  # 文件不存在，processed_queries 保持为空集合

        processed_queries_per_rollout[rollout_idx] = processed_queries

    # ====================================================================
    # 步骤 6: 构建任务列表和端口分配
    # ====================================================================
    tasks_to_run_all = []  # 存储所有待执行的任务
    per_rollout_task_counts = {i: 0 for i in range(1, roll_out_count + 1)}  # 统计每个 rollout 的任务数

    # ----------------------------------------------------------------
    # 定义 vLLM 服务端口列表
    # ----------------------------------------------------------------
    # 说明：
    #   - 假设有 8 个 vLLM 服务运行在端口 6001-6008
    #   - 这些端口由 run_react_infer.sh 脚本启动
    #   - 每个端口对应一个 GPU 上的 vLLM 实例
    planning_ports = [6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008]

    # ----------------------------------------------------------------
    # Round-robin 索引（用于负载均衡）
    # ----------------------------------------------------------------
    planning_rr_idx = 0  # 当前轮询索引
    summary_rr_idx = 0   # 保留，未来可能用于其他服务

    # ----------------------------------------------------------------
    # Sticky 端口分配策略
    # ----------------------------------------------------------------
    # 关键设计决策：同一个问题的所有 rollout 使用相同的端口
    #
    # 为什么需要 Sticky 分配？
    #   1. KV Cache 复用：
    #      - 同一问题的多个 rollout 可能有相似的前缀
    #      - 使用同一端口可以利用 vLLM 的 KV Cache
    #      - 减少重复计算，提高效率
    #
    #   2. 负载均衡：
    #      - 不同问题分配到不同端口
    #      - 避免所有请求集中到一个端口
    #
    # 实现方式：
    #   - question_to_ports: 记录每个问题分配的端口
    #   - 第一次遇到问题时，用 round-robin 分配端口
    #   - 后续 rollout 复用相同端口
    question_to_ports = {}

    # ----------------------------------------------------------------
    # 遍历所有 rollout 和问题，构建任务列表
    # ----------------------------------------------------------------
    for rollout_idx in range(1, roll_out_count + 1):
        processed_queries = processed_queries_per_rollout[rollout_idx]

        for item in items:
            # --------------------------------------------------------
            # 提取问题文本
            # --------------------------------------------------------
            question = item.get("question", "").strip()

            # 如果 question 字段为空，尝试从 messages 中提取
            if question == "":
                try:
                    user_msg = item["messages"][1]["content"]
                    question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
                    item["question"] = question
                except Exception as e:
                    print(f"Extract question from user message failed: {e}")

            # 跳过无效的问题
            if not question:
                print(f"Warning: Skipping item with empty question: {item}")
                continue

            # --------------------------------------------------------
            # 断点续传：跳过已完成的问题
            # --------------------------------------------------------
            if question not in processed_queries:
                # ------------------------------------------------
                # 端口分配逻辑
                # ------------------------------------------------
                if question not in question_to_ports:
                    # 第一次遇到此问题，使用 round-robin 分配端口
                    planning_port = planning_ports[planning_rr_idx % len(planning_ports)]
                    question_to_ports[question] = planning_port
                    planning_rr_idx += 1  # 更新轮询索引
                else:
                    # 后续 rollout 复用已分配的端口（Sticky）
                    planning_port = question_to_ports[question]

                # ------------------------------------------------
                # 添加任务到列表
                # ------------------------------------------------
                tasks_to_run_all.append({
                    "item": item.copy(),          # 问题数据
                    "rollout_idx": rollout_idx,   # Rollout 编号
                    "planning_port": planning_port,  # 分配的 vLLM 端口
                })
                per_rollout_task_counts[rollout_idx] += 1

    # ----------------------------------------------------------------
    # 打印任务统计信息
    # ----------------------------------------------------------------
    print(f"Total questions in current split: {len(items)}")
    for rollout_idx in range(1, roll_out_count + 1):
        print(f"Rollout {rollout_idx}: already successfully processed: {len(processed_queries_per_rollout[rollout_idx])}, to run: {per_rollout_task_counts[rollout_idx]}")

    # ====================================================================
    # 步骤 7: 检查是否有任务需要执行
    # ====================================================================
    if not tasks_to_run_all:
        print("All rollouts have been completed and no execution is required.")
    else:
        # ================================================================
        # 步骤 8: 配置 LLM 和 Agent
        # ================================================================
        # LLM 配置参数说明：
        llm_cfg = {
            'model': model,  # 模型路径或名称
            'generate_cfg': {
                # max_input_tokens: 最大输入 token 数
                #   - 320000: 接近 Qwen 模型的 128K context 限制
                #   - 保留一些 buffer 用于输出
                'max_input_tokens': 320000,

                # max_retries: API 调用失败时的最大重试次数
                #   - 10: 处理临时网络问题或服务器繁忙
                'max_retries': 10,

                # temperature, top_p, presence_penalty: 从命令行参数传入
                'temperature': args.temperature,
                'top_p': args.top_p,
                'presence_penalty': args.presence_penalty
            },
            'model_type': 'qwen_dashscope'  # 模型类型标识
        }

        # ----------------------------------------------------------------
        # 创建 MultiTurnReactAgent 实例
        # ----------------------------------------------------------------
        # function_list: 可用的工具列表
        #   - search: Google 网络搜索
        #   - visit: 网页访问和摘要
        #   - google_scholar: 学术搜索
        #   - PythonInterpreter: Python 代码执行
        test_agent = MultiTurnReactAgent(
            llm=llm_cfg,
            function_list=["search", "visit", "google_scholar", "PythonInterpreter"]
        )

        # ----------------------------------------------------------------
        # 为每个 rollout 创建文件写入锁
        # ----------------------------------------------------------------
        # 原因：
        #   - 多个线程可能同时完成同一 rollout 的不同问题
        #   - 需要锁来避免文件写入冲突
        #   - 每个 rollout 独立的锁，不同 rollout 可以并发写入
        write_locks = {i: threading.Lock() for i in range(1, roll_out_count + 1)}

        # ================================================================
        # 步骤 9: 使用线程池并发执行任务
        # ================================================================
        # ThreadPoolExecutor: Python 的线程池实现
        #   - max_workers: 并发线程数（从命令行参数传入）
        #   - 自动管理线程创建和销毁
        #   - 支持任务队列和结果收集
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # --------------------------------------------------------
            # 提交所有任务到线程池
            # --------------------------------------------------------
            # executor.submit() 返回 Future 对象
            # future_to_task: 将 Future 映射回原始任务信息
            future_to_task = {
                executor.submit(
                    test_agent._run,  # Agent 的执行方法
                    task,             # 任务参数
                    model             # 模型路径
                ): task for task in tasks_to_run_all
            }

            # --------------------------------------------------------
            # 收集执行结果（使用 as_completed 实时处理）
            # --------------------------------------------------------
            # as_completed: 按完成顺序yield Future（不是提交顺序）
            # tqdm: 显示进度条
            for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run_all), desc="Processing All Rollouts"):
                task_info = future_to_task[future]
                rollout_idx = task_info["rollout_idx"]
                output_file = output_files[rollout_idx]

                try:
                    # ------------------------------------------------
                    # 获取任务结果
                    # ------------------------------------------------
                    result = future.result()

                    # ------------------------------------------------
                    # 写入结果到文件（加锁）
                    # ------------------------------------------------
                    # ensure_ascii=False: 保留中文字符（不转义为 \uXXXX）
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")

                # ====================================================
                # 异常处理：超时
                # ====================================================
                # TimeoutError: 任务执行超过预设时间限制
                # （通常由 react_agent.py 中的 timeout 触发）
                except concurrent.futures.TimeoutError:
                    question = task_info["item"].get("question", "")
                    print(f'Timeout (>1800s): "{question}" (Rollout {rollout_idx})')
                    future.cancel()  # 取消任务

                    # 记录错误结果
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "error": "Timeout (>1800s)",  # 错误类型
                        "messages": [],  # 空消息列表
                        "prediction": "[Failed]"  # 失败标记
                    }

                    # 写入错误结果（加锁）
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

                # ====================================================
                # 异常处理：其他异常
                # ====================================================
                # 可能的异常：
                #   - API 调用失败
                #   - 工具执行错误
                #   - 内存不足
                #   - 网络问题
                except Exception as exc:
                    question = task_info["item"].get("question", "")
                    print(f'Task for question "{question}" (Rollout {rollout_idx}) generated an exception: {exc}')

                    # 记录错误结果
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "error": f"Future resolution failed: {exc}",
                        "messages": [],
                        "prediction": "[Failed]",
                    }

                    # 打印详细错误信息
                    print("===============================")
                    print(error_result)
                    print("===============================")

                    # 写入错误结果（加锁）
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

        print("\nAll tasks completed!")

    print(f"\nAll {roll_out_count} rollouts completed!")