"""
V0 version: call mem0 searching in `prepare_task_description` and 
    pass memories to `task_description`.
V1 version: call mem0 searching in `run_single_task` and pass memories 
    to `auxiliary_description`.
V2 version: implement a tool or subagent, let agent call and decide to 
    search whose memories.
"""

from typing import Tuple, Optional
import os
import fire
import json
import time
import hydra
import signal
import dotenv
import asyncio
from pathlib import Path
from jinja2 import Template
from rich.traceback import install
from omegaconf import DictConfig, OmegaConf

from config import config_name, config_path
from src.logging.logger import bootstrap_logger
from src.core.pipeline import execute_task_pipeline

from utils.eval_utils import verify_answer_for_datasets
from common_benchmark import (
    BenchmarkTask, BenchmarkResult, JSONLDatasetEvaluator, 
    TaskStatus, signal_handler, setup_hydra_output_dir
)


class LOCOMOMem0Evaluatorv0(JSONLDatasetEvaluator):
    """
    V0 version: call mem0 searching in `prepare_task_description` and 
    pass memories to task description
    """

    ANSWER_PROMPT_GRAPH = """You are an intelligent memory assistant tasked with retrieving accurate information from 
conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain 
timestamped information that may be relevant to answering the question. You also have 
access to knowledge graph relations for each user, showing connections between entities, 
concepts, and events relevant to that user.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the 
    memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", 
    etc.), calculate the actual date based on the memory timestamp. For example, if a 
    memory from 4 May 2022 mentions "went to India last year," then the trip occurred 
    in 2021.
6. Always convert relative time references to specific dates, months, or years. For 
    example, convert "last year" to "2022" or "two months ago" to "March 2023" based 
    on the memory timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse 
    character names mentioned in memories with the actual users who created those 
    memories.
8. The answer should be less than 5-6 words.
9. Use the knowledge graph relations to understand the user's knowledge network and 
    identify important relationships between entities in the user's world.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the 
    question
4. If the answer requires calculation (e.g., converting relative time references), 
    show your work
5. Analyze the knowledge graph relations to understand the user's knowledge context
6. Formulate a precise, concise answer based solely on the evidence in the memories
7. Double-check that your answer directly addresses the question asked
8. Ensure your final answer is specific and avoids vague time references

Memories for user {{speaker_1_user_id}}:

{{speaker_1_memories}}

Relations for user {{speaker_1_user_id}}:

{{speaker_1_graph_memories}}

Memories for user {{speaker_2_user_id}}:

{{speaker_2_memories}}

Relations for user {{speaker_2_user_id}}:

{{speaker_2_graph_memories}}

Question: {{question}}
    """


    ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain 
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), 
    calculate the actual date based on the memory timestamp. For example, if a memory from 
    4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example, 
    convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
    timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse character 
    names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories for user {{speaker_1_user_id}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_user_id}}:

{{speaker_2_memories}}

Question: {{question}}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from mem0 import MemoryClient
        self.mem0_client = MemoryClient(
            api_key=os.getenv("MEM0_API_KEY"),
            org_id=os.getenv("MEM0_ORGANIZATION_ID"),
            project_id=os.getenv("MEM0_PROJECT_ID"),
        )
        self.is_graph = os.getenv("MEM0_IS_GRAPH", False)
        self.top_k = int(os.getenv("MEM0_TOP_K", 30))

        if self.is_graph:
            self.answer_prompt = self.ANSWER_PROMPT_GRAPH
        else:
            self.answer_prompt = self.ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                if self.is_graph:
                    print("Searching with graph")
                    memories = self.mem0_client.search(
                        query,
                        user_id=user_id,
                        top_k=self.top_k,
                        filter_memories=True,
                        enable_graph=True,
                        output_format="v1.1",
                    )
                else:
                    memories = self.mem0_client.search(
                        query, user_id=user_id, top_k=self.top_k, filter_memories=True
                    )
                break
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        if not self.is_graph:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories
            ]
            graph_memories = None
        else:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories["results"]
            ]
            graph_memories = [
                {"source": relation["source"], "relationship": relation["relationship"], "target": relation["target"]}
                for relation in memories["relations"]
            ]
        return semantic_memories, graph_memories, end_time - start_time
    
    def get_answer_prompt(self, speaker_1_user_id, speaker_2_user_id, question):
        speaker_1_memories, speaker_1_graph_memories, _ = self.search_memory(
            speaker_1_user_id, question
        )
        speaker_2_memories, speaker_2_graph_memories, _ = self.search_memory(
            speaker_2_user_id, question
        )

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.answer_prompt)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            question=question,
        )
        return answer_prompt

    def prepare_task_description(
        self, task: BenchmarkTask
    ) -> Tuple[str, Optional[str]]:
        if task.file_path is None:
            full_file_path = None
        else:
            path = Path(task.file_path)
            # check if task.file_path is a relative path
            if path.is_absolute():
                full_file_path = str(path)
            # Build complete file path: data directory + relative path
            full_file_path = Path(self.data_dir) / path

            full_file_path = str(full_file_path)

        question = task.task_question
        speaker_a_user_id = task.metadata["speaker_a_user_id"]
        speaker_b_user_id = task.metadata["speaker_b_user_id"]
        answer_prompt = self.get_answer_prompt(speaker_a_user_id, speaker_b_user_id, question)
        return answer_prompt, full_file_path


async def entrypoint(cfg: DictConfig):
    """
    Main entry point for running benchmarks with Hydra.
    """
    print("Benchmark configuration:\n", OmegaConf.to_yaml(cfg, resolve=True))

    def parse_func(x: str) -> BenchmarkTask:
        data = json.loads(x)
        return BenchmarkTask(
            task_id=data["task_id"],
            task_question=data["task_question"],
            ground_truth=data["ground_truth"],
            file_path=data.get("file_path"),
            metadata=data.get("metadata", {}),
        )

    def filter_func(x: BenchmarkTask) -> bool:
        if len(cfg.benchmark.data.whitelist) > 0:
            return x.task_id in cfg.benchmark.data.whitelist
        else:
            return True

    evaluator = LOCOMOMem0Evaluatorv0(
        data_dir=cfg.benchmark.data.data_dir,
        benchmark_name=cfg.benchmark.name,
        cfg=cfg,
        metadata_file=cfg.benchmark.data.metadata_file,
        parse_func=parse_func,
        filter_func=filter_func,
    )

    """
    Run the full benchmark evaluation process
    """
    print(f"Starting evaluation for benchmark: {cfg.benchmark.name}")

    # Load tasks
    tasks = evaluator.load_tasks()
    if len(evaluator.tasks) == 0:
        print("No tasks loaded. Exiting.")
        return 0.0

    # Run inference
    print(
        f"\nStarting parallel inference with {cfg.benchmark.execution.max_concurrent} concurrent tasks..."
    )
    print(f"Using pass@{evaluator.pass_at_k} evaluation...")
    await evaluator.run_parallel_inference(
        tasks,
        max_concurrent=cfg.benchmark.execution.max_concurrent,
    )


def main(*args, config_file_name: str = ""):
    # Register signal handlers for immediate response to Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    dotenv.load_dotenv()
    LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")

    # Support load from config_file_name
    if config_file_name:
        chosen_config_name = config_file_name
    else:
        chosen_config_name = config_name()

    with hydra.initialize_config_dir(
        config_dir=os.path.abspath(config_path()), version_base=None
    ):
        cfg = hydra.compose(config_name=chosen_config_name, overrides=list(args))
        cfg = setup_hydra_output_dir(cfg, list(args))

        _ = bootstrap_logger(level=LOGGER_LEVEL)
        # Tracing functionality removed - miroflow-contrib deleted
        asyncio.run(entrypoint(cfg))


if __name__ == "__main__":
    install(suppress=[fire, hydra], show_locals=True)
    fire.Fire(main)
