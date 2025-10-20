# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import re
import asyncio
import logging
import pathlib
from pathlib import Path
import dotenv
import hydra

from src.logging.logger import bootstrap_logger
from config import config_name, config_path, debug_config
from src.core.pipeline import (
    create_pipeline_components,
    execute_task_pipeline,
)
from omegaconf import DictConfig


def to_blockquote(text: str) -> str:
    result_lines = []
    for line in text.splitlines():
        if line.strip() == '':
            result_lines.append('>')
            continue

        m = re.match(r'^(?P<prefix>(?:>\s*)+)', line)
        if m:
            level = m.group('prefix').count('>')
            content = line[m.end():]
            if content.strip() == '':
                result_lines.append('>' * (level + 1))
            else:
                result_lines.append(('>' * (level + 1)) + ' ' + content.lstrip())
        else:
            result_lines.append('> ' + line)
    return '\n'.join(result_lines) + ""


async def multi_turn_chat(
    cfg: DictConfig,
    logger: logging.Logger,
    task_id: str = "task_1",
    task_description: str = "Write a python code to say 'Hello, World!', use python to execute the code.",
    task_file_name: str = "",
) -> None:
    """Asynchrono us main function."""
    debug_config(cfg, logger)
    logs_dir = Path(cfg.output_dir)
    main_agent_tool_manager, sub_agent_tool_managers, output_formatter = (
        create_pipeline_components(cfg, logs_dir=str(logs_dir))
    )

    turn_info = []

    while True:
        turn_task_id = f"{task_id}_turn_{len(turn_info)}"

        task_name = turn_task_id
        log_path = pathlib.Path(".") / pathlib.Path(cfg.output_dir) / f"{task_name}.log"
        logger.info(f"logger_path is {log_path.absolute()}")

        if len(turn_info) > 0:
            task_description = input("Enter the task description: ")

            if task_description.strip().strip(".").lower() in ["exit", "q", "quit"]:
                break

            parts = ["---\n\n"]
            parts.append(f"There has been a conversation with the user. Please continue the conversation.\n\n")
            for turn_id, turn in enumerate(turn_info):
                parts.extend([
                    f"- Turn {turn_id + 1}\n\n",
                    f"User Prompt:\n\n{to_blockquote(turn['task_description'])}\n\n",
                    f"Summary:\n\n{to_blockquote(turn['final_summary'])}\n\n",
                    f"Boxed Answer:\n\n{to_blockquote(turn['final_boxed_answer'])}\n\n"
                ])
            parts.extend([
                f"- Below is the user's prompt on this turn:\n\n",
                f"{task_description}\n\n"
            ])
            parts.append("---\n\n")
            multi_turn_task_description = "".join(parts)
        else:
            multi_turn_task_description = task_description

        # Execute task using the pipeline
        final_summary, final_boxed_answer, _ = await execute_task_pipeline(
            cfg=cfg,
            task_name=task_name,
            task_id=turn_task_id,
            task_file_name=task_file_name,
            task_description=multi_turn_task_description,
            main_agent_tool_manager=main_agent_tool_manager,
            sub_agent_tool_managers=sub_agent_tool_managers,
            output_formatter=output_formatter,
            # relative to the folder where shell command is launched.
            log_path=log_path.absolute(),
        )

        # Print task result
        logger.info(
            f"Final Output for Task: {task_id}, turn {len(turn_info)}, summary = {final_summary}, boxed_answer = {final_boxed_answer}"
        )

        turn_info.append({
            "task_description": task_description,
            "final_summary": final_summary,
            "final_boxed_answer": final_boxed_answer,
        })


def main(
    *args,
    task_id: str = "task_1",
    task: str = "Write a python code to say 'Hello, World!', use python to execute the code.",
    task_file_name: str = "",
    config_file_name: str = "",
):
    if config_file_name:
        chosen_config_name = config_file_name
    else:
        chosen_config_name = config_name()

    dotenv.load_dotenv()
    with hydra.initialize_config_dir(config_dir=config_path(), version_base=None):
        cfg = hydra.compose(config_name=chosen_config_name, overrides=list(args))
        logger = bootstrap_logger(level="DEBUG", to_console=True)

        # Test if logger is working
        logger.info("Logger initialized successfully")

        # Tracing functionality removed - miroflow-contrib deleted
        asyncio.run(multi_turn_chat(cfg, logger, str(task_id), task, task_file_name))
