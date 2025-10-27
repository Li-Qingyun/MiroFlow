# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import os
import copy
import json
import requests
from typing import Generator, MutableMapping

from utils.prepare_benchmark.common import Task

FILE_URL = "https://raw.githubusercontent.com/snap-research/locomo/refs/heads/main/data/locomo10.json"


def check_file(file_path: str, file_name: str):
    if file_path == "":
        return None, False
    return file_path, True


def download_file(file_url: str, file_path: str):
    if os.path.exists(file_path):
        return
    response = requests.get(file_url)
    with open(file_path, "wb") as f:
        f.write(response.content)
    return file_path


def gen_locomo(data_dir: str) -> Generator[Task, None, None]:
    file_path = os.path.join(data_dir, "locomo10.json")
    download_file(FILE_URL, file_path)

    with open(file_path, "r") as f:
        dataset = json.load(f)

    for conv_idx, item in enumerate(dataset):
        item = copy.deepcopy(item)

        qa = item.pop("qa")
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{conv_idx}"
        speaker_b_user_id = f"{speaker_b}_{conv_idx}"

        for qa_idx, question_item in enumerate(qa):
            task_id = f"locomo_conversation{conv_idx}_qa{qa_idx}"
            question = question_item.get("question", "")
            gt = question_item.get("answer", "")
            category = question_item.get("category", -1)
            evidence = question_item.get("evidence", [])
            adversarial_answer = question_item.get("adversarial_answer", "")

            metadata: MutableMapping = {
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "speaker_a_user_id": speaker_a_user_id,
                "speaker_b_user_id": speaker_b_user_id,
                "category": category,
                "evidence": evidence,
                "adversarial_answer": adversarial_answer,
            }

            task = Task(
                task_id=task_id,
                task_question=question,
                ground_truth=gt,
                file_path=None,
                metadata=metadata,
            )
            yield task

    return
