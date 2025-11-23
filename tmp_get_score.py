import os
import json

N_CORRECT = 0
N_ALL = 0

for file in os.listdir("logs"):
    if file.endswith(".json"):
        with open(os.path.join("logs", file), "r") as f:
            data = json.load(f)
        judge_result = data["judge_result"]
        if judge_result == "CORRECT":
            N_CORRECT += 1
        N_ALL += 1

print(f"N_CORRECT: {N_CORRECT}, N_ALL: {N_ALL}, Score: {N_CORRECT / N_ALL}")
print(f"Score: {N_CORRECT / N_ALL * 100:.2f}%")


N_CORRECT = 0
N_ALL = 0

for file in os.listdir("logs"):
    if file.endswith(".json"):
        with open(os.path.join("logs", file), "r") as f:
            data = json.load(f)
        if data['input']['metadata']['category'] == 5:
            continue
        judge_result = data["judge_result"]
        if judge_result == "CORRECT":
            N_CORRECT += 1
        N_ALL += 1

print(f"N_CORRECT: {N_CORRECT}, N_ALL: {N_ALL}, Score: {N_CORRECT / N_ALL}")
print(f"Score: {N_CORRECT / N_ALL * 100:.2f}%")


N_CORRECT_cat = {}
N_ALL_cat = {}
for file in os.listdir("logs"):
    if file.endswith(".json"):
        with open(os.path.join("logs", file), "r") as f:
            data = json.load(f)
        if data['input']['metadata']['category'] not in N_CORRECT_cat:
            N_CORRECT_cat[data['input']['metadata']['category']] = 0
        if data['input']['metadata']['category'] not in N_ALL_cat:
            N_ALL_cat[data['input']['metadata']['category']] = 0
        if data['judge_result'] == "CORRECT":
            N_CORRECT_cat[data['input']['metadata']['category']] += 1
        N_ALL_cat[data['input']['metadata']['category']] += 1

print(N_CORRECT_cat)
print(N_ALL_cat)

for category in N_CORRECT_cat:
    print(f"Category {category}: {N_CORRECT_cat[category] / N_ALL_cat[category] * 100:.2f}%")