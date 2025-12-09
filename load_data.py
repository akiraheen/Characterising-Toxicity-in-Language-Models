import json

prompts = []

# toxic
with open("data/toxic.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        obj = json.loads(line)
        text = obj["prompt"]["text"]
        prompts.append({"prompt": text, "label": "toxic"})

# non-toxic
with open("data/nontoxic.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        obj = json.loads(line)
        text = obj["prompt"]["text"]
        prompts.append({"prompt": text, "label": "nontoxic"})

print(f"Loaded {len(prompts)} prompts")
print(prompts[0])
print(prompts[2000])
