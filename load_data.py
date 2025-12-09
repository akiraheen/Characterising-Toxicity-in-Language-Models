import json
import ollama
import time

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

models = ["mistral:7b", "llama3:8b", "gemma:7b"]

number_of_prompts = 5
selected_prompts = prompts[:number_of_prompts]

def generate_with(model_tag, prompt_text):
    response = ollama.generate(
        model=model_tag,
        prompt=prompt_text,
        options={
            "temperature": 0.7,
            "num_predict": 120, 
        },
    )
    return response["response"]


for model_tag in models:

    model_results = []

    for idx, item in enumerate(selected_prompts):
        prompt_text = item["prompt"]

        completion = generate_with(model_tag, prompt_text)

        model_results.append({
            "prompt": prompt_text,
            "completion": completion
        })

        # tiny delay so we don't hammer your machine
        time.sleep(0.01)

    # Make filename safe (no ':') and include how many prompts we used
    safe_model_tag = model_tag.replace(":", "_")
    output_filename = f"outputs_{safe_model_tag}.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(model_results, f, ensure_ascii=False, indent=2)

print("All models done.")