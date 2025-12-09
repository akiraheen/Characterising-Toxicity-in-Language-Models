import json
import ollama
import time

prompts = []

with open("data/toxic.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        obj = json.loads(line)
        text = obj["prompt"]["text"]
        prompts.append({"prompt": text, "label": "toxic"})

with open("data/nontoxic.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        obj = json.loads(line)
        text = obj["prompt"]["text"]
        prompts.append({"prompt": text, "label": "nontoxic"})

models = ["mistral:7b", "llama3:8b", "gemma:7b"]

number_of_prompts = 3
selected_prompts = prompts[:number_of_prompts]

def generate_with(model_name, prompt_text):
    response = ollama.generate(
        model=model_name,
        prompt=prompt_text,
        options={
            "temperature": 0.7,
            "num_predict": 120, 
        },
    )
    return response["response"]

counter = 0

for model in models:

    print(f"Progress: {counter} out of {len(models)}")

    model_results = []

    for prompt in selected_prompts:

        prompt_text = prompt["prompt"]
        response = generate_with(model, prompt_text)

        model_results.append({
            "prompt": prompt_text,
            "response": response
        })

        time.sleep(0.01)

    model_name = model.replace(":", "_")
    output_file = f"model_outputs/outputs_{model_name}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(model_results, f, ensure_ascii=False, indent=2)

    print(f"Output for model {model} is saved in {output_file}!\n")

    counter += 1

print(f"Progress: {counter} out of {len(models)}")
print("Complete!")