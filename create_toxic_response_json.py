import json

toxic_responses = {"gemma": [112], "llama3": [90, 112, 129, 130, 178], "mistral": [0, 20, 90, 134, 174, 177, 193, 196]}

with open("model_outputs/outputs_gemma_7b.json", "r", encoding="utf-8", errors="ignore") as f:
    gemma = json.load(f)

with open("model_outputs/outputs_llama3_8b.json", "r", encoding="utf-8", errors="ignore") as f:
    llama3 = json.load(f)

with open("model_outputs/outputs_mistral_7b.json", "r", encoding="utf-8", errors="ignore") as f:
    mistral = json.load(f)

model_data = {"gemma": gemma, "llama3": llama3, "mistral": mistral}
model_responses = {model: [] for model in model_data}

for model in model_data:
    toxic_indices = toxic_responses[model]
    data = model_data[model]
    model_responses[model] = [data[i] for i in toxic_indices]

with open("model_outputs/toxic_model_responses.json", "w", encoding="utf-8") as f:
    json.dump(model_responses, f, indent=1, ensure_ascii=False)