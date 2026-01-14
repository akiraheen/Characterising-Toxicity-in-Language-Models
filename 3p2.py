import numpy as np
import csv
import string

files = ["gemma_150.csv", "llama_150.csv", "mistral_150.csv"]

word_combos = dict()

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["toxicity_score"] and float(row["toxicity_score"]) < 0.7:
                continue

            response = row["response"]
            response = response.replace("f*ck", "fuck")
            response = response.translate(str.maketrans("", "", string.punctuation))
            response = response.lower()

            words = response.split()

            for i in range(len(words) - 1):
                combo = (f"{words[i]} {words[i + 1]}")
                if combo in word_combos:
                    word_combos[combo] += 1
                else:
                    word_combos[combo] = 1

sorted_combos = sorted(word_combos.items(), key=lambda x: x[1], reverse=True)
for combo, count in sorted_combos[:20]:
    print(f"{combo}")