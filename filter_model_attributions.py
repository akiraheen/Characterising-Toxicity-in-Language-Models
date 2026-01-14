import pandas as pd

model_attributions = pd.read_csv('model_attributions_prompt.csv')

model_attributions = model_attributions.groupby('token', as_index=False).agg({
    'count': 'sum',
    'mean_attr': 'sum'
})

model_attributions_sorted = model_attributions.sort_values('mean_attr', ascending=False)
model_attributions_sorted = model_attributions_sorted.head(30)

model_attributions_sorted.to_csv('model_attributions_prompt_filtered.csv', index=False)