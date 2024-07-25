import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
from genslm import GenSLM, SequenceDataset

model = GenSLM("genslm_250M_patric", model_cache_dir="Path to directory housing model checkpoint")
model.eval()

# Select GPU device if it is available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Sliding window
sequences = []
windows = [] 
file_path = 'Path to genome txt file'  

if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        sequence = file.read().strip()
        
        genome_length = len(sequence)
        # Adjust context_window and step_size as necessary
        context_window = 6144 
        step_size = 384
        
        # Start at the first base pair and wrap around as needed
        for start in range(0, genome_length, step_size):
            # Wrap around the sequence
            if start + context_window <= genome_length:
                trimmed_sequence = sequence[start:start + context_window]
            else:
                trimmed_sequence = sequence[start:] + sequence[:start + context_window - genome_length]
            
            sequences.append(trimmed_sequence)
            windows.append(start)
            print(f'Window start: {start}, Sequence length: {len(trimmed_sequence)}')
else:
    print(f'File not found at {file_path}')

# Validate
print(f'Total number of sequences: {len(sequences)}')

# Generate attention maps for each window (derived from original GenSLM vignette)
dataset = SequenceDataset(sequences, model.seq_length, model.tokenizer)
dataloader = DataLoader(dataset)

attention_maps = []
with torch.no_grad():
    for batch in dataloader:
        outputs = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            output_attentions=True
        )
        # outputs.attentions shape: (layers, batch_size, heads, sequence_length, sequence_length)
        last_two_layers = outputs[-1][-2:]
        # Average across the heads for each layer
        avg_heads_last_layer = last_two_layers[-1].mean(dim=1)  # shape: (batch_size, sequence_length, sequence_length)
        avg_heads_second_last_layer = last_two_layers[-2].mean(dim=1)  # shape: (batch_size, sequence_length, sequence_length)
        # Average across the two layers
        avg_attention_map = (avg_heads_last_layer + avg_heads_second_last_layer) / 2
        # Append the result to the attention_maps list
        attention_maps.append(avg_attention_map.cpu().numpy())

# Initialize final attention map with dimensions of total number of codons in sequence
token_length = 78548
summed_attention_map = np.zeros((token_length, token_length))

# Aggregate attention scores from all windows
for idx, att_map in enumerate(attention_maps):
    start_idx = windows[idx] // 3
    for i in range(att_map.shape[1]):  
        for j in range(att_map.shape[2]): 
            global_i = (start_idx + i) % token_length
            global_j = (start_idx + j) % token_length
            summed_attention_map[global_i, global_j] += att_map[0, i, j]

# Sum by columns
summed_attention_map_colsum = summed_attention_map.sum(axis=0)

# Create DataFrames from the summed columns
df_summed_map = pd.DataFrame(summed_attention_map_colsum)
df_summed_map.shape

# Save the DataFrames to CSV
df_summed_map.to_csv("summed_map_pre.csv", index=False)
