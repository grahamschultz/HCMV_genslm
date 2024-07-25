import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
from genslm import GenSLM, SequenceDataset

model = GenSLM("genslm_25M_patric", model_cache_dir="/dartfs-hpc/rc/home/j/f006zdj/GenSLM")
model.eval()

# Select GPU device if it is available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def sliding_window(filename):
    sequences = []
    windows = []
    if os.path.exists(filename):
        with open(filename, 'r') as file:  
            sequence = file.read().strip()
            genome_length = len(sequence)
            context_window = 6144
            step_size = 512
            # Start at the first base pair and wrap around as needed
            for start in range(0, genome_length, step_size):
                # Wrap around the sequence
                if start + context_window <= genome_length:
                    trimmed_sequence = sequence[start:start + context_window]
                else:
                    trimmed_sequence = sequence[start:] + sequence[:start + context_window - genome_length]
                sequences.append(trimmed_sequence)
                windows.append(start)
    else:
        print(f'File not found at {filename}') 
    return sequences, windows

# Generate sliding window averaged embedding for each txt file in folder
folder_dir = 'Path to folder directory'

all_embeddings = []

for filename in os.listdir(folder_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_dir, filename)  # Ensure full file path is used
        file_seqs = sliding_window(file_path)[0]
        dataset = SequenceDataset(file_seqs, model.seq_length, model.tokenizer)
        dataloader = DataLoader(dataset)
        # Generate embeddings
        window_embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                output_hidden_states=True,
                )
                # outputs.hidden_states shape: (layers, batch_size, sequence_length, hidden_size)
                # Use the embeddings of the last layer
                emb = outputs.hidden_states[-1].detach().cpu().numpy()
                # Compute average over sequence length
                emb = np.mean(emb, axis=1)
                window_embeddings.append(emb)
        window_embeddings = np.concatenate(window_embeddings)  # Shape: (num_windows, hidden_size)
        window_embeddings.shape
        # Compute the average embedding over all sequences
        sequence_embedding = np.mean(window_embeddings, axis=0)
        sequence_embedding.shape
        all_embeddings.append(sequence_embedding)

# Concatenate embeddings and write df out to CSV
embedding_matrix = np.array(all_embeddings)
embedding_matrix.shape # Validate expected shape
embeddings_df = pd.DataFrame(embedding_matrix)
embeddings_df['label'] = 'Virus Label' # Add label for virus if necessary
embeddings_df.to_csv('virus_embeddings_ext.csv', index=False)
            

