from transformers import Wav2Vec2Model, Wav2Vec2Processor, WavLMModel, AutoModel, AutoProcessor
import torch
from sklearn.cluster import MiniBatchKMeans
import pickle
from einops import rearrange
import gc
from tqdm import tqdm
from newdataset import DatasetNoCrop
import torch.nn.functional as F

def extract_emb(data, model, processor=None, modeltype='w2v'):
    data = rearrange(data, 'b l -> (b l)')
    model.eval()
    if modeltype == 'w2v':
        input_values = processor(data, return_tensors="pt", padding="longest", sampling_rate=16000).input_values
        with torch.no_grad():
            emb = model(input_values.to('cuda'), output_hidden_states=True)
        out = emb.hidden_states[24]
    
    elif modeltype == 'wavlm':
        input_values = processor(data, return_tensors="pt", padding="longest", sampling_rate=16000).input_values
        with torch.no_grad():
            emb = model(input_values.to('cuda'), output_hidden_states=True)
        out = emb.hidden_states[22]
        
    else:
        raise TypeError()
    
    return out

def run_kmeans_and_save(embeddings, model_name, n_clusters_list):
    for n_clusters in n_clusters_list:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024, verbose=0)
        num_batches = len(embeddings) // 1024 + 1
        for i in tqdm(range(num_batches), desc=f"KMeans Clustering for {n_clusters} clusters"):
            start = i * 1024
            end = (i + 1) * 1024
            kmeans.partial_fit(embeddings[start:end].numpy())
        
        labels = kmeans.labels_

        print(f"Labels for {n_clusters} clusters:", labels.shape)

        with open(f"kmeans_modelweight_{n_clusters}_{model_name}.pkl", 'wb') as file:
            pickle.dump(kmeans, file)

# SSL model configuration
model_configs = {
    'w2v': {
        'model_id': "facebook/wav2vec2-large-960h-lv60-self",
        'model': None,
        'processor': None,
        'modeltype': 'w2v'
    },
    'wavlm': {
        'model_id': "patrickvonplaten/wavlm-libri-clean-100h-large",
        'model': None,
        'processor': None,
        'modeltype': 'wavlm'
    }
}

# Initialize models based on user selection
selected_models = ['wavlm']  # Change this list to select models, e.g., ['w2v'] or ['wavlm'] or both

for model_key in selected_models:
    model_configs[model_key]['model'] = AutoModel.from_pretrained(model_configs[model_key]['model_id'])
    model_configs[model_key]['processor'] = AutoProcessor.from_pretrained(model_configs[model_key]['model_id'])
    model_configs[model_key]['model'].to('cuda')

# Example usage with Dataset
val_path_nb = "/ssd2/woongzip/Datasets/recon_mpeg/MP3_sr16_train_16kbps/train"
val_path_wb = "/ssd2/woongzip/Datasets/VCTK_trim/16Khz/train"
valid_dataset = DatasetNoCrop(path_dir_nb=val_path_nb, path_dir_wb=val_path_wb, iscodec=True)  # Assuming CustomDataset is defined as before

for model_key in selected_models:
    model = model_configs[model_key]['model']
    processor = model_configs[model_key]['processor']
    modeltype = model_configs[model_key]['modeltype']
    
    embs = []
    sample_rate = 16000
    chunk_duration = 10  # 5 seconds
    chunk_length = sample_rate * chunk_duration

    for idx in tqdm(range(len(valid_dataset)), desc=f"Extracting embeddings for {model_key}"):
        try:
            wav_nb, wav_wb, filename = valid_dataset[idx]
        except RuntimeError as e:
            print(f"Error loading audio for index {idx} {filename}: {e}")
            continue

        for start in range(0, wav_wb.shape[1], chunk_length):
            end = start + chunk_length
            chunk = wav_wb[:, start:end]
            if chunk.shape[1] < chunk_length:
                chunk = F.pad(chunk, (0, chunk_length - chunk.shape[1]))
            
            emb = extract_emb(chunk, model, processor, modeltype)
            embs.append(emb.detach().cpu())
    
    data_tensor = torch.cat(embs, dim=0)
    print(f"{model_key} data tensor shape:", data_tensor.shape)

    embeddings = data_tensor.reshape(-1, 1024)
    print(f"{model_key} embeddings shape:", embeddings.shape)

    run_kmeans_and_save(embeddings, model_key, [8, 16, 32, 64])
