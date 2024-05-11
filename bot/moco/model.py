# -*- coding: utf-8 -*-
import os 
import numpy as np
import torchvision
import torch
import faiss
from moco.loader import SpectrogramDatasetInferenceSingle
from moco.builder import MoCoV3Inference


# ## Первая задача. Вычисление расстояний между композициями, и внутри одной композиции

def search_nearest_vectors(vectors, vector_to_search, k=5):
    
    num_vectors, length_vectors = vectors.shape
    
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    vector_to_search = vector_to_search / np.linalg.norm(vector_to_search, axis=1)
    
    index = faiss.IndexFlatIP(length_vectors)
    index.add(vectors)
    
    D, I = index.search(vector_to_search, k)
    return D, I

if __name__ == "__main__":
    metadata = {
        "experiment_path": '/app/pgrachev/moco/experiments/exp10',

        "dataset_params": {
            "datapath": '/app/pgrachev/data/tracks_wav',
            "sr": 44100,
            "n_fft": 2048,
            "hop_length": 512,
            "n_chunks": 1,
            "duration": 10,
            "overlap": 7,
        }
    }

    dataset = SpectrogramDatasetInferenceSingle(
        datapath=metadata['dataset_params']['datapath'],
        sr=metadata['dataset_params']['sr'],
        n_fft=metadata['dataset_params']['n_fft'],
        hop_length=metadata['dataset_params']['hop_length'],
        n_chunks=metadata['dataset_params']['n_chunks'],
        duration=metadata['dataset_params']['duration'],
        overlap=metadata['dataset_params']['overlap']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoCoV3Inference(base_encoder=torchvision.models.resnet50)

    checkpoint = torch.load("model_ep0030.pt", map_location=torch.device('cpu'))

    for key in list(checkpoint['model_state_dict'].keys()):
        if 'momentum_encoder' in key:
            del checkpoint['model_state_dict'][key]

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    idx = 161
    filename, chunks = dataset[idx]
    inputs = torch.tensor(chunks).unsqueeze(0).to(device)
    embedding = model(inputs).cpu().numpy()

    embedding_database = np.load('embeddings_10sec_clean.npz')
    emdeddings = np.array(list(embedding_database.values()))

    num_tracks, embs_per_track, emb_length = emdeddings.shape

    vectors = np.mean(emdeddings, axis=1)
    D, I = search_nearest_vectors(vectors=vectors, vector_to_search=embedding, k=5)
