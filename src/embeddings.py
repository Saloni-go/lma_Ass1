"""Device detection, seeding and model loading (Bi-Encoder and Cross-Encoder)."""
import os
import random

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import BI_ENCODER_NAME, CROSS_ENCODER_NAME


def get_device():
    """Use CUDA automatically when available, otherwise fall back to CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_bi_encoder(device=None):
    return SentenceTransformer(BI_ENCODER_NAME, device=device or get_device())


def load_cross_encoder(device=None):
    return CrossEncoder(CROSS_ENCODER_NAME, device=device or get_device())


def embed_texts(model, texts, batch_size=32):
    """Encode texts into L2-normalised float32 vectors.

    With unit-length vectors, inner product == cosine similarity, which is
    what the FAISS IndexFlatIP search relies on.
    """
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    return emb
