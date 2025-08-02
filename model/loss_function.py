import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def knowledge_alignment_loss(emb1, emb2, temperature=1.0):
    emb1 = F.normalize(emb1, dim=1)  # [B, E]
    emb2 = F.normalize(emb2, dim=1)  # [B, E]

    logits = torch.matmul(emb1, emb2.T)  # Cosine similarity

    logits = logits / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    loss = (loss_i2t + loss_t2i) / 2
    return loss
