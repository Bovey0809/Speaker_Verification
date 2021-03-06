import torch
from torch.nn import functional as F
from time import time


def get_centroids(embeddings):
    centroids = embeddings.mean(dim=1)
    return centroids


def get_utterance_centroids(embeddings):
    """
    Returns the centroids for each utterance of a speaker, where
    the utterance centroid is the speaker centroid without considering
    this utterance

    Shape of embeddings should be:
    (speaker_ct, utterance_per_speaker_ct, embedding_size)
    """
    sum_centroids = embeddings.sum(dim=1)
    sum_centroids = sum_centroids.unsqueeze(1)
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids


def get_cossim(embeddings, centroids):
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)
    utterance_centroids_flat = utterance_centroids.flatten(0, 1)
    embeddings_flat = embeddings.flatten(0, 1)
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)
    centroids_expand = centroids.repeat(
        (num_utterances * embeddings.shape[0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(
        1).repeat(1, embeddings.shape[0], 1).flatten(0, 1)
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(embeddings.size(
        0), num_utterances, centroids.size(0))
    same_idx = list(range(embeddings.size(0)))
    cos_diff[same_idx, :, same_idx] = cos_same.view(
        embeddings.shape[0], num_utterances)
    cos_diff = cos_diff + 1e-6
    return cos_diff


def calc_loss(sim_matrix):
    same_idx = list(range(sim_matrix.size(0)))
    pos = sim_matrix[same_idx, :, same_idx]
    neg = (torch.exp(sim_matrix).sum(dim=2) + 1e-6).log_()
    per_embedding_loss = -1 * (pos - neg)
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss

# add timing for train


def logging(f):
    def _f(**kargs):
        print(f"PARMS: {kargs}")
        current = time()
        result = f(**kargs)
        print(f"TIME: {(time() - current):.3f} seconds")
        return result
    return _f


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_acc(sim_matrix, N, M):
    predicted = torch.argmax(sim_matrix, 2)
    groud_truth = torch.arange(N, device=sim_matrix.device).view(N, 1)
    acc = torch.sum(torch.eq(predicted, groud_truth), dtype=torch.float)/N/M
    return acc


def get_eer(embeddings, N, M):
    enroll, verify = torch.split(embeddings, M//2, dim=1)
    enroll = enroll.mean(1)
    sim_matrix = get_cossim(verify, enroll)
    diff = 1
    EER = 0
    EER_thresh = 0
    EER_FAR = 0
    EER_FRR = 0
    thresholds = [0.01*i+0.5 for i in range(50)]
    for thres in thresholds:
        sim_matrix_thresh = (sim_matrix > thres).to(torch.float)
        FAR = (sum([sim_matrix_thresh[i].sum()-sim_matrix_thresh[i, :, i].sum()
                    for i in range(N)]) / (N-1.0)/(M/2)/N)
        FRR = (sum([M/2-sim_matrix_thresh[i, :, i].sum()
                    for i in range(N)]) / (M/2)/N)

        if diff > abs(FAR-FRR):
            diff = abs(FAR-FRR)
            EER = (FAR+FRR)/2
            EER_thresh = thres
            EER_FAR = FAR
            EER_FRR = FRR
    return EER_thresh, EER, EER_FAR, EER_FRR

