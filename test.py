import torch
from torch.nn import functional as F


def get_cossim(embeddings, centroids):
    num_utterances = embeddings.shape[1]
    # Special version of getting centroids, read in the paper.
    utterance_centroids = get_utterance_centroids(embeddings)
    utterance_centroids_flat = utterance_centroids.flatten(0, 1)
    embeddings_flat = embeddings.flatten(0, 1)
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)

    # now we get the cosine distance between each utterance and the other speakers'
    # centroids
    # to do so requires comparing each utterance to each centroid. To keep the
    # operation fast, we vectorize by using matrices L (embeddings) and
    # R (centroids) where L has each utterance repeated sequentially for all
    # comparisons and R has the entire centroids frame repeated for each utterance
    centroids_expand = centroids.repeat(num_utterances * embeddings.shape[0], 1)
    embeddings_expand = embeddings_flat.unsqueeze(
        1).repeat(1, embeddings.shape[0], 1)
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1], embeddings_expand.shape[-1])
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(embeddings.size(
        0), num_utterances, centroids.size(0))
    # assign the cosine distance for same speakers to the proper idx
    same_idx = list(range(embeddings.size(0)))
    cos_diff[same_idx, :, same_idx] = cos_same.view(
        embeddings.shape[0], num_utterances)
    cos_diff = cos_diff + 1e-6
    return cos_diff


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
    这里(N, M, proj)->(N, k, proj)
    意思是对于每一个人(N), 求除了本身之外的utt的均值.
    例如, ouput.shape == (4, 3, 64), 那么一个对于output[0][0], 就是除了本身之外的另外两个的均值.
    """
    sum_centroids = embeddings.sum(dim=1)
    # we want to subtract out each utterance, prior to calculating the
    # the utterance centroid
    sum_centroids = sum_centroids.unsqueeze(1)
    # print("sumcent", sum_centroids.shape)
    # we want the mean but not including the utterance itself, so -1
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids


def calc_loss(sim_matrix):
    same_idx = list(range(sim_matrix.size(0)))
    pos = sim_matrix[same_idx, :, same_idx]
    neg = (torch.exp(sim_matrix).sum(dim=2) + 1e-6).log_()
    per_embedding_loss = -1 * (pos - neg)
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss


emb = torch.arange(120).reshape(4, 3, 10).float()
cen = emb.mean(1)
get_cossim(emb, cen)
