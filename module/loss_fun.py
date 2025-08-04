import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from kmeans_pytorch import kmeans

def sce_loss(x, y, beta=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(beta)

    loss = loss.mean()
    # to balance the loss
    return 10 * loss


def target_distribution(q) :
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def soft_assignment(device, embeddings, n_clusters, cluster_index, alpha=1):

    cluster_centers = [embeddings[cluster_index_].mean(dim=0) for cluster_index_ in cluster_index]
    cluster_centers = torch.stack(cluster_centers, dim=0)

    cluster_layer = Parameter(torch.Tensor(n_clusters, 64))

    cluster_layer.data = torch.tensor(cluster_centers).to(device)
    q = 1.0 / (1.0 + torch.sum(torch.pow(embeddings.unsqueeze(1) - cluster_layer, 2), 2) / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()

    return q

def KL_clustering(device, z_list, num_classes, dominant_index):

    pseudo_label, _ = kmeans(X=z_list[dominant_index], num_clusters=num_classes, distance='euclidean', device=device)
    cluster_index = [torch.nonzero(pseudo_label == j).flatten() for j in range(int(pseudo_label.max()))]
    clu_loss = 0
    for i in range(len(z_list)):
        qz = soft_assignment(device, z_list[i], num_classes, cluster_index, alpha=1)
        pz = target_distribution(qz)

        clu_loss += F.kl_div(qz.log(), pz, reduction='mean')

    clu_loss = clu_loss / len(z_list)

    z = torch.cat(z_list, dim=1)
    z = F.normalize(z, dim=1, p=2)
    pseudo_label, _ = kmeans(X=z, num_clusters=num_classes, distance='euclidean', device=device)
    cluster_index = [torch.nonzero(pseudo_label == j).flatten() for j in range(int(pseudo_label.max()))]
    q = soft_assignment(device, z, num_classes, cluster_index, alpha=1)
    p = target_distribution(q)
    clu_loss += F.kl_div(q.log(), p, reduction='mean')

    return clu_loss

