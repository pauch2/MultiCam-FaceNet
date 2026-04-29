"""
arcface_loss.py
---------------
ArcFace (Additive Angular Margin Loss) for face recognition.

Paper: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
       Deng et al., CVPR 2019  https://arxiv.org/abs/1801.07698

How it works
------------
Standard softmax computes logits as W·x (dot product).
ArcFace replaces this with cos(θ + m) for the ground-truth class,
where θ is the angle between the feature and the class weight vector
and m is the additive angular margin (typically 0.5 rad ≈ 28.6°).

This forces the embedding to be at least `m` radians closer (in angle)
to its own class centre than to any other class centre — much tighter
than standard cross-entropy.

Usage in training
-----------------
    head = ArcFaceHead(embedding_dim=768, num_classes=N, margin=0.5, scale=64)
    ...
    emb   = backbone(x)              # L2-normalised (D,)
    logits = head(emb, labels)       # (B, N)
    loss   = F.cross_entropy(logits, labels)

The head is ONLY used during training.  At inference, just use the
L2-normalised backbone output as the embedding — same as with Triplet Loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """
    ArcFace classification head.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the L2-normalised input embedding (e.g. 768 for Swin-T).
    num_classes : int
        Number of identity classes in the training set.
    margin : float
        Additive angular margin m (radians).  Default 0.5 (≈ 28.6°).
        Common range: 0.3 – 0.5.  Larger = stricter, slower convergence.
    scale : float
        Feature scale s.  Default 64.  Should be ≥ 30 for stable training.
        Rule of thumb: s ≈ sqrt(2) * log((num_classes - 1) / num_classes * (e^2 + 1)).
        For VGGFace2 (8,631 classes): s ≈ 64.
    easy_margin : bool
        If True, use easy-margin variant (clamps cos(θ+m) to avoid numerical
        instability when θ is close to π).  Recommended for small datasets.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes   = num_classes
        self.scale         = scale
        self.margin        = margin
        self.easy_margin   = easy_margin

        # Learnable class weight matrix — each row is a class centre
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute trig constants
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)   # threshold for easy margin
        self.mm    = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : (B, D)  L2-normalised feature vectors
        labels     : (B,)    integer class indices

        Returns
        -------
        logits : (B, num_classes)  scaled cosine logits with margin on gt class
        """
        # Normalise weight matrix row-wise → unit class centres
        W = F.normalize(self.weight, p=2, dim=1)

        # cos(θ_j) for all classes j: (B, C)
        cos_theta = F.linear(embeddings, W)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)  # numerical safety

        # sin(θ) via identity sin²+cos²=1
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)

        # cos(θ + m) = cos θ cos m − sin θ sin m
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            # Only apply margin where cos(θ) > 0  (i.e. θ < π/2)
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            # Clip to avoid the spurious minimum near θ = π
            cos_theta_m = torch.where(
                cos_theta > self.th,
                cos_theta_m,
                cos_theta - self.mm,
            )

        # Build one-hot mask for ground-truth class
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Replace gt-class cosine with the margined version
        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta

        # Scale up (counteracts the unit-sphere constraint shrinking logits)
        return logits * self.scale

    def extra_repr(self) -> str:
        return (f"embedding_dim={self.embedding_dim}, num_classes={self.num_classes}, "
                f"margin={self.margin}, scale={self.scale}")