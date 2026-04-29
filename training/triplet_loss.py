import torch
import torch.nn as nn


class SemiHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(SemiHardTripletLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, anchor, positive, negative):
        """
        Calculates Triplet Loss.
        Basic semi-hard mining concept: We focus on triplets where the negative
        is closer to the anchor than the positive + margin, but not necessarily
        closer than the positive (which would be a "hard" negative).
        """
        # Distance between anchor and positive
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        # Distance between anchor and negative
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)

        # Loss: ||a - p||^2 - ||a - n||^2 + margin
        losses = self.relu(pos_dist - neg_dist + self.margin)

        # Semi-hard mask: cases where negative is further than positive, but within margin
        # (Though often standard TripletLoss naturally handles this by zeroing out easy ones via ReLU)
        semi_hard_mask = (neg_dist > pos_dist) & (neg_dist < pos_dist + self.margin)

        if semi_hard_mask.sum() > 0:
            # Average loss over semi-hard triplets if they exist
            return losses[semi_hard_mask].mean()
        else:
            # Fallback to standard mean
            return losses.mean()