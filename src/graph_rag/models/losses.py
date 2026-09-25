import torch


def multi_positive_info_nce(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """
    InfoNCE with any number of positives per row:
        loss_i = -log( sum_{j in P_i} exp(s_ij) / sum_j exp(s_ij) )

    Every document that is relevant to a query counts as a positive, so other gold
    documents in the batch are never pushed away as negatives. Set logits of padded
    columns to -inf before calling. Rows without positives are ignored.
    """
    positive_mask = positive_mask.bool()
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return logits.sum() * 0.0
    log_denominator = torch.logsumexp(logits, dim=1)
    log_numerator = torch.logsumexp(logits.masked_fill(~positive_mask, float("-inf")), dim=1)
    return (log_denominator - log_numerator)[valid].mean()
