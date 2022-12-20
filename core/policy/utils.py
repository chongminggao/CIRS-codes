# -*- coding: utf-8 -*-

import numpy as np
import torch


def get_recommended_ids(buffer):
    if len(buffer) == 0:
        recommended_ids = None
    else:
        indices = buffer.last_index[~buffer[buffer.last_index].done]

        # is_alive = True
        recommended_ids = np.zeros([0, len(indices)], dtype=int)
        while True:
            acts = buffer.act[indices]
            recommended_ids = np.vstack([recommended_ids, acts])

            if all(indices == buffer.prev(indices)):
                break
            assert all(indices != buffer.prev(indices))

            indices = buffer.prev(indices)

        recommended_ids = recommended_ids.T

    return recommended_ids


def removed_recommended_id_from_embedding(logits, recommended_ids):
    """
    :param logits: Batch * Num_all_items
    :param recommended_ids: Batch * Num_removed
    :return:
    :rtype:
    """

    num_batch, num_action = logits.shape

    indices = np.expand_dims(np.arange(num_action), 0).repeat(num_batch, axis=0)
    indices_torch = torch.from_numpy(indices).to(logits.device)

    if recommended_ids is None:
        return logits, indices_torch

    # assert all(recommended_ids[:, -1] == num_action)
    # recommended_ids_valid = recommended_ids[:, :-1]
    # recommended_ids_valid_torch = torch.LongTensor(recommended_ids_valid).to(device=logits.device)

    recommended_ids_valid_torch = torch.LongTensor(recommended_ids).to(device=logits.device)

    mask = torch.ones_like(logits, dtype=torch.bool)
    mask_valid = mask.scatter(1, recommended_ids_valid_torch, 0)

    logits_masked = logits.masked_select(mask_valid).reshape(num_batch, -1)
    indices_masked = indices_torch.masked_select(mask_valid).reshape(num_batch, -1)

    return logits_masked, indices_masked
