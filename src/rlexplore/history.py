from collections import deque
import numpy as np
import torch


class History:
    """Bounded, episode-local observations. PPO replays complete context windows."""
    def __init__(self, length):
        self.length = length
        self.frames = deque(maxlen=length)

    def reset(self, observation):
        self.frames.clear()
        self.append(observation)

    def append(self, observation):
        self.frames.append({k: np.array(v, copy=True) for k, v in observation.items()})

    def snapshot(self):
        return list(self.frames)


def batch_histories(histories, device):
    """Right-pad on demand; contexts retain all tokens used to sample each action."""
    lengths = [len(h) for h in histories]
    maxlen = max(lengths)
    obs = {}
    for key in histories[0][0]:
        shape = histories[0][0][key].shape
        array = np.zeros((len(histories), maxlen, *shape), dtype=histories[0][0][key].dtype)
        for i, h in enumerate(histories):
            for j, frame in enumerate(h):
                array[i, j] = frame[key]
        obs[key] = torch.as_tensor(array, device=device)
    valid = torch.arange(maxlen, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    return obs, valid
