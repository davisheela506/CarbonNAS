import random


CHANNEL_OPTIONS = [
    (16, 32),
    (32, 64),
    (32, 64, 128),
    (16, 32, 64),
]

KERNEL_OPTIONS = [3, 5]

POOLING_OPTIONS = [True, False]


def sample_architecture():
    """Randomly sample one CNN architecture."""

    return {
        "channels": random.choice(CHANNEL_OPTIONS),
        "kernel_size": random.choice(KERNEL_OPTIONS),
        "use_pooling": random.choice(POOLING_OPTIONS),
    }