from collections import deque
import random
import operator

class ReplayMemory():
    # Constructor
    def __init__(self, maxlen=100_000) -> None:
        self.memory = deque([], maxlen=maxlen)

    # Add sample to memory
    def append(self, experience: tuple) -> None:
        self.memory.append(experience)

    # Get length of memory
    def __len__(self) -> int:
        return len(self.memory)

    # Get a random sample of transitions
    def sample(self, batch_size: int = 1):
        len_experience = len(self.memory)

        if len_experience == 0:
            raise RuntimeError("Replay Memory contains no experiences")

        indices = [None] * batch_size
        for i in range(batch_size):
            indices[i] = random.randrange(len_experience)

        getter = operator.itemgetter(*indices)
        return getter(self.memory)

    