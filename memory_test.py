import torch
import numpy as np
import time


start = time.time()

SIZE = 262144 * 2

BATCH = 300

data = np.ones(shape=SIZE)

tensor_data = torch.Tensor(data)

total = torch.sum(tensor_data)

probs = tensor_data / total


indexes = [i for i in range(SIZE)]

choices = np.random.choice(indexes, BATCH, p=probs)

end = time.time()

print(end - start)