import numpy as np
import multiprocessing as mp



class Node:
    def __init__(self, left, right, is_leaf: bool = False, idx = None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
        self.parent = None
        self.idx = idx  # this value is only set for leaf nodes
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf


def create_tree(input: list):
    nodes = [Node.create_leaf(v, i) for i, v in enumerate(input)]
    leaf_nodes = nodes
    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]

    return nodes[0], leaf_nodes


def retrieve(value: float, node: Node):
    if node.is_leaf:
        return node

    if node.left.value >= value:
        return retrieve(value, node.left)
    else:
        return retrieve(value - node.left.value, node.right)


def update(node: Node, new_value: float):
    change = new_value - node.value

    node.value = new_value
    propagate_changes(change, node.parent)


def propagate_changes(change: float, node: Node):
    node.value += change

    if node.parent is not None:
        propagate_changes(change, node.parent)


class PriorMemory:
    nodes_to_update = None
    is_prioritized = True

    def __init__(self, size, is_multiprocessing=False):
        self.pool = mp.Pool(8)
        self.size = size
        self.priorities = [0 for i in range(size + 1)]
        self.priorities[0] = 1
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        self.root, self.leaves = create_tree([0 for i in range(size)])
        self.is_multiprocessing = is_multiprocessing

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def append(self, element):

        self.data[self.end] = element
        max_prior = np.max(self.priorities)
        update(self.leaves[self.end], max_prior)
        self.end = (self.end + 1) % len(self.data)

        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def append_with_index(self, element, index, max_prior):
        self.data[index] = element
        update(self.leaves[index], max_prior)

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def add(self, element):
        self.append(element)

    def sample_batch(self, batch_size, current_beta):
        top_value = self.root.value
        nodes, elements, ISWeights = [], [], []
        randoms = np.random.uniform(0, top_value, batch_size)

        for rand in randoms:
            node = retrieve(rand, self.root)
            elements.append(self.data[node.idx])
            nodes.append(node)
            ISWeights.append(((1.0 / self.size) * (1 / (node.value / top_value))) ** current_beta)

        self.nodes_to_update = nodes

        return [elements, ISWeights]

    def update_nodes(self, new_values):
        if self.nodes_to_update is None:
            raise ValueError

        if self.is_multiprocessing:
            self.pool.starmap(update, zip(self.nodes_to_update, new_values))
            for i, node in enumerate(self.nodes_to_update):
                self.priorities[node.idx] = new_values[i]
        else:
            for i, node in enumerate(self.nodes_to_update):
                update(node, new_values[i])
                self.priorities[node.idx] = new_values[i]

        self.nodes_to_update = None

    def add_batch(self, new_values):

        was_full = self.end < self.start

        max_prior = np.max(self.priorities)
        max_array = [max_prior for i in range(len(new_values))]
        indexes = [(self.end + i) % self.size for i in range(len(new_values))]

        self.pool.starmap(self.append_with_index, zip(new_values, indexes, max_array))

        self.end = (self.end + len(new_values)) % len(self.data)

        if was_full:
            self.start = (self.end + 1) % len(self.data)

