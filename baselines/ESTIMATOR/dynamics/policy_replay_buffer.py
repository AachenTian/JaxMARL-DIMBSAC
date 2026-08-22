import numpy as np


class PolicyDynamicsReplayBuffer:
    """CPU replay buffer for real policy-environment transitions.

    One stored row contains the physical transition for all agents:
        x:       [A, 4]
        actions: [A, 5]
        next_x:  [A, 4]

    Terminal/reset jumps are filtered before insertion by the caller.
    """

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        seed: int,
    ):
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(int(seed))

        self.x = np.zeros(
            (self.capacity, self.num_agents, self.state_dim),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (self.capacity, self.num_agents, self.action_dim),
            dtype=np.float32,
        )
        self.next_x = np.zeros_like(self.x)

        self.size = 0
        self.write_index = 0
        self.total_added = 0

    def __len__(self):
        return self.size

    def add(self, x, actions, next_x):
        x = np.asarray(x, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        next_x = np.asarray(next_x, dtype=np.float32)

        if x.ndim != 3:
            raise ValueError(f"x must be [B,A,4], got {x.shape}")
        if actions.ndim != 3:
            raise ValueError(f"actions must be [B,A,5], got {actions.shape}")
        if next_x.shape != x.shape:
            raise ValueError(
                f"next_x shape {next_x.shape} does not match x {x.shape}"
            )

        batch_size = x.shape[0]
        if batch_size == 0:
            return

        # If a single insertion is larger than capacity, keep only the newest
        # transitions because older rows would be overwritten anyway.
        if batch_size >= self.capacity:
            x = x[-self.capacity:]
            actions = actions[-self.capacity:]
            next_x = next_x[-self.capacity:]
            batch_size = self.capacity

        first = min(batch_size, self.capacity - self.write_index)
        second = batch_size - first

        end = self.write_index + first
        self.x[self.write_index:end] = x[:first]
        self.actions[self.write_index:end] = actions[:first]
        self.next_x[self.write_index:end] = next_x[:first]

        if second > 0:
            self.x[:second] = x[first:]
            self.actions[:second] = actions[first:]
            self.next_x[:second] = next_x[first:]

        self.write_index = (self.write_index + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)
        self.total_added += batch_size

    def sample(self, batch_size: int):
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty dynamics replay buffer.")

        indices = self.rng.integers(
            0,
            self.size,
            size=int(batch_size),
        )
        return (
            self.x[indices],
            self.actions[indices],
            self.next_x[indices],
        )

    def sample_without_replacement(self, sample_size: int):
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty dynamics replay buffer.")

        sample_size = min(int(sample_size), self.size)
        indices = self.rng.choice(
            self.size,
            size=sample_size,
            replace=False,
        )
        return (
            self.x[indices],
            self.actions[indices],
            self.next_x[indices],
        )
