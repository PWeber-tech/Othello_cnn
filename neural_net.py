import torch
import torch.nn as nn
import torch.nn.functional as F


class OthelloNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 8 * 8, 64)

        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Policy head
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        # Value head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


def select_move_from_net(net, state_tensor, legal_indices, epsilon=0.1):
    """
    Use the network's policy head to pick a move from legal_indices.
    epsilon: probability of picking a random legal move instead.
    """
    with torch.no_grad():
        log_probs, _ = net(state_tensor.unsqueeze(0))
        probs = torch.exp(log_probs).squeeze(0)

    mask = torch.zeros(64)
    for idx in legal_indices:
        mask[idx] = 1.0

    probs = probs * mask
    total = probs.sum()
    if total == 0:
        # Fallback: uniform over legal moves
        return random_move(legal_indices)
    probs = probs / total

    if torch.rand(1).item() < epsilon:
        return random_move(legal_indices)
    else:
        return torch.multinomial(probs, 1).item()


def random_move(legal_indices):
    import random
    return random.choice(legal_indices)


def train_step(net, optimizer, batch):
    """
    Run one gradient update on a batch of (state, move_index, value) tuples.
    Returns scalar loss value.
    """
    states, moves, values = zip(*batch)

    states = torch.stack(states)
    moves = torch.tensor(moves)
    values = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

    log_probs, predicted_values = net(states)

    policy_loss = F.nll_loss(log_probs, moves)
    value_loss = F.mse_loss(predicted_values, values)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
