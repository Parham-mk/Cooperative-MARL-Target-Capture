"""Small, shared helpers for tabular Q-learning updates."""


def q_learning_value(
    current_q: float,
    reward: float,
    best_next_q: float,
    learning_rate: float,
    gamma: float,
    terminated: bool = False,
) -> float:
    """Return one Q-learning update, with no bootstrap after termination."""
    bootstrap = 0.0 if terminated else gamma * best_next_q
    target = reward + bootstrap
    return current_q + learning_rate * (target - current_q)
