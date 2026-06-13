"""
UCB Multi-Armed Bandit client selector for Federated Learning.

Replaces uniform-random client selection with UCB1-based selection that
prioritises clients contributing the most to improving macro recall,
while guaranteeing every client is explored at least once (cold-start).

Mathematical formulation:
    Score_k(t) = V_k_hat + c * sqrt(ln(t+1) / (N_k(t)+1))

    V_k_hat  : EMA of reward received from client k
    N_k(t)   : number of times client k has been selected by round t
    c        : exploration constant (default 1.0)

Reward:
    R_k = macro_recall_after - macro_recall_before
    Shared equally among all selected clients in the round.
"""

import math
import numpy as np


class UCBClientSelector:
    """UCB1 client selector with EMA reward estimates and cold-start guarantee."""

    def __init__(self, clients: list, c: float = 1.0, ema_alpha: float = 0.1):
        """
        Args:
            clients:   Full list of all available client identifiers.
            c:         UCB exploration constant. Higher = more exploration.
            ema_alpha: EMA smoothing factor for reward estimates (0 < alpha <= 1).
        """
        self.clients = list(clients)
        self.c = c
        self.ema_alpha = ema_alpha

        self.n_selected = {k: 0 for k in self.clients}   # N_k(t)
        self.value_est = {k: 0.0 for k in self.clients}  # V_k_hat
        self.last_macro_recall = 0.0
        self._cold_start_idx = 0  # tracks round-robin progress

    def _cold_start_done(self) -> bool:
        return self._cold_start_idx >= len(self.clients)

    def select(self, n: int, t: int) -> list:
        """
        Select n clients for round t.

        During the first len(clients) rounds, each client is selected
        exactly once in round-robin order (cold-start guarantee).
        After that, UCB1 scoring is used.

        Args:
            n: Number of clients to select.
            t: Current round index (0-based).

        Returns:
            List of n selected client identifiers.
        """
        if not self._cold_start_done():
            # Cold-start: ensure every client is seen once
            start = self._cold_start_idx
            selected = self.clients[start:start + n]
            # Wrap around if n pushes past the end
            if len(selected) < n:
                selected += self.clients[:n - len(selected)]
            return selected

        # UCB1 scoring
        scores = {}
        for k in self.clients:
            exploration = self.c * math.sqrt(
                math.log(t + 1) / (self.n_selected[k] + 1)
            )
            scores[k] = self.value_est[k] + exploration

        selected = sorted(self.clients, key=lambda k: scores[k], reverse=True)[:n]
        return selected

    def update(self, selected_clients: list, rewards: dict, macro_recall_after: float):
        """
        Update EMA estimates and selection counts after a round.

        Args:
            selected_clients:  Clients that participated in this round.
            rewards:           Dict mapping client_id -> scalar reward.
            macro_recall_after: Global macro recall after this round
                                (stored for next round's delta computation).
        """
        if not self._cold_start_done():
            self._cold_start_idx += len(selected_clients)

        for k in selected_clients:
            self.n_selected[k] += 1
            r = rewards.get(k, 0.0)
            # EMA update
            self.value_est[k] = (
                (1 - self.ema_alpha) * self.value_est[k] +
                self.ema_alpha * r
            )

        self.last_macro_recall = macro_recall_after

    def get_counts(self) -> dict:
        """Return selection counts — useful for logging and debugging."""
        return dict(self.n_selected)

    def get_state(self) -> dict:
        """Return serializable state for checkpointing."""
        return {
            'clients': self.clients,
            'c': self.c,
            'ema_alpha': self.ema_alpha,
            'n_selected': self.n_selected,
            'value_est': self.value_est,
            'last_macro_recall': self.last_macro_recall,
            '_cold_start_idx': self._cold_start_idx,
        }

    @classmethod
    def from_state(cls, state: dict) -> 'UCBClientSelector':
        """Restore a UCBClientSelector from a saved state dict."""
        obj = cls(state['clients'], c=state['c'], ema_alpha=state['ema_alpha'])
        obj.n_selected = state['n_selected']
        obj.value_est = state['value_est']
        obj.last_macro_recall = state['last_macro_recall']
        obj._cold_start_idx = state['_cold_start_idx']
        return obj
