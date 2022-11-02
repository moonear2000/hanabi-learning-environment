import random
from hanabi_learning_environment.rl_env import Agent


class GreedyAgent(Agent):
  """Agent that takes random legal actions."""

  def __init__(self, config, epsilon=0, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    # epsilon determines how much exploration is done
    self.epsilon = epsilon

  def act(self, observation, encoded_state, q_table):
    """Act based on an observation, and q_table."""
    # Explore with probability epsilon
    if random.uniform(0,1)<self.epsilon:
        random_action = random.choice(q_table[encoded_state])
        return random_action['action']
    else:
        best_action = {}
        best_value = -1
        for action in q_table[encoded_state]:
            if action['value']>best_value:
                best_action = action['action']
                best_value = action['value']
        return best_action