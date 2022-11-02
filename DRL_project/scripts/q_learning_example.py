from __future__ import print_function

import sys
import getopt
import json
import io
import os
from tqdm import tqdm

import rl_env
from myAgents.epsilon_greedy import GreedyAgent

"""Some notes regarding this script:
    1) define an epsilon greedy agent that starts exploring using random states and then explores more common states
    2) define the tabular method for maintaining q values
    3) define update rules for q-values"""

class Trainer(object):
  """Runner class"""
  def __init__(self, flags):
    """Initialize runner."""
    self.flags = flags
    self.environment = rl_env.make('Hanabi-Very-Small', num_players=flags['players'])
    self.agent_config = {'players': flags['players'], 'information_tokens': 3}
    self.epsilon = 0.5
    self.agent_class = GreedyAgent
    self.gamma = 0.9
    self.lr = 0.1
    dirname = os.path.dirname(__file__)
    self.filename = os.path.join(dirname, 'tables/q_table.json')
    try:
        with open(self.filename, 'r') as fp:
            self.q_table = json.load(fp)
    except IOError:
        print('No old table found, starting a new table')
        self.q_table = {}
    
  def encode_state(self, observation):
    # The purpose of this function is to encode the state into a string that can be used as a key to the q_dict

    state= ''
    color = 'R'
    # life tokens [0,1]
    state += str(observation['life_tokens'])
    # information tokens [0,1,2,3]
    state += str(observation['information_tokens'])
    # fireworks [0,1,2,3,4]
    state += str(observation['fireworks'][color])
    # discard pile: list of discarded numbers
    discard_pile = ''
    for card in observation['discard_pile']:
        discard_pile += str(card['rank'])
    state += 'X'*(10 - len(discard_pile)) + ''.join(sorted(discard_pile))
    # current agent hand knowledge:
    hand_knowledge = ''
    for card in observation['card_knowledge'][0]:
        if card['rank'] == None:
            hand_knowledge += 'X'
            continue
        hand_knowledge += str(card['rank'])
    if len(hand_knowledge)<2:
        hand_knowledge += 'X'
    state += ''.join(sorted(hand_knowledge))
    # his hand
    hand = ''
    for card in observation['observed_hands'][-1]:
        hand += str(card['rank'])
    if len(hand) <2:
        hand += 'X'
    state += ''.join(sorted(hand))
    # his hand knowledge"
    hand_knowledge = ''
    for card in observation['card_knowledge'][1]:
        if card['rank'] == None:
            hand_knowledge += 'X'
            continue
        hand_knowledge += str(card['rank'])
    if len(hand_knowledge) <2:
        hand_knowledge += 'X'
    state += ''.join(sorted(hand_knowledge))
    
    # the length of state string should be 19
    if len(state) != 19:
        print(observation)
    assert len(state) == 19
    return state

  def find_q(self, state, action = None):

    if action == None: # Find over all actions
        max_value = 0
        for a in self.q_table[state]:
            max_value = max(a['value'], max_value)
        return max_value
    else:
        for a in self.q_table[state]:
            if a['action'] == action:
                return a['value']
        return 0

  def update_q_table(self, old_state, new_state, action, reward):
    
    old_q = self.find_q(old_state, action)
    new_q = self.find_q(new_state)
    q = old_q + self.lr *(reward + self.gamma*(new_q - old_q))

    for a in self.q_table[old_state]:
        if a['action'] == action:
            a['value'] = q

  def add_state_to_table(self, state, legal_moves):
    self.q_table[state] = []
    for a in legal_moves:
        self.q_table[state].append({'action': a, 'value': 0})

  
  def train(self):
    "Run episodes"
    rewards = []
    for episode in tqdm(range(self.flags['num_episodes'])):
        # Reset game
        observations = self.environment.reset()
        # Create agents
        agents = [self.agent_class(self.agent_config, self.epsilon) for _ in range(self.flags['players'])]
        done = False
        episode_reward = 0
        # q-parameters
        old_state = 'X'
        if old_state not in self.q_table:
            self.add_state_to_table(old_state, ['Start Game'])
        reward = 0
        action = 'Start Game'
        # Start game
        while not done:
            # At each turn, iterate through all agents
            for agent_id, agent in enumerate(agents):
                
                # Only proceed for current agent
                if observations['player_observations'][agent_id]['current_player_offset'] == 0:
                    # Find that agents observations
                    observation = observations['player_observations'][agent_id]
                    new_state = self.encode_state(observation)
                    if new_state not in self.q_table:
                        self.add_state_to_table(new_state, observation['legal_moves'])
                    self.update_q_table(old_state, new_state, action, reward)
                    action = agent.act(observation, new_state, self.q_table)
                    assert action is not None
                    observations, reward, done, unused_info = self.environment.step(action)
                    break

            old_state = new_state
            # Note when playing a correct card, reward is +1
            episode_reward += reward
        rewards.append(episode_reward)
        if episode%200000 == 0:
            print('Saving table')
            try:
                with open(self.filename, 'w') as fp:
                    json.dump(self.q_table, fp, sort_keys=True, indent=4)
            except:
                print('Could not save')

if __name__ == "__main__":

  flags = {'players': 2, 'num_episodes': 1000000, 'agent_class': 'RandomAgent'}
  options, arguments = getopt.getopt(sys.argv[1:], '',
                                     ['players=',
                                      'num_episodes=',
                                      'agent_class='])
  for flag, value in options:
    flag = flag[2:]  # Strip leading --.
    flags[flag] = type(flags[flag])(value)
  runner = Trainer(flags)
  runner.train()