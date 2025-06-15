import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque, namedtuple

from game.players import BasePokerPlayer
from game.engine.card import Card
from game.engine.hand_evaluator import HandEvaluator

# --- 1. Hyperparameters ---
# Training parameters
GAMMA = 0.99  # Discount factor for rewards
BATCH_SIZE = 128  # Batch size for experience replay
REPLAY_MEMORY_SIZE = 100000  # Maximum capacity of the replay memory buffer
TARGET_UPDATE_FREQUENCY = 1000  # How often to update the target network
LEARNING_RATE = 0.0001  # Learning rate for the optimizer

# Epsilon-Greedy policy parameters (Exploration/Exploitation)
EPSILON_START = 0.9  # Initial value of epsilon
EPSILON_END = 0.05  # Final value of epsilon
EPSILON_DECAY = 20000  # Slower epsilon decay rate

# AI Action Space Definition
ACTION_SPACE_SIZE = 5  # { Fold, Call, Raise 0.5*Pot, Raise 1.0*Pot, All-in }

# State Vector Dimension
STATE_SPACE_SIZE = 14

# Model Save Path
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepq_model.pth")

# Define a single transition record for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# --- 2. Replay Memory Buffer ---
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Samples a batch of transitions randomly"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- 3. Q-Network Model (DQN) ---
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.layer3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.layer4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.layer5 = nn.Linear(256, n_actions)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # BatchNorm1d and Dropout work correctly on batches.
        # When batch size is 1 during action selection, model should be in eval() mode.
        x = self.dropout(F.leaky_relu(self.bn1(self.layer1(x))))
        x = self.dropout(F.leaky_relu(self.bn2(self.layer2(x))))
        x = self.dropout(F.leaky_relu(self.bn3(self.layer3(x))))
        x = self.dropout(F.leaky_relu(self.bn4(self.layer4(x))))
        return self.layer5(x)

# --- 4. DeepQPlayer Agent ---
class DeepQPlayer(BasePokerPlayer):

    def __init__(self):
        super().__init__()
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize policy and target networks
        self.policy_net = DQN(STATE_SPACE_SIZE, ACTION_SPACE_SIZE).to(self.device)
        self.target_net = DQN(STATE_SPACE_SIZE, ACTION_SPACE_SIZE).to(self.device)
        self._load_model()  # Attempt to load a pre-trained model
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not in training mode

        # Initialize optimizer and replay memory
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)

        # Internal state tracking
        self.steps_done = 0
        self.last_state = None
        self.last_action_idx = None
        self.my_stack_at_last_action = 0
        self.initial_stack = 1000 # Initial stack for the game
        self.max_round = 0
        
        # Define a mapping for hand strengths
        self.hand_strength_map = {
            "HIGHCARD": 0, "ONEPAIR": 1, "TWOPAIR": 2, "THREECARD": 3,
            "STRAIGHT": 4, "FLASH": 5, "FULLHOUSE": 6, "FOURCARD": 7, "STRAIGHTFLASH": 8
        }

    def _get_state(self, hole_card, round_state, valid_actions):
        """Extracts a feature vector from the game state"""
        # 1. Hand Strength (Pre-flop)
        hole_cards = [Card.from_str(c) for c in hole_card]
        preflop_strength = self._calculate_hole_card_strength(hole_cards) / 10.0

        # 2. Hand Strength (Post-flop) and Draw Potential
        community_cards = [Card.from_str(c) for c in round_state["community_card"]]
        outs = 0.0
        is_strong_draw = 0.0
        if community_cards:
            hand_info = HandEvaluator.gen_hand_rank_info(hole_cards, community_cards)
            postflop_strength = self.hand_strength_map[hand_info["hand"]["strength"]] / 8.0
            # Calculate draw potential
            outs_count, strong_draw_bool = self._calculate_draw_potential(hole_cards, community_cards)
            outs = outs_count / 10.0 # Normalize (e.g., max outs for flush + straight draw can be 15)
            is_strong_draw = 1.0 if strong_draw_bool else 0.0
        else:
            postflop_strength = 0.0

        # 3. Game Stage (Street)
        street = round_state['street']
        street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        street_one_hot = [0] * 4
        if street in street_map:
            street_one_hot[street_map[street]] = 1
        
        # 4. Stack and Pot Information
        my_seat = next(s for s in round_state['seats'] if s['uuid'] == self.uuid)
        opp_seat = next(s for s in round_state['seats'] if s['uuid'] != self.uuid)
        my_stack = my_seat['stack'] / self.initial_stack
        opp_stack = opp_seat['stack'] / self.initial_stack
        pot_size = round_state['pot']['main']['amount']
        
        # 5. Cost to Call and Pot Odds
        call_amount = valid_actions[1]['amount']
        # Avoid division by zero
        amount_to_call_ratio = call_amount / (my_seat['stack'] + 1e-6)
        pot_odds = call_amount / (pot_size + call_amount + 1e-6) if (pot_size + call_amount) > 0 else 0.0
        
        # 6. Positional Information
        is_sb = round_state['seats'][round_state['dealer_btn']]['uuid'] == self.uuid
        position = 1.0 if is_sb else 0.0

        # Normalize pot_size
        pot_size_normalized = pot_size / self.initial_stack

        state = np.array([
            preflop_strength, postflop_strength, my_stack, opp_stack, pot_size_normalized,
            amount_to_call_ratio, position, pot_odds, outs, is_strong_draw
        ] + street_one_hot)
        
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _select_action(self, state, valid_actions):
        """Selects a discrete action based on the Epsilon-Greedy policy"""
        sample = random.random()
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        np.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                self.policy_net.eval()  # Switch to evaluation mode for inference with BatchNorm and Dropout
                # Exploitation: select the action with the highest Q-value
                action_q_values = self.policy_net(state)
                self.policy_net.train() # Switch back to training mode
                # Filter out invalid actions
                for i in range(ACTION_SPACE_SIZE):
                    if not self._is_action_valid(i, valid_actions):
                        action_q_values[0][i] = -float('inf')
                return action_q_values.max(1)[1].view(1, 1)
        else:
            # Exploration: select a random valid action
            valid_action_indices = [i for i in range(ACTION_SPACE_SIZE) if self._is_action_valid(i, valid_actions)]
            chosen_action = random.choice(valid_action_indices) if valid_action_indices else 1 # Default to call
            return torch.tensor([[chosen_action]], device=self.device, dtype=torch.long)

    def declare_action(self, valid_actions, hole_card, round_state):
        current_state = self._get_state(hole_card, round_state, valid_actions)
        my_current_stack = next(s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid)

        # If this is not the start of a decision-making sequence, calculate reward and store the last transition
        if self.last_state is not None:
            reward = (my_current_stack - self.my_stack_at_last_action) / self.initial_stack
            reward_tensor = torch.tensor([reward], device=self.device)
            self.memory.push(self.last_state, self.last_action_idx, reward_tensor, current_state, torch.tensor([False], device=self.device))
            self._optimize_model()

        # Select and execute an action
        action_idx_tensor = self._select_action(current_state, valid_actions)
        action_idx = action_idx_tensor.item()
        
        # Record the current state and action for the next callback
        self.last_state = current_state
        self.last_action_idx = action_idx_tensor
        self.my_stack_at_last_action = my_current_stack

        # Map the discrete action to a game engine-compatible action
        return self._map_action_to_game(action_idx, valid_actions, round_state)

    def receive_round_result_message(self, winners, hand_info, round_state):
        my_final_stack = next(s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid)
        
        # Calculate the reward for the final decision in the round
        if self.last_state is not None:
            reward = (my_final_stack - self.my_stack_at_last_action) / self.initial_stack
            
            # If it's the last round, add a terminal reward for winning/losing the game
            if round_state['round_count'] == self.max_round:
                opp_final_stack = next(s['stack'] for s in round_state['seats'] if s['uuid'] != self.uuid)
                if my_final_stack > opp_final_stack:
                    reward += 1.0 # Large reward for winning the entire game
                elif my_final_stack < opp_final_stack:
                    reward -= 1.0 # Large penalty for losing the game
                
                # self._save_model() # TODO:Save the model at the end of the game

            reward_tensor = torch.tensor([reward], device=self.device)
            # The "next_state" for this transition is None because the round ended
            self.memory.push(self.last_state, self.last_action_idx, reward_tensor, None, torch.tensor([True], device=self.device))
            self._optimize_model()

        # Reset state for the next round
        self.last_state = None
        self.last_action_idx = None

    def _optimize_model(self):
        """Samples from replay memory and trains the network"""
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Calculate Q(s,a) for the current state
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Calculate V(s') for the next state
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            # Double DQN: use policy_net to select the best action, and target_net to evaluate its value
            next_state_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_state_actions).squeeze(1)
        
        # Calculate the expected Q-values (Bellman equation)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute the loss (Smooth L1 Loss / Huber Loss)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) # Gradient clipping
        self.optimizer.step()

        # Periodically update the target network
        if self.steps_done % TARGET_UPDATE_FREQUENCY == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # --- Helper Functions ---
    def _is_action_valid(self, action_idx, valid_actions):
        """Checks if a discrete action is valid in the current game context"""
        if action_idx == 0: # Fold
            return True # Fold is always possible if it's our turn
        if action_idx == 1: # Call
            return True # Call/Check is always valid
        if action_idx >= 2: # Raise
            return valid_actions[2]['amount']['min'] != -1
        return False
        
    def _map_action_to_game(self, action_idx, valid_actions, round_state):
        """Maps a discrete action to the game engine's action format"""
        fold_action, call_action, raise_action = valid_actions[0], valid_actions[1], valid_actions[2]
        pot_amount = round_state['pot']['main']['amount']

        if action_idx == 0: # Fold
            return fold_action['action'], fold_action['amount']
        
        if action_idx == 1: # Call
            return call_action['action'], call_action['amount']
        
        if action_idx == 2: # Raise 0.5 * Pot
            raise_amount = int(pot_amount * 0.5)
        
        elif action_idx == 3: # Raise 1.0 * Pot
            raise_amount = int(pot_amount * 1.0)
        
        elif action_idx == 4: # All-in
            raise_amount = raise_action['amount']['max']
        
        # Ensure the bet amount is within the valid range
        if raise_action['amount']['min'] != -1:
            final_amount = max(raise_action['amount']['min'], raise_amount)
            final_amount = min(raise_action['amount']['max'], final_amount)
            return raise_action['action'], final_amount
        else:
            # If raising is not possible, default to calling
            return call_action['action'], call_action['amount']

    def _calculate_hole_card_strength(self, hole_cards):
        """Pre-flop hand strength evaluation function, adapted from expert_player"""
        card1, card2 = hole_cards[0], hole_cards[1]
        rank1, rank2 = sorted([c.rank for c in hole_cards], reverse=True)
        is_pair = rank1 == rank2
        is_suited = card1.suit == card2.suit

        if is_pair:
            if rank1 >= 11: return 10  # JJ+
            if rank1 >= 8: return 8   # TT, 99, 88
            return 5
        if rank1 == 14:  # Ace
            if rank2 >= 13: return 9  # AK
            if rank2 >= 10: return 7 if is_suited else 6
            if is_suited: return 4
        if rank1 >= 11 and rank2 >= 10: return 7 if is_suited else 5
        if is_suited and (abs(rank1 - rank2) <= 4): return 3
        if (not is_suited) and (abs(rank1 - rank2) <= 3) and rank1 >= 8: return 1
        return 0
    
    def _calculate_draw_potential(self, hole_cards, community_cards):
        """Calculates draw potential (number of outs and if it's a strong draw), adapted from expert_player"""
        all_cards = hole_cards + community_cards
        outs = 0
        is_strong_draw = False

        # 1. Flush Draw
        suits = [c.suit for c in all_cards]
        for s in [Card.CLUB, Card.DIAMOND, Card.HEART, Card.SPADE]:
            if suits.count(s) == 4:
                outs += 9  # 9 remaining cards of the same suit
                is_strong_draw = True
                break

        # 2. Straight Draw
        # To handle A-2-3-4-5 straights, we treat Ace as rank 1 as well
        ranks = sorted(list(set([c.rank for c in all_cards])))
        
        # Check all possible 5-card windows
        for i in range(1, 11): # From A-5 to T-A
            window = range(i, i + 5)
            
            # Special case for A-5 (ranks: 14, 2, 3, 4, 5)
            if i == 1:
                window_ranks = {14, 2, 3, 4, 5}
            else:
                window_ranks = set(window)
            
            common_ranks = window_ranks.intersection(ranks)
            
            if len(common_ranks) == 4:
                missing_ranks = window_ranks.difference(common_ranks)
                missing_rank = missing_ranks.pop()
                
                # Exclude cards that are already on the board
                if not any(card.rank == missing_rank for card in all_cards):
                    # Open-ended straight draw
                    if (i > 1 and i < 10) and (missing_rank == i or missing_rank == i + 4):
                        # If the missing card is at either end, there are 8 outs (e.g., holding 5,6,7,8 -> need 4 or 9)
                        outs += 8
                        is_strong_draw = True
                    # Gutshot straight draw
                    else:
                        # Missing a middle card, or A/5 for a wheel draw, or T/A for a broadway draw
                        outs += 4

        return outs, is_strong_draw

    def _save_model(self):
        """Saves the model weights"""
        torch.save(self.policy_net.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    def _load_model(self):
        """Loads the model weights"""
        if os.path.exists(MODEL_PATH):
            self.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            print(f"Model loaded from {MODEL_PATH}")

    # --- Game Engine Message Handlers ---
    def receive_game_start_message(self, game_info):
        self.initial_stack = game_info['rule']['initial_stack']
        self.max_round = game_info['rule']['max_round']

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.last_state = None
        self.last_action_idx = None

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def set_uuid(self, uuid):
        self.uuid = uuid

def setup_ai():
    """Function to export the AI instance"""
    return DeepQPlayer()
