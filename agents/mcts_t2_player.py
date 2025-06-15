import random
import math
import time
import collections

from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card
from game.engine.deck import Deck
from game.engine.poker_constants import PokerConstants as Const

class MCTSNode:
    def __init__(self, parent=None, action=None, valid_actions=None):
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = valid_actions
        self.player_count = 0

    def select_child(self, exploration_weight=1.414): # sqrt(2) is a common exploration weight
        # UCB1 formula
        best_score = -1
        best_child = None
        for child in self.children:
            if child.visits == 0:
                score = float('inf')
            else:
                exploit = child.wins / child.visits
                explore = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                score = exploit + explore
            
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, action):
        child = MCTSNode(parent=self, action=action, valid_actions=[]) # Child nodes are leaves in this model
        self.untried_actions.remove(action)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result


class MonteCarloPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.hole_card = []
        self.community_card = []
        self.seats = None
        self.round_state = None
        self.my_uuid = None
        self.game_info = None
        self.round_count = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        start_time = time.time()

        # 2. Absolute Win Lock-in Logic
        if self.game_info:
            my_seat = next((s for s in self.seats if s['uuid'] == self.my_uuid), None)
            if my_seat:
                my_stack = my_seat['stack']
                initial_stack = self.game_info['rule']['initial_stack']
                max_round = self.game_info['rule']['max_round']
                big_blind = self.game_info['rule']['small_blind_amount'] * 2
                
                # If my profit is greater than the max possible loss of folding every remaining round
                if my_stack - initial_stack > (max_round - self.round_count + 1) * big_blind:
                    # We can safely fold to victory
                    return "fold", 0

        self.hole_card = [Card.from_str(c) for c in hole_card]
        self.community_card = [Card.from_str(c) for c in round_state["community_card"]]
        self.round_state = round_state

        # 3. Pre-flop intelligence to avoid being a calling station
        is_preflop = len(self.community_card) == 0
        if is_preflop:
            strength = self._preflop_hand_strength(self.hole_card)
            call_action = valid_actions[1]
            big_blind = self.game_info['rule']['small_blind_amount'] * 2
            
            # If facing a raise (call > big blind) with a weak hand, just fold.
            if strength < 25 and call_action['amount'] > big_blind:
                return valid_actions[0]['action'], 0 # Fold

        num_players = len([p for p in round_state["seats"] if p["state"] == "participating"])

        root = MCTSNode(valid_actions=[a['action'] for a in valid_actions])
        root.player_count = num_players
        
        # --- EXPANSION: Create children for all possible actions immediately ---
        if root.untried_actions:
            for action_str in list(root.untried_actions):
                root.expand(action_str)
        
        # Time limit for MCTS is slightly less than the timeout

        MIN_SIMULATIONS_FOR_EARLY_STOP = 1600  # Minimum simulations before we can stop early
        WIN_RATE_THRESHOLD_FOR_EARLY_STOP = 0.65  # If best action's win rate is over this, stop.
        MAX_SIMULATIONS_TIME = 8  # Maximum time to run MCTS in seconds

        while time.time() - start_time < MAX_SIMULATIONS_TIME:
            # --- SELECTION: Select a child (action) to simulate ---
            action_node = root.select_child()
            if not action_node: break # Should not happen if there are children

            # --- SIMULATION ---
            if action_node.action == 'fold':
                win_rate = 0.0
            else:
                # For now, treat call and raise simulations the same way (showdown equity)
                # A more advanced model could estimate fold equity for raises.
                win_rate = self._run_simulation(num_simulations=30)
            
            # --- BACKPROPAGATION ---
            # We use win_rate directly as the result
            action_node.update(win_rate)
            root.update(win_rate)
            
            # --- Early Stopping Check ---
            if len(root.children) > 1 and root.visits > MIN_SIMULATIONS_FOR_EARLY_STOP and root.visits % 200 == 0: # Check every 500 sims
                best_child = max(root.children, key=lambda c: c.visits)
                if best_child.visits / root.visits > WIN_RATE_THRESHOLD_FOR_EARLY_STOP:
                    break
        
        # After MCTS, choose the best action
        if not root.children: # No time to run or only one action
            call_action_info = valid_actions[1]
            return call_action_info["action"], call_action_info["amount"]

        # Choose action that was most explored (most robust choice)
        best_child = max(root.children, key=lambda c: c.visits)
        best_action_str = best_child.action
        
        # Check for profit lock-in fold
        # A simple heuristic: if we are ahead and face a big bet, consider folding
        my_stack = [s['stack'] for s in self.seats if s['uuid'] == self.uuid][0]
        initial_stack = self.game_info['rule']['initial_stack']
        if my_stack > initial_stack * 1.5 and best_action_str == 'raise':
             # If we have a significant lead, be more cautious
            fold_action_info = valid_actions[0]
            if fold_action_info['action'] == 'fold':
                return fold_action_info['action'], 0


        best_action_info = next((a for a in valid_actions if a['action'] == best_action_str), None)
        action = best_action_info['action']
        amount = best_action_info['amount']

        if action == 'raise':
            # If the action is raise, we must choose a specific amount.
            pot_total = round_state['pot']['main']['amount']
            # A standard bet is 2/3 or 3/4 of the pot. Let's use 75%.
            raise_amount = round(pot_total * 0.75)
            
            # The chosen amount must be within the valid range.
            min_raise = amount['min']
            max_raise = amount['max']
            
            # Clamp the amount to the allowed range and ensure it's not smaller than min_raise
            amount = max(min_raise, min(raise_amount, max_raise))

        # Simple BE calculation
        if best_action_str == 'call':
            call_amount = amount
            pot_total = round_state['pot']['main']['amount'] + round_state['pot']['side'][0]['amount'] if round_state['pot']['side'] else round_state['pot']['main']['amount']
            pot_odds = call_amount / (pot_total + call_amount)
            win_prob = best_child.wins / best_child.visits if best_child.visits > 0 else 0
            
            # If our win probability is lower than pot odds, maybe we should fold
            if win_prob < pot_odds:
                if 'fold' in [a['action'] for a in valid_actions]:
                     return valid_actions[0]['action'], valid_actions[0]['amount']
        
        return action, amount

    def _run_simulation(self, num_simulations=100):
        wins = 0
        losses = 0

        num_opponents = len([p for p in self.round_state["seats"] if p["uuid"] != self.my_uuid and p["state"] == "participating"])
        if num_opponents == 0:
            return 1 # We are the only one left

        for _ in range(num_simulations):
            # 1. Determine unknown cards
            deck = Deck()
            known_cards = self.hole_card + self.community_card
            deck.deck = [card for card in deck.deck if card not in known_cards]
            deck.shuffle()
            
            # 2. Randomly sample opponent hands and remaining community cards
            try:
                opponent_hands = [[deck.draw_card() for _ in range(2)] for _ in range(num_opponents)]
                
                num_remaining_community = 5 - len(self.community_card)
                sim_community_card = self.community_card + [deck.draw_card() for _ in range(num_remaining_community)]
            except IndexError:
                # This can happen if deck runs out of cards, which implies a very specific board.
                # In this rare case, we just skip this simulation run.
                continue

            # 3. Evaluate hands
            my_hand_value = HandEvaluator.eval_hand(self.hole_card, sim_community_card)
            
            opponent_best_value = 0
            for hand in opponent_hands:
                opp_val = HandEvaluator.eval_hand(hand, sim_community_card)
                if opp_val > opponent_best_value:
                    opponent_best_value = opp_val

            # 4. Compare and record result
            if my_hand_value > opponent_best_value:
                wins += 1
            elif my_hand_value < opponent_best_value:
                losses += 1
            # Ties are ignored in win rate calculation as per user instructions
        
        if wins + losses == 0:
            return 0
        
        return wins / (wins + losses)

    def _preflop_hand_strength(self, hole_card):
        """A simple heuristic to evaluate pre-flop hand strength."""
        card1, card2 = hole_card[0], hole_card[1]
        rank1, rank2 = card1.rank, card2.rank
        
        score = 0
        # 1. High card points
        score += (rank1 + rank2)
        
        # 2. Pair bonus
        if rank1 == rank2:
            score += (rank1 * 2)
            # Give a big bonus for high pairs
            if rank1 >= 10: # TT, JJ, QQ, KK, AA
                score += 20
        
        # 3. Suited bonus
        if card1.suit == card2.suit:
            score += 10
            
        # 4. Connector bonus
        gap = abs(rank1 - rank2)
        if gap == 1 or (rank1 == 14 and rank2 == 2) or (rank2 == 14 and rank1 == 2): # Connected
            score += 6
        elif gap == 2: # 1-gap
            score += 4
        
        return score

    def receive_game_start_message(self, game_info):
        self.game_info = game_info
        self.my_uuid = self.uuid

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count
        self.hole_card = [Card.from_str(c) for c in hole_card]
        self.seats = seats

    def receive_street_start_message(self, street, round_state):
        self.community_card = [Card.from_str(c) for c in round_state["community_card"]]
        self.round_state = round_state

    def receive_game_update_message(self, action, round_state):
        self.round_state = round_state

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass # Reset for the next round might be needed if state carries over

def setup_ai():
    return MonteCarloPlayer()
