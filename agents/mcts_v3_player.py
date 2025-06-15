import random
import math
import time
from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card
from game.engine.deck import Deck
from game.engine.poker_constants import PokerConstants as Const

# MCTS Node
class Node:
    def __init__(self, parent=None, action=None, state=None, valid_actions=None):
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.action = action  # The action that led to this state
        self.state = state  # The game state this node represents
        self.untried_actions = valid_actions
        self.player_count = state['player_count']

    def select_child(self, exploration_constant=1.414):
        # UCB1 formula
        best_child = max(self.children, key=lambda c: (c.wins / c.visits) + exploration_constant * math.sqrt(math.log(self.visits) / c.visits))
        return best_child

    def expand(self, action, next_state, valid_actions):
        child = Node(parent=self, action=action, state=next_state, valid_actions=valid_actions)
        self.children.append(child)
        self.untried_actions.remove(action)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

class MonteCarloPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.hole_card = []
        self.community_card = []
        self.round_state = None
        self.my_uuid = None
        self.initial_stack = 1000  # Default, will be updated
        self.my_stack = 1000 # Default, will be updated
        self.round_count = 0
        self.small_blind_amount = 5 # Default, will be updated
        self.total_round = 20 # Default, will be updated

    def declare_action(self, valid_actions, hole_card, round_state):
        # Ultimate fallback for empty valid_actions to prevent crash
        if not valid_actions:
            return 'fold', 0

        # --- Absolute Win Lock-in ---
        # If my profit is larger than the maximum I can lose by paying blinds for all remaining rounds, just fold to secure the win.
        fold_action_info = next((a for a in valid_actions if a['action'] == 'fold'), None)
        if fold_action_info and self.round_count > 0: # Ensure round_count is initialized
            # We assume the big blind is the max we can lose per round if we fold immediately.
            # This is a safe estimation.
            max_possible_loss = (self.total_round - self.round_count + 1) * (self.small_blind_amount * 2)
            current_profit = self.my_stack - self.initial_stack
            if current_profit > max_possible_loss:
                return fold_action_info['action'], fold_action_info['amount']

        start_time = time.time()
        
        self.hole_card = [Card.from_str(c) for c in hole_card]
        self.community_card = [Card.from_str(c) for c in round_state['community_card']]
        self.round_state = round_state

        player_count = sum(1 for seat in round_state['seats'] if seat['state'] != 'folded')
        if player_count == 0: player_count = len(round_state['seats'])


        current_state = {
            'hole_card': self.hole_card,
            'community_card': self.community_card,
            'player_count': player_count,
            'round_state': round_state
        }

        # We don't want to explore folding, but we need it for final decisions.
        # The MCTS root will explore non-fold actions.
        untried_actions_for_mcts = [a for a in valid_actions if a['action'] != 'fold']
        
        # If only fold is possible, just fold.
        if not untried_actions_for_mcts:
            fold_action = next((a for a in valid_actions if a['action'] == 'fold'), valid_actions[0])
            return fold_action['action'], fold_action['amount']

        root = Node(state=current_state, valid_actions=untried_actions_for_mcts)


        # MCTS loop
        MIN_SIMULATIONS_FOR_EARLY_STOP = 2000  # Minimum simulations before we can stop early
        WIN_RATE_THRESHOLD_FOR_EARLY_STOP = 0.6  # If best action's win rate is over this, stop.
        MAX_SIMULATIONS_TIME = 8  # Maximum time to run MCTS in seconds

        while time.time() - start_time < MAX_SIMULATIONS_TIME:
            node = root
            state_clone = self._clone_state(current_state)

            # 1. Selection
            while not node.untried_actions and node.children:
                node = node.select_child()
                state_clone = self._apply_action_to_state(state_clone, node.action)

            # 2. Expansion
            if node.untried_actions:
                action = random.choice(node.untried_actions)
                next_state_clone = self._apply_action_to_state(state_clone, action)
                node = node.expand(action, next_state_clone, valid_actions) 

            # 3. Simulation (Rollout)
            result = self._run_simulation(node.state)

            # 4. Backpropagation
            while node is not None:
                node.update(result)
                node = node.parent

            # Early stopping logic: Check if we are confident enough in our best move
            if root.visits > MIN_SIMULATIONS_FOR_EARLY_STOP and root.children:
                best_child = max(root.children, key=lambda c: c.wins / (c.visits + 1e-6))
                if best_child.visits > 0:
                    win_rate = best_child.wins / best_child.visits
                    if win_rate > WIN_RATE_THRESHOLD_FOR_EARLY_STOP:
                        break  # Confident enough, stop early

        return self._make_final_decision(root, valid_actions, round_state)

    def _make_final_decision(self, root, valid_actions, round_state):
        # Find essential actions safely, without relying on index
        fold_action = next((a for a in valid_actions if a['action'] == 'fold'), None)
        call_action = next((a for a in valid_actions if a['action'] == 'call'), None)

        # Fallback if MCTS didn't run or produced no results
        if not root.children:
            if call_action: return call_action["action"], call_action["amount"]
            if fold_action: return fold_action['action'], fold_action['amount']
            return valid_actions[0]['action'], valid_actions[0]['amount']  # Last resort, if fold is somehow not available

        # --- Strategic Layer ---
        # 1. Profit Lock-in: If we are winning big, play super safe
        profit = self.my_stack - self.initial_stack
        target_profit = (self.round_count / 15) * self.initial_stack if self.round_count <= 18 else self.initial_stack * 0.2
        # Only apply profit-lockdown after the flop when we have a hand to evaluate
        if profit > target_profit and len(self.community_card) >= 3:
            my_score = HandEvaluator.eval_hand(self.hole_card, self.community_card)
            # Check if my_score is not None (safety) and hand is weaker than two-pair
            if my_score is not None and my_score < HandEvaluator.TWOPAIR:
                if fold_action: return fold_action['action'], fold_action['amount']

        # 2. Risk Factor based on game stage
        if self.round_count <= 5:
            risk_factor = 1.15  # Be braver in early game
        elif self.round_count >= 16:
            risk_factor = 0.85  # Be more cautious in late game
        else:
            risk_factor = 1.0   # Neutral

        # 3. Decision based on MCTS results + strategy
        # If pot odds are good, or we can check for free, trust the MCTS result.
        # We must select a concrete amount if the action is 'raise'.
        
        # --- New Decision Logic: Choose action with the highest win rate, not most visits ---
        # This makes the AI more aggressive and value-oriented.
        # Add a small epsilon to the denominator to avoid division by zero and favor nodes with more visits in case of win_rate ties.
        best_action_node = max(root.children, key=lambda c: c.wins / (c.visits + 1e-6))
        best_action = best_action_node.action
        win_rate = best_action_node.wins / best_action_node.visits if best_action_node.visits > 0 else 0
        
        # Re-check pot odds for the best win-rate action
        if call_action and call_action['amount'] > 0:
            call_amount = call_action['amount']
            pot_total = round_state['pot']['main']['amount']
            for side_pot in round_state['pot']['side']:
                pot_total += side_pot['amount']
            
            if (pot_total + call_amount) > 0:
                break_even_rate = call_amount / (pot_total + call_amount)
                # If the BEST action's win rate is still not good enough, fold.
                if win_rate * risk_factor < break_even_rate:
                    if fold_action:
                        return fold_action['action'], fold_action['amount']

        chosen_action, chosen_amount = best_action['action'], best_action['amount']
        
        if chosen_action == 'raise':
            pot_total = round_state['pot']['main']['amount']
            for side_pot in round_state['pot']['side']:
                pot_total += side_pot['amount']
            chosen_amount = self._calculate_raise_amount(win_rate, chosen_amount['min'], chosen_amount['max'], pot_total)
        
        return chosen_action, chosen_amount

    def _calculate_raise_amount(self, win_rate, min_raise, max_raise, pot_size):
        """
        Calculates a smart raise amount based on win_rate and pot_size.
        """
        # Strong hand (high confidence): Bet for value, typically a significant portion of the pot.
        if win_rate > 0.85:
            # Bet 75% of the pot
            amount_to_raise = int(pot_size * 0.75)
        elif win_rate > 0.70:
            # Bet 66% of the pot
            amount_to_raise = int(pot_size * 0.66)
        elif win_rate > 0.55:
            # Bet 50% of the pot (standard value bet)
            amount_to_raise = int(pot_size * 0.50)
        else:
            # Low confidence raise (e.g., a semi-bluff), just make the minimum raise.
            amount_to_raise = min_raise
            
        # Ensure the chosen amount is within the valid min/max raise bounds.
        return max(min_raise, min(amount_to_raise, max_raise))

    def _run_simulation(self, state):
        hole_card = state['hole_card']
        community_card = state['community_card']
        player_count = state['player_count']

        if player_count <= 1: return 1.0 # If everyone else folded, we win.
        
        deck = Deck()
        deck.shuffle()
        
        known_cards = hole_card + community_card
        for card in known_cards:
            if card in deck.deck:
                deck.deck.remove(card)

        community_card_sim = community_card[:]
        while len(community_card_sim) < 5:
            community_card_sim.append(deck.draw_card())
            
        opponents_hands = []
        for _ in range(player_count - 1):
            if len(deck.deck) >= 2:
                opponents_hands.append([deck.draw_card(), deck.draw_card()])

        my_score = HandEvaluator.eval_hand(hole_card, community_card_sim)
        opponents_scores = [HandEvaluator.eval_hand(hand, community_card_sim) for hand in opponents_hands if hand]

        if not opponents_scores: return 1.0 # No opponents left to compare

        max_opponent_score = max(opponents_scores)
        
        if my_score > max_opponent_score:
            return 1.0  # Win
        elif my_score == max_opponent_score:
            return 0.5  # Tie
        else:
            return 0.0  # Loss
        
    def _clone_state(self, state):
        return {
            'hole_card': list(state['hole_card']),
            'community_card': list(state['community_card']),
            'player_count': state['player_count'],
            'round_state': state['round_state'].copy() # Shallow copy is enough
        }

    def _apply_action_to_state(self, state, action):
        # This is a simplification. A full implementation would need to update
        # the pot, player stacks, etc., based on the action.
        # For this MCTS, we mainly care about the cards, which don't change within a street's action phase.
        return state

    def receive_game_start_message(self, game_info):
        self.my_uuid = self.uuid
        self.initial_stack = game_info['rule']['initial_stack']
        self.small_blind_amount = game_info['rule']['small_blind_amount']
        self.total_round = game_info['rule']['max_round']

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count
        self.hole_card = [Card.from_str(c) for c in hole_card]
        self.community_card = []
        my_seat = next((s for s in seats if s['uuid'] == self.my_uuid), None)
        if my_seat:
            self.my_stack = my_seat['stack']
        self.round_state = None # Reset round state

    def receive_street_start_message(self, street, round_state):
        self.community_card = [Card.from_str(c) for c in round_state['community_card']]
        self.round_state = round_state

    def receive_game_update_message(self, action, round_state):
        self.round_state = round_state
        # Update my stack after my own action or someone else's
        my_seat = next((s for s in round_state['seats'] if s['uuid'] == self.my_uuid), None)
        if my_seat:
            self.my_stack = my_seat['stack']

    def receive_round_result_message(self, winners, hand_info, round_state):
        # Update my stack at the end of the round
        my_seat = next((s for s in round_state['seats'] if s['uuid'] == self.my_uuid), None)
        if my_seat:
            self.my_stack = my_seat['stack']

def setup_ai():
    return MonteCarloPlayer()
