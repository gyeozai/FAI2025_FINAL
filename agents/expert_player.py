from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card
import random


class ExpertPlayer(
    BasePokerPlayer
):  # Do not forget to make parent class as "BasePokerPlayer"
    def __init__(self):
        super().__init__()
        self.uuid = None
        self.game_info = None
        self.round_count = 0

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # Absolute Win Lock-in Logic
        if self.game_info and self.round_count > 0:
            my_seat = next((s for s in round_state['seats'] if s['uuid'] == self.uuid), None)
            if my_seat:
                my_stack = my_seat['stack']
                initial_stack = self.game_info['rule']['initial_stack']
                max_round = self.game_info['rule']['max_round']
                big_blind = self.game_info['rule']['small_blind_amount'] * 2

                # If our profit is greater than the max possible loss of folding every remaining round,
                # we can safely fold to victory.
                # This is a safe upper bound on the loss (assuming we pay BB every round).
                max_future_loss = (max_round - self.round_count + 1) * big_blind
                current_profit = my_stack - initial_stack

                if current_profit > max_future_loss:
                    return valid_actions[0]['action'], 0  # Safely fold to win

        hole_cards = [Card.from_str(c) for c in hole_card]
        community_cards = [Card.from_str(c) for c in round_state["community_card"]]
        street = round_state["street"]
        call_action = valid_actions[1]
        call_amount = call_action["amount"]
        raise_action = valid_actions[2]
        raise_amount_options = raise_action["amount"]
        pot_amount = round_state["pot"]["main"]["amount"]

        if street == "preflop":
            is_sb = round_state['seats'][round_state['dealer_btn']]['uuid'] == self.uuid
            strength = self._calculate_hole_card_strength(hole_cards)

            # Heads-up strategy: Aggressive on the button (SB), defensive on the BB.
            if is_sb:
                if strength >= 8:  # Premium hands
                    return self._make_aggressive_move(valid_actions, call_amount, raise_amount_options, pot_amount)
                elif strength >= 2:  # Any playable hand in heads-up from button
                    return self._make_standard_raise(raise_amount_options, pot_amount, valid_actions)
                else:  # Limp with trash
                    return valid_actions[1]["action"], call_amount
            else: # Big Blind
                is_facing_raise = call_amount > 0
                if not is_facing_raise: # Opponent limped
                    if strength >= 3:
                        return self._make_standard_raise(raise_amount_options, pot_amount, valid_actions)
                    else:
                        return valid_actions[1]["action"], 0 # Check
                else: # Opponent raised
                    pot_odds = call_amount / (pot_amount + call_amount)
                    if strength >= 7: # Re-raise with premium
                        return self._make_aggressive_move(valid_actions, call_amount, raise_amount_options, pot_amount)
                    elif strength >= 4: # Call with decent hands
                        return valid_actions[1]["action"], call_amount
                    elif strength >= 1 and pot_odds < 0.4: # Call with speculative hands if odds are good
                        return valid_actions[1]["action"], call_amount
                    else:
                        return valid_actions[0]["action"], 0
        else:  # Post-flop (flop, turn, river)
            hand_strength_info = HandEvaluator.gen_hand_rank_info(
                hole_cards, community_cards
            )
            hand_strength = hand_strength_info["hand"]["strength"]

            # Monster hands
            if hand_strength in ["STRAIGHTFLASH", "FOURCARD", "FULLHOUSE"]:
                # Slow play 20% of the time on the flop
                if street == "flop" and random.random() < 0.2:
                    return valid_actions[1]["action"], call_amount
                return self._make_aggressive_move(
                    valid_actions, call_amount, raise_amount_options, pot_amount
                )

            # Very strong hands
            if hand_strength in ["FLASH", "STRAIGHT", "THREECARD", "TWOPAIR"]:
                return self._make_aggressive_move(
                    valid_actions, call_amount, raise_amount_options, pot_amount
                )

            # One pair
            if hand_strength == "ONEPAIR":
                # Bet for value, roughly half pot
                if call_amount == 0:
                    bet_amount = int(pot_amount * 0.5)
                    return self._get_bet_action(bet_amount, raise_amount_options, valid_actions)
                else:  # Face a bet
                    pot_odds = call_amount / (pot_amount + call_amount)
                    if pot_odds < 0.3:  # Call if getting good odds
                        return valid_actions[1]["action"], call_amount
                    else:
                        return valid_actions[0]["action"], 0

            # Drawing hands or weak hands
            outs, is_strong_draw = self._calculate_draw_potential(
                hole_cards, community_cards
            )
            if outs > 0:
                pot_odds = call_amount / (pot_amount + call_amount)
                hand_odds = outs / (52 - len(hole_cards) - len(community_cards))
                if hand_odds > pot_odds:
                    if (
                        is_strong_draw and random.random() < 0.15
                    ):  # Semi-bluff with strong draws
                        return self._make_standard_raise(
                            raise_amount_options, pot_amount, valid_actions
                        )
                    return valid_actions[1]["action"], call_amount

            # Bluffing
            # Bluff 5% of the time if no one has bet and board seems dry
            if call_amount == 0 and random.random() < 0.05:
                bet_amount = int(pot_amount * 0.4)
                return self._get_bet_action(bet_amount, raise_amount_options, valid_actions)

            # Default to check or fold
            return (
                valid_actions[1]["action"]
                if call_amount == 0
                else valid_actions[0]["action"]
            ), call_amount

    def _calculate_hole_card_strength(self, hole_cards):
        card1, card2 = hole_cards[0], hole_cards[1]
        rank1, rank2 = sorted([c.rank for c in hole_cards], reverse=True)
        is_pair = rank1 == rank2
        is_suited = card1.suit == card2.suit

        if is_pair:
            if rank1 >= 11:
                return 10  # JJ+
            if rank1 >= 8:
                return 8  # TT, 99, 88
            return 5  # 77-22 (upgraded from 4)
        if rank1 == 14:  # Ace
            if rank2 >= 13:
                return 9  # AK
            if rank2 >= 10:
                return 7 if is_suited else 6  # AQs, AJs, ATs, AQo... (upgraded from 6/5)
            if is_suited:
                return 4 # A9s-A2s
        if rank1 >= 11 and rank2 >= 10:
            return 7 if is_suited else 5  # KQs, KJs, QJs...
        if is_suited and (abs(rank1 - rank2) <= 4):
            return 3  # Suited connectors/gappers
        if (not is_suited) and (abs(rank1 - rank2) <= 3) and rank1 >= 8:
            return 1 # High broadway connectors e.g. T9o, JTo
        return 0

    def _calculate_draw_potential(self, hole_cards, community_cards):
        all_cards = hole_cards + community_cards
        outs = 0
        is_strong_draw = False

        # Flush draw
        suits = [c.suit for c in all_cards]
        for s in [Card.CLUB, Card.DIAMOND, Card.HEART, Card.SPADE]:
            if suits.count(s) == 4:
                outs += 9
                is_strong_draw = True
                break

        # Straight draw
        ranks = sorted(list(set([c.rank for c in all_cards])))
        for i in range(len(ranks) - 3):
            is_straight = ranks[i + 3] - ranks[i] == 3 and len(ranks[i : i + 4]) == 4
            if is_straight:
                if ranks[i] == 2 and ranks[i+3] == 5: # Wheel draw A-2-3-4
                    outs += 4
                elif ranks[i] > 2 and ranks[i+3] < 14: # Open-ended
                    outs += 8
                    is_strong_draw = True
                else: # Gutshot
                    outs += 4
                break
        return outs, is_strong_draw

    def _make_aggressive_move(
        self, valid_actions, call_amount, raise_amount_options, pot_amount
    ):
        if call_amount > 0:  # Re-raise
            raise_amount = min(
                raise_amount_options["max"], max(raise_amount_options["min"], int(pot_amount * 0.75 + call_amount))
            )
        else:  # Bet
            raise_amount = min(
                raise_amount_options["max"], max(raise_amount_options["min"], int(pot_amount * 0.75))
            )
        return self._get_bet_action(raise_amount, raise_amount_options, valid_actions)

    def _make_standard_raise(self, raise_amount_options, pot_amount, valid_actions):
        bet_amount = min(
            raise_amount_options["max"], max(raise_amount_options["min"], int(pot_amount * 0.5))
        )
        return self._get_bet_action(bet_amount, raise_amount_options, valid_actions)

    def _get_bet_action(self, amount, raise_amount_options, valid_actions):
        action = "raise"
        bet_amount = max(raise_amount_options["min"], amount)
        bet_amount = min(raise_amount_options["max"], bet_amount)
        if bet_amount == -1:  # Cannot raise
            return "call", valid_actions[1]["amount"]
        return action, bet_amount

    def receive_game_start_message(self, game_info):
        self.game_info = game_info

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count
        my_seat = next(s for s in seats if s["uuid"] == self.uuid)

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def set_uuid(self, uuid):
        self.uuid = uuid


def setup_ai():
    return ExpertPlayer()
