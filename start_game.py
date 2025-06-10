import json
import argparse
import os
from contextlib import redirect_stdout
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.nottako import setup_ai as nottako

from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai

baselines = [
    baseline0_ai,
    baseline1_ai,
    baseline2_ai,
    baseline3_ai,
    baseline4_ai,
    baseline5_ai,
    baseline6_ai,
    baseline7_ai,
]

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-b", type=int)
parser.add_argument("-p", action="store_true")
args = parser.parse_args()

def run_game(player1_ai, player2_ai, verbose=1, suppress_output=False):
    config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
    config.register_player(name="baseline", algorithm=player1_ai())
    config.register_player(name="nottako", algorithm=player2_ai())
    if suppress_output:
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
            return start_poker(config, verbose=verbose)
    else:
        return start_poker(config, verbose=verbose)

if args.b is not None:
    if args.p:
        print(f"baseline {args.b}")
        result = run_game(baselines[args.b], nottako, verbose=1, suppress_output=True)
        print(json.dumps(result, indent=4))
    else:
        print(f"baseline {args.b}")
        result = run_game(baselines[args.b], nottako, verbose=1, suppress_output=False)
        print(json.dumps(result, indent=4))
else:
    for i, setup_ai in enumerate(baselines):
        print(f"baseline {i}")
        if args.p:
            result = run_game(setup_ai, nottako, verbose=0, suppress_output=True)
            print(json.dumps(result, indent=4))
        else:
            result = run_game(setup_ai, nottako, verbose=0, suppress_output=False)
            print(json.dumps(result, indent=4))





