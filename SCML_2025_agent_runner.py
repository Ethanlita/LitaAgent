#%% Standard imports
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

import math
from typing import Iterable
from rich.jupyter import print

from negmas import SAOResponse, ResponseType, Outcome, SAOState
from scml.std import *
from scml.runner import WorldRunner

from litaagent_std.litaagent_cir import LitaAgentCIR
from litaagent_std.litaagent_n import LitaAgentN
from litaagent_std.litaagent_y import LitaAgentY

# create a runner that encapsulates a number of configs to evaluate agents
# in the same conditions every time
CONFIGS, REPS, STEPS = 10, 5, 25
context = ANACStdContext( # what are the rounds here, number of trials, process (needs processes or not), etc. and align with the actual parameters of the live competitoin.
    n_steps=STEPS, n_processes=3, world_params=dict(construct_graphs=True)
)
single_agent_runner = WorldRunner(
    context, n_configs=CONFIGS, n_repetitions=REPS, save_worlds=True
)
full_market_runner = WorldRunner.from_runner(
    single_agent_runner, control_all_agents=True
)

#%% create a world with a single agent and run it
single_agent_runner(LitaAgentCIR)
single_agent_runner.draw_worlds_of(LitaAgentCIR)

#%% plot the results
single_agent_runner.plot_stats(agg=False)
single_agent_runner.score_summary()
plt.show()
def analyze_contracts(worlds, exogenous_only=False):
    """
    Analyzes the contracts signed in the given world
    """
    dfs = []
    for world in worlds:
        dfs.append(pd.DataFrame.from_records(world.saved_contracts))
    data = pd.concat(dfs)
    if exogenous_only:
        data = data.loc[
            (data["seller_name"] == "SELLER") | (data["buyer_name"] == "BUYER"), :
        ]
    return data.groupby(["seller_name", "buyer_name"])[["quantity", "unit_price"]].agg(
        dict(quantity=("sum", "count"), unit_price="mean")
    )


print(analyze_contracts(single_agent_runner.worlds_of()))


"""
#%% create a world with a number of agents and run it
full_market_runner(LitaAgentN)
full_market_runner.draw_worlds_of(LitaAgentN)

#%% plot the results
full_market_runner.plot_stats(agg=False)
plt.show()
print("Plotting stats")
"""
