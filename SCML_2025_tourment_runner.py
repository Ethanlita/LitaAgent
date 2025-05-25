import matplotlib.pyplot as plt
from scml.std import *
from scml.oneshot import *

from litaagent_std.litaagent_y import LitaAgentY

agent_types = [
    SyncRandomStdAgent,
    RandDistOneShotAgent,
    GreedyOneShotAgent,
    RandomStdAgent,
    LitaAgentY,
]

world = SCML2024StdWorld(
    **SCML2024StdWorld.generate(agent_types=agent_types, n_steps=50),
    construct_graphs=True,
)

_, _ = world.draw()

# 这是调试代码
# input()

# Run the tournament
world.run_with_progress()  # may take few minutes

# Plot the results
world.plot_stats("n_negotiations", ylegend=1.25)
plt.show()

world.plot_stats("n_negotiations")
plt.show()

world.plot_stats("bankrupt", ylegend=1.25)
plt.show()

world.plot_stats()
plt.show()