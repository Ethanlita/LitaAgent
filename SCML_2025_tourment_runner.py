import matplotlib.pyplot as plt

try:
    from scml.std import SCML2025StdWorld
except ImportError:
    from scml.std import SCML2024StdWorld as SCML2025StdWorld
from scml.std.agents import GreedyStdAgent, RandomStdAgent, SyncRandomStdAgent

from litaagent_std.litaagent_cir import LitaAgentCIR
from litaagent_std.litaagent_y import LitaAgentY

agent_types = [
    SyncRandomStdAgent,
    GreedyStdAgent,
    RandomStdAgent,
    LitaAgentCIR,
    LitaAgentY
]

world = SCML2025StdWorld(
    **SCML2025StdWorld.generate(agent_types=agent_types, n_steps=50),
    construct_graphs=True,
)

_, _ = world.draw()

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
