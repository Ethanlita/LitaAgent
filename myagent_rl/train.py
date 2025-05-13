# trains an RL model
#
import logging
from typing import Any

from negmas.sao import SAOResponse
from rich import print
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.rl.action import FlexibleActionManager
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.common import model_wrapper
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.reward import DefaultRewardFunction

# sys.path.append(str(Path(__file__).parent))
from .common import MODEL_PATH, MyObservationManager, TrainingAlgorithm, make_context

NTRAINING = 100  # number of training steps


class MyRewardFunction(DefaultRewardFunction):
    """My reward function"""

    def before_action(self, awi: OneShotAWI) -> float:
        return super().before_action(awi)

    def __call__(self, awi: OneShotAWI, action: dict[str, SAOResponse], info: float):
        return super().__call__(awi, action, info)


def make_env(as_supplier, log: bool = False) -> OneShotEnv:
    log_params: dict[str, Any] = (
        dict(
            no_logs=False,
            log_stats_every=1,
            log_file_level=logging.DEBUG,
            log_screen_level=logging.ERROR,
            save_signed_contracts=True,
            save_cancelled_contracts=True,
            save_negotiations=True,
            save_resolved_breaches=True,
            save_unresolved_breaches=True,
            debug=True,
        )
        if log
        else dict(debug=True)
    )
    log_params.update(
        dict(
            ignore_agent_exceptions=False,
            ignore_negotiation_exceptions=False,
            ignore_contract_execution_exceptions=False,
            ignore_simulation_exceptions=False,
        )
    )
    context = make_context(as_supplier)
    return OneShotEnv(
        action_manager=FlexibleActionManager(context=context),
        observation_manager=MyObservationManager(context=context),  # type: ignore
        reward_function=MyRewardFunction(),
        context=context,
        extra_checks=False,
    )


def try_a_model(
    model,
    as_supplier: bool,
):
    """Runs a single simulation with one agent controlled with the given model"""

    obs_type = MyObservationManager
    # Create a world context compatibly with the model
    context = make_context(as_supplier)
    # sample a world and the RL agents (always one in this case)
    world, _ = context.generate(
        types=(OneShotRLAgent,),
        params=(
            dict(
                models=[model_wrapper(model)],
                observation_managers=[obs_type(context)],
                action_managers=[FlexibleActionManager(context)],
            ),
        ),
    )
    # run the world simulation
    world.run_with_progress()
    return world


def main(ntrain: int = NTRAINING):
    # choose the type of the model. Possibilities supported are:
    # fixed: Supports a single world configuration
    # limited: Supports a limited range of world configuration
    # unlimited: Supports any range of world configurations

    for as_supplier in (False, True):
        print(f"Training as {'supplier' if as_supplier else 'consumer'}")
        # create a gymnasium environment for training
        env = make_env(as_supplier)

        # choose a training algorithm
        model = TrainingAlgorithm(  # type: ignore learning_rate must be passed by the algorithm itself
            "MlpPolicy", env, verbose=0
        )

        # train the model
        model.learn(total_timesteps=ntrain, progress_bar=True)
        print(
            f"\tFinished training the model for {ntrain} steps ... Testing it on a single world simulation"
        )

        # decide the model path to save to
        model_path = (
            MODEL_PATH.parent
            / f"{MODEL_PATH.name}{'_supplier' if as_supplier else '_consumer'}"
        )

        # save the model
        model.save(model_path)
        # remove the in-memory model
        del model
        # load the model
        model = TrainingAlgorithm.load(model_path)
        # try the model in a single simulation
        world = try_a_model(model, as_supplier)
        print(world.scores())


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else NTRAINING)
