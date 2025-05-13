"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML competition.
**提交给 ANAC 2024 SCML（OneShot 赛道）***
*团队* 在此处键入团队名称
*作者* 在此处键入团队成员的姓名和电子邮件地址

本代码可自由使用或更新，但须注明出处
作者和 ANAC 2024 SCML 竞赛。
"""

from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.action import FlexibleActionManager
from scml.oneshot.rl.common import model_wrapper

from .common import MODEL_PATH, MyObservationManager, TrainingAlgorithm, make_context

# used to repeat the response to every negotiator.


class MyAgent(OneShotRLAgent):
    """
    This is the only class you *need* to implement. The current skeleton simply loads a single model
    that is supposed to be saved in MODEL_PATH (train.py can be used to train such a model).
    这是您唯一*需要实现的类。 当前的骨架只加载一个模型
    该模型应保存在 MODEL_PATH（train.py 可用于训练该模型）。
    """

    def __init__(self, *args, **kwargs):
        # get full path to models (supplier and consumer models).
        base_name = MODEL_PATH.name
        self.paths = [
            MODEL_PATH.parent / f"{base_name}_supplier",
            MODEL_PATH.parent / f"{base_name}_consumer",
        ]
        models = tuple(model_wrapper(TrainingAlgorithm.load(_)) for _ in self.paths)
        contexts = (make_context(as_supplier=True), make_context(as_supplier=False))
        # update keyword arguments
        # 这里看起来像是分别初始化了作为supplier和consumer的Observation 和 Action Manager
        kwargs.update(
            dict(
                # load models from MODEL_PATH
                models=models,
                # create corresponding observation managers
                observation_managers=(
                    MyObservationManager(context=contexts[0]),
                    MyObservationManager(context=contexts[1]),
                ),
                action_managers=(
                    FlexibleActionManager(context=contexts[0]),
                    FlexibleActionManager(context=contexts[1]),
                ),
            )
        )
        # Initialize the base OneShotRLAgent with model paths and observation managers.
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    from .helpers.runner import run

    run([MyAgent])
