#!/usr/bin/env python
# Temporarily deprecated!! Don't use this agent for the competition at this moment
"""
**Submitted to ANAC 2024 SCML (Std track)**
*Authors* type-your-team-member-names-with-their-emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.

AWI: Agent World Interface
*代理世界接口*
SAO: Supplier Agent Offer
*供应商代理报价*
SAOState: Supplier Agent Offer State
*供应商代理报价状态*
SAOResponse: Supplier Agent Offer Response
*供应商代理报价响应*
NMI: Negotiation Manager Interface
*谈判管理接口*

"""
from __future__ import annotations

# required for typing
from typing import Any

# required for development
from scml.std import StdAWI, StdSyncAgent, StdRLAgent

# required for typing
from negmas import Contract, Outcome, SAOResponse, SAOState

from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.action import FlexibleActionManager
from scml.oneshot.rl.common import model_wrapper

from common import MODEL_PATH, MyObservationManager, TrainingAlgorithm, make_context

__all__ = ["LitaAgent"]

class LitaAgent(StdRLAgent):
    """
    This is the only class you *need* to implement. The current skeleton has a
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by
    calling methods in the agent-world-interface instantiated as `self.awi`
    in your agent. See the documentation for more details
    这是您唯一需要实现的类。当前的骨架代码包含一个基本的空实现。
    您可以根据需要修改其中的任何部分。您可以通过调用代理与世界接口（以 `self.awi` 实例化）中的方法来与世界交互。如需更多详细信息，请参阅文档。



    **Please change the name of this class to match your agent name**

    Remarks:
        - You can get a list of partner IDs using `self.negotiators.keys()`. This will
          always match `self.awi.my_suppliers` (buyer) or `self.awi.my_consumers` (seller).
        - You can get a dict mapping partner IDs to `NegotiationInfo` (including their NMIs)
          using `self.negotiators`. This will include negotiations currently still running
          and concluded negotiations for this day. You can limit the dict to negotiations
          currently running only using `self.active_negotiators`
        - You can access your ufun using `self.ufun` (See `StdUFun` in the docs for more details).
        - 您可以使用 `self.negotiators.keys（）` 获取合作伙伴 ID 的列表。这将始终与 `self.awi.my_suppliers`
          （买家）或 `self.awi.my_consumers`（卖家）匹配。
        - 您可以使用 `self.negotiators` 获取一个字典，该字典将合作伙伴 ID 映射到 `NegotiationInfo`（包括其 NMIs）。
          这将包括当天仍在进行的谈判以及已完成的谈判。您可以使用 `self.active_negotiators` 将字典限制为仅包含当前正在进行的谈判。
        - 您可以使用 `self.ufun` 访问您的 ufun（有关更多详细信息，请参阅文档中的 `StdUFun`）。


    """
    # =====================
    # Initialization
    # =====================
    def __init__(self, *args, **kwargs):
        # get full path to models (supplier and consumer models).
        # 拟使用的RL模型的路径
        base_name = MODEL_PATH.name
        self.paths = [
            MODEL_PATH.parent / f"{base_name}_supplier",
            MODEL_PATH.parent / f"{base_name}_consumer",
            ]

        # 把模型包起来，大抵相当于initialize？
        # 以及这模型到底是哪儿来的？
        models = tuple(model_wrapper(TrainingAlgorithm.load(_)) for _ in self.paths)

        # 生成上下文 这context似乎是为supplier和consumer分别生成的
        # 但是我还是不明白这context究竟是什么，是否会用来生成world（看起来不会，因为是world调用的agent，那这个context是干嘛的？）
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
        super().__init__(*args, production_level=0.25, future_concession=0.1, **kwargs)

        # 看看自己在哪一层，头尾还是中间
        # 如果有购买的外生协议，则不作为买方谈判者
        # 如果有销售的外生协议，则不作为卖方谈判者
        if self.awi.is_first_level:
            total_needs = self.awi.needed_sales
            self.is_seller_negotiator = True
            self.is_buyer_negotiator = False
        elif self.awi.is_last_level:
            total_needs = self.awi.needed_supplies
            self.is_seller_negotiator = False
            self.is_buyer_negotiator = True
        else:
            # n_lines是生产线数量，乘以一个production level即生产线的生产水平（可以理解为产量，在0-1之间）
            # official的skeleton做了一个假设，即假设p_l是0.25，但这个值应该调整
            total_needs = self.production_level * self.awi.n_lines
            self.is_seller_negotiator = True
            self.is_buyer_negotiator = True

    # =====================
    # Negotiation Callbacks
    # =====================

    def first_proposals(self) -> dict[str, Outcome | None]:
        """
        Decide a first proposal for every partner.

        Remarks:
            - During this call, self.active_negotiators and self.negotiators will return the same dict
            - The negotiation issues will ALWAYS be the same for all negotiations running concurrently.
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """
        #see what we have here
        pass

        # 在这里采用什么策略提出第一轮的初始报价？
        pass





        return dict()

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        """
        Decide how to respond to every partner with which negotiations are still running.

        Remarks:
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """
        return dict()

    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        """Called once after the agent-world interface is initialized"""

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""

    def step(self):
        """Called at at the END of every production step (day)"""

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_negotiation_failure(  # type: ignore
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: StdAWI,
        state: SAOState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without agreement"""

    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:  # type: ignore
        """Called when a negotiation the agent is a party of ends with agreement"""


if __name__ == "__main__":
    import sys

    from .helpers.runner import run

    run([LitaAgent], sys.argv[1] if len(sys.argv) > 1 else "std")
