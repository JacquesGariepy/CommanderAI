#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BaseAgent, CognitiveAgent, ReactiveAgent, MediatorAgent
"""
import os
import logging
from typing import Dict, Any, List, Union, Optional, Tuple, Callable

from .types import KQMLMessage, KQMLPerformative, Message, CommunicationProtocol, AgentType
from .planner import STRIPSPlanner, State

# On permet l'injection du manager
agent_manager = None

class BaseAgent:
    def __init__(self, agent_id: int, agent_type: AgentType):
        logging.debug("[BaseAgent] init called")

        self.id = agent_id
        self.type = agent_type

        # BDI
        self.beliefs: Dict[str, Any] = {}
        self.desires: List[str] = []
        self.intentions: List[str] = []
        self.plans: List[str] = []
        self.goals: List[Dict[str, Any]] = []

        # Engagements
        self.commitments: Dict[int, List[str]] = {}

        # Tools
        self.tools: Dict[str, Any] = {}

        # Inbox
        self.inbox: List[Union[Message, KQMLMessage]] = []

        # STRIPS Planner
        self.strips_planner: Optional[STRIPSPlanner] = None

    def add_tool(self, tool: Any):
        self.tools[tool.name] = tool

    def add_planner(self, planner: STRIPSPlanner):
        self.strips_planner = planner

    def perceive(self, context: Dict[str, Any]):
        logging.debug(f"[Agent {self.id}] perceive => {context}")
        self.beliefs.update(context)

    def reason(self):
        logging.debug(f"[Agent {self.id}] reason start.")
        self._process_inbox()
        self._update_intentions_from_desires()
        self._plan_for_goals()
        logging.debug(f"[Agent {self.id}] reason end.")

    def act(self):
        logging.debug(f"[Agent {self.id}] act start.")
        if self.intentions:
            intent = self.intentions.pop(0)
            logging.info(f"[Agent {self.id}] executing intention => {intent}")
            # Ici, on peut implémenter la logique d’appel direct au TaskExecutor,
            # si l’intention correspond à un plan (ex.: "open:notepad")
            return

        if self.plans:
            actions = self.plans[0].split(",")
            if actions:
                next_action = actions.pop(0).strip()
                self.plans[0] = ",".join(actions).strip()

                # Vérifier si c’est un outil
                for tname, t in self.tools.items():
                    if next_action.startswith(tname):
                        inp = next_action.replace(tname + " ", "")
                        res = t.use(inp)
                        logging.info(f"[Agent {self.id}] used {tname} => {inp}, result={res}")
                        self.beliefs["last_tool_result"] = res
                        return

                logging.info(f"[Agent {self.id}] local plan action => {next_action}")

            if not self.plans[0]:
                self.plans.pop(0)
        logging.debug(f"[Agent {self.id}] act end.")

    async def handle_user_request(self, user_request: str, interpreter, executor):
        """
        Reçoit une requête (texte). Utilise TaskInterpreter => plan JSON => exécution via TaskExecutor.
        Met à jour beliefs.
        """
        logging.info(f"[Agent {self.id}] handle_user_request => {user_request}")
        try:
            plan = await interpreter.interpret_task(user_request)
            logging.info(f"[Agent {self.id}] plan from LLM => {plan}")

            result = await executor.execute_task(plan)
            logging.info(f"[Agent {self.id}] execution result => {result}")
            self.beliefs["last_task_result"] = result
        except Exception as e:
            logging.error(f"[Agent {self.id}] handle_user_request error => {e}")

    def _update_intentions_from_desires(self):
        for d in self.desires:
            if d not in self.beliefs.values():
                new_int = f"achieve_{d}"
                if new_int not in self.intentions:
                    self.intentions.append(new_int)
                    logging.debug(f"[Agent {self.id}] new intention => {new_int}")

    def _plan_for_goals(self):
        if not self.strips_planner:
            return
        for g in self.goals:
            if not self._goal_satisfied(g):
                logging.info(f"[Agent {self.id}] planning STRIPS for => {g}")
                plan_ops = self.strips_planner.plan(State(self.beliefs), g)
                if plan_ops:
                    self.plans.append(",".join(plan_ops))
                    logging.info(f"[Agent {self.id}] appended plan => {plan_ops}")
                else:
                    logging.warning(f"[Agent {self.id}] no plan for => {g}")

    def _goal_satisfied(self, goal: Dict[str, Any]) -> bool:
        for k, v in goal.items():
            if k not in self.beliefs or self.beliefs[k] != v:
                return False
        return True

    def _process_inbox(self):
        if not self.inbox:
            return
        to_remove = []
        for msg in self.inbox:
            if isinstance(msg, KQMLMessage):
                self._handle_kqml(msg)
            elif isinstance(msg, Message):
                logging.info(f"[Agent {self.id}] Received => {msg}")
            to_remove.append(msg)
        for m in to_remove:
            self.inbox.remove(m)

    def _handle_kqml(self, kmsg: KQMLMessage):
        logging.debug(f"[Agent {self.id}] handle_kqml => {kmsg}")
        if kmsg.performative == KQMLPerformative.PROPOSE:
            # Rejet par défaut
            from .types import KQMLPerformative
            rej = KQMLMessage(KQMLPerformative.REJECT, f"Rejected: {kmsg.content}", self.id, kmsg.sender)
            self.deliver_message(rej)
        elif kmsg.performative == KQMLPerformative.ACCEPT:
            logging.info(f"[Agent {self.id}] KQML ACCEPT => {kmsg.content}")
        elif kmsg.performative == KQMLPerformative.REJECT:
            logging.info(f"[Agent {self.id}] KQML REJECT => {kmsg.content}")
        elif kmsg.performative == KQMLPerformative.REQUEST:
            logging.info(f"[Agent {self.id}] KQML REQUEST => {kmsg.content}")
        elif kmsg.performative == KQMLPerformative.INFORM:
            logging.info(f"[Agent {self.id}] KQML INFORM => {kmsg.content}")

    def deliver_message(self, msg_obj: Union[KQMLMessage, Message]):
        logging.debug(f"[Agent {self.id}] deliver_message => {msg_obj}")
        if agent_manager:
            protocol = msg_obj.protocol if isinstance(msg_obj, Message) else CommunicationProtocol.DIRECT
            agent_manager.deliver_message(self.id, msg_obj, protocol)
        else:
            logging.warning("No agent_manager defined -> can't deliver message.")

    def receive_message(self, msg_obj: Union[KQMLMessage, Message]):
        logging.debug(f"[Agent {self.id}] receive_message => {msg_obj}")
        self.inbox.append(msg_obj)


class CognitiveAgent(BaseAgent):
    def __init__(self, agent_id: int):
        logging.debug("[CognitiveAgent] init called")
        super().__init__(agent_id, AgentType.COGNITIVE)

class ReactiveAgent(BaseAgent):
    def __init__(self, agent_id: int):
        logging.debug("[ReactiveAgent] init called")
        super().__init__(agent_id, AgentType.REACTIVE)
        self.rules: List[Tuple[Callable[[Dict[str, Any]], bool], Callable[['ReactiveAgent'], None]]] = []

    def add_rule(self, condition_fn: Callable[[Dict[str, Any]], bool],
                 action_fn: Callable[['ReactiveAgent'], None]):
        self.rules.append((condition_fn, action_fn))

    def reason(self):
        self._process_inbox()
        for c, a in self.rules:
            if c(self.beliefs):
                self.intentions.append(a.__name__)
                logging.info(f"[ReactiveAgent {self.id}] triggered => {a.__name__}")
                break

class MediatorAgent(BaseAgent):
    def __init__(self, agent_id: int):
        logging.debug("[MediatorAgent] init called")
        super().__init__(agent_id, AgentType.MEDIATOR)

    def _process_inbox(self):
        from .types import CommunicationProtocol
        if not self.inbox:
            return
        to_remove = []
        for msg in self.inbox:
            if isinstance(msg, Message) and msg.protocol == CommunicationProtocol.MEDIATED:
                logging.info(f"[MediatorAgent {self.id}] routing MEDIATED => {msg}")
                if agent_manager:
                    agent_manager.deliver_message(self.id, msg, CommunicationProtocol.DIRECT)
            elif isinstance(msg, KQMLMessage):
                logging.info(f"[MediatorAgent {self.id}] routing KQML => {msg}")
                if agent_manager:
                    agent_manager.deliver_message(self.id, msg, CommunicationProtocol.DIRECT)
            to_remove.append(msg)
        for m in to_remove:
            self.inbox.remove(m)
