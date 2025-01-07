#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AgentManager : gestion globale des agents, routage interne.
"""
import os
import logging
from typing import Dict, Union
from .types import KQMLMessage, Message, CommunicationProtocol, AgentType
from .base_agents import BaseAgent

logging.basicConfig(level=logging.DEBUG)

class AgentManager:
    def __init__(self):
        self.agents: Dict[int, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent):
        self.agents[agent.id] = agent
        logging.info(f"[AgentManager] Registered agent {agent.id} ({agent.type.name})")

    def deliver_message(self, sender_id: int, msg_obj: Union[KQMLMessage, Message],
                        protocol: CommunicationProtocol):
        if isinstance(msg_obj, Message) and msg_obj.protocol != protocol:
            msg_obj.protocol = protocol

        if protocol == CommunicationProtocol.DIRECT:
            rid = msg_obj.receiver
            if rid in self.agents:
                self.agents[rid].receive_message(msg_obj)
            else:
                logging.warning(f"[AgentManager] unknown receiver {rid}")
        elif protocol == CommunicationProtocol.BROADCAST:
            for a_id, ag in self.agents.items():
                if a_id != sender_id:
                    ag.receive_message(msg_obj)
        elif protocol == CommunicationProtocol.MEDIATED:
            # On cherche un agent MEDIATOR
            mediator_id = None
            for a_id, ag in self.agents.items():
                if ag.type == AgentType.MEDIATOR:
                    mediator_id = a_id
                    break
            if mediator_id is not None:
                self.agents[mediator_id].receive_message(msg_obj)
            else:
                logging.warning("[AgentManager] no mediator found for MEDIATED.")
        else:
            logging.warning(f"[AgentManager] unknown protocol {protocol}")

    def step_all(self):
        for ag in self.agents.values():
            ag.reason()
        for ag in self.agents.values():
            ag.act()
