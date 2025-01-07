#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Définitions des messages (KQML, ACL), protocoles, AgentType, etc.
"""
import os
import logging
from enum import Enum
from typing import Union

logging.basicConfig(level=logging.DEBUG)

class KQMLPerformative(Enum):
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"
    INFORM = "inform"
    REQUEST = "request"

class CommunicationProtocol(Enum):
    DIRECT = 1
    BROADCAST = 2
    MEDIATED = 3

class KQMLMessage:
    def __init__(self, performative: KQMLPerformative, content: str, sender: int, receiver: int):
        self.performative = performative
        self.content = content
        self.sender = sender
        self.receiver = receiver
    def __str__(self):
        return f"KQML({self.performative.value}, {self.sender}->{self.receiver}, {self.content})"

class Message:
    def __init__(self, sender: int, receiver: int, content: str,
                 protocol: CommunicationProtocol = CommunicationProtocol.DIRECT):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.protocol = protocol
    def __str__(self):
        return f"[Message] {self.sender}->{self.receiver} : {self.content} (via {self.protocol.name})"

class AgentType(Enum):
    COGNITIVE = 1
    REACTIVE = 2
    MEDIATOR = 3
