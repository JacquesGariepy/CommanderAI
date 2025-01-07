#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Planificateur STRIPS-like BFS (State, ActionOperator, STRIPSPlanner).
"""
import os
import logging
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Set

logging.basicConfig(level=logging.DEBUG)

class State:
    def __init__(self, facts: Dict[str, Any]):
        self.facts = dict(facts)

    def is_goal_satisfied(self, goal: Dict[str, Any]) -> bool:
        for k, v in goal.items():
            if k not in self.facts or self.facts[k] != v:
                return False
        return True

    def __eq__(self, other):
        return isinstance(other, State) and self.facts == other.facts

    def __hash__(self):
        items = tuple(sorted(self.facts.items()))
        return hash(items)

class ActionOperator:
    def __init__(self, name: str, preconds: Dict[str, Any], effects: Dict[str, Any]):
        self.name = name
        self.preconds = dict(preconds)
        self.effects = dict(effects)
    def is_applicable(self, state: State) -> bool:
        for k, v in self.preconds.items():
            if k not in state.facts or state.facts[k] != v:
                return False
        return True
    def apply(self, state: State) -> State:
        new_facts = dict(state.facts)
        for k, v in self.effects.items():
            if v is None:
                if k in new_facts: del new_facts[k]
            else:
                new_facts[k] = v
        return State(new_facts)
    def __str__(self):
        return f"ActionOperator({self.name})"

class STRIPSPlanner:
    def __init__(self, operators: List[ActionOperator]):
        self.operators = operators

    def plan(self, init_state: State, goal: Dict[str, Any]) -> Optional[List[str]]:
        visited: Set[State] = set()
        queue: deque[Tuple[State, List[str]]] = deque()
        queue.append((init_state, []))
        visited.add(init_state)

        while queue:
            current_state, path = queue.popleft()
            if current_state.is_goal_satisfied(goal):
                return path

            for op in self.operators:
                if op.is_applicable(current_state):
                    nxt = op.apply(current_state)
                    if nxt not in visited:
                        visited.add(nxt)
                        new_path = path + [op.name]
                        queue.append((nxt, new_path))

        return None
