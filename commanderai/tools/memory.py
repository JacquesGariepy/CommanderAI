#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PersistentMemory pour stocker config et apprentissage.
"""
import os
import json
import logging
from typing import Any, Dict

logging.basicConfig(level=logging.DEBUG)

MEMORY_FILE = "memory.json"

class PersistentMemory:
    def __init__(self):
        logging.debug("Initializing PersistentMemory")
        self.memory: Dict[str, Any] = {}
        self.load_memory()

    def load_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding='utf-8') as f:
                    self.memory = json.load(f)
                logging.info("[PersistentMemory] loaded successfully.")
            except Exception as e:
                logging.error(f"[PersistentMemory] load error: {e}")
                self.memory = {}
        else:
            logging.info("[PersistentMemory] No existing memory. Creating new.")

    def save_memory(self):
        try:
            with open(MEMORY_FILE, "w", encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2)
            logging.info("[PersistentMemory] saved successfully.")
        except Exception as e:
            logging.error(f"[PersistentMemory] save error: {e}")

    def update_memory(self, key: str, value: Any):
        logging.debug(f"[PersistentMemory] update {key} = {value}")
        self.memory[key] = value
        self.save_memory()

    def get(self, key: str, default: Any = None):
        return self.memory.get(key, default)

    def record_success(self, action: str):
        if "learning" not in self.memory:
            self.memory["learning"] = {}
        if action not in self.memory["learning"]:
            self.memory["learning"][action] = {"successes": 0, "failures": 0}
        self.memory["learning"][action]["successes"] += 1
        self.save_memory()

    def record_failure(self, action: str):
        if "learning" not in self.memory:
            self.memory["learning"] = {}
        if action not in self.memory["learning"]:
            self.memory["learning"][action] = {"successes": 0, "failures": 0}
        self.memory["learning"][action]["failures"] += 1
        self.save_memory()
