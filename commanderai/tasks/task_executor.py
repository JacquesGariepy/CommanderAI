#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exécute un plan JSON (open, interact, capture_screen).
Valide et loggue succès/échecs.
"""
import os
import logging
import asyncio
import json
import time
import numpy as np
from threading import Event
from typing import Dict, Any, List

from .screen_analyzer import ScreenAnalyzer
from .window_locator import WindowLocator
from ..tools.memory import PersistentMemory
from ..tools.system_tools import Tool

import pywinauto
import pyautogui
import psutil

logging.basicConfig(level=logging.DEBUG)

def compare_screens(before: np.ndarray, after: np.ndarray) -> bool:
    if before.shape != after.shape:
        return True
    diff = float((abs(before.astype(float) - after.astype(float))).sum())
    return diff > 1_000_000

class InteractionStrategies:
    def __init__(self):
        self.last_interaction_time = 0
        self.MIN_DELAY = 0.5

    def _enforce_delay(self):
        now = time.time()
        if (now - self.last_interaction_time) < self.MIN_DELAY:
            time.sleep(self.MIN_DELAY - (now - self.last_interaction_time))
        self.last_interaction_time = time.time()

    def interact(self, window, action_description: str, max_retries: int = 3) -> bool:
        from langchain.prompts import PromptTemplate
        from langchain.chat_models import ChatOpenAI
        from langchain.chains import LLMChain
        from langchain.schema import AIMessage

        for attempt in range(max_retries):
            try:
                self._enforce_delay()
                prompt_txt = f"""
                You are an AI assistant specialized in automating user interfaces in Python.
                The user wants to: "{action_description}".
                1. Provide fully functional code using 'pywinauto' or 'pyautogui'.
                2. Code must be self-contained, no syntax errors.
                3. Return only python code, no markdown.
                """
                llm = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), model=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
                chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_txt))
                resp: AIMessage = chain.run({})
                code = resp.strip().replace("```python", "").replace("```", "")

                try:
                    ast.parse(code)
                except SyntaxError as e:
                    logging.error(f"[Interaction] syntax error => {e}")
                    continue

                exec_locals = {"window": window, "pywinauto": pywinauto, "pyautogui": pyautogui}
                try:
                    exec(code, {}, exec_locals)
                except Exception as e:
                    logging.error(f"[Interaction] exec error => {e}")
                    continue

                logging.info(f"[Interaction] success => {action_description}")
                return True
            except Exception as e:
                logging.error(f"[Interaction] attempt {attempt+1} => {e}")
                continue
        return False


class TaskExecutor:
    def __init__(self, memory: PersistentMemory, app_registry):
        self.memory = memory
        self.app_registry = app_registry
        self.stop_event = Event()
        self.strategies = InteractionStrategies()
        self.screen_analyzer = ScreenAnalyzer()
        self.last_analysis_time = 0
        self.MIN_INTERVAL = 1.0

    async def analyze_current_screen(self) -> Dict[str, Any]:
        now = time.time()
        if (now - self.last_analysis_time) < self.MIN_INTERVAL:
            await asyncio.sleep(self.MIN_INTERVAL - (now - self.last_analysis_time))

        txt, els = await asyncio.to_thread(self.screen_analyzer.analyze_screen)
        self.last_analysis_time = time.time()
        return {"screen_text": txt, "ui_elements": els, "timestamp": self.last_analysis_time}

    async def execute_task(self, task_plan: Dict[str, Any]) -> str:
        results = []
        logging.info("[TaskExecutor] Starting execute_task...")

        for step in task_plan.get("steps", []):
            if self.stop_event.is_set():
                break

            action = step.get("action", "")
            step_result = {"action": action, "status": "failure", "details": None}

            screen_before = await self.screen_analyzer.capture_screen()
            try:
                if action == "capture_screen":
                    analysis = await self.analyze_current_screen()
                    step_result["status"] = "success"
                    step_result["details"] = analysis
                    self.memory.record_success("capture_screen")

                elif action == "open":
                    detail = await self._handle_open(step)
                    analysis = await self.analyze_current_screen()
                    step_result["details"] = detail
                    step_result["screen_state"] = analysis
                    if "opened successfully" in detail:
                        step_result["status"] = "success"
                        self.memory.record_success("open_app")
                    else:
                        self.memory.record_failure("open_app")

                elif action == "interact":
                    detail = await self._handle_interact(step)
                    analysis = await self.analyze_current_screen()
                    step_result["details"] = detail
                    step_result["screen_state"] = analysis
                    if "successful" in detail:
                        step_result["status"] = "success"
                        self.memory.record_success("interact")
                    else:
                        self.memory.record_failure("interact")

                else:
                    step_result["details"] = f"Unknown action: {action}"

            except Exception as e:
                step_result["details"] = str(e)

            screen_after = await self.screen_analyzer.capture_screen()
            if step_result["status"] == "success" and screen_before is not None and screen_after is not None:
                changed = compare_screens(screen_before, screen_after)
                logging.info(f"[TaskExecutor] screen changed after '{action}' => {changed}")

            results.append(step_result)
            if step_result["status"] == "failure":
                break

        final = {
            "status": "complete",
            "success": all(r["status"] == "success" for r in results),
            "results": results
        }
        return json.dumps(final, indent=2)

    async def _handle_open(self, step: Dict[str, Any]) -> str:
        application_name = step.get("application", "")
        if not application_name:
            return "No application specified."

        from ..tools.memory import PersistentMemory

        logging.info(f"[TaskExecutor] Checking registry...")
        path = self.app_registry.find_executable(application_name)
        if not path:
            self.app_registry.update_tool_stats(application_name, success=False)
            return f"Executable not found: {application_name}"

        try:
            await asyncio.to_thread(pywinauto.Application(backend="uia").start, path)
            from .window_locator import WindowLocator
            for _ in range(10):
                w = await asyncio.to_thread(WindowLocator.find_window_by_executable, application_name)
                if w:
                    self.app_registry.update_tool_stats(application_name, success=True)
                    return f"{application_name} opened successfully"
                await asyncio.sleep(0.5)

            self.app_registry.update_tool_stats(application_name, success=False)
            return f"Failed to validate launch of {application_name}"
        except Exception as e:
            self.app_registry.update_tool_stats(application_name, success=False)
            return f"Failed to open {application_name}: {str(e)}"

    async def _handle_interact(self, step: Dict[str, Any]) -> str:
        details = step.get("details", {})
        proc_name = details.get("process_name", "")
        desc = details.get("action_description", "")

        if not proc_name or not desc:
            return "Missing process_name or action_description."

        from .window_locator import WindowLocator
        try:
            w = await asyncio.to_thread(WindowLocator.find_window_by_executable, proc_name)
            if not w:
                logging.error(f"[TaskExecutor] Window not found for {proc_name}")
                return f"Window not found for {proc_name}"

            success = self.strategies.interact(w, desc)
            return "Interaction successful" if success else "Interaction failed"
        except Exception as e:
            logging.error(f"[TaskExecutor] _handle_interact => {e}")
            return f"Interaction failed: {str(e)}"
