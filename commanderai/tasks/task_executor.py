#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exécute un plan JSON (open, interact, capture_screen).
Valide et loggue succès/échecs.
"""
import os
import ast
import logging
import asyncio
import json
import time
import numpy as np
from threading import Event
from typing import Dict, Any, List

from ..core.config import Config
from .screen_analyzer import ScreenAnalyzer
from .window_locator import WindowLocator
from ..tools.memory import PersistentMemory
from ..tools.system_tools import Tool

import pywinauto
import pyautogui
import psutil

def compare_screens(before: np.ndarray, after: np.ndarray) -> bool:
    logging.debug(f"[compare_screens] before: {before.shape}, after: {after.shape}")
    if before.shape != after.shape:
        return True
    diff = float((abs(before.astype(float) - after.astype(float))).sum())
    return diff > 1_000_000

# not used at the moment but could be useful in the future
def compare_memory_files(agents_memory_path: str, main_memory_path: str):
    logging.debug(f"[compare_memory_files] {agents_memory_path} vs {main_memory_path}")
    try:
        with open(agents_memory_path, "r", encoding="utf-8") as f1, \
             open(main_memory_path, "r", encoding="utf-8") as f2:
            agents_data = json.load(f1)
            main_data = json.load(f2)

        agents_registry = agents_data.get("registry", {})
        main_registry = main_data.get("registry", {})

        missing_in_agents = set(main_registry.keys()) - set(agents_registry.keys())
        missing_in_main = set(agents_registry.keys()) - set(main_registry.keys())

        if missing_in_agents:
            logging.info(f"Applications present in main but absent in agents: {missing_in_agents}")
        if missing_in_main:
            logging.info(f"Applications present in agents but absent in main: {missing_in_main}")
    except Exception as e:
        logging.error(f"Error comparing memory files: {e}")

class InteractionStrategies:
    def __init__(self):
        logging.debug("[InteractionStrategies] init called")
        self.last_interaction_time = 0
        self.MIN_DELAY = 0.5
        self.openai_api_key = Config.OPENAI_API_KEY
        # self.LLM_MODEL = Config.LLM_MODEL 
        # self.MAX_RETRIES = Config.MAX_RETRIES
        # self.MAX_TOKENS = Config.MAX_TOKENS
        # self.TEMPERATURE = Config.TEMPERATURE
        # self.TIMEOUT = Config.TIMEOUT

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
                You are an AI assistant specialized in automating user interfaces on Windows using Python.  
                The user wants to: "{action_description}".

                Constraints:  
                1. You must return strictly executable Python code with no special characters or Markdown formatting.  
                2. The code must work with `pywinauto` or `pyautogui`.  
                3. You may use the `window` object (already connected or previously obtained) to interact with the target window.  
                4. Provide fully functional code using `pywinauto` or `pyautogui`.  
                5. Code must be self-contained with no syntax errors.  
                6. Return only Python code, no Markdown formatting or additional text.  

                If the action requires a click, keyboard input, or any other type of interaction, code it using `pywinauto` or `pyautogui`.  
                If necessary, handle exceptions (for example, by capturing errors if the window or control does not exist).  

                Usage examples:  
                - To click on an identified button, use `window['ButtonName'].click_input()` (`pywinauto`).  
                - To type text, use `pyautogui.typewrite("my text")` or equivalent functions in `pywinauto`.  

                Strictly follow these instructions and return only the final Python code.
                """
                llm = ChatOpenAI(api_key=self.openai_api_key, model=self.LLM_MODEL)
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
        logging.debug("[TaskExecutor] init called")
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
        logging.debug(f"[TaskExecutor] Received task plan: {json.dumps(task_plan, indent=2)}")
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
                    step_result["details"] = await self._handle_open_with_validation(step)
                    if "opened successfully" in step_result["details"]:
                        step_result["status"] = "success"
                    analysis = await self.analyze_current_screen()
                    step_result["screen_state"] = analysis
                    if "opened successfully" in step_result["details"]:
                        step_result["status"] = "success"
                        self.memory.record_success("open_app")
                    else:
                        self.memory.record_failure("open_app")

                elif action == "interact":
                    interaction_result = await self._handle_interact_with_validation(step)
                    step_result["details"] = interaction_result
                    if "successful" in interaction_result:
                        step_result["status"] = "success"
                    analysis = await self.analyze_current_screen()
                    step_result["screen_state"] = analysis
                    if "successful" in interaction_result:
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

    async def _handle_open_with_validation(self, step: Dict[str, Any]) -> str:
        application_name = step.get("application", "")
        if not application_name:
            return "No application specified."

        logging.debug(f"[TaskExecutor] Trying to open => '{application_name}'")
        path = self.app_registry.find_executable(application_name)
        logging.debug(f"[TaskExecutor] find_executable('{application_name}') => {path}")
        if not path or not os.path.isfile(path):
            self.app_registry.update_tool_stats(application_name, success=False)
            return f"Executable not found: {application_name}"

        try:
            await asyncio.to_thread(pywinauto.Application(backend='uia').start, path)
        except Exception as e:
            self.app_registry.update_tool_stats(application_name, success=False)
            return f"Failed to open {application_name}: {e}"

        if await self._validate_open_action(application_name):
            self.app_registry.update_tool_stats(application_name, success=True)
            return f"{application_name} opened successfully"
        else:
            self.app_registry.update_tool_stats(application_name, success=False)
            return f"Failed to validate launch of {application_name}"

    async def _validate_open_action(self, application_name: str, retries: int = 10, delay: float = 0.5) -> bool:
        for attempt in range(retries):
            window = await asyncio.to_thread(WindowLocator.find_window_by_executable, application_name)
            if window:
                logging.debug(f"[TaskExecutor] found window on attempt {attempt+1}")
                return True
            logging.debug(f"[TaskExecutor] not found, attempt {attempt+1}/{retries}")
            await asyncio.sleep(delay)
        return False

    async def _handle_interact_with_validation(self, step: Dict[str, Any]) -> str:
        details = step.get("details", {})
        proc_name = details.get("process_name", "")
        desc = details.get("action_description", "")
        if not proc_name or not desc:
            return "Missing process_name or action_description."

        logging.debug(f"[TaskExecutor] Interact => process: {proc_name}, action: {desc}")
        window = await asyncio.to_thread(WindowLocator.find_window_by_executable, proc_name)
        if not window:
            return f"Window not found for {proc_name}"

        success = self.strategies.interact(window, desc)
        return "Interaction successful" if success else "Interaction failed"
