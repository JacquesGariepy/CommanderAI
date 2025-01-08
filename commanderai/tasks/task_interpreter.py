#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interprète la requête utilisateur (LLM) => JSON plan (open, interact, capture_screen).
"""
import os
import logging
import re
import json
import asyncio
from typing import Dict, Any

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage
from langchain.chains import LLMChain

from ..core.config import Config

class TaskInterpreter:
    def __init__(self, registry):
        logging.debug("[TaskInterpreter] init called")
        self.registry = registry
        self.openai_api_key = Config.OPENAI_API_KEY
        self.LLM_MODEL = Config.LLM_MODEL 
        # self.MAX_RETRIES = Config.MAX_RETRIES
        # self.MAX_TOKENS = Config.MAX_TOKENS
        # self.TEMPERATURE = Config.TEMPERATURE
        # self.TIMEOUT = Config.TIMEOUT
        #logging.debug(f"[TaskInterpreter] config => OPENAI_API_KEY: {self.openai_api_key}, LLM_MODEL: {self.LLM_MODEL}, MAX_RETRIES: {self.MAX_RETRIES}, MAX_TOKENS: {self.MAX_TOKENS}, TEMPERATURE: {self.TEMPERATURE}, TIMEOUT: {self.TIMEOUT}")
        
    async def interpret_task(self, user_request: str) -> Dict[str, Any]:
        logging.info(f"[TaskInterpreter] interpret_task: {user_request}")
        tool_names = [t["name"] for t in self.registry.list_tools()]
        available_apps = ", ".join(tool_names)

        logging.debug("[TaskInterpreter] Building prompt and chain.")
        prompt_template = PromptTemplate(
            input_variables=["user_request", "available_apps"],
            template="""
            You are CommanderAI, an AI assistant for Windows designed to provide a plan of automated actions.  
            You master Windows and want to help users accomplish tasks.
            You must provide a plan of actions to respond to the user's request using only the available applications.
            You are proficient with the following applications: {available_apps}.
            These actions will address the user's request by using only the available applications.

            Context & Constraints:
            1. **You master the available applications, list of available applications**: {available_apps}.
            2. **User request**: {user_request}.
            3. You must return **only** a JSON object following this exact format:

            ```json
            {{
            "steps": [
                {{
                "action": "...",
                ...
                }},
                ...
            }}
            ```

            4. Possible values for "action" include (but are not limited to):
            - "open": to open an application  
                - Mandatory field: "application" containing the name of the application to open (as listed in the available applications, if applicable).  
            - "interact": to interact with a window or component  
                - Mandatory field: "details", which itself contains:  
                - "process_name": the name of the process (i.e., the executable) already open to act upon.  
                - "action_description": a precise textual description of what to do (click at a specific location, type text, etc.).  
            - "capture_screen": to take a screenshot and analyze its content.  

            5. Each step must be minimal and **strictly** contained within a JSON object in the "steps" list.  
            6. **No** text, phrases, or comments outside the JSON object should be returned.  
            7. If no solution is possible or the user's request is beyond scope, still return a valid JSON object with `"steps": []` or include the reasons for impossibility in a field (e.g., `"details"`).

            Follow this process to generate the step-by-step action plan to best address the request:
            - Analyze the user request {user_request}.  
            - Check which available applications {available_apps} are useful.  
            - Determine the sequence of required actions.  
            - Return this sequence of actions as follows:

            ```json
            {{
            "steps": [
                {{ "action": "open", "application": "..." }},
                {{ "action": "interact", "details": {{
                    "process_name": "...",
                    "action_description": "..."
                }}}},
                ...
            ]
            ```

            **Important**: No text, phrases, or comments outside the JSON braces. Only the final JSON object must appear.
        """
        )

        llm = ChatOpenAI(api_key=self.openai_api_key, model=self.LLM_MODEL)
        chain = prompt_template | llm

        logging.debug("[TaskInterpreter] Invoking LLM chain...")
        response = await chain.ainvoke({
            "user_request": user_request,
            "available_apps": available_apps
        })

        logging.debug("[TaskInterpreter] Parsing LLM response for JSON.")
        match = re.search(r"{.*}", response.content, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in LLM response")

        logging.debug("[TaskInterpreter] JSON successfully extracted.")
        plan = json.loads(match.group(0))
        if "steps" not in plan or not isinstance(plan["steps"], list):
            raise ValueError("Invalid plan from LLM")

        logging.info(f"[TaskInterpreter] Plan steps => {plan['steps']}")
        return plan
