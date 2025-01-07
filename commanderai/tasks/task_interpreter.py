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

logging.basicConfig(level=logging.DEBUG)

class TaskInterpreter:
    def __init__(self, registry):
        self.registry = registry

    async def interpret_task(self, user_request: str) -> Dict[str, Any]:
        tool_names = [t["name"] for t in self.registry.list_tools()]
        available_apps = ", ".join(tool_names)

        prompt_template = PromptTemplate(
            input_variables=["user_request", "available_apps"],
            template="""
            You are an AI assistant tasked with executing user requests in a Windows environment.
            The available applications are:
            {available_apps}

            User request: "{user_request}"

            Generate a task plan in JSON format:
            {{
                "steps": [
                    {{"action": "open", "application": "application_name"}},
                    {{"action": "interact", "details": {{
                        "process_name": "process_name",
                        "action_description": "detailed description of the action"
                    }}}},
                    {{"action": "capture_screen"}}
                ]
            }}
            """
        )

        llm = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), model=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
        chain = prompt_template | llm

        response = await chain.ainvoke({
            "user_request": user_request,
            "available_apps": available_apps
        })

        match = re.search(r"{.*}", response.content, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in LLM response")

        plan = json.loads(match.group(0))
        if "steps" not in plan or not isinstance(plan["steps"], list):
            raise ValueError("Invalid plan from LLM")

        return plan
