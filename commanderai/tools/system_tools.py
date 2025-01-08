#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Outils divers : ex. CalculatorTool, etc.
"""
import os
import logging

class Tool:
    def __init__(self, name: str, description: str):
        logging.debug("[Tool] init called")
        self.name = name
        self.description = description

    def use(self, input_data: str) -> str:
        raise NotImplementedError()

# Tool example: calculator
class CalculatorTool(Tool):
    def __init__(self):
        logging.debug("[CalculatorTool] init called")
        super().__init__("calculator", "Simple calculator tool.")

    def use(self, input_data: str) -> str:
        try:
            return str(eval(input_data))
        except Exception as e:
            logging.error(f"[CalculatorTool] error with input '{input_data}': {e}")
            return "Calculation error."
