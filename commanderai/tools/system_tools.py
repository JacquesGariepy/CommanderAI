#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Outils divers : ex. CalculatorTool, etc.
"""
import os
import logging

logging.basicConfig(level=logging.DEBUG)

class Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def use(self, input_data: str) -> str:
        raise NotImplementedError()

class CalculatorTool(Tool):
    def __init__(self):
        super().__init__("calculator", "Effectue des calculs mathématiques.")

    def use(self, input_data: str) -> str:
        try:
            return str(eval(input_data))
        except Exception as e:
            logging.error(f"[CalculatorTool] error with input '{input_data}': {e}")
            return "Calculation error."
