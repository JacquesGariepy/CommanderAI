#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Localise la fenêtre d’un process par PID ou nom d’exécutable,
puis la prépare (restore, focus).
"""
import os
import logging
import psutil
import pywinauto
import win32gui
import win32process
import time
from typing import Optional, Any

logging.basicConfig(level=logging.DEBUG)

class WindowLocator:
    @staticmethod
    def find_window_by_pid(pid: int) -> Optional[Any]:
        try:
            def callback(handle, arr):
                try:
                    _, p_id = win32process.GetWindowThreadProcessId(handle)
                    if p_id == pid and win32gui.IsWindowVisible(handle):
                        arr.append(handle)
                except:
                    pass
                return True

            found = []
            win32gui.EnumWindows(callback, found)
            if found:
                app = pywinauto.Application().connect(handle=found[0])
                return app.window(handle=found[0])
        except Exception as e:
            logging.error(f"[WindowLocator] find_window_by_pid: {e}")
            return None
        return None

    @staticmethod
    def find_window_by_executable(exe_name: str, retries=20, delay=0.5) -> Optional[Any]:
        for _ in range(retries):
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if WindowLocator._match(proc, exe_name):
                        w = WindowLocator.find_window_by_pid(proc.info['pid'])
                        if w:
                            WindowLocator._prepare_window(w)
                            return w
            except:
                pass
            time.sleep(delay)
        return None

    @staticmethod
    def _match(proc: psutil.Process, exe_name: str) -> bool:
        if not proc or not proc.info:
            return False
        n = (proc.info.get('name') or '').lower()
        cmd = proc.info.get('cmdline', []) or []
        return (exe_name.lower() in n) or any(exe_name.lower() in c.lower() for c in cmd)

    @staticmethod
    def _prepare_window(w) -> None:
        try:
            if w.is_minimized():
                w.restore()
            if not w.is_visible():
                w.set_focus()
        except Exception as e:
            logging.error(f"[WindowLocator] prepare_window => {e}")
