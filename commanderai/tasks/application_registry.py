import os
import logging
import difflib
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..tools.memory import PersistentMemory

class ApplicationRegistry:
    def __init__(self, memory: PersistentMemory):
        logging.debug("[ApplicationRegistry] init called")
        self.memory = memory
        self.registry: Dict[str, Dict[str, Any]] = self.memory.get("registry", {})
        self.ensure_defaults()

    def ensure_defaults(self):
        default = {
            "path": None,
            "type": "unknown",
            "source": "dynamic",
            "launch_count": 0,
            "success_count": 0,
            "failure_count": 0
        }
        for app_name, details in self.registry.items():
            for k, v in default.items():
                if k not in details:
                    details[k] = v

    def discover_tools(self):
        logging.info("[ApplicationRegistry] discovering tools in PATH...")
        paths = os.environ.get("PATH", "").split(os.pathsep)
        for p in paths:
            p_obj = Path(p)
            if p_obj.is_dir():
                for exe in p_obj.glob("*.exe"):
                    tname = exe.stem.lower()
                    if tname not in self.registry:
                        self.registry[tname] = {
                            "path": str(exe), "type": "executable",
                            "source": "PATH", "launch_count": 0,
                            "success_count": 0, "failure_count": 0
                        }
        self.memory.update_memory("registry", self.registry)

    def find_executable(self, appname: str) -> Optional[str]:
        """
        Attempt to find an executable by `appname`. 
        1) Exact lookup in self.registry
        2) If not found, tries an approximate match using difflib
        3) Return None if no suitable candidate is found
        """
        if not appname:
            return None

        search = appname.strip().lower()

        # 1) Exact match
        if search in self.registry:
            path = self.registry[search].get("path")
            if path and os.path.isfile(path):
                return path

        # 2) Fallback: approximate matching across the existing registry keys
        all_keys = list(self.registry.keys())
        matches = difflib.get_close_matches(search, all_keys, n=1, cutoff=0.6)
        #  -> n=1 means we only care about the single best match
        #  -> cutoff=0.6 requires at least 60% similarity to consider

        if matches:
            best_match = matches[0]
            path = self.registry[best_match].get("path")
            if path and os.path.isfile(path):
                logging.debug(
                    f"[ApplicationRegistry] Using approximate match '{best_match}' for '{search}' => {path}"
                )
                return path

        # 3) If no match found, return None
        return None

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": k, **v}
            for k, v in self.registry.items()
            if v.get("path") and os.path.isfile(v["path"])
        ]

    def update_tool_stats(self, app_name: str, success: bool):
        try:
            low = app_name.lower()
            if low not in self.registry:
                self.registry[low] = {
                    "path": None,
                    "type": "unknown",
                    "source": "dynamic",
                    "launch_count": 0,
                    "success_count": 0,
                    "failure_count": 0
                }
            self.registry[low]["launch_count"] += 1
            if success:
                self.registry[low]["success_count"] += 1
            else:
                self.registry[low]["failure_count"] += 1
            self.memory.update_memory("registry", self.registry)
        except Exception as e:
            logging.error(f"Error updating tool stats: {e}")
