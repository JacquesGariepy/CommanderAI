import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..tools.memory import PersistentMemory
logging.basicConfig(level=logging.DEBUG)

class ApplicationRegistry:
    def __init__(self, memory: PersistentMemory):
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
        logging.debug(f"Attempting to find executable: {appname}")
        search = appname.lower()
        if search in self.registry:
            path = self.registry[search]["path"]
            if path and os.path.isfile(path):
                return path
        return None

    def list_tools(self) -> List[Dict[str, Any]]:
        out = []
        for k, v in self.registry.items():
            if v["path"]:
                out.append({"name": k, **v})
        return out

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
