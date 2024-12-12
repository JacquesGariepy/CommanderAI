import ast
import locale
import os
import json
import asyncio
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import subprocess
import pyautogui
import psutil
import pywinauto
import win32gui
import win32process
from pywinauto import Application
from threading import Event
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage
from langchain.chains import LLMChain
import sys
import mss
import pytesseract
import numpy as np
from PIL import Image
import cv2
import speech_recognition as sr
import pyttsx3

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s - [%(pathname)s:%(lineno)d]",
    filename="automation.log",
    filemode="a",
)

# Constants
MEMORY_FILE = "memory.json"
TESSERACT_CONFIG = r'--oem 3 --psm 6'
MIN_ELEMENT_WIDTH = 20
MAX_ELEMENT_WIDTH = 300
MIN_ELEMENT_HEIGHT = 20
MAX_ELEMENT_HEIGHT = 100

system_language = locale.getdefaultlocale()[0]
llm_model = "gpt-4o-mini"

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Set speech rate

def speak_message(message: str):
    """Speak a message using text-to-speech."""
    logging.debug(f"speak_message: {message}")
    engine.say(message)
    engine.runAndWait()

def recognize_speech() -> str:
    """Recognize voice command and convert to text."""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            speak_message("I'm listening...")
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio, language="en-US")
                logging.info(f"Recognized command: {command}")
                return command.lower()
            except sr.UnknownValueError:
                speak_message("I didn't understand the command.")
                return ""
            except sr.WaitTimeoutError:
                speak_message("Command timeout.")
                return ""
            except Exception as e:
                logging.error(f"Voice recognition error: {e}")
                speak_message("Error during voice recognition.")
                return ""
    except sr.RequestError as e:
        logging.error(f"Request error for voice recognition service: {e}")
        speak_message("Request error for voice recognition service.")
        return ""
    except sr.MicrophoneUnavailableError:
        logging.error("No default input device available.")
        speak_message("No default input device available.")
        return ""
    except Exception as e:
        logging.error(f"Unexpected error during voice recognition: {e}")
        speak_message("Unexpected error during voice recognition.")
        return ""

class PersistentMemory:
    def __init__(self):
        logging.debug("Initializing PersistentMemory")
        self.memory = {}
        self.load_memory()

    def load_memory(self):
        try:
            if os.path.exists(MEMORY_FILE):
                with open(MEMORY_FILE, "r", encoding='utf-8') as f:
                    self.memory = json.load(f)
                logging.info("Memory loaded successfully.")
            else:
                logging.info("No existing memory. Initializing.")
        except Exception as e:
            logging.error(f"Error loading memory: {e}")
            self.memory = {}

    def save_memory(self):
        try:
            with open(MEMORY_FILE, "w", encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2)
            logging.info("Memory saved successfully.")
        except Exception as e:
            logging.error(f"Error saving memory: {e}")

    def update_memory(self, key: str, value: Any):
        logging.debug(f"Updating memory: {key} = {value}")
        self.memory[key] = value
        self.save_memory()

    def get(self, key: str, default: Any = None) -> Any:
        logging.debug(f"Getting memory: {key}")
        return self.memory.get(key, default)

class ScreenAnalyzer:
    def __init__(self):
        try:
            self.screen_capture = mss.mss()
            self.screen_capture_lock = asyncio.Lock()
            logging.info("ScreenAnalyzer initialized.")
        except Exception as e:
            logging.error(f"Error initializing ScreenAnalyzer: {e}")
            raise

    async def capture_screen(self) -> Optional[np.ndarray]:
        """Capture the screen safely with a lock."""
        try:
            async with self.screen_capture_lock:
                screen = await asyncio.to_thread(self._do_capture)
                if screen and hasattr(screen, 'rgb') and screen.rgb:
                    img = Image.frombytes('RGB', screen.size, screen.rgb)
                    return np.array(img)
                else:
                    logging.error("Screen capture failed or 'screen.rgb' is invalid.")
                    return None
        except Exception as e:
            logging.error(f"Screen capture failed with details: {e}", exc_info=True)
            return None

    def _do_capture(self):
        """Perform screen capture in a separate thread."""
        try:
            with mss.mss() as screen_capture:
                return screen_capture.grab(screen_capture.monitors[0])
        except Exception as e:
            logging.error(f"Error during native screen capture: {e}", exc_info=True)
            return None

    def analyze_screen(self) -> Tuple[str, List[Dict[str, Any]]]:
        """Analyze the current state of the screen."""
        try:
            screen_image = asyncio.run(self.capture_screen())
            if screen_image is None:
                logging.error("No screen image captured.")
                return "", []

            text = self._extract_text(screen_image)
            elements = self._detect_ui_elements(screen_image)

            return text, elements
        except Exception as e:
            logging.error(f"Screen analysis failed: {e}")
            return "", []

    def _extract_text(self, image: np.ndarray) -> str:
        """Extract text from the image using OCR."""
        try:
            if image is None or not isinstance(image, np.ndarray):
                logging.error("Invalid image for text extraction.")
                return ""
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            denoised = cv2.medianBlur(thresh, 3)
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(denoised, kernel, iterations=1)
            text = pytesseract.image_to_string(dilated, config=TESSERACT_CONFIG, lang='fra')
            return text.strip()
        except Exception as e:
            logging.error(f"Text extraction failed: {e}")
            return ""

    def _detect_ui_elements(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect UI elements in the image."""
        try:
            if image is None or not isinstance(image, np.ndarray):
                logging.error("Invalid image for UI element detection.")
                return []
            elements = []
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                if self._validate_element_size(w, h):
                    roi = gray[y:y + h, x:x + w]
                    element_type = self._classify_element(roi, w, h)
                    confidence = self._calculate_confidence(roi)

                    if confidence > 30:
                        element = {
                            "type": element_type,
                            "position": (x, y),
                            "size": (w, h),
                            "center": (x + w // 2, y + h // 2),
                            "confidence": confidence
                        }
                        elements.append(element)

            return elements
        except Exception as e:
            logging.error(f"UI element detection failed: {e}")
            return []

    def _validate_element_size(self, width: int, height: int) -> bool:
        """Validate the dimensions of the element."""
        return (MIN_ELEMENT_WIDTH < width < MAX_ELEMENT_WIDTH and
                MIN_ELEMENT_HEIGHT < height < MAX_ELEMENT_HEIGHT)

    def _classify_element(self, roi: np.ndarray, width: int, height: int) -> str:
        """Classify the detected element."""
        try:
            aspect_ratio = width / height if height != 0 else 0
            std_dev = np.std(roi) if roi.size > 0 else 0

            if 2.5 < aspect_ratio < 8 and std_dev > 30:
                return "text_field"
            elif 0.8 < aspect_ratio < 1.2:
                return "button"
            elif aspect_ratio > 8:
                return "menu"
            else:
                return "unknown"
        except Exception as e:
            logging.error(f"Element classification failed: {e}")
            return "unknown"

    def _calculate_confidence(self, roi: np.ndarray) -> float:
        """Calculate the confidence score of the element."""
        try:
            if roi.size == 0:
                return 0.0

            std_dev = np.std(roi)
            mean_val = np.mean(roi)
            base_confidence = (std_dev / 128.0) * 100
            mean_factor = (mean_val / 255.0) * 0.5 + 0.5

            confidence = base_confidence * mean_factor
            return min(100.0, max(0.0, confidence))
        except Exception as e:
            logging.error(f"Confidence calculation failed: {e}")
            return 0.0

class ApplicationRegistry:
    def __init__(self, memory: PersistentMemory):
        logging.debug("Initializing ApplicationRegistry")
        self.registry: Dict[str, Dict[str, Any]] = memory.get("registry", {})
        self.memory = memory
        logging.debug("ApplicationRegistry initialized.")
        self.ensure_default_keys()

    def ensure_default_keys(self):
        try:
            default_keys = {
                "path": None,
                "type": "unknown",
                "source": "dynamic",
                "launch_count": 0,
                "success_count": 0,
                "failure_count": 0
            }

            for app_name, app_details in self.registry.items():
                for key, default_value in default_keys.items():
                    if key not in app_details:
                        app_details[key] = default_value
                        logging.info(f"Added missing key '{key}' to '{app_name}'")
        except Exception as e:
            logging.error(f"Error adding default keys: {e}")

    def discover_tools(self):
        try:
            logging.info("Dynamically discovering tools...")
            paths = os.environ.get("PATH", "").split(os.pathsep)

            for path in paths:
                path_obj = Path(path)
                if path_obj.is_dir():
                    for exe in path_obj.glob("*.exe"):
                        tool_name = exe.stem.lower()
                        if tool_name not in self.registry:
                            self.registry[tool_name] = {
                                "path": str(exe),
                                "type": "executable",
                                "source": "PATH",
                                "launch_count": 0,
                                "success_count": 0,
                                "failure_count": 0
                            }
                            logging.debug(f"Discovered tool: {tool_name}")

            self.memory.update_memory("registry", self.registry)
            logging.info("Tool discovery completed.")
        except Exception as e:
            logging.error(f"Error discovering tools: {e}")

    def find_executable(self, application_name: str) -> Optional[str]:
        """Find the executable path of the application."""
        try:
            search_term = application_name.lower()
            logging.debug(f"Searching for '{search_term}' in the registry.")
            if search_term in self.registry:
                path = self.registry[search_term]["path"]
                if path and os.path.isfile(path):
                    return path
                logging.warning(f"Entry found but path is invalid: {path}")
            return None
        except Exception as e:
            logging.error(f"Error finding executable: {e}")
            return None

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        try:
            tools = []
            for name, details in self.registry.items():
                if self._validate_tool_entry(name, details):
                    tools.append({"name": name, **details})
            return tools
        except Exception as e:
            logging.error(f"Error listing tools: {e}")
            return []

    def _validate_tool_entry(self, name: str, details: Dict[str, Any]) -> bool:
        """Validate a tool registry entry."""
        required_keys = ["path", "type", "source"]
        return all(key in details for key in required_keys)

    def update_tool_stats(self, app_name: str, success: bool = True):
        """Update tool usage statistics."""
        try:
            app_name_lower = app_name.lower()
            if app_name_lower not in self.registry:
                self.registry[app_name_lower] = {
                    "path": None,
                    "type": "unknown",
                    "source": "dynamic",
                    "launch_count": 0,
                    "success_count": 0,
                    "failure_count": 0
                }

            self.registry[app_name_lower]["launch_count"] += 1
            if success:
                self.registry[app_name_lower]["success_count"] += 1
            else:
                self.registry[app_name_lower]["failure_count"] += 1

            self.memory.update_memory("registry", self.registry)
        except Exception as e:
            logging.error(f"Error updating tool stats: {e}")

class WindowLocator:
    @staticmethod
    def find_window_by_pid(pid: int) -> Optional[Any]:
        """Locate a window by its PID."""
        try:
            def callback(handle, windows):
                try:
                    _, process_id = win32process.GetWindowThreadProcessId(handle)
                    if process_id == pid:
                        visible = win32gui.IsWindowVisible(handle)
                        logging.debug(f"Window found - PID: {pid}, Handle: {handle}, Visible: {visible}")
                        if visible:
                            windows.append(handle)
                except Exception as e:
                    logging.error(f"Error in window callback: {e}", exc_info=True)
                return True

            windows = []
            win32gui.EnumWindows(callback, windows)

            if windows:
                logging.info(f"Windows found for PID {pid}: {len(windows)}")
                try:
                    window = Application().connect(handle=windows[0])
                    logging.debug(f"Successfully connected to handle {windows[0]}")
                    return window.window(handle=windows[0])
                except Exception as e:
                    logging.error(f"Failed to connect to window: {e}", exc_info=True)
                    return None
            else:
                logging.warning(f"No visible window found for PID {pid}")
                return None
        except Exception as e:
            logging.error(f"Error finding window by PID: {e}", exc_info=True)
            return None

    @staticmethod
    def find_window_by_executable(executable_name: str, retries: int = 20, delay: float = 0.5) -> Optional[Any]:
        """Find a window by the executable name."""
        for attempt in range(retries):
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if WindowLocator._match_process(proc, executable_name):
                            window = WindowLocator.find_window_by_pid(proc.info['pid'])
                            if window:
                                WindowLocator._prepare_window(window)
                                return window
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

            except Exception as e:
                logging.error(f"Error during attempt {attempt} to find window: {e}")

            time.sleep(delay)

        logging.error(f"Failed to find window for '{executable_name}' after {retries} attempts")
        return None

    @staticmethod
    def _match_process(proc: psutil.Process, executable_name: str) -> bool:
        """Check if the process matches the executable name."""
        try:
            if not proc or not proc.info:
                logging.debug(f"Invalid process or info for {executable_name}")
                return False

            name = proc.info.get('name', '').lower()
            cmdline = proc.info.get('cmdline', [])

            # Ensure cmdline is a list
            if cmdline is None:
                cmdline = []

            logging.debug(f"Checking process - Name: {name}, Cmd: {cmdline}")

            if not name and not cmdline:
                logging.debug("Process without name and command line")
                return False

            name_match = executable_name.lower() in name
            cmdline_match = any(executable_name.lower() in (cmd.lower() if cmd else '') for cmd in cmdline)

            logging.debug(f"Match - Name: {name_match}, Cmd: {cmdline_match}")
            return name_match or cmdline_match

        except Exception as e:
            logging.debug(f"Error matching process: {e}", exc_info=True)
            return False

    @staticmethod
    def _prepare_window(window) -> None:
        """Prepare the window for interaction."""
        try:
            if window.is_minimized():
                window.restore()
            if not window.is_visible():
                window.set_focus()
            logging.info(f"Window prepared: {window.window_text()}")
        except Exception as e:
            logging.error(f"Error preparing window: {e}")

class InteractionStrategies:
    """Strategies for interacting with the user interface."""

    def __init__(self):
        logging.debug("Initializing InteractionStrategies")
        self.last_interaction_time = 0
        self.MIN_INTERACTION_DELAY = 0.5  # Minimum delay between interactions

    def _enforce_interaction_delay(self):
        """Ensure a minimum delay between interactions."""
        current_time = time.time()
        time_since_last = current_time - self.last_interaction_time
        if time_since_last < self.MIN_INTERACTION_DELAY:
            time.sleep(self.MIN_INTERACTION_DELAY - time_since_last)
        self.last_interaction_time = time.time()

    def interact(self, window, action_description: str, max_retries: int = 3) -> bool:
        """
        Interact with an application using a natural language action description.
        Generate Python code with an LLM, validate it, and execute it.
        Retries up to 'max_retries' times if an error occurs.
        """
        for attempt in range(1, max_retries + 1):
            try:
                self._enforce_interaction_delay()

                # Create a detailed prompt with explicit examples
                prompt = f"""
                You are an AI assistant specialized in automating user interfaces in Python.
                The user wants to: "{action_description}".
                Follow the instructions below:
                1. Provide functional Python code using 'pywinauto' or 'pyautogui'.
                2. Validate that your code is self-contained and contains no syntax errors.
                3. ** Important all necessary imports and references must be included. The script will be executed as is, do not forget anything. **
                4. Return only the Python code, without any additional text and without markdown like ```python or ```, the code will be executed directly.

                Example to type "hello" in Notepad:
                
                from pywinauto import Application

                # Connect to Notepad
                app = Application().connect(path="notepad.exe")

                # Access the Notepad window
                notepad = app.top_window()

                # Type text into Notepad
                notepad.type_keys("hello")
            
                Provide only the Python code.
                """
                llm = ChatOpenAI(model=llm_model)
                chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
                response: AIMessage = chain.run({})
                logging.debug(f"Generated code for interaction: {response}")
                response = response.strip().replace("```python", "").replace("```", "")

                # Syntax check of the generated code
                try:
                    ast.parse(response)  # Syntax analysis
                except SyntaxError as e:
                    logging.error(f"Syntax error in generated code: {e}")
                    continue  # Retry on syntax error

                # Execute the generated code
                exec_locals = {"window": window, "pyautogui": pyautogui, "pywinauto": pywinauto}
                try:
                    exec(response, {}, exec_locals)
                except Exception as e:
                    logging.error(f"Execution error in generated code: {e}")
                    continue  # Retry on execution error

                logging.info(f"Interaction successful with action: {action_description}")
                return True

            except Exception as e:
                logging.error(f"Interaction failed on attempt {attempt}: {e}")
                continue  # Retry on any other error

        logging.error(f"Interaction failed after {max_retries} attempts.")
        return False


class TaskExecutor:
    """Task executor with comprehensive error handling and validation."""

    def __init__(self, app_registry: ApplicationRegistry):
        logging.debug("Initializing TaskExecutor")
        self.app_registry = app_registry
        self.stop_event = Event()
        self.interaction_strategies = InteractionStrategies()
        self.screen_analyzer = ScreenAnalyzer()
        self.last_analysis_time = 0
        self.MIN_ANALYSIS_INTERVAL = 1.0  # Minimum interval between screen analyses

    async def analyze_current_screen(self) -> Dict[str, Any]:
        """Analyze the current state of the screen with rate limiting."""
        try:
            current_time = time.time()
            if current_time - self.last_analysis_time < self.MIN_ANALYSIS_INTERVAL:
                await asyncio.sleep(self.MIN_ANALYSIS_INTERVAL - (current_time - self.last_analysis_time))

            text, elements = await asyncio.to_thread(self.screen_analyzer.analyze_screen)
            self.last_analysis_time = time.time()

            logging.debug(f"Screen analysis result: text='{text}', elements={elements}")
            return {
                "screen_text": text,
                "ui_elements": elements,
                "timestamp": self.last_analysis_time
            }
        except Exception as e:
            logging.error(f"Screen analysis failed: {e}")
            return {}

    def validate_step(self, step: Dict[str, Any], timeout: int = 10) -> bool:
        """Validate the execution of a step."""
        action = step.get("action")
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.stop_event.is_set():
                return False

            try:
                if action == "open":
                    if self._validate_open_action(step):
                        return True
                elif action == "interact":
                    if self._validate_interact_action(step):
                        return True
            except Exception as e:
                logging.error(f"Validation error: {e}")

            time.sleep(0.5)

        logging.warning(f"Validation timeout for step: {step}")
        return False

    def _validate_open_action(self, step: Dict[str, Any]) -> bool:
        """Validate the open application action."""
        try:
            application_name = step.get("application")
            for proc in pywinauto.findwindows.find_elements():
                if application_name.lower() in proc.name.lower():
                    logging.info(f"Validation: '{application_name}' is open")
                    return True
            return False
        except Exception as e:
            logging.error(f"Failed to validate open action: {e}")
            return False

    def _validate_interact_action(self, step: Dict[str, Any]) -> bool:
        """Validate the interaction action."""
        # For dynamic validation, we can compare the screen state before and after the action
        # or check for the presence of certain elements on the screen
        return True  # Simplification for this example

    async def execute_task(self, task_plan: Dict[str, Any]) -> str:
        """Execute the task plan."""
        try:
            results = []
            logging.info("Starting task execution...")

            for step in task_plan.get("steps", []):
                if self.stop_event.is_set():
                    return json.dumps({"status": "interrupted", "results": results})

                action = step.get("action")
                step_result = {"action": action, "status": "failure", "details": None}

                try:
                    if action == "capture_screen":
                        analysis = await self.analyze_current_screen()
                        step_result.update({
                            "status": "success",
                            "details": analysis
                        })
                    else:
                        result = await self._execute_step(step)
                        analysis = await self.analyze_current_screen()
                        step_result.update({
                            "status": "success" if result else "failure",
                            "details": result,
                            "screen_state": analysis
                        })

                except Exception as step_error:
                    logging.error(f"Step execution failed: {step_error}")
                    step_result["details"] = str(step_error)

                results.append(step_result)

                if step_result["status"] == "failure":
                    break

            return json.dumps({
                "status": "complete",
                "success": all(r["status"] == "success" for r in results),
                "results": results
            }, indent=2)

        except Exception as e:
            logging.error(f"Task execution failed: {e}")
            return json.dumps({
                "status": "error",
                "error": str(e),
                "results": results
            }, indent=2)

    async def _execute_step(self, step: Dict[str, Any]) -> str:
        """Execute a single step of the task plan."""
        action = step.get("action")

        if action == "open":
            return await self._handle_open_action(step)
        elif action == "interact":
            return await self._handle_interact_action(step)
        else:
            return f"Unknown action: {action}"

    async def _handle_open_action(self, step: Dict[str, Any]) -> str:
        """Handle the open application action."""
        application_name = step.get("application")
        executable_path = self.app_registry.find_executable(application_name)

        if not executable_path or not os.path.isfile(executable_path):
            self.app_registry.update_tool_stats(application_name, success=False)
            return f"Executable not found: {application_name}"

        try:
            process = await asyncio.to_thread(
                Application(backend="uia").start,
                executable_path
            )

            for _ in range(10):
                try:
                    window = await asyncio.to_thread(
                        WindowLocator.find_window_by_executable,
                        application_name
                    )
                    if window:
                        if self.validate_step(step):
                            self.app_registry.update_tool_stats(application_name)
                            return f"{application_name} opened successfully"
                        break
                except Exception:
                    await asyncio.sleep(0.5)

            self.app_registry.update_tool_stats(application_name, success=False)
            return f"Failed to validate launch of {application_name}"

        except Exception as e:
            self.app_registry.update_tool_stats(application_name, success=False)
            return f"Failed to open {application_name}: {str(e)}"

    async def _handle_interact_action(self, step: Dict[str, Any]) -> str:
        """Handle the interaction with the application."""
        details = step.get("details", {})
        process_name = details.get("process_name")
        action_description = details.get("action_description")

        try:
            window = await asyncio.to_thread(
                WindowLocator.find_window_by_executable,
                process_name
            )

            if not window:
                logging.error(f"Window not found for {process_name}")
                return f"Window not found for {process_name}"

            success = self.interaction_strategies.interact(window, action_description)
            logging.debug(f"Interaction result for '{action_description}': {success}")

            return "Interaction successful" if success else "Interaction failed"
        except Exception as e:
            logging.error(f"Interaction failed: {e}")
            return f"Interaction failed: {str(e)}"

class TaskInterpreter:
    """Interpret tasks from user commands."""

    def __init__(self, app_registry: ApplicationRegistry):
        logging.debug("Initializing TaskInterpreter")
        self.app_registry = app_registry

    async def interpret_task(self, user_request: str) -> Dict[str, Any]:
        """Interpret the user request to generate a task plan."""
        try:
            tool_names = [tool["name"] for tool in self.app_registry.list_tools()]
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
                            "action_description": "detailed description of the action to perform"
                        }}}},
                        {{"action": "capture_screen"}}
                    ]
                }}
                """
            )

            llm = ChatOpenAI(model=llm_model)
            chain = prompt_template | llm

            response = await chain.ainvoke({
                "user_request": user_request,
                "available_apps": available_apps
            })

            json_pattern = r"{.*}"
            match = re.search(json_pattern, response.content, re.DOTALL)
            if not match:
                raise ValueError("Invalid response format: no JSON found")

            task_plan = json.loads(match.group(0))
            self._validate_task_plan(task_plan)

            return task_plan

        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            raise ValueError(f"Invalid task plan format: {e}")
        except AttributeError as e:
            logging.error(f"Attribute error: {e}")
            raise ValueError(f"Attribute error: {e}")
        except Exception as e:
            logging.error(f"Task interpretation failed: {e}")
            raise

    def _validate_task_plan(self, task_plan: Dict[str, Any]) -> None:
        """Validate the structure and content of the task plan."""
        if not isinstance(task_plan, dict):
            raise ValueError("Task plan must be a dictionary")

        if "steps" not in task_plan:
            raise ValueError("Task plan must contain the 'steps' key")

        if not isinstance(task_plan["steps"], list):
            raise ValueError("Steps must be a list")

        for step in task_plan["steps"]:
            self._validate_step(step)

    def _validate_step(self, step: Dict[str, Any]) -> None:
        """Validate the structure of an individual step."""
        if not isinstance(step, dict):
            raise ValueError("A step must be a dictionary")

        if "action" not in step:
            raise ValueError("A step must contain the 'action' key")

        action = step["action"]
        if action not in ["open", "interact", "capture_screen"]:
            raise ValueError(f"Invalid action: {action}")

        if action == "open":
            if "application" not in step:
                raise ValueError("The 'open' action must contain the 'application' key")
        elif action == "interact":
            if "details" not in step:
                raise ValueError("The 'interact' action must contain the 'details' key")
            self._validate_interaction_details(step["details"])

    def _validate_interaction_details(self, details: Dict[str, Any]) -> None:
        """Validate the interaction details."""
        required_keys = ["process_name", "action_description"]
        for key in required_keys:
            if key not in details:
                raise ValueError(f"Interaction details must contain '{key}'")

async def main():
    """Main entry point of the application."""
    try:
        memory = PersistentMemory()
        app_registry = ApplicationRegistry(memory)

        if not memory.get("registry"):
            app_registry.discover_tools()

        task_interpreter = TaskInterpreter(app_registry)
        task_executor = TaskExecutor(app_registry)

        speak_message("Automation system initialized. Say 'help' for commands.")

        while True:
            try:
                # Ask the user if they want to use voice or text
                mode = input("\nDo you want to use voice or text to enter a command? (voice/text): ").strip().lower()

                if mode == "voice":
                    user_request = recognize_speech()
                elif mode == "text":
                    user_request = input("\nEnter a command (or 'exit' to quit): ").strip().lower()
                else:
                    speak_message("Unrecognized mode. Please choose 'voice' or 'text'.")
                    continue

                if "exit" in user_request:
                    speak_message("Shutting down the system. Goodbye.")
                    break
                elif "help" in user_request:
                    speak_message("You can ask me to perform tasks on your computer.")
                    continue
                elif "list" in user_request:
                    tools = app_registry.list_tools()
                    tool_names = [tool["name"] for tool in tools]
                    speak_message(f"Available applications: {', '.join(tool_names)}")
                    continue

                if user_request:
                    speak_message("Analyzing your request...")
                    task_plan = await task_interpreter.interpret_task(user_request)

                    speak_message("Executing the plan.")
                    result = await task_executor.execute_task(task_plan)
                    logging.info(f"Execution result: {result}")
                    speak_message("Task executed successfully.")

            except ValueError as e:
                logging.error(f"Request error: {e}")
                speak_message(f"Request error: {e}")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                speak_message(f"Unexpected error: {e}")

    except Exception as e:
        logging.critical(f"Critical error: {e}")
        speak_message(f"Critical error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        speak_message("Program interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        speak_message(f"Fatal error: {e}")
        sys.exit(1)
