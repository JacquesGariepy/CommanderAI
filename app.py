import ast
import locale
import os
import json
import asyncio
import logging
import re
import time
import configparser
import shlex
from pathlib import Path
from typing import List, Dict, Any, Literal, Tuple, Optional, TypedDict, Union
import subprocess
import pyautogui
import psutil
import platform

IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# Platform-specific imports
if IS_WINDOWS:
    import pywinauto
    import win32gui
    import win32process
    from pywinauto import Application
else:
    import pygetwindow as gw
    # For macOS, we may need additional imports
    if IS_MACOS:
        try:
            from AppKit import NSWorkspace, NSRunningApplication
            APPKIT_AVAILABLE = True
        except ImportError:
            APPKIT_AVAILABLE = False
            logging.warning("AppKit not available on macOS - some features may be limited")
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
from dotenv import load_dotenv
load_dotenv()

# Configure logging - use platform-appropriate log path
LOG_FILE = str(get_app_data_dir() / "automation.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s - [%(pathname)s:%(lineno)d]",
    filename=LOG_FILE,
    filemode="a",
)

# Constants - Platform-appropriate paths
def get_app_data_dir() -> Path:
    """Get the appropriate application data directory for the current platform."""
    if IS_WINDOWS:
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif IS_MACOS:
        base = Path.home() / "Library" / "Application Support"
    else:  # Linux and others
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    app_dir = base / "CommanderAI"
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir

MEMORY_FILE = str(get_app_data_dir() / "memory.json")
TESSERACT_CONFIG = r'--oem 3 --psm 6'

# Configure Tesseract path based on platform
def configure_tesseract():
    """Configure Tesseract OCR path for the current platform."""
    import shutil
    if IS_WINDOWS:
        # Common Windows installation paths
        windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for path in windows_paths:
            if os.path.isfile(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return
    elif IS_MACOS:
        # Common macOS installation paths (Homebrew)
        macos_paths = [
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
        ]
        for path in macos_paths:
            if os.path.isfile(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return
    # Linux typically has tesseract in PATH, but check common locations
    linux_paths = ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]
    for path in linux_paths:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return
    # Fallback: assume tesseract is in PATH
    if shutil.which("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = "tesseract"

configure_tesseract()

# Map system locale to Tesseract language codes
def get_tesseract_language() -> str:
    """Get the appropriate Tesseract language code based on system locale."""
    locale_to_tesseract = {
        "en": "eng", "fr": "fra", "de": "deu", "es": "spa", "it": "ita",
        "pt": "por", "nl": "nld", "ru": "rus", "zh": "chi_sim", "ja": "jpn",
        "ko": "kor", "ar": "ara", "hi": "hin", "pl": "pol", "tr": "tur",
    }
    try:
        lang_code = (system_language or "en").split("_")[0].lower()
        return locale_to_tesseract.get(lang_code, "eng")
    except Exception:
        return "eng"

TESSERACT_LANG = get_tesseract_language()

MIN_ELEMENT_WIDTH = 20
MAX_ELEMENT_WIDTH = 300
MIN_ELEMENT_HEIGHT = 20
MAX_ELEMENT_HEIGHT = 100

# Get system language (compatible with Python 3.15+)
try:
    system_language = locale.getlocale()[0] or os.environ.get("LANG", "en_US").split(".")[0]
except Exception:
    system_language = "en_US"

load_dotenv()

# LLM Provider Configuration
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()  # openai, lmstudio, ollama
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", None)  # For LM Studio: http://localhost:1234/v1
openai_api_key = os.environ.get("OPENAI_API_KEY", "")


class LLMProviderConfig:
    """Configuration for different LLM providers."""

    PROVIDERS = {
        "openai": {
            "name": "OpenAI",
            "default_model": "gpt-4o-mini",
            "base_url": None,
            "requires_api_key": True
        },
        "lmstudio": {
            "name": "LM Studio",
            "default_model": "local-model",
            "base_url": "http://localhost:1234/v1",
            "requires_api_key": False
        },
        "ollama": {
            "name": "Ollama",
            "default_model": "llama3.2",
            "base_url": "http://localhost:11434/v1",
            "requires_api_key": False
        },
        "anthropic": {
            "name": "Anthropic",
            "default_model": "claude-3-5-sonnet-20241022",
            "base_url": None,
            "requires_api_key": True
        },
        "groq": {
            "name": "Groq",
            "default_model": "llama-3.3-70b-versatile",
            "base_url": "https://api.groq.com/openai/v1",
            "requires_api_key": True
        },
        "together": {
            "name": "Together AI",
            "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "base_url": "https://api.together.xyz/v1",
            "requires_api_key": True
        }
    }

    @classmethod
    def get_config(cls, provider: str) -> Dict[str, Any]:
        """Get configuration for a provider."""
        return cls.PROVIDERS.get(provider.lower(), cls.PROVIDERS["openai"])

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all available providers."""
        return list(cls.PROVIDERS.keys())


def get_llm_client():
    """Get the appropriate LLM client based on configuration."""
    provider_config = LLMProviderConfig.get_config(LLM_PROVIDER)

    # Determine base URL
    base_url = LLM_BASE_URL or provider_config.get("base_url")

    # Determine model
    model = LLM_MODEL if LLM_MODEL != "gpt-4o-mini" else provider_config.get("default_model", "gpt-4o-mini")

    # Determine API key
    api_key = openai_api_key
    if not provider_config.get("requires_api_key"):
        api_key = "not-needed"  # LM Studio and Ollama don't need real API keys

    # Create ChatOpenAI instance (works with OpenAI-compatible APIs)
    kwargs = {
        "api_key": api_key,
        "model": model
    }
    if base_url:
        kwargs["base_url"] = base_url

    logging.info(f"Using LLM provider: {provider_config['name']}, model: {model}")
    return ChatOpenAI(**kwargs)


# Legacy variable for backward compatibility
llm_model = LLM_MODEL


class AICLITools:
    """Manages AI CLI tools like Claude Code, GitHub Copilot, Gemini, etc."""

    # Known AI CLI tools with their commands and detection methods
    KNOWN_TOOLS = {
        "claude": {
            "command": "claude",
            "name": "Claude Code",
            "check_args": ["--version"],
            "description": "Anthropic's Claude Code CLI"
        },
        "gh_copilot": {
            "command": "gh",
            "name": "GitHub Copilot CLI",
            "check_args": ["copilot", "--help"],
            "description": "GitHub Copilot in the CLI"
        },
        "gemini": {
            "command": "gemini",
            "name": "Google Gemini CLI",
            "check_args": ["--version"],
            "description": "Google Gemini CLI"
        },
        "aider": {
            "command": "aider",
            "name": "Aider",
            "check_args": ["--version"],
            "description": "AI pair programming in your terminal"
        },
        "copilot": {
            "command": "github-copilot-cli",
            "name": "GitHub Copilot CLI (standalone)",
            "check_args": ["--version"],
            "description": "GitHub Copilot CLI standalone"
        },
        "cody": {
            "command": "cody",
            "name": "Sourcegraph Cody",
            "check_args": ["--version"],
            "description": "Sourcegraph Cody AI assistant"
        },
        "cursor": {
            "command": "cursor",
            "name": "Cursor",
            "check_args": ["--version"],
            "description": "Cursor AI-powered editor CLI"
        },
        "continue": {
            "command": "continue",
            "name": "Continue",
            "check_args": ["--version"],
            "description": "Continue AI coding assistant"
        },
        "ollama": {
            "command": "ollama",
            "name": "Ollama",
            "check_args": ["--version"],
            "description": "Run LLMs locally"
        },
        "llm": {
            "command": "llm",
            "name": "LLM CLI",
            "check_args": ["--version"],
            "description": "Simon Willison's LLM CLI tool"
        },
        "lmstudio": {
            "command": "lms",
            "name": "LM Studio",
            "check_args": ["--version"],
            "description": "LM Studio CLI for local LLMs"
        }
    }

    def __init__(self):
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self._discover_tools()

    def _discover_tools(self):
        """Discover available AI CLI tools on the system."""
        import shutil

        logging.info("Discovering AI CLI tools...")

        for tool_id, tool_info in self.KNOWN_TOOLS.items():
            cmd = tool_info["command"]

            # Check if command exists in PATH
            cmd_path = shutil.which(cmd)
            if not cmd_path:
                continue

            # Verify the tool works
            try:
                result = subprocess.run(
                    [cmd] + tool_info["check_args"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                # Tool is available if it doesn't error out completely
                if result.returncode in [0, 1]:  # Some tools return 1 for --help
                    self.available_tools[tool_id] = {
                        **tool_info,
                        "path": cmd_path,
                        "available": True
                    }
                    logging.info(f"Found AI CLI tool: {tool_info['name']} at {cmd_path}")
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                logging.debug(f"Tool {cmd} not available: {e}")

        logging.info(f"Discovered {len(self.available_tools)} AI CLI tools")

    def list_available(self) -> List[Dict[str, Any]]:
        """List all available AI CLI tools."""
        return [
            {"id": k, **v}
            for k, v in self.available_tools.items()
        ]

    def is_available(self, tool_id: str) -> bool:
        """Check if a specific tool is available."""
        return tool_id in self.available_tools

    def get_preferred_tool(self) -> Optional[str]:
        """Get the preferred AI CLI tool (first available)."""
        # Priority order
        priority = ["claude", "gh_copilot", "aider", "cody", "gemini", "ollama", "llm"]
        for tool_id in priority:
            if tool_id in self.available_tools:
                return tool_id
        # Return first available if none in priority list
        if self.available_tools:
            return next(iter(self.available_tools))
        return None

    async def execute_with_claude(self, prompt: str, working_dir: Optional[str] = None) -> str:
        """Execute a prompt using Claude Code CLI."""
        if "claude" not in self.available_tools:
            raise RuntimeError("Claude Code CLI is not available")

        cmd = ["claude", "--print", prompt]
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=working_dir
            )
            return result.stdout or result.stderr
        except subprocess.TimeoutExpired:
            return "Error: Claude Code CLI timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    async def execute_with_gh_copilot(self, prompt: str, mode: str = "explain") -> str:
        """Execute a prompt using GitHub Copilot CLI."""
        if "gh_copilot" not in self.available_tools:
            raise RuntimeError("GitHub Copilot CLI is not available")

        # Modes: explain, suggest
        cmd = ["gh", "copilot", mode, prompt]
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.stdout or result.stderr
        except subprocess.TimeoutExpired:
            return "Error: GitHub Copilot CLI timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    async def execute_with_ollama(self, prompt: str, model: str = "llama2") -> str:
        """Execute a prompt using Ollama."""
        if "ollama" not in self.available_tools:
            raise RuntimeError("Ollama is not available")

        cmd = ["ollama", "run", model, prompt]
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.stdout or result.stderr
        except subprocess.TimeoutExpired:
            return "Error: Ollama timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    async def execute_with_llm(self, prompt: str, model: Optional[str] = None) -> str:
        """Execute a prompt using LLM CLI."""
        if "llm" not in self.available_tools:
            raise RuntimeError("LLM CLI is not available")

        cmd = ["llm", prompt]
        if model:
            cmd = ["llm", "-m", model, prompt]

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.stdout or result.stderr
        except subprocess.TimeoutExpired:
            return "Error: LLM CLI timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    async def execute_with_lmstudio(self, prompt: str, model: Optional[str] = None) -> str:
        """Execute a prompt using LM Studio CLI."""
        if "lmstudio" not in self.available_tools:
            raise RuntimeError("LM Studio CLI is not available")

        cmd = ["lms", "chat", prompt]
        if model:
            cmd = ["lms", "chat", "--model", model, prompt]

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.stdout or result.stderr
        except subprocess.TimeoutExpired:
            return "Error: LM Studio CLI timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    async def execute_with_aider(self, prompt: str, working_dir: Optional[str] = None) -> str:
        """Execute a prompt using Aider."""
        if "aider" not in self.available_tools:
            raise RuntimeError("Aider is not available")

        cmd = ["aider", "--message", prompt, "--yes"]
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=working_dir
            )
            return result.stdout or result.stderr
        except subprocess.TimeoutExpired:
            return "Error: Aider timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    async def execute_prompt(self, prompt: str, tool_id: Optional[str] = None) -> str:
        """Execute a prompt with the specified or preferred AI CLI tool."""
        if tool_id is None:
            tool_id = self.get_preferred_tool()

        if tool_id is None:
            return "Error: No AI CLI tools available"

        if tool_id == "claude":
            return await self.execute_with_claude(prompt)
        elif tool_id == "gh_copilot":
            return await self.execute_with_gh_copilot(prompt)
        elif tool_id == "ollama":
            return await self.execute_with_ollama(prompt)
        elif tool_id == "llm":
            return await self.execute_with_llm(prompt)
        elif tool_id == "lmstudio":
            return await self.execute_with_lmstudio(prompt)
        elif tool_id == "aider":
            return await self.execute_with_aider(prompt)
        else:
            return f"Error: Tool {tool_id} execution not implemented"


# Global AI CLI tools instance
ai_cli_tools: Optional[AICLITools] = None

def get_ai_cli_tools() -> AICLITools:
    """Get or create the AI CLI tools instance."""
    global ai_cli_tools
    if ai_cli_tools is None:
        ai_cli_tools = AICLITools()
    return ai_cli_tools


# Initialize text-to-speech engine with platform-specific handling
def init_tts_engine():
    """Initialize text-to-speech engine with platform-specific configuration."""
    try:
        if IS_WINDOWS:
            # Windows uses SAPI5
            engine = pyttsx3.init('sapi5')
        elif IS_MACOS:
            # macOS uses NSSpeechSynthesizer
            engine = pyttsx3.init('nsss')
        else:
            # Linux uses espeak
            engine = pyttsx3.init('espeak')

        engine.setProperty("rate", 150)

        # Set a voice appropriate for the system language
        voices = engine.getProperty('voices')
        if voices:
            # Try to find a voice matching system language
            lang_prefix = (system_language or "en").split("_")[0].lower()
            for voice in voices:
                if lang_prefix in voice.id.lower() or lang_prefix in str(voice.languages).lower():
                    engine.setProperty('voice', voice.id)
                    break

        logging.info(f"TTS engine initialized for {platform.system()}")
        return engine
    except Exception as e:
        logging.warning(f"Failed to initialize TTS engine: {e}")
        return None


# Global TTS engine (may be None if initialization fails)
tts_engine = init_tts_engine()


def speak_message(message: str):
    """Speak a message using text-to-speech."""
    logging.debug(f"speak_message: {message}")
    if tts_engine:
        try:
            tts_engine.say(message)
            tts_engine.runAndWait()
        except Exception as e:
            logging.error(f"TTS error: {e}")
            print(f"[TTS unavailable] {message}")
    else:
        # Fallback: just print the message
        print(f"[TTS unavailable] {message}")

def check_microphone_available() -> bool:
    """Check if a microphone is available on the system."""
    try:
        # Check if PyAudio can find any input devices
        mics = sr.Microphone.list_microphone_names()
        return len(mics) > 0
    except (OSError, AttributeError, Exception) as e:
        logging.debug(f"Microphone check failed: {e}")
        return False


def get_speech_recognition_language() -> str:
    """Get the appropriate language code for speech recognition based on system locale."""
    # Map common locale prefixes to Google Speech API language codes
    locale_to_speech = {
        "en": "en-US", "fr": "fr-FR", "de": "de-DE", "es": "es-ES", "it": "it-IT",
        "pt": "pt-BR", "nl": "nl-NL", "ru": "ru-RU", "zh": "zh-CN", "ja": "ja-JP",
        "ko": "ko-KR", "ar": "ar-SA", "hi": "hi-IN", "pl": "pl-PL", "tr": "tr-TR",
    }
    try:
        lang_code = (system_language or "en").split("_")[0].lower()
        return locale_to_speech.get(lang_code, "en-US")
    except Exception:
        return "en-US"


SPEECH_LANGUAGE = get_speech_recognition_language()


def recognize_speech() -> str:
    """Recognize voice command and convert to text."""
    # First check if microphone is available
    if not check_microphone_available():
        logging.warning("No microphone available - voice input disabled")
        speak_message("No microphone available. Please use text input.")
        return ""

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise on first use
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            speak_message("I'm listening...")
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio, language=SPEECH_LANGUAGE)
                logging.info(f"Recognized command: {command}")
                return command.lower()
            except sr.UnknownValueError:
                speak_message("I didn't understand the command.")
                return ""
            except sr.WaitTimeoutError:
                speak_message("Command timeout.")
                return ""
            except sr.RequestError as e:
                logging.error(f"Speech API request error: {e}")
                speak_message("Speech recognition service unavailable.")
                return ""
            except Exception as e:
                logging.error(f"Voice recognition error: {e}")
                speak_message("Error during voice recognition.")
                return ""
    except OSError as e:
        # Common on systems without audio (Docker, CI, servers)
        logging.error(f"Audio system error: {e}")
        speak_message("Audio system not available.")
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
            text = pytesseract.image_to_string(dilated, config=TESSERACT_CONFIG, lang=TESSERACT_LANG)
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
                if not path_obj.is_dir():
                    continue

                if IS_WINDOWS:
                    candidates = path_obj.glob("*.exe")
                else:
                    candidates = (
                        p for p in path_obj.iterdir()
                        if p.is_file() and os.access(p, os.X_OK)
                    )

            for exe in candidates:
                tool_name = exe.stem.lower() if IS_WINDOWS else exe.name.lower()
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

            if IS_MACOS:
                # Discover macOS applications from /Applications and ~/Applications
                app_dirs = [
                    Path("/Applications"),
                    Path.home() / "Applications",
                ]
                for app_dir in app_dirs:
                    if not app_dir.is_dir():
                        continue
                    for app_bundle in app_dir.glob("*.app"):
                        try:
                            # Extract app name from bundle name
                            app_name = app_bundle.stem.lower()
                            # The executable is typically in Contents/MacOS/
                            macos_dir = app_bundle / "Contents" / "MacOS"
                            if macos_dir.is_dir():
                                # Usually the main executable has the same name as the app
                                exec_path = macos_dir / app_bundle.stem
                                if not exec_path.is_file():
                                    # Try to find any executable in the MacOS folder
                                    executables = list(macos_dir.iterdir())
                                    if executables:
                                        exec_path = executables[0]
                                    else:
                                        continue

                                if app_name not in self.registry:
                                    self.registry[app_name] = {
                                        "path": str(app_bundle),  # Store the .app bundle path
                                        "type": "macos_app",
                                        "source": "Applications",
                                        "launch_count": 0,
                                        "success_count": 0,
                                        "failure_count": 0,
                                    }
                                    logging.debug(f"Discovered macOS app: {app_name} -> {app_bundle}")
                        except Exception as e:
                            logging.debug(f"Failed to parse {app_bundle}: {e}")

            elif IS_LINUX:
                # Discover Linux applications from .desktop files
                desktop_dirs = [
                    Path("/usr/share/applications"),
                    Path.home() / ".local/share/applications",
                ]
                for ddir in desktop_dirs:
                    if not ddir.is_dir():
                        continue
                    for desktop_file in ddir.glob("*.desktop"):
                        try:
                            config = configparser.ConfigParser(interpolation=None)
                            config.read(desktop_file, encoding="utf-8")
                            if "Desktop Entry" not in config:
                                continue
                            entry = config["Desktop Entry"]
                            name = entry.get("Name")
                            exec_cmd = entry.get("Exec", "").strip()
                            if not name or not exec_cmd:
                                continue
                            exec_path = shlex.split(exec_cmd)[0]
                            tool_name = name.lower()
                            if tool_name not in self.registry:
                                self.registry[tool_name] = {
                                    "path": exec_path,
                                    "type": "desktop",
                                    "source": "desktop",
                                    "launch_count": 0,
                                    "success_count": 0,
                                    "failure_count": 0,
                                }
                                logging.debug(
                                    f"Discovered desktop entry: {tool_name} -> {exec_path}"
                                )
                        except Exception as e:
                            logging.debug(f"Failed to parse {desktop_file}: {e}")

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
                app_type = self.registry[search_term].get("type", "")

                # macOS .app bundles are directories, not files
                if app_type == "macos_app":
                    if path and os.path.isdir(path):
                        return path
                else:
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
            if IS_WINDOWS:
                def callback(handle, windows):
                    try:
                        _, process_id = win32process.GetWindowThreadProcessId(handle)
                        if process_id == pid:
                            visible = win32gui.IsWindowVisible(handle)
                            logging.debug(
                                f"Window found - PID: {pid}, Handle: {handle}, Visible: {visible}"
                            )
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
                logging.warning(f"No visible window found for PID {pid}")
                return None
            else:
                # On Linux, pygetwindow supports .pid
                # On macOS, .pid is NOT supported - use title matching instead
                if IS_LINUX:
                    for window in gw.getAllWindows():
                        try:
                            if hasattr(window, '_hWnd') or hasattr(window, 'pid'):
                                if window.pid == pid:
                                    logging.info(f"Window found for PID {pid}")
                                    return window
                        except (AttributeError, Exception):
                            continue
                elif IS_MACOS:
                    # macOS: pygetwindow doesn't support PID lookup
                    # Return the first visible window as fallback
                    # Better approach: use process name matching
                    try:
                        proc = psutil.Process(pid)
                        proc_name = proc.name().lower()
                        for window in gw.getAllWindows():
                            try:
                                # Match by window title containing process name
                                if proc_name in window.title.lower():
                                    logging.info(f"Window found for PID {pid} by title match")
                                    return window
                            except Exception:
                                continue
                    except (psutil.NoSuchProcess, Exception) as e:
                        logging.debug(f"Could not find process {pid}: {e}")

                logging.warning(f"No window found for PID {pid}")
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

        logging.error(
            f"Failed to find window for '{executable_name}' after {retries} attempts"
        )
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
            if IS_WINDOWS:
                if window.is_minimized():
                    window.restore()
                if not window.is_visible():
                    window.set_focus()
                logging.info(f"Window prepared: {window.window_text()}")
            else:
                if window.isMinimized:
                    window.restore()
                if not window.isActive:
                    window.activate()
                logging.info(f"Window prepared: {window.title}")
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

                # Platform-specific prompt and examples
                current_platform = "Windows" if IS_WINDOWS else ("macOS" if IS_MACOS else "Linux")

                if IS_WINDOWS:
                    automation_lib = "'pywinauto' or 'pyautogui'"
                    example = """
                from pywinauto import Application

                # Connect to Notepad
                app = Application().connect(path="notepad.exe")

                # Access the Notepad window
                notepad = app.top_window()

                # Type text into Notepad
                notepad.type_keys("hello")
                """
                else:
                    automation_lib = "'pyautogui'"
                    if IS_MACOS:
                        example = """
                import pyautogui
                import time

                # Give focus time to settle
                time.sleep(0.5)

                # Type text using pyautogui (works cross-platform)
                pyautogui.typewrite("hello", interval=0.05)
                """
                    else:  # Linux
                        example = """
                import pyautogui
                import time

                # Give focus time to settle
                time.sleep(0.5)

                # Type text using pyautogui (works cross-platform)
                pyautogui.typewrite("hello", interval=0.05)
                """

                # Create a detailed prompt with explicit examples
                prompt = f"""
                You are an AI assistant specialized in automating user interfaces in Python on {current_platform}.
                The user wants to: "{action_description}".
                Follow the instructions below:
                1. Provide functional Python code using {automation_lib}.
                2. Validate that your code is self-contained and contains no syntax errors.
                3. ** Important all necessary imports and references must be included. The script will be executed as is, do not forget anything. **
                4. Return only the Python code, without any additional text and without markdown like ```python or ```, the code will be executed directly.

                Example:
                {example}

                Provide only the Python code.
                """
                llm = get_llm_client()
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

                # Execute the generated code with platform-appropriate locals
                exec_locals = {"window": window, "pyautogui": pyautogui}
                if IS_WINDOWS:
                    exec_locals["pywinauto"] = pywinauto
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

            if IS_WINDOWS:
                # Use pywinauto on Windows
                for proc in pywinauto.findwindows.find_elements():
                    if application_name.lower() in proc.name.lower():
                        logging.info(f"Validation: '{application_name}' is open")
                        return True
            else:
                # Use psutil for cross-platform process checking
                for proc in psutil.process_iter(['name', 'cmdline']):
                    try:
                        proc_name = proc.info.get('name', '').lower()
                        cmdline = proc.info.get('cmdline', []) or []
                        cmdline_str = ' '.join(cmdline).lower()

                        if application_name.lower() in proc_name or application_name.lower() in cmdline_str:
                            logging.info(f"Validation: '{application_name}' is open")
                            return True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
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

        # Validate path - on macOS, .app bundles are directories
        path_valid = executable_path and (os.path.isfile(executable_path) or os.path.isdir(executable_path))
        if not path_valid:
            self.app_registry.update_tool_stats(application_name, success=False)
            return f"Executable not found: {application_name}"

        try:
            if IS_WINDOWS:
                # Use pywinauto on Windows
                process = await asyncio.to_thread(
                    Application(backend="uia").start,
                    executable_path
                )
            elif IS_MACOS:
                # Use 'open' command on macOS
                await asyncio.to_thread(
                    subprocess.Popen,
                    ["open", "-a", executable_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                # Use subprocess on Linux
                await asyncio.to_thread(
                    subprocess.Popen,
                    [executable_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
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

            # Determine current platform for context
            current_platform = "Windows" if IS_WINDOWS else ("macOS" if IS_MACOS else "Linux")

            prompt_template = PromptTemplate(
                input_variables=["user_request", "available_apps", "platform"],
                template="""
                You are an AI assistant tasked with executing user requests in a {platform} environment.
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

                Important: Use appropriate application names for {platform}:
                - Windows: notepad.exe, calc.exe, etc.
                - macOS: TextEdit, Calculator, etc.
                - Linux: gedit, gnome-calculator, etc.
                """
            )

            llm = get_llm_client()
            chain = prompt_template | llm

            response = await chain.ainvoke({
                "user_request": user_request,
                "available_apps": available_apps,
                "platform": current_platform
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
class OpenStep(TypedDict):
    action: Literal["open"]
    application: str

class InteractDetails(TypedDict):
    process_name: str
    action_description: str

class InteractStep(TypedDict):
    action: Literal["interact"]
    details: InteractDetails

class CaptureStep(TypedDict):
    action: Literal["capture_screen"]

StepType = Union[OpenStep, InteractStep, CaptureStep]           
      
class TaskPlan(TypedDict):
    steps: List[StepType]
    
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
                mode = input("\nDo you want to use voice or text to enter a command? (voice/text/sample): ").strip().lower()

                if mode == "voice":
                    user_request = recognize_speech()
                elif mode == "text":
                    user_request = input("\nEnter a command (or 'exit' to quit): ").strip().lower()
                elif mode == "sample":
                    user_request = "sample"
                else:
                    speak_message("Unrecognized mode. Please choose 'voice', 'text' or 'sample'.")
                    continue

                if "exit" in user_request:
                    speak_message("Shutting down the system. Goodbye.")
                    break
                elif "help" in user_request:
                    help_text = """Available commands:
- 'list': List available applications
- 'ai tools': List available AI CLI tools (Claude, Copilot, etc.)
- 'provider': Show current LLM provider
- 'providers': List all supported LLM providers
- 'sample': Run a sample automation task
- 'exit': Quit the program
Or just describe what you want to do!"""
                    print(help_text)
                    speak_message("Check the console for available commands.")
                    continue
                elif "ai tools" in user_request or "ai cli" in user_request:
                    # List available AI CLI tools
                    cli_tools = get_ai_cli_tools()
                    available = cli_tools.list_available()
                    if available:
                        tool_list = ", ".join([t["name"] for t in available])
                        print(f"\nAvailable AI CLI tools: {tool_list}")
                        for tool in available:
                            print(f"  - {tool['name']}: {tool['description']}")
                        speak_message(f"Found {len(available)} AI CLI tools: {tool_list}")
                    else:
                        speak_message("No AI CLI tools found. Install claude, gh copilot, ollama, or aider.")
                    continue
                elif user_request == "provider":
                    # Show current provider
                    provider_config = LLMProviderConfig.get_config(LLM_PROVIDER)
                    print(f"\nCurrent LLM Provider: {provider_config['name']}")
                    print(f"Model: {LLM_MODEL}")
                    if LLM_BASE_URL:
                        print(f"Base URL: {LLM_BASE_URL}")
                    speak_message(f"Using {provider_config['name']} with model {LLM_MODEL}")
                    continue
                elif "providers" in user_request:
                    # List all providers
                    print("\nSupported LLM Providers:")
                    for name, config in LLMProviderConfig.PROVIDERS.items():
                        print(f"  - {name}: {config['name']} (default model: {config['default_model']})")
                    print("\nSet provider via environment variables:")
                    print("  LLM_PROVIDER=lmstudio  (or openai, ollama, groq, etc.)")
                    print("  LLM_MODEL=your-model")
                    print("  LLM_BASE_URL=http://localhost:1234/v1  (for local providers)")
                    speak_message("Check the console for supported providers.")
                    continue
                elif "list" in user_request:
                    tools = app_registry.list_tools()
                    tool_names = [tool["name"] for tool in tools]
                    speak_message(f"Available applications: {', '.join(tool_names[:10])}...")
                    print(f"\nAll applications ({len(tool_names)} total):")
                    for name in sorted(tool_names)[:20]:
                        print(f"  - {name}")
                    if len(tool_names) > 20:
                        print(f"  ... and {len(tool_names) - 20} more")
                    continue
                elif "sample" in user_request:
                    # Example task plan - platform-adaptive
                    speak_message("Executing the sample plan.")

                    # Choose appropriate text editor for the platform
                    if IS_WINDOWS:
                        app_name = "notepad"
                        process_name = "notepad.exe"
                    elif IS_MACOS:
                        app_name = "textedit"
                        process_name = "TextEdit"
                    else:  # Linux
                        app_name = "gedit"
                        process_name = "gedit"

                    example_task: TaskPlan = {
                        "steps": [
                            {
                                "action": "open",
                                "application": app_name
                            },
                            {
                                "action": "interact",
                                "details": {
                                    "process_name": process_name,
                                    "action_description": "Create a new file and type 'Hello, World!'"
                                }
                            }
                        ]
                    }

                    await task_executor.execute_task(example_task)
                    speak_message("Task executed successfully.")
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
