# CommanderAI / LLM-Driven Action Generation on Windows

> The system interprets requests and dynamically generates Python code to interact with applications with Langchain (openai).

*Note: This project is a **proof of concept** demonstrating the automation capabilities described below. Be patient. It is not intended for production use without further validation, security measures, and adaptations.*

## Table of Contents

1. [General Description](#general-description)
2. [Key Features](#key-features)
3. [Architecture and Components](#architecture-and-components)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage](#usage)
   - [Text Mode vs. Voice Mode](#text-mode-vs-voice-mode)
   - [Usage Examples](#usage-examples)
8. [Screen Analysis and OCR](#screen-analysis-and-ocr)
9. [Speech Recognition and Text-to-Speech](#speech-recognition-and-text-to-speech)
10. [Task Interpretation and LLM Pipeline](#task-interpretation-and-llm-pipeline)
11. [Persistent Memory and Application Registry](#persistent-memory-and-application-registry)
12. [UI Interaction](#ui-interaction)
13. [Logging and Debugging](#logging-and-debugging)
14. [Error Handling and Exceptions](#error-handling-and-exceptions)
15. [Limitations and Usage Tips](#limitations-and-usage-tips)
16. [Contribution](#contribution)
17. [License](#license)

---

## General Description

This project is an intelligent automation system designed to interact with Windows applications and LLM, analyze the screen, recognize speech, speak responses, execute planned tasks, and interact with user interfaces in an automated manner. It leverages advanced language models (via the OpenAI API), langchain, speech recognition, text-to-speech, screen capture, text recognition (OCR), and application interaction through tools like pywinauto and pyautogui.

As a **proof of concept**, it demonstrates how various components can be integrated to automate different tasks. It serves as an example and inspiration, but further study and hardening are needed before using it in a real-world environment.

---

## Key Features

- **Launch and interact with Windows applications** via pywinauto and pyautogui.
- **Screen analysis**: Uses `mss` for screenshots, `pytesseract` for OCR, and OpenCV for detecting UI elements.
- **Speech recognition**: Allows the user to give voice commands (via `speech_recognition` and Google Speech API).
- **Text-to-speech**: Uses `pyttsx3` to provide spoken responses.
- **Dynamic Python code generation**: Utilizes a LLM (ChatOpenAI) to produce UI interaction code based on natural language descriptions.
- **Persistent memory**: Stores state and application registry in `memory.json`.
- **Detailed logging**: Actions, errors, and debug info are logged to `automation.log`.

---
## Exemples

```markdown
Navigate to wikipedia.com with chrome and find the definition of AI
```
```markdown
Write me a text on automation
```
In the whispering woods, where the tall trees sway,Nature's beauty unfolds in a magical display.With petals like silk and leaves like a song,The chorus of life hums, inviting us along.The rivers dance lightly, reflecting the sky,While mountains stand guard, reaching up high.Each sunrise a canvas, each sunset a dream,In the heart of the forest, life flows like a stream.The fragrance of flowers, the warmth of the sun,In nature's embrace, we find joy, we are one.So let us wander where the wild things grow,In the splendor of nature, let our spirits flow.
```markdown 
draw me a cat
```
![cat_3](https://github.com/user-attachments/assets/f3088765-8f1d-4ecf-9ea1-7ab0701326db)

---
## Pre-prepared tasks - TaskPlan  

It can accept pre-prepared tasks as input in JSON format, for example, if the tool is connected to an external utility and provides it with the steps. In this POC, simply type "sample.".
```python 
TaskPlan = {
              "steps": [    
                  {
                     "action": "open",
                     "application": "notepad"
                  },
                  {
                     "action": "interact",
                     "details": {
                         "process_name": "notepad.exe",
                         "action_description": "Create a new file and type 'Hello, World!'"
                     }
                  }
              ]
            }
```
---

## Architecture and Components

- **Main file**: The script (within the `if __name__ == "__main__": ...` block) is the program’s entry point.
- **Key Classes**:
  - `PersistentMemory`: Manages persistent state (read/write from `memory.json`).
  - `ApplicationRegistry`: Maintains a registry of discovered tools/applications and usage statistics.
  - `ScreenAnalyzer`: Captures and analyzes the screen, performs OCR, and detects UI elements.
  - `WindowLocator`: Finds windows associated with a given process or application.
  - `InteractionStrategies`: Logic to automatically interact with the window (executes code generated by the LLM).
  - `TaskInterpreter`: Interprets user requests into a task plan (uses LLM).
  - `TaskExecutor`: Executes the task plan step-by-step (open app, interact, capture screen, etc.).

---

## Prerequisites

- **Operating System**: Windows (required for pywinauto, win32gui, etc.).
- Python 3.8+ recommended.
- An OpenAI API key
- Microphone and speakers (for voice recognition and text-to-speech).
- Tesseract OCR installed on the machine (and accessible in the PATH). Installer here https://github.com/UB-Mannheim/tesseract/wiki.
- Internet connection (for Google Speech Recognition and OpenAI API).

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JacquesGariepy/CommanderAI.git
   cd CommanderAI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Dependencies include (not exhaustive):
   - `pyautogui`, `psutil`, `pywinauto`, `win32gui`, `win32process`
   - `mss`, `pytesseract`, `numpy`, `Pillow`, `opencv-python-headless`
   - `speech_recognition`, `pyttsx3`
   - `langchain`, `openai`
   
   Plus standard libraries: `ast`, `locale`, `os`, `json`, `asyncio`, `logging`, `re`, `time`, `pathlib`, `typing`, `subprocess`, `sys`

3. **Install Tesseract**:
   - Download and install Tesseract: <https://github.com/UB-Mannheim/tesseract/wiki>
   - Ensure `tesseract.exe` is in your PATH.

---

## Configuration

- **OpenAI API Key**: Set the `OPENAI_API_KEY` environment variable with your key in ".env"
- **System Language**: The code detects your system language via `locale.getdefaultlocale()[0]`.
- **Tesseract Settings**: Default is `TESSERACT_CONFIG = '--oem 3 --psm 6'` and `lang='fra'`. Adjust if needed.
- **Memory File**: `MEMORY_FILE = "memory.json"`. Ensure it’s writable.

---

## Usage

Run the script:

```bash
python main.py
```

The program:
- Initializes components.
- Asks if you want to use voice or text mode.
- You can then say or type a command like "Open Notepad" or "Take a screenshot".
- The program interprets your command, generates a task plan, and executes it.
- Results are displayed and logged in `automation.log`.

### Text Mode vs. Voice Mode

- **Text Mode**: Type commands directly.
- **Voice Mode**: The script prompts you to speak. It listens, transcribes, and executes your commands.

### Usage Examples

- **Open an application**:
  - Text mode: type "open notepad".
  - Voice mode: say "open notepad".
- **List available applications**: say or type "list".
- **Help**: say or type "help".

---

## Screen Analysis and OCR

`ScreenAnalyzer`:
- Uses `mss` for screenshot.
- Converts and threshold the image, then uses OCR with Tesseract.
- Detects UI elements (buttons, text fields) via OpenCV.

Screen analysis results can validate UI states after actions.

---

## Speech Recognition and Text-to-Speech

- **Speech Recognition**: `speech_recognition` + Google Speech API (requires Internet).
- **Text-to-Speech**: `pyttsx3` speaks the program’s responses.

Users can interact entirely by voice if desired.

---

## Task Interpretation and LLM Pipeline

- `TaskInterpreter` sends user requests to the LLM (ChatOpenAI via LangChain).
- The LLM returns a JSON task plan.
- `TaskExecutor` executes these steps.
- For interaction steps, code is dynamically generated by the LLM, validated, and executed.

This integration showcases how LLMs can aid in automation scenarios.

---

## Persistent Memory and Application Registry

- `PersistentMemory` reads/writes `memory.json` to persist state across sessions.
- `ApplicationRegistry` maintains a list of discovered apps and usage stats.

---

## UI Interaction

- `WindowLocator` finds application windows.
- `pywinauto` and `pyautogui` handle window interactions (clicking, typing, selecting menus).
- `InteractionStrategies` dynamically generates interaction code from action descriptions.

---

## Logging and Debugging

- All steps and errors are logged in `automation.log`.
- Adjust `logging` level (e.g., `logging.DEBUG`) for detailed diagnostics.

---

## Error Handling and Exceptions

- Critical operations (OCR, app launch, speech recognition) are in try/except blocks.
- On error, a message is logged, possibly spoken, and the script attempts to continue if possible.

---

## Limitations and Usage Tips

- **Limitations**:
  - Speech recognition depends on microphone quality and Internet connectivity.
  - OCR and UI detection depend on the visual quality and context of the screen.
  - The LLM can generate imperfect code; this proof of concept attempts to handle retries.

- **Tips**:
  - Test with simple apps (Notepad, Calculator).
  - Verify Tesseract installation.
  - If voice recognition fails, use text mode.
  - Ensure a valid OpenAI key.

---

## Contribution

Contributions are welcome to:
- Fix bugs.
- Add features.
- Improve code robustness and validation.
- Adapt this proof of concept to real-world environments.

Submit a pull request with your changes and a clear description.

---

## License

This proof of concept is not explicitly licensed. For use beyond experimental purposes, contact the author or define an appropriate license.
