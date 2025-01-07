import os
import logging
import sys
import asyncio
import click
import speech_recognition as sr
import pyttsx3

from .tools.memory import PersistentMemory
from .tasks.task_executor import TaskExecutor
from .tasks.task_interpreter import TaskInterpreter
from .agents.base_agents import CognitiveAgent, MediatorAgent, agent_manager
from .agents.agent_manager import AgentManager
from .agents.types import CommunicationProtocol, Message
from .tools.system_tools import CalculatorTool
from .tasks.application_registry import ApplicationRegistry
from .tasks.screen_analyzer import ScreenAnalyzer
from .tasks.window_locator import WindowLocator
from .agents.planner import STRIPSPlanner, ActionOperator, State

# Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - %(message)s - [%(pathname)s:%(lineno)d]",
#     filename="commanderai.log",
#     filemode="a",
# )
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
def speak_message(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak_message("Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio).lower()
    except sr.UnknownValueError:
        speak_message("Sorry, I did not understand that.")
        return ""
    except sr.RequestError:
        speak_message("Sorry, my speech service is down.")
        return ""

@click.command()
@click.argument('command')
def run(command):
    logger.debug(f"CommanderAI command: {command}")
    if command == 'run':
        try:
            memory = PersistentMemory()

            # "ApplicationRegistry"
            global agent_manager_registry
            agent_manager_registry = ApplicationRegistry(memory)
            if not memory.get("registry"):
                agent_manager_registry.discover_tools()

            # Prepare TaskInterpreter and TaskExecutor
            from .tasks.task_interpreter import TaskInterpreter
            from .tasks.task_executor import TaskExecutor

            interpreter = TaskInterpreter(agent_manager_registry)
            executor = TaskExecutor(memory, agent_manager_registry)

            # Agents and manager
            global agent_manager
            agent_manager = AgentManager()

            # Create a cognitive agent
            cagent = CognitiveAgent(1)
            agent_manager.register_agent(cagent)

            # Create a mediator
            medi = MediatorAgent(2)
            agent_manager.register_agent(medi)

            # Add a calculator tool
            calc_tool = CalculatorTool()
            cagent.add_tool(calc_tool)

            speak_message("CommanderAI is ready. Type or say 'help' for usage.")

            async def execute_requests():
                while True:
                    mode = input("\nMode? (voice/text/exit/help/list): ").strip().lower()
                    if mode == "exit":
                        speak_message("Shutting down CommanderAI.")
                        break
                    elif mode == "help":
                        speak_message("Ask me to open apps, interact, or capture screen.")
                        continue
                    elif mode == "list":
                        apps = agent_manager_registry.list_tools()
                        speak_message("Available: " + ", ".join(a["name"] for a in apps))
                        continue
                    elif mode == "voice":
                        user_req = recognize_speech()
                    elif mode == "text":
                        user_req = input("Request => ").strip().lower()
                    else:
                        speak_message("Unknown mode.")
                        continue

                    if not user_req:
                        continue

                    # Send the request to the cognitive agent
                    speak_message("Sending your request to the agent.")
                    await cagent.handle_user_request(user_req, interpreter, executor)

                    # Execute the reason/act logic in parallel
                    agent_manager.step_all()

            asyncio.run(execute_requests())

        except KeyboardInterrupt:
            speak_message("CommanderAI interrupted by user.")
        except Exception as e:
            logging.critical(f"Fatal error => {e}")
            speak_message(f"Fatal error => {e}")
            sys.exit(1)
    else:
        logging.error(f"Unknown command: {command}")
        speak_message(f"Unknown command: {command}")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logging.critical(f"Unhandled => {e}")
        speak_message(f"Unhandled => {e}")
        sys.exit(1)