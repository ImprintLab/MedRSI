"""MedRSI: recursive self-improvement of medical agents through clinically aligned self-evolution."""
from .agent import MedAgent
from .data import load_cases, make_cohorts, split_cases
from .judge import LLMJudge, RubricJudge
from .llm import LLM, AnthropicChat, FunctionLLM, OpenAIChat, make_llm
from .registry import registration_decision
from .rsi import Config, MedRSI, load_agent
from .task import Case, Diagnosis, Task
from .tools import Library, Tool, ToolContext

__version__ = "0.1.0"
__all__ = [
    "MedRSI", "Config", "load_agent", "MedAgent", "Task", "Case", "Diagnosis",
    "load_cases", "split_cases", "make_cohorts", "RubricJudge", "LLMJudge",
    "LLM", "OpenAIChat", "AnthropicChat", "FunctionLLM", "make_llm",
    "Tool", "ToolContext", "Library", "registration_decision",
]
