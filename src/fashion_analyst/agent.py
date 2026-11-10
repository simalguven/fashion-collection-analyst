"""A small LangGraph agent that answers questions using the collection tools."""

import os
import re
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import DEFAULT_AGENT_MODEL, ROOT
from .tools import CollectionStore, build_tools

SYSTEM_PROMPT = """You are a fashion collection analyst. You answer questions about a \
sample of product photos that has been grouped into style clusters.

Base every factual statement on tool results from this conversation. If a tool returns an \
error, tell the user what went wrong. The collection only contains product images and \
catalogue labels."""

MAX_GRAPH_STEPS = 12
MAX_INVALID_TOOL_CALL_RETRIES = 2
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class AgentRun:
    answer: str = ""
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    total_tokens: int = 0
    seconds: float = 0.0
    error: str | None = None


def build_agent(store: CollectionStore, model_name: str | None = None,
                system_prompt: str = SYSTEM_PROMPT):
    """Compile a graph that alternates between the model and its tools."""
    load_dotenv(ROOT / ".env")
    name = model_name or os.getenv("AGENT_MODEL", DEFAULT_AGENT_MODEL)
    tools = build_tools(store)
    model = init_chat_model(name, temperature=0).bind_tools(tools)

    tool_names = ", ".join(t.name for t in tools)

    def call_model(state: MessagesState) -> dict:
        messages = [SystemMessage(system_prompt), *state["messages"]]
        for attempt in range(MAX_INVALID_TOOL_CALL_RETRIES + 1):
            try:
                return {"messages": [model.invoke(messages)]}
            except Exception as error:
                # Groq rejects a request when the model emits a malformed tool call.
                if "tool_use_failed" not in str(error) or attempt == MAX_INVALID_TOOL_CALL_RETRIES:
                    raise
                messages.append(
                    SystemMessage(f"Your last tool call was invalid. The only tools are: {tool_names}.")
                )

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    return graph.compile()


def ask(agent, question: str) -> AgentRun:
    """Run one question and collect the answer, tool trace and token usage."""
    start = time.perf_counter()
    result = agent.invoke(
        {"messages": [HumanMessage(question)]}, {"recursion_limit": MAX_GRAPH_STEPS}
    )
    run = AgentRun(seconds=round(time.perf_counter() - start, 2))
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            run.tool_calls += [
                {"name": call["name"], "args": call["args"]} for call in message.tool_calls
            ]
            run.total_tokens += (message.usage_metadata or {}).get("total_tokens", 0)
        elif isinstance(message, ToolMessage):
            run.tool_results.append(message.content)
    run.answer = THINK_BLOCK.sub("", result["messages"][-1].content).strip()
    return run
