"""LangGraph supervisor graph definition."""

from typing import Any, Literal

import structlog
from agent.app.nodes.action import action_node
from agent.app.nodes.comms import comms_node
from agent.app.nodes.supervisor import supervisor_node
from agent.app.nodes.triage import triage_node
from agent.app.schemas.state import AgentState
from langgraph.graph import END, START, StateGraph

log = structlog.get_logger()


def create_graph() -> StateGraph:
    """
    Create the LangGraph supervisor graph.

    Topology:
    supervisor -> triage -> supervisor -> action (if MEDIUM/HIGH) -> supervisor -> comms -> END

    Note: triage, action, and comms are async nodes. Use ainvoke() to run the graph.
    """
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("triage", triage_node)
    graph.add_node("action", action_node)
    graph.add_node("comms", comms_node)

    graph.add_edge(START, "supervisor")

    def supervisor_route(state: dict[str, Any]) -> Literal["triage", "action", "comms", "__end__"]:
        next_node = state.get("next")
        if next_node == "END":
            return END
        if next_node == "triage":
            return "triage"
        if next_node == "action":
            return "action"
        if next_node == "comms":
            return "comms"
        return END

    graph.add_conditional_edges(
        "supervisor",
        supervisor_route,
        path_map={
            "triage": "triage",
            "action": "action",
            "comms": "comms",
            "__end__": END,
        },
    )
    graph.add_edge("triage", "supervisor")
    graph.add_edge("action", "supervisor")
    graph.add_edge("comms", END)

    return graph.compile()


__all__ = ["create_graph"]
