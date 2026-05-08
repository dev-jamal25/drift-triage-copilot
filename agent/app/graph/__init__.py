"""LangGraph supervisor graph definition."""

from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph

from agent.app.nodes.action import action_node
from agent.app.nodes.comms import comms_node
from agent.app.nodes.supervisor import supervisor_node
from agent.app.nodes.triage import triage_node

log = structlog.get_logger()


def create_graph() -> StateGraph:
	"""
	Create the LangGraph supervisor graph.

	Topology:
	supervisor -> triage -> supervisor -> action (if MEDIUM/HIGH) -> supervisor -> comms -> END
	"""
	graph = StateGraph(dict)

	graph.add_node("supervisor", supervisor_node)
	graph.add_node("triage", triage_node)
	graph.add_node("action", action_node)
	graph.add_node("comms", comms_node)

	graph.add_edge(START, "supervisor")

	def supervisor_route(state: dict[str, Any]) -> str:
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

	graph.add_conditional_edges("supervisor", supervisor_route)
	graph.add_edge("triage", "supervisor")
	graph.add_edge("action", "supervisor")
	graph.add_edge("comms", END)

	return graph.compile()


__all__ = ["create_graph"]
