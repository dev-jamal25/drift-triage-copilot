"""LangGraph Studio entry point for drift triage agent.

Exports the compiled graph for visualization and manual testing in LangGraph Studio.
Studio can invoke the graph directly without FastAPI context.
"""

from agent.app.graph import create_graph

# Export graph for Studio
graph = create_graph()
