from typing import TypedDict, List, Any, Optional
from langgraph.graph import StateGraph, END
from PIL import Image

class AgentState(TypedDict):
    question: str
    image: Any
    rag_enabled: bool
    context: List[dict]
    grade: str
    web_results: str
    final_response: str

def build_graph(worker):
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", worker.retrieve_node)
    workflow.add_node("grade", worker.grade_node)
    workflow.add_node("web_search", worker.web_search_node)
    workflow.add_node("generate", worker.generate_node)
    
    # Conditional Entry Point
    def route_start(state):
        if state.get("rag_enabled"):
            return "retrieve"
        return "generate"

    workflow.set_conditional_entry_point(
        route_start,
        {
            "retrieve": "retrieve",
            "generate": "generate"
        }
    )
    
    workflow.add_edge("retrieve", "grade")
    
    def route_grade(state):
        #add_conditiona_edge needs a function that returns a string and cant accept a string key directly
        if state.get("grade") == "lacking":
            return "web_search"
        return "generate"

    workflow.add_conditional_edges(
        "grade",
        route_grade,
        {
            "web_search": "web_search",
            "generate": "generate"
        }
    )
    
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
