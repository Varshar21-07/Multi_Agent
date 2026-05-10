import os
import sys
from typing import TypedDict, List, Annotated
from datetime import datetime

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
except ImportError:
    # Fallback if rich is not installed
    class Console:
        def print(self, *args, **kwargs):
            if 'style' in kwargs: del kwargs['style']
            print(*args, **kwargs)
    console = Console()
    Panel = lambda x, **kwargs: x
    Table = lambda **kwargs: None
    Markdown = lambda x: x

# Load environment variables from .env file
load_dotenv()

# --- 1. STATE MANAGEMENT ---
# This defines the shared memory between all agents
class StudyPlanState(TypedDict):
    """Shared state for the Multi-Agent Study Planner."""
    # User Inputs
    user_query: str
    subject: str
    exam_date: str
    available_hours: float
    weak_topics: List[str]
    
    # Agent Outputs
    analysis: str
    study_strategy: str
    resources: str
    motivation_tips: str
    final_plan: str
    
    # Tracking
    current_agent: str

# --- 2. LLM INITIALIZATION ---
def get_llm():
    """Initializes the LLM. Supports Groq (preferred) or OpenAI."""
    # Check for Groq first
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            api_key=groq_key
        )
    
    # Fallback to OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=openai_key
        )
        
    console.print("[bold red]Error: Neither GROQ_API_KEY nor OPENAI_API_KEY found.[/bold red]")
    console.print("Please add your API key to the .env file.")
    sys.exit(1)

llm = get_llm()

# --- 3. AGENT NODES ---

def input_analyzer_node(state: StudyPlanState):
    """
    Input Analyzer Agent:
    Responsible for understanding the user's raw requirements and 
    extracting key information like goals, subjects, and weak areas.
    """
    console.print("\n[bold cyan]🔍 Analyzer Agent is processing your request...[/bold cyan]")
    
    prompt = f"""
    You are an Expert Academic Consultant. 
    Analyze the following student request and extract the core details:
    Request: {state['user_query']}
    
    Identify:
    1. Subject of study
    2. Exam/Deadline date
    3. Daily study availability
    4. Specific weak topics or pain points
    
    Format your response as a structured summary.
    """
    
    response = llm.invoke(prompt)
    
    # Update state
    return {
        "analysis": response.content,
        "current_agent": "Study Strategy Agent"
    }

def study_strategy_node(state: StudyPlanState):
    """
    Study Strategy Agent:
    Responsible for creating the roadmap, deciding priorities, 
    and allocating study hours based on the analysis.
    """
    console.print("[bold green]📅 Strategy Agent is designing your roadmap...[/bold green]")
    
    prompt = f"""
    You are a professional Study Strategist. Based on the analysis below, create a high-level study roadmap.
    
    Analysis: {state['analysis']}
    
    Your roadmap must:
    - Prioritize weak topics early in the schedule.
    - Break down the subject into logical weekly/daily milestones.
    - Allocate specific hours for theory vs. practice.
    """
    
    response = llm.invoke(prompt)
    
    return {
        "study_strategy": response.content,
        "current_agent": "Resource Recommendation Agent"
    }

def resource_recommendation_node(state: StudyPlanState):
    """
    Resource Recommendation Agent:
    Suggests high-quality study materials like books, YouTube channels, and websites.
    """
    console.print("[bold yellow]📚 Resource Agent is finding the best materials...[/bold yellow]")
    
    prompt = f"""
    You are a Resource Librarian. Based on the Study Strategy, suggest specific resources.
    
    Study Strategy: {state['study_strategy']}
    
    Suggest:
    - Top 2 Books
    - 3 YouTube Channels or specific Playlists
    - 2 Practice websites or documentation links
    - 1 Active learning technique appropriate for this subject
    """
    
    response = llm.invoke(prompt)
    
    return {
        "resources": response.content,
        "current_agent": "Motivation & Productivity Agent"
    }

def motivation_productivity_node(state: StudyPlanState):
    """
    Motivation & Productivity Agent:
    Provides wellness tips, Pomodoro schedules, and motivational guidance.
    """
    console.print("[bold magenta]🚀 Motivation Agent is adding finishing touches...[/bold magenta]")
    
    prompt = f"""
    You are a Student Success Coach. Given the intense study plan, provide productivity hacks.
    
    Context: Creating a plan for {state['subject']} until {state['exam_date']}.
    
    Provide:
    - A suggested daily break schedule (e.g., Pomodoro).
    - 3 wellness tips to avoid burnout.
    - A personalized motivational quote for this student's specific challenge.
    """
    
    response = llm.invoke(prompt)
    
    return {
        "motivation_tips": response.content,
        "current_agent": "Final Planner Agent"
    }

def final_planner_node(state: StudyPlanState):
    """
    Final Planner Agent:
    Combines all outputs into a single, cohesive, and beautiful markdown layout.
    """
    console.print("[bold white]📋 Final Planner Agent is generating your document...[/bold white]")
    
    prompt = f"""
    You are the Lead Coordinator. Combine all previous outputs into a FINAL PERSONALIZED STUDY PLAN.
    
    STRATEGY: {state['study_strategy']}
    RESOURCES: {state['resources']}
    MOTIVATION: {state['motivation_tips']}
    
    Format the output using clear Markdown headers (##), bullet points, and tables where appropriate.
    Make it look professional, encouraging, and easy to follow.
    """
    
    response = llm.invoke(prompt)
    
    return {
        "final_plan": response.content,
        "current_agent": "Complete"
    }

# --- 4. GRAPH CONSTRUCTION ---

def create_study_graph():
    """Initializes and connects the LangGraph."""
    workflow = StateGraph(StudyPlanState)
    
    # Add Nodes
    workflow.add_node("analyzer", input_analyzer_node)
    workflow.add_node("strategist", study_strategy_node)
    workflow.add_node("resource_specialist", resource_recommendation_node)
    workflow.add_node("motivator", motivation_productivity_node)
    workflow.add_node("finalizer", final_planner_node)
    
    # Define Edges (Linear workflow)
    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "strategist")
    workflow.add_edge("strategist", "resource_specialist")
    workflow.add_edge("resource_specialist", "motivator")
    workflow.add_edge("motivator", "finalizer")
    workflow.add_edge("finalizer", END)
    
    return workflow.compile()

# --- 5. MAIN EXECUTION ---

def main():
    """Main terminal application entry point."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    console.print(Panel(
        "[bold white]🎓 AI STUDY PLANNER: MULTI-AGENT SYSTEM[/bold white]\n"
        "[italic]Your personalized AI academic coordination team[/italic]",
        style="blue",
        expand=False
    ))
    
    # Step 1: Collect User Input
    console.print("\n[bold]Please enter your study details:[/bold]")
    subject = console.input("[bold cyan]What subject are you studying?[/bold cyan] ")
    exam_date = console.input("[bold cyan]When is your exam/deadline? (e.g., June 15)[/bold cyan] ")
    hours = console.input("[bold cyan]How many hours can you study daily?[/bold cyan] ")
    weaknesses = console.input("[bold cyan]What are your weak topics? (Optional)[/bold cyan] ")
    
    # Construct the query for the agent system
    user_query = f"I am studying {subject}. My exam is on {exam_date}. I can study {hours} hours a day. My weak topics are: {weaknesses}."
    
    # Initialize State
    initial_state: StudyPlanState = {
        "user_query": user_query,
        "subject": subject,
        "exam_date": exam_date,
        "available_hours": float(hours) if hours.replace('.','',1).isdigit() else 2.0,
        "weak_topics": [w.strip() for w in weaknesses.split(",")] if weaknesses else [],
        "analysis": "",
        "study_strategy": "",
        "resources": "",
        "motivation_tips": "",
        "final_plan": "",
        "current_agent": "Start"
    }
    
    # Create and run the graph
    app = create_study_graph()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Agents collaborating...", total=None)
        final_state = app.invoke(initial_state)
    
    # Final Output Display
    console.print("\n" + "="*50)
    console.print(Markdown(final_state['final_plan']))
    console.print("="*50)
    
    console.print("\n[bold green]✅ Your Study Plan is ready! Good luck with your studies![/bold green]\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Terminated by user. Goodbye![/bold red]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]An error occurred: {e}[/bold red]")
        sys.exit(1)
