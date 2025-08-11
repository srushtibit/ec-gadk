
"""
Demo script for showcasing emergent communication in the multi-agent system.
"""

import asyncio
import os
import sys
from pathlib import Path
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[0]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import system components
from agents.base_agent import AgentCoordinator
from agents.communication.communication_agent import CommunicationAgent
from agents.retrieval.retrieval_agent import RetrievalAgent
from agents.critic.critic_agent import CriticAgent
from agents.escalation.escalation_agent import EscalationAgent
from rl.agent_lightning.enhanced_coordinator import create_enhanced_coordinator

def display_results(result: dict):
    """Displays the results of the agent interaction in a structured format."""
    console = Console()

    console.print(Panel("[bold green]Emergent Communication Demo[/bold green]", expand=False))
    console.print(f"\n[bold]User Query:[/bold] {result['workflow_steps'][0]['content'] if result['workflow_steps'] else 'N/A'}")
    console.print(f"\n[bold]Final Response:[/bold] {result['response']}\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Step", style="dim", width=5)
    table.add_column("Agent", style="cyan")
    table.add_column("Action", style="green")
    table.add_column("Communication Intent", style="yellow")
    table.add_column("Emergent Protocol", style="blue")

    for i, step in enumerate(result['workflow_steps']):
        protocol = step.get('emergent_protocol', {})
        protocol_str = f"Pattern: {protocol.get('communication_pattern', 'N/A')}\n" \
                       f"Efficiency: {protocol.get('protocol_efficiency', 0):.2f}"

        table.add_row(
            str(i + 1),
            step.get('agent', 'N/A'),
            step.get('action', 'N/A'),
            step.get('communication_intent', 'N/A'),
            protocol_str
        )

    console.print(table)

    console.print("\n[bold]Explanation:[/bold]")
    console.print("The table above shows the flow of communication between the agents. Notice how each step has a [bold yellow]Communication Intent[/bold yellow] and an [bold blue]Emergent Protocol[/bold blue].")
    console.print("This is the 'emergent' part of the communication. The system is not just passing messages, but it's also reasoning about the *purpose* of the communication (the intent) and using a *protocol* that can be adapted and optimized over time.")
    console.print("For example, the system can learn that for a 'technical_assistance' intent, the 'retrieval_agent_search_knowledge_base' protocol is the most efficient.")


async def main():
    """Main function to run the demo."""
    print("Initializing the multi-agent system...")

    # Initialize agents
    agents = {
        "communication_agent": CommunicationAgent(),
        "retrieval_agent": RetrievalAgent(),
        "critic_agent": CriticAgent(),
        "escalation_agent": EscalationAgent(),
    }

    # Initialize the enhanced coordinator
    # Note: AgentOps API key is not required for this demo
    enhanced_coordinator = create_enhanced_coordinator(agents)

    print("System initialized.")
    print("\n--------------------------------------------------\n")

    # Define a sample query
    # query = "I can't share my screen during zoom calls"
    query = "urgent security breach detected"


    print(f"Processing query: '{query}'")
    print("\n--------------------------------------------------\n")


    # Process the query
    result = await enhanced_coordinator.process_query_enhanced(query)

    # Display the results
    display_results(result)


if __name__ == "__main__":
    # To run this script, you might need to install the dependencies from requirements.txt
    # pip install -r requirements.txt
    asyncio.run(main())
