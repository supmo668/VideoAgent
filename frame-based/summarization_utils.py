import operator
from typing import Annotated, List, Literal, TypedDict
import asyncio
from dotenv import load_dotenv
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Token limit for collapsing summaries
token_max = 1000

def get_llm():
    """Get ChatOpenAI instance, initializing it if necessary."""
    return ChatOpenAI(model="gpt-4o-mini")

# Define prompt templates
map_prompt = ChatPromptTemplate.from_messages(
    [("human", "Write a concise summary of the following laboratory action frame captions:\n\n{context}")]
)
reduce_template = """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated abstract of the scientific workflow that took place.
"""
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
title_prompt = ChatPromptTemplate.from_messages(
    [("human", """
      The following is a set of summaries:{docs}:
      \n\nGenerate a concise, informative and logical title.""")]
)

# Define chains lazily
def get_chains():
    """Get chains with initialized LLM."""
    llm = get_llm()
    return {
        'map_chain': map_prompt | llm | StrOutputParser(),
        'reduce_chain': reduce_prompt | llm | StrOutputParser(),
        'title_chain': title_prompt | llm | StrOutputParser()
    }

# Function to calculate total token count
def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents."""
    llm = get_llm()
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

# Overall state of the graph
class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[List[str], operator.add]
    collapsed_summaries: List[Document]
    abstract: str
    title: str

# State for individual summaries
class SummaryState(TypedDict):
    content: str

# Generate a summary for a document
async def generate_summary(state: SummaryState):
    chains = get_chains()
    response = await chains['map_chain'].ainvoke(state["content"])
    return {"summaries": [response]}

# Map contents to summary generation nodes
def map_summaries(state: OverallState):
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]

# Collect summaries into collapsed summaries
def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(page_content=summary) for summary in state["summaries"]]
    }

# Collapse summaries if needed
async def collapse_summaries(state: OverallState):
    chains = get_chains()
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, token_max
    )
    results = []
    for doc_list in doc_lists:
        results.append(await acollapse_docs(doc_list, chains['reduce_chain'].ainvoke))
    return {"collapsed_summaries": results}

# Determine whether to collapse summaries or proceed to final summary
def should_collapse(state: OverallState) -> Literal["collapse_summaries", "generate_abstract"]:
    num_tokens = length_function(state["collapsed_summaries"])
    return "collapse_summaries" if num_tokens > token_max else "generate_abstract"

# Generate the final summary
async def generate_abstract(state: OverallState):
    chains = get_chains()
    response = await chains['reduce_chain'].ainvoke(state["collapsed_summaries"])
    return {"abstract": response}

# Generate a title from the final summary
async def generate_title(state: OverallState):
    chains = get_chains()
    response = await chains['title_chain'].ainvoke(state["collapsed_summaries"])
    return {"title": response}

# Construct the graph
graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)
graph.add_node("collect_summaries", collect_summaries)
graph.add_node("collapse_summaries", collapse_summaries)
graph.add_node("generate_abstract", generate_abstract)
graph.add_node("generate_title", generate_title)

# Add edges
graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("generate_abstract", "generate_title")
graph.add_edge("generate_title", END)

# Compile the graph
app = graph.compile()

# Async entry point to summarize contents and generate a title
async def summarize_and_generate_title_async(contents: List[str]):
    """Async entry point to summarize contents and generate a title."""
    input_state = {"contents": contents}
    result = await app.ainvoke(input_state)
    return result["abstract"], result["title"]

# Main function to test the summarization pipeline
if __name__ == "__main__":
    async def main():
        example_input = [
          "In this image frame, a laboratory setting is depicted. The following elements are notable:\n\n- **Action**: A person is using a dropper pipet to add a cabbage indicator solution to a test tube. This action involves carefully dispensing the solution into the pH standard solutions arranged in a test tube rack.\n  \n- **Equipment**: \n  - **Dropper Pipet**: Being used to transfer the cabbage indicator solution.\n  - **Test Tube Rack**: Holding several test tubes containing pH standard solutions, each labeled from 1 to 3. The colors of the solutions vary, likely indicating different pH levels.\n  \n- **Additional Materials**: A purple liquid is visible in a beaker, possibly the cabbage indicator solution, and there appears to be another bottle or container on the table.\n\nThe environment is well-lit with modern laboratory furnishings, and the person is wearing gloves for safety while conducting the experiment.",
          "The image depicts a biological science lab environment with various laboratory items on a countertop. Here’s a detailed description of the scene:\n\n1. **Countertop Area**: A wooden table is visible on the right side, covered with several objects including:\n   - A few bottles containing liquid reagents, with at least one labeled.\n   - A variety of colorful markers and sticky notes, possibly for labeling or notation purposes.\n   - A white box that appears to hold testing kits or samples, suggested by the design and presence of pipettes nearby.\n   - Several small, clear plastic slides and labels are scattered across the counter.\n\n2. **Cleaning Supplies**: On the left side of the countertop, there are cleaning agents displayed, including a spray bottle (likely for disinfectant) and a gallon jug, which may be for cleaning or diluting solutions.\n\n3. **Sink Area**: To the left, there is a stainless steel sink installed, indicating this area is equipped for cleaning equipment or materials.\n\n4. **Storage Area**: In the background, there are shelves and a door, suggesting storage for lab supplies and equipment. \n\n5. **Seating**: A black chair is positioned in front of the table, suggesting that the area is designed for conducting experiments or analysis.\n\nOverall, the scene illustrates a typical setup for a biological science lab, showcasing laboratory supplies and equipment essential for conducting experiments or sample analyses.",
          "In this laboratory frame, a person is engaged in adding a cabbage indicator solution to pH standard solutions. Below are the key elements and actions observed in the scene:\n\n1. **Action Taking Place**: The individual is using a dropper pipet to dispense the cabbage indicator solution into test tubes that are arranged in a test tube rack.\n\n2. **Equipment Present**:\n   - **Dropper Pipet**: Being used by the individual to transfer the indicator solution.\n   - **Test Tube Rack**: Holds several test tubes, clearly marked with labels (1-3) indicating different pH standards.\n   - **Beaker**: Contains a purple solution, likely the cabbage indicator solution, positioned to the left of the dropper pipet.\n\n3. **Laboratory Environment**: \n   - The workspace is well-lit with overhead lighting.\n   - There are other lab materials and equipment visible on the countertop, including what appears to be a storage container.\n\n4. **Safety Precautions**: The individual is wearing gloves, indicating adherence to standard laboratory safety protocols.\n\nThis scene illustrates a typical procedure in a biological science lab focusing on pH testing using natural indicators."
        ]
        abstract, title = await summarize_and_generate_title_async(example_input)
        print("Final Abstract:\n", abstract)
        print("Generated Title:\n", title)

    asyncio.run(main())
