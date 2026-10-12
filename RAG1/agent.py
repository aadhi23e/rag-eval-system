import RAG1.tools as tl
from .prompt import system_agent_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from .tools import collecting_info
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from .objects import Answer

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    # model="gemini-3.5-flash",
    temperature=0
)

agent = create_agent(
    model=llm,
    tools=[collecting_info],
    system_prompt=system_agent_prompt,
    response_format=Answer
)

def ask(question) -> dict:
    """Answer the user's question about the pdf"""

    tl.last_retrieved_context = []

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    structured_response = result["structured_response"]

    return {
        "answer": structured_response.answer,
        "context": structured_response.context
    }

# print(ask("what is the diets  of penguin"))