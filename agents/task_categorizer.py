# agents/task_categorizer.py
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from core.models import CategorizedTask
from core.config import OLLAMA_BASE_URL, LLM_MODEL, CATEGORIES, PRIORITIES

# Initialize Ollama LLM
llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1) 

def categorize_and_prioritize(task_input: str) -> CategorizedTask:
    """
    Analyzes natural language input and returns a structured Pydantic object 
    using the Ollama LLM.
    """
    parser = PydanticOutputParser(pydantic_object=CategorizedTask)
    
    category_list = ", ".join([f'"{c}"' for c in CATEGORIES])
    priority_list = ", ".join([f'"{p}"' for p in PRIORITIES])

    system_prompt = (
        "You are an expert Task Categorizer and Prioritizer. "
        "Analyze the user's task description and strictly output a JSON object "
        "that adheres to the following schema. "
        f"Allowed Categories: {category_list}. Allowed Priorities: {priority_list}.\n"
        "Schema: {format_instructions}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Task: {task_input}")
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    
    structured_task = chain.invoke({"task_input": task_input})
    return structured_task