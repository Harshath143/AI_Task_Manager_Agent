# core/models.py
from pydantic import BaseModel, Field
from typing import Literal

# Define structured output for the LLM
class CategorizedTask(BaseModel):
    category: Literal["Development", "Research", "Admin", "Urgent"] = Field(
        ..., description="The primary category of the task. Choose from the allowed list."
    )
    priority: Literal["High", "Medium", "Low"] = Field(
        ..., description="The priority level of the task. Choose from the allowed list."
    )
    summary: str = Field(
        ..., description="A concise, one-sentence summary of the task."
    )

# Define the full Task structure for storage
class Task(BaseModel):
    task_id: str
    original_input: str
    category: str
    priority: str
    summary: str
    status: Literal["Pending", "Complete", "Blocked"] = "Pending"
    timestamp: str # Use ISO format