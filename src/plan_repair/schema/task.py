"""Agent task data contract.

Only ``available_tools`` is consumed by the Ticket 001 validator (tool existence check).
The coverage-related fields are declared here so the contract is fixed early, but they are
unused until the evidence/operation coverage ticket.
"""

from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):
    """A tool the agent is allowed to call. Only its name matters for now."""

    model_config = ConfigDict(extra="forbid")

    name: str


class AgentTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    user_query: str
    available_tools: list[ToolSpec]
    required_evidence: list[str] = Field(default_factory=list)
    required_operations: list[str] = Field(default_factory=list)
    max_tool_calls: int | None = None

    def tool_names(self) -> set[str]:
        return {tool.name for tool in self.available_tools}
