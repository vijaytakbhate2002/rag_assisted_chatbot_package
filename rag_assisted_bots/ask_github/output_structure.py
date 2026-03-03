from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Optional, Annotated, Union


class RagActivation(BaseModel):
    """ RagActivation is pydantic dictionary used to control output of rag_activation_model."""
    rag_activation: Annotated[Union[str, Literal["yes", "no"]], "rag activation represent whether to activate RAG model or not"] = "no"


class InterViewResponse(BaseModel):
    """ InterViewResponse is pydantic dictionary used to control output of interview_model.
        output should be answer of HR question based on resume provided."""
    response_message: Optional[str] = Field(description="response message to HR question based on context provided.")
    reference_links: Optional[List[str]] = Field(description="List of reference links of github projects, blogs, articles etc.")

