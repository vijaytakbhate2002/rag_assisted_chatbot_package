from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Optional, Annotated, Union


class QuestionCategory(BaseModel):
    """ Question category is pydantic dictionary used to control output of question_category_model.
        output should be one word out of all literals."""
    question_category: Annotated[Union[str,
        Literal["project", "personal", "experience", "education", "soft_skills", "others"]],
        "question category represent category of question"
    ] = "other"
    rag_activation: Annotated[Union[str, Literal["yes", "no"]], "rag activation represent whether to activate RAG model or not"] = "no"



class InterViewResponse(BaseModel):
    """ InterViewResponse is pydantic dictionary used to control output of interview_model.
        output should be answer of HR question based on resume provided."""
    response_message: Optional[str] = Field(description="response message to HR question based on context provided.")
    reference_links: Optional[List[str]] = Field(description="List of reference links of github projects, blogs, articles etc.")
    # confidence_score: Optional[float] = Field(description="Confidence score of the response message ranging from 0 to 1.")
    # follow_up_question: Optional[str] = Field(description="A relevant follow-up question to keep the interview engaging.")
    # additional_resources: Optional[List[str]] = Field(description="List of additional resources for further reading or exploration.")