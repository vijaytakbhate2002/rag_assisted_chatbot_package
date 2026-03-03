from langchain_core.messages import SystemMessage
from typing import Union, List
from rag_assisted_bots.ask_github.logging_config import configure_file_logger
from rag_assisted_bots.ask_github.prompts import SystemPromptTemplate

class ConversationManager:
    """ Manages conversation history and system prompt generation based on RAG activation and assistant type. 
    - Initializes with assistant type and sets up system prompt template.
    - Provides a method to update conversation history with the appropriate system prompt based on RAG context
    and activation status.
    """
    def __init__(self, assistant_type:str):
        """Initializes the ConversationManager with the specified assistant type and sets up the system prompt template.
         - assistant_type: A string indicating the type of assistant (e.g., "github", "medium").
        """
        self.assistant_type = assistant_type
        self.system_prompt_template = SystemPromptTemplate(assistant_type=assistant_type)
        
    def manage(self, rag_context: Union[str, List[str]], top_k_matches: int, rag_activation: str) -> list:
        """Manages the conversation based on the provided context and settings.
         - Builds the appropriate system prompt based on RAG activation and assistant type.
         - Initializes conversation with the system prompt.
         - Returns the updated conversation list.
        """
        conversation = []

        if isinstance(rag_context, list):
            rag_context = "\n\n".join(str(chunk) for chunk in rag_context)
            
        system_prompt_templates = {
            "github": self.system_prompt_template.system_prompt_github,
            "medium": self.system_prompt_template.system_prompt_medium
        }
        system_prompt = system_prompt_templates.get(self.assistant_type, self.system_prompt_template.system_prompt_github)(
            rag_context=rag_context,
            rag_activation=rag_activation,
            top_k_matches=top_k_matches
        )
        conversation.append(SystemMessage(content=system_prompt.strip()))
        return conversation

