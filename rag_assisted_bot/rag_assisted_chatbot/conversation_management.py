from langchain_core.messages import SystemMessage
from typing import Union
from rag_assisted_bot.rag_assisted_chatbot.logging_config import configure_file_logger

logger = configure_file_logger(__name__)


def conversationUpdate(conversation:list, context:str, rag_context:Union[str, list], top_k_matches:int) -> list:
    """ Updates the conversation list, keeping first conversation related to question type predicted by question_category model
        It remembers only last 4 conversations by keeping SystemMessage() at first index"""
    
    if len(conversation) > 4:
        logger.info("Conversation compressed to keep last 4 items (plus system message)")
        conversation = [conversation[0]] + conversation[-4:]
        
    conversation[0] = SystemMessage(f"""
        You are Vijay Dipak Takbhate, a candidate attending an HR interview.
        You will be provided with your resume and HR questions.

        Use the resume information below to answer naturally, confidently, and concisely.
        Keep your tone conversational yet professional to maintain engagement.

        Resume:
        {context}

        Your task: Respond to each HR question wisely with a short, meaningful, and authentic answer.

        Other than the resume I am going to provide you top {top_k_matches} relevant chunks from the RAG model.
        Use them to answer the question if they are relevant to the question asked.
        If they are not relevant, do not use them in the answer.
        
        RAG Context:
        {rag_context}
        """)
    logger.debug("Conversation[0] (system) updated with context and rag context summary")
    
    return conversation


    