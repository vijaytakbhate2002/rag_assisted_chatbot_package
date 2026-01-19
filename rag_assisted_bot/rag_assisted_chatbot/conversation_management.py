from langchain_core.messages import SystemMessage
from typing import Union
from rag_assisted_bot.rag_assisted_chatbot.logging_config import configure_file_logger


def conversationUpdate(conversation:list, context:str, rag_context:Union[str, list], top_k_matches:int) -> list:
    """ Updates the conversation list, keeping first conversation related to question type predicted by question_category model
        It remembers only last 3 conversations by keeping SystemMessage() at first index"""
    
    if len(conversation) > 3:
        conversation = [conversation[0]] + conversation[-3:]
        
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

        you can use reference links or facts from rag context, but do not make up any links.
        
        RAG Context:
        {rag_context}
        """)
    
    return conversation


    