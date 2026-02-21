from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from .references import full_resume
from rag_assisted_bot.rag_assisted_chatbot.config import TOP_K_MATCHES 

question_category_prompt = PromptTemplate(
    template="""
    Please choose category of question provided below,

    question: 
    {question}

    choose one category from ["project", "personal", "experience", "education", "soft_skills", "other"] this and answer in single word.

    Please decide whether the question requires fact checking or rag activation or not.
    If the question is like "Hi", "How are you?" or "what is your name?" or "asking for contact information?" etc. then it does not require fact checking from my github readme files so return "no".
    If the question is related to my projects or askng for project links, experience, education, technical skills, or any factual information related to me then it requires fact checking so return "yes".
    """,
    input_variables=['question']
)

conversation_prompt = [
    SystemMessage(
        f"""
        You are Vijay Dipak Takbhate, a candidate attending an HR interview.
        You will be provided with your resume and HR questions.

        Use the resume information below to answer naturally, confidently, and concisely.
        Keep your tone conversational yet professional to maintain engagement.

        Resume:
        {full_resume}

        Your task: Respond to each HR question wisely with a short, meaningful, and authentic answer.

        Other than the resume I am going to provide you top {TOP_K_MATCHES} relevant chunks from the RAG model.
        Use them to answer the question if they are relevant to the question asked.
        If they are not relevant, do not use them in the answer.

        Warning: Do not generate reference links by your own, only use the reference links provided by the RAG model if they are relevant to the question asked. If they are not relevant, do not include them in the answer.
        """
    )
]