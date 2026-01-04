from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.prompts import question_category_prompt, conversation_prompt   
from src.output_structure import InterViewResponse, QuestionCategory
from typing import List, Annotated, Union, Literal, Optional
from pydantic import BaseModel, Field
from src.references import full_resume
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model_name="gpt-5-mini",
    temperature=0.7
)

structured_output = model.with_structured_output(QuestionCategory)

chain = question_category_prompt | structured_output

if __name__ == "__main__":

    while True:
        print("Please enter your HR questions (or type 'exit' to quit):")
        HR_question = input()
        if HR_question.lower() == 'exit':
            break

        response = chain.invoke({
            "question": HR_question
        })

        print("question_category:")
        print(response.question_category)



assistant = Assistant(gpt_model_name="gpt-5-mini", temperature=0.7, rag_activated=False)
assistant.run(question, )