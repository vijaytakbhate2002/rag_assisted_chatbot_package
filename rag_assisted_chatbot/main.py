from rag_assisted_chatbot.conversation_management import conversationUpdate
from rag_assisted_chatbot.output_structure import InterViewResponse, QuestionCategory
from rag_assisted_chatbot.prompts import conversation_prompt, question_category_prompt
from rag_assisted_chatbot.references import full_resume, personal, project, soft_skills, others, education, experience
from rag_assisted_chatbot.ask_vectordb import AskToVectorDB
from rag_assisted_chatbot.config import GPT_MODEL_NAME, TOP_K_MATCHES, EMBEDDING_MODEL_NAME, VECTORDB_PATH
import chromadb
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


from rag_assisted_chatbot.logging_config import configure_file_logger
logger = configure_file_logger(__name__) 


class RAGModel:
    """ This is VectorDB communicator which takes question as input and returns the relevant chunks from the VectorDB. """
    def __init__(self, vectordb_path:str, collection_name:str, embedding_model_name:str):
        self.vectordb_path = vectordb_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name


    def build_config(self):
        """
        Build ChromaDB PersistentClient and get the collection. Plus initialize AskToVectorDB 
        Returns:
            None
        """
        client = chromadb.PersistentClient(path=self.vectordb_path)
        self.collection = client.get_collection(name=self.collection_name)
        self.asker = AskToVectorDB(collection=self.collection, embedding_model_name=self.embedding_model_name)

    
    def ask(self, question, n_results) -> str:
        """ This is helper function to ask question to asked """
        response = self.asker.ask(question, n_results=n_results) 
        documents = response['documents']
        return documents



class Assistant:
    """ This is an LLM gpt-5-min whcih uses RAG plus resume context to answer interview questions asked by HR."""

    context_category = {
        "personal": personal,
        "project" : project,
        "education" : education,
        "experience" : experience,
        "soft_skills" : soft_skills,
        "others" : others
    }

    updated_conversation = conversation_prompt.copy()

    def __init__(self, gpt_model_name:str, temperature:float, rag_activated:bool):
        self.gpt_model_name = gpt_model_name
        self.temperature = temperature
        self.rag_activated = rag_activated
        if self.rag_activated:
            self.rag_model = RAGModel(
                                    vectordb_path=VECTORDB_PATH,
                                    collection_name="my_embeddings",
                                    embedding_model_name=EMBEDDING_MODEL_NAME
                                    )
            self.rag_model.build_config()

        if self.gpt_model_name:
            self.model = ChatOpenAI(
                                    model_name=self.gpt_model_name, 
                                    temperature=self.temperature
                                    )
            

    def RAG_context_fetcher(self, question:str, n_results:int) -> str:
        """ This function fetches context from RAG model based on question asked."""
        documents = self.rag_model.ask(question, n_results=n_results)
        context = "\n".join([doc for doc in documents[0]])
        return context

    
    def get_context_for_question_category(self, question_category:str) -> str:
        """ This function returns context based on question category predicted."""
        return self.context_category.get(question_category, full_resume)


    def build_chains(self, question_category_prompt):
        """ This function will initialize question category model and conversational model. """
        conversational_model = self.model.with_structured_output(InterViewResponse)
        question_category_model = self.model.with_structured_output(QuestionCategory)
        question_category_model = question_category_prompt | question_category_model

        return conversational_model, question_category_model


    def chat_with_model(self, question:str) -> dict:
        """ Takes input question and return answer of that question with updating conversation list.
            conversation list used to make model remember last 4 conversation messages
            Args:
                question: input question

            Returns:
                    (answer, question_category): returns the response of llm """

        conversation_model, question_category_chain = self.build_chains(question_category_prompt)
        question_category = question_category_chain.invoke(question)
        print(question_category, type(question_category))

        context=self.get_context_for_question_category(question_category=question_category.question_category)
        print("Context used for question category:  ", context)

        rag_context = self.RAG_context_fetcher(
                                                    question=question,
                                                    n_results=TOP_K_MATCHES
                                                    ) if self.rag_activated else ""

        print("RAG Context fetched: ", rag_context)

        self.updated_conversation = conversationUpdate(
                                                conversation=self.updated_conversation,
                                                context=self.get_context_for_question_category(
                                                                        question_category=question_category.question_category
                                                                    ),
                                                rag_context=rag_context,
                                                top_k_matches=TOP_K_MATCHES
                                                    )

        self.updated_conversation.append(HumanMessage(question))
        response = conversation_model.invoke(self.updated_conversation)
        self.updated_conversation.append(AIMessage(response.response_message))

        return {
                "response":  response,
                "question_category":question_category.question_category
        }



if __name__ == "__main__":
    assistant = Assistant(
                        gpt_model_name=GPT_MODEL_NAME,
                        temperature=0.7,
                        rag_activated=True
                        )
    
    while True:
        question = input("You: ")
        if question == 'exit':
            break
        ai_response = assistant.chat_with_model(question)

        print("Question Category:", ai_response['question_category'])
        print("Answer -------------------------- :")
        for key, value in ai_response['response'].model_dump().items():
            print(f"{key}: {value}")
        print("\n")