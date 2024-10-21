from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate

from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()
class Chatbot:
    def __init__(self):
        # Load documents
        self.loader = TextLoader('./knowledge-db.txt')
        self.documents = self.loader.load()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.docs = self.text_splitter.split_documents(self.documents)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings()

        # Initialize Pinecone client
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

        # Define Index Name
        index_name = "langchain-demo"

        # Checking or creating Pinecone index
        if index_name not in pc.list_indexes().names():
            pc.create_index(name=index_name, metric="cosine", dimension=768, spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        self.docsearch = PineconeVectorStore.from_documents(self.docs, self.embeddings, index_name=index_name)

        # Initialize the LLM (Mixtral model from Hugging Face)
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        # Define prompt template
        template = """
        You are a fortune teller. These Human will ask you a questions about their life. 
        Use following piece of context to answer the question. 
        If you don't know the answer, just say you don't know. 
        Keep the answer within 2 sentences and concise.

        Context: {context}
        Question: {question}
        Answer: 

        """
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        def handle_question(question):
            return question

        # Define the RAG chain
        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

# Get input from the user and run the chain
#bot = Chatbot()
#input = input("Ask me anything: ")
#result = bot.rag_chain.invoke(input)
#print(result)
