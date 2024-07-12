from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models.ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langsmith.wrappers import wrap_openai
from langsmith import traceable

DB_FAISS_PATH = 'vectorstore/db_faiss'

@traceable
class ChatPDF:
    vector_store = None
    retriever = None
    chain = None
    prompt = None

    def __init__(self):
        self.model = ChatOllama(model="llama3:latest")
        self.text_splitter = RecursiveCharacterTextSplitter(
             chunk_size=256, 
             chunk_overlap=16, 
             separators=["\n\n", "\n", "(?<=\. )", " ", ""])
        self.embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')#768 dimension
        #self.embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')#384
        self.prompt = PromptTemplate.from_template(
            """
            [INST]<<SYS>>You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.<</SYS>> 
            Question: {input} 
            Context: {context} 
            Answer: Answer the question as best as you can using only the information from the document. If you don't know the answer, just say that you don't know.[/INST]
            """
        ) 
        
        vector_store = Chroma(persist_directory="./chroma_db", embedding_function=self.embedding)
        self.retriever = vector_store.as_retriever(
            # search_type="similarity_score_threshold",
            # search_kwargs={
            #     'k': 10,
            #     'score_threshold': 0.5
            # },
        )
        self.load_chain()
        # self.chain =  (
        #      {
        #     "context" : self.retriever,
        #     "question" : RunnablePassthrough()
        #                }
        #                 | self.prompt
        #                 | self.model
        #                 | StrOutputParser()
        #                )
    def load_chain(self):
        question_answer_chain = create_stuff_documents_chain(self.model, self.prompt)
        self.chain = create_retrieval_chain(self.retriever, question_answer_chain) 
    
    def load_default_data(self):
        loader = PyPDFDirectoryLoader("data/")
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        vector_store = Chroma.from_documents(documents=chunks, embedding=self.embedding, persist_directory="./chroma_db")
        self.retriever = vector_store.as_retriever()
        self.load_chain()
        print("chunks len: ", len(chunks))
        print("3rd chunk: ", chunks[3])

    def ingest(self, pdf_path):
        docs = PyPDFLoader(file_path=pdf_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        vector_store = Chroma.from_documents(documents=chunks, embedding=self.embedding, persist_directory="./chroma_db")
        
        print("chunks len: ",len(chunks))
        print("3rd chunk: ", chunks[3])
        # db = FAISS.from_documents(chunks, embeddings)
        # db.save_local(DB_FAISS_PATH)
        # print(db.index.ntotal)
        self.retriever = vector_store.as_retriever(
            #search_type="similarity_score_threshold",
            # search_kwargs={
            #     'score_threshold': 0.5
            # },
        )
        self.load_chain()
        # self.chain = ({
        #     "context" : self.retriever,
        #     "question" : RunnablePassthrough()
        #                }
        #                 | self.prompt
        #                 | self.model
        #                 | StrOutputParser()
        #                )
    def update_template(self, template: str):
        self.prompt = PromptTemplate.from_template(template)
        self.load_chain()

    def ask(self, query: str):
        if not self.chain:
            return "Please ingest a PDF file first."
        #return self.chain.invoke(query)
        for chunk in self.chain.stream({"question": query}): 
            print (chunk, end="|", flush=True)
            return "chunk"
    
    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None