from langchain_community.chat_models.ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from chromadb.config import Settings
import chromadb

model = ChatOllama(model="llama3:latest")
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    "{input}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
loader = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/Barack_Obama",),
    # bs_kwargs=dict(
    #     parse_only=bs4.SoupStrainer(
    #         class_=("post-content", "post-title", "post-header")
    #     )
    # ),
)
loader.requests_kwargs = {'verify':False}
docs = loader.load()
print(docs)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=16)
splits = text_splitter.split_documents(docs)
#client = chromadb.HttpClient(settings=Settings(allow_reset=True))
#client.reset()  # resets the database
vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'), persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                'score_threshold': 0.5
            },)

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#chain = model | StrOutputParser()
for chunk in rag_chain.stream({"input":"Please give me details of president Obama and how he got the president?"}):
    if answer_chunk := chunk.get("answer"):
        print(f"{answer_chunk}|", end="")