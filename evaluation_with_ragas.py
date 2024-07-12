import os
import asyncio
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langsmith import Client
from langsmith import traceable
from langchain.smith import RunEvalConfig
from langchain.chains import create_retrieval_chain
from ragas import RunConfig
from ragas.integrations.langchain import EvaluatorChain
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from ragas import evaluate

from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
    faithfulness,
)
from ragas.metrics import ContextPrecision, ContextRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import sys

from rag_bot import NaiveRagBot
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from embedding import EmbeddingGenerator  # NOQA
from query import QueryProcessor  # NOQA
from extractor import PyMuPDFExtractor  # NOQA
from splitter import TextSplitter  # NOQA

os.environ["OPENAI_API_KEY"] = 'sk-'
LLM_MODELS = ['llama3:latest']
EMBEDDING_MODELS = ['sentence-transformers/paraphrase-MiniLM-L6-v2']
PDF_PARSERS = ['PyMuPDF', 'PyPDF2']
SPLIT_METHODS = ['split_documents', 'split_text']
CHUNK_SIZE = [256]  # , 512, 1024
CHUNK_OVERLAP = [50, 100, 200]
DATASET_NAME = 'ds_greenkern_owl'


def get_huggingface_embeddings(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)


def get_huggingface_llm(model_name):
    return HuggingFacePipeline.from_model_id(model_id=model_name)


def create_dataset(client: Client):
    datasets = client.list_datasets(dataset_name=DATASET_NAME)
    dataset = None
    while True:
        try:
            dataset = next(datasets)
        except StopIteration:
            break

    # not client.has_dataset(dataset_name=dataset_name):
    if dataset is None:
        dataset = client.create_dataset(
            DATASET_NAME, description="Dataset for GreenKern OWL")
        client.create_examples(
            inputs=[
                {"question": "What is the team structure?"},
                {"question": "What is total cost of finalcial proposal?"},
            ],
            outputs=[
                {"ground_truth": "Project Manager, Lead Architect"},
                {"ground_truth": "175000"},
            ],
            dataset_id=dataset.id,
        )
        return dataset


def list_pdf_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.pdf')]


def read_pdf(file_path, extractor: PyMuPDFExtractor):
    pdf_text = extractor.load_pdf_and_extract(file_path)
    return pdf_text


def read_pdfs_from_directory(directory, extractor: PyMuPDFExtractor):
    pdf_files = list_pdf_files(directory)
    for pdf_file in pdf_files:
        file_path = os.path.join(directory, pdf_file)
        print(f"Reading {file_path}")
        content = read_pdf(file_path, extractor)
        return content


def get_documents(directory, extractor: PyMuPDFExtractor):
    pdf_files = list_pdf_files(directory)
    langchain_documents = []
    for pdf_file in pdf_files:
        file_path = os.path.join(directory, pdf_file)
        print(f"Loading {pdf_file}")
        loader = DirectoryLoader(file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["filename"] = doc.metadata["source"]
        # documents = [doc for doc in documents if len(
        #     doc.page_content.split()) > 1000]
        langchain_documents.extend(documents)
    return langchain_documents


def create_qa_chain(llm, retriever, return_context=True):
    qa_chain = create_retrieval_chain(
        llm,
        retriever
    )
    return qa_chain


async def run_RAGAS_metrics(client: Client, llm_or_chain_factory, llm_judge, embedding_judge, llm_model, embedding_model):
    langchain_llm = LangchainLLMWrapper(llm_judge)
    langchain_embeddings = LangchainEmbeddingsWrapper(embedding_judge)
    # Wrap the RAGAS metrics to use in LangChain
    contextprecision = ContextPrecision(langchain_llm)

    contextprecision.llm = langchain_llm
    contextprecision.embeddings = langchain_embeddings
    contextrelevancy = ContextRelevancy(langchain_llm)
    contextrelevancy.llm = langchain_llm
    contextrelevancy.embeddings = langchain_embeddings


# Wrap the RAGAS metrics to use in LangChain
    evaluators = [
        EvaluatorChain(metric)
        for metric in [
            contextprecision,
            contextrelevancy,
        ]
    ]

    project_metadata = dict({'llm_model': llm_model})
    project_metadata = {
        **project_metadata,
        "embedding_model": embedding_model,
    }
    eval_config = RunEvalConfig(
        custom_evaluators=evaluators, prediction_key="result",)

    results = await client.arun_on_dataset(
        dataset_name=DATASET_NAME,
        llm_or_chain_factory=llm_or_chain_factory,
        evaluation=eval_config,
        project_metadata=project_metadata,
        input_mapper=lambda x: x,)
    results
# [v['feedback'] for k,v in results['results'].items()]


async def evaluate_model(client: Client, llm_judge, embedding_judge, llm_model, embedding_model, text, chunk_size, overlap_size):
    splitter = TextSplitter(chunk_size, overlap_size)
    embedding_generator = EmbeddingGenerator(embedding_model)
    chunks = splitter.split_documents(text)
    vector_store = embedding_generator.generate_embeddings(chunks)
    processor = QueryProcessor(llm_model, vector_store)

    rag_bot = NaiveRagBot(vector_store.as_retriever(), llm=llm_judge)

    # rag_chain = create_qa_chain(
    #    llm=llm_judge, retriever=vector_store.as_retriever(), return_context=True)
    await run_RAGAS_metrics(client, rag_bot.get_answer2,
                            llm_judge, embedding_judge, llm_model, embedding_model)

    # results = client.arun_on_dataset(
    #     dataset_name=DATASET_NAME,
    #     llm_or_chain_factory=processor.create_combine_docs_chain,
    #     evaluation=eval_config,
    # )

    # for metric, result in results.items():
    #     print(f"{metric}: {result}")
    #     return results


def generate_test_data(documents, llm_model, embedding, run_config):
    llm = LangchainLLMWrapper(llm_model)
    embedder = LangchainEmbeddingsWrapper(embedding)
    generator = TestsetGenerator.from_langchain(
        llm_model, llm_model, embedding, chunk_size=256, run_config=run_config)
    # Change resulting question type distribution
    distributions = {
        simple: 0.5,
        multi_context: 0.4,
        reasoning: 0.1
    }
    testset = generator.generate_with_langchain_docs(
        documents[:10], 10, distributions, with_debugging_logs=True, run_config=run_config)

    df = testset.to_pandas()
    print(df.head())


if __name__ == "__main__":
    generate_task = False
    client = Client()
    dataset = create_dataset(client)
    extractor = PyMuPDFExtractor()
    llm_judge = ChatOllama(llm_model='llama3:latest')
    embedding_judge = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-mpnet-base-v2')
    run_config = RunConfig(thread_timeout=1000, timeout=10000)

    if generate_task:
        loader = PyPDFDirectoryLoader("data/")
        documents = loader.load()
        # for doc in documents:
        #     doc.metadata["filename"] = doc.metadata["source"]
        len(documents)
        generate_test_data(documents, llm_judge, embedding_judge, run_config)
    else:
        text = read_pdfs_from_directory('data', extractor)
        results = []
        for llm_model in LLM_MODELS:
            # llm = get_huggingface_llm(llm_model)
            for embedding_model in EMBEDDING_MODELS:
                # embedding = get_huggingface_embeddings(embedding_model)
                for idx, chunk_size in enumerate(CHUNK_SIZE):
                    overlap_size = CHUNK_OVERLAP[idx]
                    asyncio.run(evaluate_model(client, llm_judge, embedding_judge, llm_model, embedding_model,
                                               text, chunk_size, overlap_size))
                    # chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # for result in results:
    #     print(f"LLM: {result['llm']}, Embedding: {result['embedding']}, "
    #           f"Chunk Size: {result['chunk_size']}, Retrieval Score: {result['retrieval_score']}, "
    #           f"Quality Score: {result['quality_score']}")
