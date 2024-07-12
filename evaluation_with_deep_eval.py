import os
import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from embedding import EmbeddingGenerator
from llama_embedding_model import LlamaEmbeddingModel
from query import QueryProcessor
from splitter import TextSplitter
from extractor import PyMuPDFExtractor
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langsmith import Client
from langsmith import traceable
from langchain.smith import RunEvalConfig
from langchain.chains import retrieval_qa
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
import ollama

os.environ["OPENAI_API_KEY"] = ''
LLM_MODELS = ['llama3:latest', 'llama2:latest', 'mistral:latest']
EMBEDDING_MODELS = ['thenlper/gte-base', 'sentence-transformers/all-MiniLM-L6-v2',
                    'sentence-transformers/paraphrase-MiniLM-L6-v2']
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
                {"must_mention": ["Project Manager", "Lead Architect"]},
                {"must_mention": ["175000"]},
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
    documents = []
    for pdf_file in pdf_files:
        file_path = os.path.join(directory, pdf_file)
        print(f"Reading {file_path}")
        content = read_pdf(file_path, extractor)
        documents.append({'filename': file_path, 'text': content})
    return documents


def evaluate_model(client, llm_judge, embedding_judge, llm_model, embedding_model, text, chunk_size, overlap_size):
    splitter = TextSplitter(chunk_size, overlap_size)
    embedding_generator = EmbeddingGenerator(embedding_model)
    chunks = splitter.split_documents(text)
    vector_store = embedding_generator.generate_embeddings(chunks)
    processor = QueryProcessor(llm_model, vector_store)


def generate_test_data(documents, llm, embedding):
    embedding_model = LlamaEmbeddingModel(model_name="llama3")

    synthesizer = Synthesizer(model='gpt-4o')
    dataset = EvaluationDataset()
    dataset.generate_goldens_from_docs(synthesizer=synthesizer, document_paths=[
                                       'rfp-2020-001-sample-4.pdf'])

    dataset.save_as(
        file_type='json',  # or 'csv'
        directory="./synthetic_data"
    )


if __name__ == "__main__":
    generate_task = True
    client = Client()
    dataset = create_dataset(client)
    extractor = PyMuPDFExtractor()
    llm_judge = ChatOllama(llm_model='llama3:latest')
    embedding_judge = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-mpnet-base-v2')
    documents = read_pdfs_from_directory('data', extractor)

    if generate_task:
        generate_test_data(documents, llm_judge, embedding_judge)
    else:
        text = read_pdfs_from_directory('data', extractor)
        results = []
        for llm_model in LLM_MODELS:
            # llm = get_huggingface_llm(llm_model)
            for embedding_model in EMBEDDING_MODELS:
                # embedding = get_huggingface_embeddings(embedding_model)
                for idx, chunk_size in enumerate(CHUNK_SIZE):
                    overlap_size = CHUNK_OVERLAP[idx]
                    # chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
                    evaluate_model(client, llm_judge, embedding_judge, llm_model, embedding_model,
                                   text, chunk_size, overlap_size)

    # for result in results:
    #     print(f"LLM: {result['llm']}, Embedding: {result['embedding']}, "
    #           f"Chunk Size: {result['chunk_size']}, Retrieval Score: {result['retrieval_score']}, "
    #           f"Quality Score: {result['quality_score']}")
