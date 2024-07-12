import os
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from embedding import EmbeddingGenerator
from query import QueryProcessor
from splitter import TextSplitter
from extractor import PyMuPDFExtractor
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langsmith import Client
from langsmith import traceable
from langchain.smith import RunEvalConfig
from langchain.chains import retrieval_qa
from langchain_openai import ChatOpenAI
from langsmith.schemas import Example, Run
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langsmith.evaluation import evaluate

os.environ["OPENAI_API_KEY"] = 'dsfsfd'
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


@traceable
def rag_pipeline(question):
    query = generate_wiki_search(question)
    context = "\n\n".join([doc["page_content"] for doc in retrieve(query)])
    answer = generate_answer(question, context)
    return answer


def run_RAGAS_metrics(predict_rag_answer):
    experiment_results = evaluate(
        predict_rag_answer,
        data=DATASET_NAME,
        evaluators=[document_relevance_grader, answer_hallucination_grader],
        experiment_prefix="LCEL context, gpt-3.5-turbo"
    )


def document_relevance_grader(root_run: Run, example: Example) -> dict:
    """
    A simple evaluator that checks to see if retrieved documents are relevant to the question
    """

    # Get documents and question
    rag_pipeline_run = next(
        run for run in root_run.child_runs if run.name == "get_answer")
    retrieve_run = next(
        run for run in rag_pipeline_run.child_runs if run.name == "retrieve_docs")
    doc_txt = "\n\n".join(
        doc.page_content for doc in retrieve_run.outputs["output"])
    question = retrieve_run.inputs["question"]

    # Data model for grade
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""
        binary_score: int = Field(
            description="Documents are relevant to the question, 1 or 0")

    # LLM with function call
    # llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    llm = ChatOllama("llama3:latest")
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 1 or 0 score, where 1 means that the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    score = retrieval_grader.invoke(
        {"question": question, "document": doc_txt})
    return {"key": "document_relevance", "score": int(score.binary_score)}


def answer_hallucination_grader(root_run: Run, example: Example) -> dict:
    """
    A simple evaluator that checks to see the answer is grounded in the documents
    """

    # Get documents and answer
    rag_pipeline_run = next(
        run for run in root_run.child_runs if run.name == "get_answer")
    retrieve_run = next(
        run for run in rag_pipeline_run.child_runs if run.name == "retrieve_docs")
    doc_txt = "\n\n".join(
        doc.page_content for doc in retrieve_run.outputs["output"])
    generation = rag_pipeline_run.outputs["answer"]

    # Data model
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: int = Field(
            description="Answer is grounded in the facts, 1 or 0")

    # LLM with function call
    # llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    llm = ChatOllama("llama3:latest")
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
         Give a binary score 1 or 0, where 1 means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader
    score = hallucination_grader.invoke(
        {"documents": doc_txt, "generation": generation})
    return {"key": "answer_hallucination", "score": int(score.binary_score)}


def evaluate_model(client, llm_judge, embedding_judge, llm_model, embedding_model, text, chunk_size, overlap_size):
    splitter = TextSplitter(chunk_size, overlap_size)
    embedding_generator = EmbeddingGenerator(embedding_model)
    chunks = splitter.split_documents(text)
    vector_store = embedding_generator.generate_embeddings(chunks)
    processor = QueryProcessor(llm_model, vector_store)

    run_RAGAS_metrics(processor.create_combine_docs_chain,
                      llm_judge, embedding_judge)

    # results = client.arun_on_dataset(
    #     dataset_name=DATASET_NAME,
    #     llm_or_chain_factory=processor.create_combine_docs_chain,
    #     evaluation=eval_config,
    # )

    # for metric, result in results.items():
    #     print(f"{metric}: {result}")
    #     return results


def generate_test_data(documents, llm, embedding):
    generator = TestsetGenerator.from_langchain(
        llm, llm, embedding, chunk_size=256)
    # Change resulting question type distribution
    distributions = {
        simple: 0.5,
        multi_context: 0.4,
        reasoning: 0.1
    }
    testset = generator.generate_with_langchain_docs(
        documents[:10], 10, distributions, with_debugging_logs=True)

    df = testset.to_pandas()
    print(df.head())


if __name__ == "__main__":
    generate_task = True
    client = Client()
    dataset = create_dataset(client)
    extractor = PyMuPDFExtractor()
    llm_judge = ChatOllama(llm_model='llama3:latest')
    embedding_judge = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-mpnet-base-v2')

    if generate_task:
        loader = DirectoryLoader('data')
        documents = loader.load()
        for doc in documents:
            doc.metadata["filename"] = doc.metadata["source"]
        len(documents)
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
