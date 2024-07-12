from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from ragas.metrics import (

    answer_relevancy,

    answer_similarity,

    answer_correctness,

    faithfulness,

    context_recall,

    context_precision,

    context_relevancy,

)
from ragas import evaluate
from langchain_core.embeddings import Embeddings

from ragas.llms.base import LangchainLLMWrapper

from ragas.embeddings.base import LangchainEmbeddingsWrapper

import numpy as np


import matplotlib.pyplot as plt

from ragas.llms import BaseRagasLLM

from langchain_huggingface.llms import HuggingFacePipeline

from langchain_community.embeddings import HuggingFaceEmbeddings

from datasets import load_dataset, Dataset
from ragas import RunConfig
from transformers import GenerationConfig
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models.ollama import ChatOllama

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# pipe = pipeline("text-generation", model=model,
#                 tokenizer=tokenizer, max_new_tokens=10)
# hf = HuggingFacePipeline(pipeline=pipe)
ollama = ChatOllama(model="llama3:latest")
llm = LangchainLLMWrapper(ollama)
model_name = "sentence-transformers/all-MiniLM-L6-v2"

model_kwargs = {'device': 'cpu'}

encode_kwargs = {'normalize_embeddings': False}

hf_e = HuggingFaceEmbeddings(
    model_name=model_name,
    # model_kwargs=model_kwargs,
    # encode_kwargs=encode_kwargs

)

embedder = LangchainEmbeddingsWrapper(hf_e)

print("llm loaded")


test_data = {

    'question': ['What is the capital of France?', 'Who wrote "Romeo and Juliet"?'],
    'contexts': [['Bananas are an excellent source of potassium', 'France is known for its cuisine.', 'Data is more valulable than oil'],
                 ['William Shakespeare wrote "Romeo and Juliet".', 'The play is a tragedy.']],
    'answer': ['Paris', 'William Shakespeare'],
    'ground_truth': ['Paris', 'William Shakespeare']
}


dataset_eval = Dataset.from_dict(test_data)
run_config = RunConfig(thread_timeout=1000, timeout=10000)


results = evaluate(dataset_eval,
                   metrics=[
                       answer_similarity,
                       context_relevancy,
                   ],
                   llm=llm, embeddings=embedder, run_config=run_config)


print(results)
