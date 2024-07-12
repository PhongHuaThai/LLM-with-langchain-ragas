from langsmith.wrappers import wrap_openai
from typing import List

import numpy as np
import openai
from langsmith import traceable
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    async def from_docs(cls, docs, oai_client):
        embeddings = await oai_client.embeddings.create(
            model="text-embedding-3-small", input=[doc["content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    @classmethod
    async def from_docs2(cls, docs, oai_client):
        embeddings = await oai_client.embeddings.create(
            model="text-embedding-3-small", input=[doc["content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    @traceable
    async def query(self, query: str, k: int = 5) -> List[dict]:
        embed = await self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


class NaiveRagBot:
    def __init__(self, retriever, llm, model: str = "gpt-4-turbo-preview"):
        self._retriever = retriever
        # Wrapping the client instruments the LLM
        # and is completely optional
        # self._client = wrap_openai(openai.AsyncClient())
        self._model = model
        if llm is not None:
            self._llm = llm

    @traceable
    async def get_answer(self, question: str):
        similar = await self._retriever.query(question)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                    " Use the following docs to help answer the user's question.\n\n"
                    f"## Docs\n\n{similar}",
                },
                {"role": "user", "content": question},
            ],
        )

        # The RAGAS evaluators expect the "answer" and "contexts"
        # keys to work properly. If your pipeline does not return these values,
        # you should wrap in a function that provides them.
        return {
            "answer": response.choices[0].message.content,
            "contexts": [str(doc) for doc in similar],
        }

    @traceable
    async def get_answer2(self, question: str):
        similar = self._retriever.invoke(question)
        context = [doc.page_content for doc in similar]
        # messages = [
        #     (
        #         content="You are a helpful AI assistant.Use the following docs to help answer the user's question."
        #     ),
        #     HumanMessage(content="##Docs:\n\n {similar}"),
        #     HumanMessage(
        #         content="question: {question}"
        #     ),
        # ]

        prompt = PromptTemplate.from_template(
            """
            [INST]<<SYS>>You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.<</SYS>> 
            Question: {input} 
            Context: {context} 
            Answer: Answer the question as best as you can using only the information from the document. If you don't know the answer, just say that you don't know.[/INST]
            """
        )

        # chain =  prompt | self._llm | StrOutputParser()
        # chain = (
        #     {
        #         "context": self.retriever,
        #         "question": RunnablePassthrough()
        #     }
        #     | prompt
        #     | self.model
        #     | StrOutputParser()
        # )
        stuff_documents_chain = create_stuff_documents_chain(
            self._llm, prompt)
        chain = create_retrieval_chain(self._retriever, stuff_documents_chain)
        response = await chain.ainvoke({'input': question})
        return {
            "answer": response['answer'],
            "contexts": [doc.page_content for doc in response['context']]
        }
        # response = stuff_documents_chain.invoke(
        #    {'context', similar}, {'input': question})

# The RAGAS evaluators expect the "answer" and "contexts"
# keys to work properly. If your pipeline does not return these values,
# you should wrap in a function that provides them.
        # return {
        #     "answer": response,
        #     "contexts": context,
        # }
