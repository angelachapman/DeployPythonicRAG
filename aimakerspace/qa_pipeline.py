from rank_bm25 import BM25Plus

from .openai_utils.prompts import (
    SystemRolePrompt
)
from .vectordatabase import VectorDatabase
from .openai_utils.chatmodel import ChatOpenAI

# Utility function for reranking
def bm25plus_rerank(corpus, query, initial_ranking, top_n=3):
    tokenized_corpus = [corpus[i].split() for i in initial_ranking]
    tokenized_query = query.split()

    bm25 = BM25Plus(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    ranked_indices = [initial_ranking[i] for i in bm25_scores.argsort()[::-1]]    
    return ranked_indices[:top_n]


class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI(), vector_db_retriever: VectorDatabase) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever

    async def arun_pipeline(self, user_query: str):
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)

        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"

        formatted_system_prompt = system_role_prompt.create_message()

        formatted_user_prompt = user_role_prompt.create_message(question=user_query, context=context_prompt)

        async def generate_response():
            async for chunk in self.llm.astream([formatted_system_prompt, formatted_user_prompt]):
                yield chunk

        return {"response": generate_response(), "context": context_list}
    
class RerankedQAPipeline(RetrievalAugmentedQAPipeline):
    # Extends the RetrievalAugmentedQAPipeline class with reranking
    
    async def arun_pipeline(self, user_query: str, rerank: bool=False) -> str:
        # Retrieve the top 10 results. Either return the top 3, or rerank with BM25 and then return 
        # the new top 3
        context_list = self.vector_db_retriever.search_by_text(user_query, k=10)

        # Convert from tuples to strings
        context_list_str = [context_list[i][0] for i in range(len(context_list))]

        # Optionally re-rank the retrieved context using BM25
        n = 3
        reranked_contexts = context_list_str[0:n]

        if rerank:
            initial_ranking = list(range(len(context_list_str)))
            reranked_indices = bm25plus_rerank(context_list_str, user_query, initial_ranking, top_n=n)
            reranked_contexts = [context_list_str[i] for i in reranked_indices]

        context_prompt = "\n\n".join(context for context in reranked_contexts) + "\n\n"

        formatted_system_prompt = SystemRolePrompt(system_prompt).create_message() if system_prompt else rag_prompt.create_message()
        formatted_user_prompt = user_prompt.create_message(user_query=user_query, context=context_prompt)

        async def generate_response():
            async for chunk in self.llm.astream([formatted_system_prompt, formatted_user_prompt]):
                yield chunk

        return {"response": generate_response(), "context": context_list}