import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_ollama import ChatOllama

import config
from src.retriever import retrieve

PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
Be brief. For yes/no questions, answer "yes" or "no".
If the context does not contain the answer, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""


def answer(question: str, k: int = config.TOP_K) -> dict:
    """Simple RAG: retrieve passages, then generate an answer from them."""
    t0 = time.perf_counter()
    documents = retrieve(question, k=k)
    retrieval_s = time.perf_counter() - t0

    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # temperature=0 for reproducible answers (important for evaluation)
    llm = ChatOllama(model=config.LLM_MODEL, temperature=0)
    t1 = time.perf_counter()
    response = llm.invoke(prompt)
    generation_s = time.perf_counter() - t1

    # Return documents too: needed later to evaluate retrieval.
    # The two timings feed the per-stage latency in eval/run_eval.py.
    return {
        "question": question,
        "answer": response.content,
        "documents": documents,
        "retrieval_s": round(retrieval_s, 3),
        "generation_s": round(generation_s, 3),
    }


if __name__ == "__main__":
    for q in [
        "Was Abraham Lincoln the sixteenth President of the United States?",
        "What is the capital of Uruguay?",
    ]:
        result = answer(q)
        print("Q:", result["question"])
        print("A:", result["answer"])
        print("Top passage:", result["documents"][0].page_content[:150], "\n")