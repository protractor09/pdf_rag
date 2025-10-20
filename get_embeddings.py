from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from db import query_collection, open_db


def create_rag_chain(db_path: str = "db"):
    """Create a simple RAG chain using local `mistral` via Ollama and Chroma retrieval."""
    llm = ChatOllama(model="mistral", validate_model_on_init=True)

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using the provided context. "
        "If the answer is not in the context, say you don't know. Be concise."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    def retrieve_context(question: str, k: int = 4):
        collection = open_db(db_path)
        results = query_collection(collection, question, k=k)
        context_text = "\n\n".join(r["text"] for r in results)
        return context_text, results

    def chain_func(inputs: dict) -> str:
        question = inputs["question"]
        context, sources = retrieve_context(question)
        messages = prompt.format_messages(context=context, question=question)
        response = llm.invoke(messages)
        return {"answer": response.content, "sources": sources}

    return chain_func


def ask_question(chain, question: str):
    return chain({"question": question})
