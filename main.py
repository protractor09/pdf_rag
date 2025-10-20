import argparse
from data_format import extract_pdf, chunk_text
from get_embeddings import  create_rag_chain, ask_question
from db import make_db, open_db

def main():
    parser = argparse.ArgumentParser(description="CLI RAG with local Mistral")
    parser.add_argument("--pdf", type=str, help="Path to PDF file (ingest)")
    parser.add_argument("--question", type=str, help="Question to ask (query mode)")
    parser.add_argument("--db", type=str, default="db", help="Chroma DB path")
    args = parser.parse_args()

    # Ingest mode if a PDF path is provided
    if args.pdf:
        print("📄 Extracting PDF...")
        text = extract_pdf(args.pdf)
        chunks = chunk_text(text)
        print("🧠 Embedding & storing chunks...")
        make_db(chunks, db_path=args.db)

    # Query mode if a question is provided
    print("🤖 Loading local Mistral & creating RAG chain...")
    chain = create_rag_chain(db_path=args.db)

    # One-shot query if provided
    if args.question:
        result = ask_question(chain, args.question)
        print("\n💡 Answer:")
        print(result["answer"]) 
        if result.get("sources"):
            print("\n📎 Sources (top-k):")
            for i, src in enumerate(result["sources"], 1):
                print(f"  {i}. id={src['id']} score={src['score']:.4f}")
        return

    # Interactive REPL if no --question
    print("\nType your question and press Enter. Type 'exit' to quit.")
    while True:
        try:
            user_q = input("❓ Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting.")
            break
        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit", ":q", "q"}:
            print("👋 Bye!")
            break
        result = ask_question(chain, user_q)
        print("💡", result["answer"]) 
        if result.get("sources"):
            print("📎 Sources (top-k):")
            for i, src in enumerate(result["sources"], 1):
                print(f"  {i}. id={src['id']} score={src['score']:.4f}")

if __name__ == "__main__":
    main()
