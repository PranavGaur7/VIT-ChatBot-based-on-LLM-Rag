import os
import logging
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

# Updated imports for new LangChain structure
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# CHANGED: Import Google's LLM instead of Ollama's
from langchain_google_genai import ChatGoogleGenerativeAI

# (These imports from your original file are not used in the RAG chain,
# but we'll leave them in case you use them elsewhere)
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.reasoning import ReasoningTools

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv() # This will now load your GOOGLE_API_KEY

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/academics"
COLLECTION_NAME = "vit_academics"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  PROMPT UPDATED HERE 
chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an Academic Assistant for VIT University. "
     "You must answer ONLY using information explicitly present in the retrieved CSV rows provided to you. "
     "The CSV context is the single source of truth. "
     "If the answer is not fully supported by the CSV context, respond ONLY with: 'I don’t know'. "
     "Do not infer, guess, or use outside knowledge. "
     "When presenting data, be extremely precise and faithful to the CSV. "
     "If a structured answer is needed, you MUST use bullet points, sub-headings, "
     "and HTML tables (<table>, <tr>, <th>, <td>). "
     "Never use Markdown tables. "
     "Do not rewrite or reformat CSV rows beyond organizing them for clarity."
    ),
    ("system",
     "Your job in every response:\n"
     "1. Read the question.\n"
     "2. Read the retrieved CSV rows in 'CSV_CONTEXT'.\n"
     "3. Extract ONLY the relevant rows.\n"
     "4. Summarize them accurately without adding new details.\n"
     "5. If no relevant rows exist, say 'I don’t know'."
    ),
    ("human",
     "QUESTION:\n{question}\n\n"
     "CSV_CONTEXT:\n{context}\n\n"
     "Generate the final answer strictly based on the CSV context. "
     "If missing, say 'I don’t know'.")
])



# CHANGED: This function now initializes Gemini
def initialize_llm() -> ChatGoogleGenerativeAI:
    """Initialize Google Gemini LLM."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in .env file.")
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
    
    # Using a reliable, versioned model name
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Using latest
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True 
    )


def initialize_vectorstore() -> Chroma:
    """Initialize Chroma vector store with embedding model."""
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"trust_remote_code": True}
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=str(VECTORSTORE_DIR)
    )


def process_csvs_to_vectorstore(csv_paths: list[str], vector_store: Chroma, force_rebuild: bool = False):
    """Load academic data from multiple CSV files and store in the vector DB."""
    existing_docs = vector_store._collection.count()
    if existing_docs > 0 and not force_rebuild:
        logger.info("Vector store already has %d documents. Skipping rebuild.", existing_docs)
        return

    all_docs = []
    for csv_path in csv_paths:
        if not Path(csv_path).exists():
            logger.warning("CSV file not found: %s", csv_path)
            continue

        logger.info("Loading CSV data from %s", csv_path)
        loader = CSVLoader(
            file_path=csv_path,
            csv_args={"delimiter": ",", "quotechar": '"'}
        )
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        docs = text_splitter.split_documents(data)
        all_docs.extend(docs)

    if not all_docs:
        logger.warning("No documents found in the provided CSVs.")
        return

    logger.info("Adding %d documents from %d CSV files to vector store in batches...", len(all_docs), len(csv_paths))
    uuids = [str(uuid4()) for _ in range(len(all_docs))]
    
    # Define a batch size (safely under the 5461 limit)
    BATCH_SIZE = 5000 
    
    for i in range(0, len(all_docs), BATCH_SIZE):
        # Get the chunk of documents and their corresponding IDs
        batch_docs = all_docs[i : i + BATCH_SIZE]
        batch_uuids = uuids[i : i + BATCH_SIZE]
        
        # Calculate batch number for logging
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(all_docs) // BATCH_SIZE) + 1
        
        logger.info(f"Adding batch {batch_num}/{total_batches} (size: {len(batch_docs)})...")
        vector_store.add_documents(batch_docs, ids=batch_uuids)
    
    logger.info("Data successfully added to vector store.")


#  CHANGED: Updated the type hint for the llm parameter
def generate_answer(query: str, llm: ChatGoogleGenerativeAI, vector_store: Chroma) -> str:
    """Run retrieval and LLM chain to answer query."""
    retriever = vector_store.as_retriever()

    chain = (
        retriever
        | (lambda docs: {
            "context": "\n\n".join([d.page_content for d in docs]),
            "question": query
        })
        | chat_prompt
        | llm
    )

    result = chain.invoke(query)
    return result.content if hasattr(result, "content") else str(result)


def academic_assistant(query: str, csvs: list[str], rebuild: bool = False) -> str:
    """Main RAG entrypoint."""
    llm = initialize_llm()
    vector_store = initialize_vectorstore()
    process_csvs_to_vectorstore(csvs, vector_store, force_rebuild=rebuild)
    return generate_answer(query, llm, vector_store)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Academic Assistant (CSV-based)")
    parser.add_argument("--csv", nargs="+", type=str, help="Paths to academic CSV files", default=["result.csv","faculty_data.csv","VIT_Alumni_Data_10000.csv","VIT_Originated_Startups_100.csv","VIT_Research_Publications_Indian_2000.csv"])
    parser.add_argument("--query", type=str, help="Academic question to answer", default="List electives related to AI")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vector DB from CSVs")

    args = parser.parse_args()

    llm = initialize_llm()
    vector_store = initialize_vectorstore()

    process_csvs_to_vectorstore(args.csv, vector_store, force_rebuild=args.rebuild)
    answer = generate_answer(args.query, llm, vector_store)

    print("\n=== Answer ===\n")
    print(answer)
