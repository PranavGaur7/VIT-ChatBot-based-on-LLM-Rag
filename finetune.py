import os
import logging
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader  
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Optional: AGNO / Gemini (commented for now)
# from agno.agent import Agent
# from agno.models.google import Gemini
# from agno.tools.reasoning import ReasoningTools

# =====================================================
#               CONFIGURATION
# =====================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/academics"
COLLECTION_NAME = "vit_academics"

# CSV files to integrate
CSV_FILES = [
    # "resources/academics/vit_courses.csv",
    # "resources/academics/vit_subjects.csv",
    # "resources/academics/vit_labs.csv",
    # "resources/academics/vit_prerequisites.csv",
    # "resources/academics/vit_electives.csv"
    "result.csv","faculty_data.csv","VIT_Alumni_Data_10000.csv","VIT_Originated_Startups_100.csv","VIT_Research_Publications_Indian_2000.csv"
]

# =====================================================
#               LOGGING CONFIG
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
#               CHAT PROMPT TEMPLATE
# =====================================================
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an **Academic Assistant** for VIT University. "
     "Your role is to help students by providing clear, structured, and accurate academic information "
     "about courses, subjects, credits, prerequisites, labs, and other program-related details. "
     "Always present answers in a structured format using tables when possible. "
     "The database context is loaded from CSV files, so use only the dataset context directly "
     "for accuracy and avoid assumptions. "
     "If information is missing or not available in the dataset, clearly reply with: 'I don’t know'. "
     "Do not hallucinate or guess. "
     "Always ensure responses are student-friendly, concise, and easy to understand."
    ),
    ("human", 
     "Question: {question}\n\nCSV Context:\n{context}\n\nAnswer in a structured format:")
])

# =====================================================
#               INITIALIZATION FUNCTIONS
# =====================================================
def initialize_llm() -> OllamaLLM:
    """Initialize local Ollama LLM."""
    return OllamaLLM(model="llama2", temperature=0.3)

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

# =====================================================
#               DATA INGESTION (MULTI CSV)
# =====================================================
def process_multiple_csvs_to_vectorstore(csv_files: list[str], vector_store: Chroma, force_rebuild: bool = False):
    """Load multiple CSVs and store all content in one Chroma collection."""
    existing_docs = vector_store._collection.count()
    if existing_docs > 0 and not force_rebuild:
        logger.info("Vector store already contains %d documents. Skipping rebuild.", existing_docs)
        return

    all_docs = []
    for csv_path in csv_files:
        if not os.path.exists(csv_path):
            logger.warning("CSV file not found: %s", csv_path)
            continue

        logger.info("Loading CSV data from %s", csv_path)
        loader = CSVLoader(
            file_path=csv_path,
            csv_args={"delimiter": ",", "quotechar": '"'}
        )
        data = loader.load()

        # Add source metadata for better context
        for doc in data:
            doc.metadata["source"] = os.path.basename(csv_path)

        all_docs.extend(data)

    if not all_docs:
        logger.warning("No valid documents found from the provided CSVs.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(all_docs)

    logger.info("Adding %d documents to vector store...", len(docs))
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)
    logger.info("✅ All CSV data successfully added to vector store.")

# =====================================================
#               QUERY GENERATION (RAG)
# =====================================================
def generate_answer(query: str, llm: OllamaLLM, vector_store: Chroma) -> str:
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

# =====================================================
#               MAIN FUNCTION
# =====================================================
def academic_assistant(query: str, rebuild: bool = False):
    """Main RAG entrypoint."""
    llm = initialize_llm()
    vector_store = initialize_vectorstore()
    process_multiple_csvs_to_vectorstore(CSV_FILES, vector_store, force_rebuild=rebuild)
    return generate_answer(query, llm, vector_store)

# =====================================================
#               CLI EXECUTION
# =====================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Academic Assistant (Multi-CSV)")
    parser.add_argument("--query", type=str, help="Academic question to answer", default="List electives related to AI")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vector DB from CSVs")

    args = parser.parse_args()

    logger.info("🚀 Starting Academic Assistant...")
    answer = academic_assistant(args.query, rebuild=args.rebuild)

    print("\n=== Answer ===\n")
    print(answer)
