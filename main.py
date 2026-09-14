import os
from pathlib import Path

from dotenv import load_dotenv

# ============================================================
# ENVIRONMENT SETUP
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small"
)

YOUTUBE_URL = os.getenv(
    "YOUTUBE_URL",
    "https://www.youtube.com/watch?v=wqU7yYb35yo"
)

if not OPENAI_API_KEY:
    print("[ERROR] OPENAI_API_KEY is not configured.")
    print("Please create a .env file and add your OpenAI API key.")
    raise SystemExit(1)


# ============================================================
# LANGCHAIN IMPORTS
# ============================================================

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from langchain_community.retrievers import WikipediaRetriever
from langchain_community.document_loaders import (
    TextLoader,
    YoutubeLoader
)

from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

# New LangChain versions use langchain-classic for these
from langchain_classic.retrievers import (
    MultiQueryRetriever,
    ContextualCompressionRetriever
)

from langchain_classic.retrievers.document_compressors import (
    LLMChainExtractor
)


# ============================================================
# MODEL INITIALIZATION
# ============================================================

print("\n" + "=" * 70)
print("ASSIGNMENT 23 - OPENAI & RETRIEVAL-AUGMENTED GENERATION")
print("=" * 70)

print("\nInitializing OpenAI models...")

llm = ChatOpenAI(
    model=OPENAI_CHAT_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.1
)

embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY
)

print("[OK] Chat model initialized:", OPENAI_CHAT_MODEL)
print("[OK] Embedding model initialized:", OPENAI_EMBEDDING_MODEL)


# ============================================================
# TASK 1
# OPENAI SETUP, CHAT MODEL AND BASIC PROMPT
# ============================================================

print("\n" + "=" * 70)
print("TASK 1: OpenAI Setup and Basic Prompt Execution")
print("=" * 70)

task1_prompt = (
    "Explain Retrieval-Augmented Generation (RAG) "
    "in two simple sentences."
)

task1_response = llm.invoke(task1_prompt)

print("\nQuestion:")
print(task1_prompt)

print("\nAnswer:")
print(task1_response.content)


# ============================================================
# TASK 2
# WIKIPEDIA RETRIEVER
# ============================================================

print("\n" + "=" * 70)
print("TASK 2: WikipediaRetriever Document Retrieval")
print("=" * 70)

wiki_retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

try:
    wiki_documents = wiki_retriever.invoke(
        "Phishing attacks"
    )

    print("\nWikipedia Results:\n")

    if not wiki_documents:
        print("No Wikipedia documents were returned.")

    for index, document in enumerate(
        wiki_documents,
        start=1
    ):
        title = document.metadata.get(
            "title",
            "Unknown Title"
        )

        content = document.page_content[:500]
        content = content.replace("\n", " ")

        print(f"[{index}] {title}")
        print(content)
        print("-" * 60)

except Exception as error:
    print("[WARNING] Wikipedia retrieval failed.")
    print("Reason:", error)


# ============================================================
# TASK 3
# VECTOR STORE USING OPENAI EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("TASK 3: FAISS Vector Store with OpenAI Embeddings")
print("=" * 70)

# Create documents folder
Path("documents").mkdir(
    exist_ok=True
)

document_path = Path(
    "documents/phishguard_doc.txt"
)

# Small local document for the assignment
document_text = """
Project PhishGuard is an AI-based cybersecurity system designed
to detect phishing threats across URLs, emails and QR codes.

The system uses machine learning models such as Random Forest
and XGBoost for phishing detection.

The system is designed for low-latency execution and can process
security requests in under 120 milliseconds.

The system also records important security metadata for auditing.

Artificial intelligence and retrieval techniques can be combined
to improve cybersecurity analysis and threat detection.
"""

document_path.write_text(
    document_text,
    encoding="utf-8"
)

# Load document
loader = TextLoader(
    str(document_path),
    encoding="utf-8"
)

raw_documents = loader.load()

print("\nOriginal documents:", len(raw_documents))

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=40
)

document_chunks = text_splitter.split_documents(
    raw_documents
)

print("Document chunks:", len(document_chunks))

# Create FAISS vector database
vector_db = FAISS.from_documents(
    document_chunks,
    embeddings
)

print("[OK] FAISS vector store created.")

# Create normal retriever
base_retriever = vector_db.as_retriever(
    search_kwargs={
        "k": 2
    }
)

question = (
    "What machine learning models does "
    "PhishGuard use?"
)

retrieved_documents = base_retriever.invoke(
    question
)

print("\nQuestion:")
print(question)

print("\nRetrieved Documents:")

for document in retrieved_documents:
    print(
        "-",
        document.page_content.strip()
    )


# ============================================================
# TASK 4
# SIMILARITY SEARCH VS MMR
# ============================================================

print("\n" + "=" * 70)
print("TASK 4: Similarity Search vs MMR")
print("=" * 70)

search_question = (
    "What machine learning techniques "
    "are implemented in PhishGuard?"
)

# Standard similarity search
similarity_retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 2
    }
)

# MMR search
mmr_retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 6,
        "lambda_mult": 0.5
    }
)

print("\nQuestion:")
print(search_question)

print("\n--- Standard Similarity Search ---")

similarity_results = similarity_retriever.invoke(
    search_question
)

for index, document in enumerate(
    similarity_results,
    start=1
):
    print(
        f"{index}. {document.page_content.strip()}"
    )

print("\n--- MMR Search ---")

mmr_results = mmr_retriever.invoke(
    search_question
)

for index, document in enumerate(
    mmr_results,
    start=1
):
    print(
        f"{index}. {document.page_content.strip()}"
    )


# ============================================================
# TASK 5
# MULTI QUERY RETRIEVER
# ============================================================

print("\n" + "=" * 70)
print("TASK 5: MultiQueryRetriever")
print("=" * 70)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True
)

multiquery_question = (
    "How fast does the PhishGuard system "
    "process requests?"
)

print("\nOriginal Question:")
print(multiquery_question)

multiquery_results = multiquery_retriever.invoke(
    multiquery_question
)

print("\nMultiQuery Retrieved Documents:")

for index, document in enumerate(
    multiquery_results,
    start=1
):
    print(
        f"{index}. {document.page_content.strip()}"
    )


# Also show example query reformulations
print("\nExample Query Reformulations:")

reformulation_prompt = ChatPromptTemplate.from_template(
    """
Rewrite the following search question in three
different ways.

Return only the three rewritten questions,
one per line.

Question:
{question}
"""
)

reformulation_chain = reformulation_prompt | llm

reformulation_response = reformulation_chain.invoke(
    {
        "question": multiquery_question
    }
)

print(reformulation_response.content)


# ============================================================
# TASK 6
# CONTEXTUAL COMPRESSION
# ============================================================

print("\n" + "=" * 70)
print("TASK 6: ContextualCompressionRetriever")
print("=" * 70)

compressor = LLMChainExtractor.from_llm(
    llm
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

compression_question = (
    "What is the processing speed of PhishGuard?"
)

# Retrieve normal documents
raw_results = base_retriever.invoke(
    compression_question
)

# Retrieve compressed documents
compressed_results = compression_retriever.invoke(
    compression_question
)

print("\nQuestion:")
print(compression_question)

print("\n--- Before Compression ---")

if raw_results:
    for document in raw_results:
        print(
            document.page_content.strip()
        )
else:
    print("No documents found.")

print("\n--- After Compression ---")

if compressed_results:
    for document in compressed_results:
        print(
            document.page_content.strip()
        )
else:
    print("No compressed documents returned.")


# ============================================================
# TASK 7 & 8
# YOUTUBE TRANSCRIPT + FAISS
# ============================================================

print("\n" + "=" * 70)
print("TASK 7 & 8: YouTube Transcript and FAISS Indexing")
print("=" * 70)

print("\nYouTube URL:")
print(YOUTUBE_URL)

try:

    youtube_loader = YoutubeLoader.from_youtube_url(
        YOUTUBE_URL,
        add_video_info=False
    )

    youtube_documents = youtube_loader.load()

    print(
        "\nOriginal transcript documents:",
        len(youtube_documents)
    )

    # Split transcript
    youtube_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60
    )

    youtube_chunks = youtube_splitter.split_documents(
        youtube_documents
    )

    print(
        "Transcript chunks:",
        len(youtube_chunks)
    )

    if len(youtube_chunks) == 0:
        raise RuntimeError(
            "The YouTube transcript is empty."
        )

    # Create vector database
    youtube_vector_db = FAISS.from_documents(
        youtube_chunks,
        embeddings
    )

    youtube_retriever = youtube_vector_db.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    print(
        "[OK] YouTube transcript indexed successfully."
    )

except Exception as error:

    print(
        "\n[WARNING] YouTube transcript could not be loaded."
    )

    print("Reason:", error)

    print(
        "\nUsing a small fallback transcript "
        "so the remaining RAG tasks can still run."
    )

    fallback_text = """
    Retrieval-Augmented Generation, commonly called RAG,
    combines information retrieval with a language model.

    In a RAG system, documents are divided into smaller chunks.
    These chunks are converted into numerical embeddings.

    A vector database such as FAISS stores the embeddings.

    When a user asks a question, the system searches the vector
    database and retrieves relevant chunks.

    The retrieved information is then provided to the language
    model so that it can generate a grounded answer.

    LangChain can be used to connect document loading,
    text splitting, embeddings, vector stores and language models.

    Conversational RAG systems can also maintain chat history
    so that follow-up questions can be understood in context.
    """

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30
    )

    youtube_chunks = fallback_splitter.create_documents(
        [fallback_text]
    )

    youtube_vector_db = FAISS.from_documents(
        youtube_chunks,
        embeddings
    )

    youtube_retriever = youtube_vector_db.as_retriever(
        search_kwargs={
            "k": 2
        }
    )

    print(
        "[OK] Fallback transcript indexed."
    )

print(
    "\nTotal YouTube/fallback chunks:",
    len(youtube_chunks)
)


# ============================================================
# TASK 9
# CONVERSATIONAL YOUTUBE RAG CHATBOT
# ============================================================

print("\n" + "=" * 70)
print("TASK 9: Conversational YouTube RAG Chatbot")
print("=" * 70)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful YouTube RAG assistant.

Answer questions using only the supplied transcript
context and conversation history.

If the answer cannot be supported by the transcript,
say exactly:

"I am sorry, but the video does not mention this."

Do not invent information.

Transcript Context:
{context}
"""
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            "{question}"
        )
    ]
)

rag_chain = qa_prompt | llm

chat_history = []


def run_chat(question):
    """
    Retrieve relevant YouTube transcript chunks,
    then generate an answer using the transcript
    and previous conversation history.
    """

    retrieved_docs = youtube_retriever.invoke(
        question
    )

    context = "\n\n".join(
        document.page_content.strip()
        for document in retrieved_docs
    )

    response = rag_chain.invoke(
        {
            "context": context,
            "chat_history": chat_history,
            "question": question
        }
    )

    answer = response.content.strip()

    # Store conversation
    chat_history.append(
        HumanMessage(
            content=question
        )
    )

    chat_history.append(
        AIMessage(
            content=answer
        )
    )

    return answer


# ============================================================
# TASK 10
# FIVE QUESTIONS + OUT OF SCOPE QUESTION
# ============================================================

print("\n" + "=" * 70)
print("TASK 10: YouTube RAG Chatbot Testing")
print("=" * 70)

test_questions = [
    "What is the main topic explained in this video?",

    "What concepts or tools are mentioned for building a RAG system?",

    "Can you explain the first concept you mentioned?",

    "What was my second question to you?",

    "What is the capital of France?"
]

for index, question in enumerate(
    test_questions,
    start=1
):

    print("\n" + "-" * 70)

    print(
        f"Question {index}: {question}"
    )

    answer = run_chat(
        question
    )

    print(
        f"Answer {index}: {answer}"
    )


# ============================================================
# TASK 11
# CONCEPTUAL QUESTIONS
# ============================================================

print("\n" + "=" * 70)
print("TASK 11: Conceptual Questions")
print("=" * 70)

conceptual_questions = [
    (
        "1. What is RAG and why is it necessary?",

        """
RAG stands for Retrieval-Augmented Generation.
It retrieves relevant information from an external
knowledge source before the language model generates
an answer.

This helps the model produce answers that are grounded
in the retrieved information instead of relying only on
its internal knowledge.
"""
    ),

    (
        "2. How does vector similarity search work in FAISS?",

        """
Documents are converted into numerical vectors called
embeddings.

The user's question is also converted into an embedding.
FAISS then searches for document vectors that are closest
to the query vector according to the configured distance
or similarity measure.

The closest documents are returned as relevant context.
"""
    ),

    (
        "3. What is the difference between Similarity Search and MMR?",

        """
Similarity Search mainly selects documents that are most
similar to the user's query.

MMR, or Maximal Marginal Relevance, considers both relevance
and diversity. It tries to avoid returning several documents
that contain nearly identical information.
"""
    ),

    (
        "4. Why is Contextual Compression useful?",

        """
A retriever may return entire chunks containing information
that is not necessary for the question.

Contextual compression uses an LLM to extract the parts
that are relevant to the query. This can reduce unnecessary
context and make the final prompt more focused.
"""
    ),

    (
        "5. Why is conversation history important in a RAG chatbot?",

        """
Conversation history allows the chatbot to understand
follow-up questions.

For example, after asking about a concept, a user can say
"Can you explain that again?" The history helps the system
understand what "that" refers to.
"""
    )
]

for question, answer in conceptual_questions:

    print("\n" + question)
    print(answer.strip())


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ASSIGNMENT 23 COMPLETED")
print("=" * 70)

print(
    """
Tasks completed:

1. OpenAI setup and basic prompt
2. WikipediaRetriever
3. FAISS vector store with OpenAI embeddings
4. Similarity Search vs MMR
5. MultiQueryRetriever
6. ContextualCompressionRetriever
7. YouTube transcript loading and splitting
8. YouTube FAISS vector store
9. Conversational YouTube RAG chatbot
10. Five-question evaluation and out-of-scope handling
11. Conceptual RAG questions
"""
)