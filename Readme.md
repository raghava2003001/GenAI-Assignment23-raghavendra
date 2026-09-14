# Assignment 23: OpenAI & Retrieval-Augmented Generation (RAG)

This project demonstrates an end-to-end Retrieval-Augmented Generation
(RAG) pipeline using Python, LangChain, OpenAI, FAISS, Wikipedia retrieval,
multiple retrieval strategies, contextual compression, and a conversational
YouTube chatbot.

The project was developed as part of the Generative AI assignment and covers
Tasks 1 to 11 from the module.

---

## Features

The project includes:

- OpenAI Chat Model integration
- OpenAI text embeddings
- WikipediaRetriever
- Document loading and text splitting
- FAISS vector database
- Vector similarity search
- Maximal Marginal Relevance (MMR)
- MultiQueryRetriever
- ContextualCompressionRetriever
- LLMChainExtractor
- YouTube transcript loading
- YouTube transcript chunking
- YouTube FAISS vector store
- Conversational RAG chatbot
- Conversation history
- Out-of-scope question handling
- Conceptual RAG questions

---

## Assignment Tasks

### Task 1 - OpenAI Setup

The project initializes an OpenAI chat model using LangChain's
`ChatOpenAI` class and executes a basic prompt.

### Task 2 - WikipediaRetriever

WikipediaRetriever is used to retrieve information related to
phishing attacks.

### Task 3 - Vector Store Retriever

A local cybersecurity document is loaded, divided into chunks,
converted into OpenAI embeddings and stored in a FAISS vector database.

### Task 4 - Similarity Search vs MMR

The project compares:

- Standard Similarity Search
- Maximal Marginal Relevance (MMR)

Similarity search focuses on relevance while MMR also considers
diversity between retrieved documents.

### Task 5 - MultiQueryRetriever

MultiQueryRetriever generates alternative interpretations of a
user question to improve retrieval.

### Task 6 - Contextual Compression

ContextualCompressionRetriever with LLMChainExtractor is used to
reduce irrelevant information from retrieved documents.

The project displays:

- Retrieved content before compression
- Retrieved content after compression

### Task 7 - YouTube Transcript

YoutubeLoader is used to retrieve the transcript from the configured
YouTube video.

The transcript is then divided into smaller chunks.

### Task 8 - YouTube Vector Store

The transcript chunks are converted into OpenAI embeddings and
stored in a FAISS vector database.

### Task 9 - Conversational YouTube RAG

A conversational RAG chatbot is created using:

- YouTube transcript retrieval
- FAISS
- OpenAI embeddings
- OpenAI chat model
- Conversation history

### Task 10 - Chatbot Testing

The chatbot is tested using at least five questions.

The tests include:

- Questions about the video
- Follow-up questions
- Questions requiring conversation history
- An out-of-scope question

For unsupported information, the chatbot is instructed to respond:

> I am sorry, but the video does not mention this.

### Task 11 - Conceptual Questions

The project answers conceptual questions covering:

- RAG architecture
- FAISS vector search
- Similarity Search
- MMR
- Contextual Compression
- Conversation history

---

## Project Structure

```text
GenAI-Assignment23/
│
├── main.py
├── assignment23_rag_openai.ipynb
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
└── documents/
    └── phishguard_doc.txt