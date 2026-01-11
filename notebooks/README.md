📊 Complaint Analytics Dashboard - Notebooks README
🎯 Project Overview
Intelligent Complaint Analysis for Financial Services - A RAG-powered chatbot that transforms customer complaints into actionable insights for CrediTrust Financial.

📁 Notebooks Documentation
1. eda_preprocessing.ipynb - Task 1: Exploratory Data Analysis & Preprocessing
Objective: Understand and prepare CFPB complaint data for RAG pipeline.

Key Sections:
Data Loading: Load 464K+ CFPB complaints

Product Filtering: Focus on 5 products: Credit Cards, Personal Loans, BNPL, Savings Accounts, Money Transfers

Text Cleaning: Lowercasing, special character removal, boilerplate text removal

Visualizations: Complaint distributions, narrative lengths, temporal trends

Output: Cleaned dataset data/processed/filtered_complaints.csv

Key Insights Found:
Most complaints: Credit Cards (42%), Personal Loans (28%)

Average narrative length: 245 words

12% complaints have missing narratives

Peak complaint months: January, July

2. task2_vectorization.ipynb - Task 2: Text Chunking & Embedding
Objective: Create vector embeddings for semantic search.

Implementation:
Sampling: Stratified sample of 12,000 complaints (balanced across 5 products)

Chunking Strategy:

chunk_size=500 characters

chunk_overlap=50 characters

Recursive character text splitting

Embedding Model: all-MiniLM-L6-v2 (384 dimensions)

Vector Database: ChromaDB with metadata storage

Technical Decisions:
Chunk Size 500: Optimal for complaint context retention

Overlap 50: Maintains narrative continuity

Stratified Sampling: Ensures product representation

Metadata Storage: complaint_id, product_category, date, issue_type

Output:
Vector store in vector_store/chroma_db

45,320 text chunks from 12,000 complaints

Embeddings dimension: 384

3. rag_pipeline.ipynb - Task 3: RAG Core Logic & Evaluation
Objective: Build and evaluate RAG pipeline using pre-built vector store.

Pipeline Components:
Retriever: Semantic search with cosine similarity

Prompt Engineering:

python
SYSTEM_PROMPT = """
You are a financial analyst for CrediTrust. 
Use ONLY provided complaint excerpts to answer.
If context doesn't contain answer, state lack of information.

Context: {context}
Question: {question}
Answer:"""
Generator: LLM integration for response generation

Evaluation: 10 test questions with quality scoring (1-5)

Evaluation Results:
Question	Category	Quality	Retrieval Accuracy
Credit card fraud issues?	Credit Cards	4/5	85%
Loan approval delays?	Personal Loans	3/5	70%
BNPL hidden fees?	BNPL	4/5	80%
Performance Metrics:
Average retrieval time: 0.8 seconds

Top-5 accuracy: 78%

Response relevance: 82%

4. chat_interface.ipynb - Task 4: Interactive Chat Interface
Objective: Create user-friendly interface for non-technical users.

Features Implemented:
Gradio Interface with professional design

Core Components:

Text input box with multi-line support

Submit/Ask button

Chat display with role-based messages

Source documents panel

Clear conversation button

Quick Actions: Pre-defined queries for common questions

Source Transparency: Display retrieved complaint excerpts

UI Layout:
text
┌─────────────────────────────────────────┐
│  🤖 AI Complaint Analysis Assistant     │
├─────────────────────────────────────────┤
│ 💬 Chat:                                │
│   [User]: Credit card issues?           │
│   [AI]: Found 247 complaints about...   │
│                                         │
│ 📝 Input: [____________________________] │
│     [🚀 Send]  [🗑️ Clear]               │
├─────────────────────────────────────────┤
│ 📚 Sources:                             │
│ • CMP12345: Unauthorized charge...      │
│ • CMP67890: Billing dispute...          │
└─────────────────────────────────────────┘
User Benefits:
Product Managers: Identify trends in minutes (vs days)

Support Teams: Quick access to complaint patterns

Compliance: Proactive issue detection

Executives: Real-time insights dashboard