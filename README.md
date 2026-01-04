# Intelligent Complaint Analysis Chatbot – Week 7 Challenge

## Overview
This project implements an **intelligent RAG-powered chatbot** for **CrediTrust Financial**, designed to analyze **customer complaints** across five major financial products:
- Credit Cards
- Personal Loans
- Buy Now, Pay Later (BNPL)
- Savings Accounts
- Money Transfers

The system transforms unstructured complaint narratives into actionable insights using **semantic search**, **vector databases**, and **large language models (LLMs)**. Internal teams can ask natural-language questions and get evidence-backed answers, reducing manual analysis time from **days to minutes**.

---

## Project Structure

rag-complaint-chatbot/
├── .vscode/ # VSCode settings
├── .github/workflows/ # CI/CD workflow
│ └── unittests.yml
├── data/
│ ├── raw/ # Raw CFPB dataset
│ └── processed/ # Filtered and cleaned data
├── vector_store/ # Persisted FAISS/ChromaDB index
├── notebooks/ # EDA and experiments
│ └── task1_eda_preprocessing.ipynb
├── src/ # Python modules
│ ├── init.py
│ ├── preprocessing.py
│ └── utils.py
├── tests/ # Unit tests
├── app.py # Gradio/Streamlit UI
├── requirements.txt
├── README.md
└── .gitignore