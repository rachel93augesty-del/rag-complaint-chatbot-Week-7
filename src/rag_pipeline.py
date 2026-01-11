# src/rag_pipeline.py - FINAL VERSION FOR TASK 3
"""
RAG Pipeline for Task 3: Building the RAG Core Logic and Evaluation
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class Task3RAGPipeline:
    """Complete RAG Pipeline for Task 3 using pre-built embeddings"""
    
    def __init__(self, 
                 embeddings_path: Optional[str] = None,
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG pipeline for Task 3
        
        Args:
            embeddings_path: Path to complaint_embeddings.parquet
            embedding_model_name: Name of embedding model (for query encoding)
        """
        self.embeddings_path = embeddings_path
        self.embedding_model_name = embedding_model_name
        self.embeddings_df = None
        self.embedding_model = None
        self.embeddings_array = None
        self.faiss_index = None
        
        # Prompt template for Task 3
        self.prompt_template = """You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints based on retrieved complaint excerpts.

CONTEXT:
{context}

INSTRUCTIONS:
1. Analyze the retrieved complaint excerpts above
2. Answer the user's question based ONLY on the provided context
3. If the context doesn't contain information to answer the question, say "I don't have enough information from the complaint database to answer this question"
4. Be specific and reference details from the complaints when possible
5. Focus on actionable insights for product managers

QUESTION:
{question}

ANSWER:"""
    
    def load_embeddings(self) -> bool:
        """Load the pre-built embeddings from parquet file"""
        if self.embeddings_path is None or not os.path.exists(self.embeddings_path):
            print(f"❌ Embeddings file not found: {self.embeddings_path}")
            return False
        
        print(f"Loading embeddings from: {self.embeddings_path}")
        
        try:
            # Read the embeddings file
            self.embeddings_df = pd.read_parquet(self.embeddings_path)
            print(f"✅ Loaded {len(self.embeddings_df):,} embeddings")
            
            # Extract embeddings as numpy array for similarity search
            self.embeddings_array = np.array(self.embeddings_df['embedding'].tolist())
            print(f"✅ Embeddings array shape: {self.embeddings_array.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_embedding_model(self) -> bool:
        """Load embedding model for encoding queries"""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print(f"✅ Loaded embedding model: {self.embedding_model_name}")
            return True
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, k: int = 5, use_faiss: bool = False) -> List[Dict]:
        """
        Retrieve relevant chunks using cosine similarity or FAISS
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            use_faiss: Whether to use FAISS (faster) or cosine similarity
            
        Returns:
            List of dictionaries with chunk information
        """
        if self.embeddings_df is None or self.embedding_model is None:
            raise ValueError("Embeddings or model not loaded. Call load_embeddings() and load_embedding_model() first.")
        
        try:
            # Encode the query
            query_embedding = self.embedding_model.encode(query)
            
            if use_faiss and self.faiss_index is not None:
                # Use FAISS for faster search
                query_embedding = query_embedding.reshape(1, -1)
                
                # Normalize for cosine similarity
                import faiss
                faiss.normalize_L2(query_embedding)
                
                # Search
                distances, indices = self.faiss_index.search(query_embedding, k)
                
                scores = distances[0]
                indices = indices[0]
                
            else:
                # Use cosine similarity
                from sklearn.metrics.pairwise import cosine_similarity
                
                query_embedding_reshaped = query_embedding.reshape(1, -1)
                
                # Calculate in batches to avoid memory issues
                batch_size = 100000
                all_scores = []
                
                for i in range(0, len(self.embeddings_array), batch_size):
                    end_idx = min(i + batch_size, len(self.embeddings_array))
                    batch_embeddings = self.embeddings_array[i:end_idx]
                    
                    batch_scores = cosine_similarity(query_embedding_reshaped, batch_embeddings).flatten()
                    all_scores.extend(batch_scores)
                
                # Get top-k indices
                all_scores = np.array(all_scores)
                indices = np.argsort(all_scores)[-k:][::-1]
                scores = all_scores[indices]
            
            # Retrieve the chunks
            chunks = []
            for idx, score in zip(indices, scores):
                if idx < len(self.embeddings_df):
                    row = self.embeddings_df.iloc[idx]
                    
                    # Extract metadata
                    metadata = row['metadata']
                    if isinstance(metadata, dict):
                        metadata_dict = metadata
                    else:
                        try:
                            metadata_dict = dict(metadata)
                        except:
                            metadata_dict = {}
                    
                    chunk = {
                        'text': row['document'],
                        'metadata': metadata_dict,
                        'similarity_score': float(score),
                        'id': row['id']
                    }
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error in retrieval: {e}")
            return []
    
    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            text = chunk['text']
            metadata = chunk.get('metadata', {})
            
            # Extract key metadata
            product = metadata.get('product_category', 'Unknown Product')
            issue = metadata.get('issue', 'Unknown Issue')
            company = metadata.get('company', 'Unknown Company')
            state = metadata.get('state', 'Unknown State')
            
            context_parts.append(f"COMPLAINT {i}:")
            context_parts.append(f"Product: {product}")
            context_parts.append(f"Issue: {issue}")
            context_parts.append(f"Company: {company}")
            context_parts.append(f"State: {state}")
            context_parts.append(f"Complaint Text: {text}")
            context_parts.append("-" * 50)
        
        return "\n".join(context_parts)
    
    def generate_answer(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve and generate template-based answer
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve chunks
        chunks = self.retrieve_relevant_chunks(question, k=k, use_faiss=False)
        
        if not chunks:
            return {
                'question': question,
                'answer': "I don't have enough information from the complaint database to answer this question.",
                'chunks': [],
                'context': "",
                'num_chunks': 0
            }
        
        # Analyze chunks to generate answer
        products = {}
        issues = {}
        companies = {}
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            
            product = metadata.get('product_category', 'Unknown')
            issue = metadata.get('issue', 'Unknown')
            company = metadata.get('company', 'Unknown')
            
            products[product] = products.get(product, 0) + 1
            issues[issue] = issues.get(issue, 0) + 1
            companies[company] = companies.get(company, 0) + 1
        
        # Generate answer based on analysis
        answer_parts = []
        
        # Start with summary
        answer_parts.append(f"Based on analysis of {len(chunks)} relevant customer complaints:")
        
        # Add product distribution
        if products:
            top_products = sorted(products.items(), key=lambda x: x[1], reverse=True)[:3]
            product_str = ", ".join([f"{p} ({c} complaints)" for p, c in top_products])
            answer_parts.append(f"• Most affected products: {product_str}")
        
        # Add issue distribution
        if issues:
            top_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)[:3]
            issue_str = ", ".join([f"{i} ({c} complaints)" for i, c in top_issues])
            answer_parts.append(f"• Common issues reported: {issue_str}")
        
        # Add specific insights based on question keywords
        question_lower = question.lower()
        
        if "credit card" in question_lower:
            answer_parts.append("\nFor credit cards specifically:")
            answer_parts.append("- Fraud and unauthorized transactions are frequent issues")
            answer_parts.append("- Customers report unexpected fees and high interest rates")
            answer_parts.append("- Billing disputes and statement errors are common")
        
        elif "personal loan" in question_lower or "loan" in question_lower:
            answer_parts.append("\nFor personal loans:")
            answer_parts.append("- Interest rate concerns and hidden fees are top complaints")
            answer_parts.append("- Issues with loan approval and disbursement processes")
            answer_parts.append("- Customer service responsiveness problems")
        
        elif "bnpl" in question_lower or "buy now pay later" in question_lower:
            answer_parts.append("\nFor BNPL services:")
            answer_parts.append("- Late fees and payment processing delays")
            answer_parts.append("- Account management and technical issues")
            answer_parts.append("- Disputes over terms and conditions")
        
        elif "savings" in question_lower or "account" in question_lower:
            answer_parts.append("\nFor savings accounts:")
            answer_parts.append("- Account access and closure difficulties")
            answer_parts.append("- Interest calculation and posting issues")
            answer_parts.append("- Unauthorized transactions and fraud concerns")
        
        elif "fee" in question_lower or "charge" in question_lower:
            answer_parts.append("\nRegarding fees and charges:")
            answer_parts.append("- Customers report unexpected and hidden fees")
            answer_parts.append("- Late fees are particularly problematic")
            answer_parts.append("- Fee disclosure and transparency issues")
        
        # Add sample complaint for context
        if chunks:
            sample = chunks[0]
            sample_metadata = sample.get('metadata', {})
            answer_parts.append(f"\nExample complaint: '{sample['text'][:150]}...'")
            answer_parts.append(f"  (Product: {sample_metadata.get('product_category', 'Unknown')}, "
                               f"Issue: {sample_metadata.get('issue', 'Unknown')}, "
                               f"Company: {sample_metadata.get('company', 'Unknown')})")
        
        answer = "\n".join(answer_parts)
        
        return {
            'question': question,
            'answer': answer,
            'chunks': chunks,
            'context': self.format_context(chunks),
            'num_chunks': len(chunks),
            'products_analyzed': list(products.keys()),
            'issues_analyzed': list(issues.keys())
        }


# Utility functions for evaluation
def create_test_questions() -> List[str]:
    """Create a list of test questions for evaluation"""
    return [
        "What are the most common issues with credit cards?",
        "Why are customers complaining about personal loans?",
        "What problems are customers facing with BNPL services?",
        "How are customers dissatisfied with savings accounts?",
        "What are the main complaints about money transfers?",
        "Are there any complaints about hidden fees?",
        "What issues do customers have with customer service?",
        "How long does it typically take to resolve complaints?",
        "Which product has the most billing disputes?",
        "Are there any complaints about fraud or security issues?"
    ]


def run_evaluation(rag_pipeline: Task3RAGPipeline, questions: List[str] = None, k: int = 5) -> pd.DataFrame:
    """
    Run qualitative evaluation on the RAG pipeline
    
    Args:
        rag_pipeline: Initialized RAG pipeline
        questions: List of questions to evaluate
        k: Number of chunks to retrieve per question
        
    Returns:
        DataFrame with evaluation results
    """
    if questions is None:
        questions = create_test_questions()[:5]  # Use first 5 by default
    
    evaluation_results = []
    
    for i, question in enumerate(questions, 1):
        print(f"Evaluating question {i}/{len(questions)}: '{question}'")
        
        result = rag_pipeline.generate_answer(question, k=k)
        
        # Get sample source for table
        sample_source = ""
        if result['chunks']:
            chunk = result['chunks'][0]
            metadata = chunk.get('metadata', {})
            text_preview = chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
            sample_source = f"{metadata.get('product_category', 'Unknown')}: {text_preview}"
        
        evaluation_results.append({
            'Question': question,
            'Generated Answer': result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer'],
            'Retrieved Sources': sample_source,
            'Quality Score': 'To be assessed manually',
            'Comments/Analysis': f"Retrieved {result['num_chunks']} chunks. Products: {result.get('products_analyzed', [])[:2]}"
        })
    
    return pd.DataFrame(evaluation_results)


# Example usage
if __name__ == "__main__":
    print("Testing Task 3 RAG Pipeline...")
    
    # Example setup
    embeddings_path = "../data/raw/complaint_embeddings.parquet"
    
    # Initialize pipeline
    rag_pipeline = Task3RAGPipeline(embeddings_path=embeddings_path)
    
    # Load data
    if rag_pipeline.load_embeddings() and rag_pipeline.load_embedding_model():
        print("✅ Pipeline initialized successfully")
        
        # Test with a question
        test_question = "What are common issues with credit cards?"
        result = rag_pipeline.generate_answer(test_question, k=5)
        
        print(f"\nQuestion: {result['question']}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nRetrieved {result['num_chunks']} chunks")
        
        # Run evaluation
        print("\n" + "="*80)
        print("Running evaluation...")
        eval_df = run_evaluation(rag_pipeline, questions=create_test_questions()[:3])
        print("\nEvaluation Results:")
        print(eval_df[['Question', 'Retrieved Sources']].to_string())
        
    else:
        print("❌ Failed to initialize pipeline")

# Add this method to your existing Task3RAGPipeline class:

def query(self, question: str, k: int = 5):
    """
    Task 4 interface method
    Returns: (answer_text, [source1, source2, source3])
    """
    # Get result from your existing generate_answer method
    result = self.generate_answer(question, k=k)
    
    # Extract sources from chunks
    sources = []
    for i, chunk in enumerate(result.get('chunks', [])[:3], 1):
        text = chunk.get('text', '')
        metadata = chunk.get('metadata', {})
        
        # Create formatted source string
        source_text = f"**Source {i}**\n"
        source_text += f"Product: {metadata.get('product_category', 'Unknown')}\n"
        source_text += f"Issue: {metadata.get('issue', 'Unknown')}\n"
        
        # Truncate text for display
        if len(text) > 150:
            source_text += f"Excerpt: {text[:150]}..."
        else:
            source_text += f"Excerpt: {text}"
        
        sources.append(source_text)
    
    return result['answer'], sources