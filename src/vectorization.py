"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
import sys
from pathlib import Path
import logging
from dataclasses import dataclass
import json
import re

# Custom text splitter (no langchain dependency)
from typing import List

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not installed. Please run: pip install sentence-transformers")

# Vector store
try:
    import faiss
except ImportError:
    print("Warning: faiss-cpu not installed. Please run: pip install faiss-cpu")

# Progress bar
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x: x  # Fallback if tqdm not installed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom Text Splitter (replacement for LangChain's RecursiveCharacterTextSplitter)
class CustomTextSplitter:
    """Custom text splitter that mimics LangChain's RecursiveCharacterTextSplitter"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        if not text:
            return []
        
        # If text is already shorter than chunk size, return as is
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If we're at the end of the text
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to find a good break point
            chunk = text[start:end]
            
            # Look for the last occurrence of each separator
            best_break = -1
            for separator in self.separators:
                if separator:
                    # Find the last occurrence of this separator in the chunk
                    break_pos = chunk.rfind(separator)
                    if break_pos > best_break and break_pos > self.chunk_size * 0.3:
                        best_break = break_pos + len(separator)
            
            # If we found a good break point, use it
            if best_break > 0:
                end = start + best_break
            
            # Add the chunk
            chunks.append(text[start:end])
            
            # Move start position for next chunk, accounting for overlap
            start = end - self.chunk_overlap
            
            # Ensure we're making progress
            if start <= end - self.chunk_overlap:
                start = end - self.chunk_overlap
        
        return chunks
    
    def __call__(self, text: str) -> List[str]:
        return self.split_text(text)

@dataclass
class VectorStoreConfig:
    """Configuration for vector store creation"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sample_size: int = 12000
    vector_store_path: str = "../vector_store/faiss_index"
    metadata_path: str = "../vector_store/metadata.json"
    random_state: int = 42

class Task2Pipeline:
    """Pipeline for text chunking, embedding, and vector store indexing"""
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self.text_splitter = None
        self.embedding_model = None
        self.vector_store = None
        self.metadata = []
        
    def create_stratified_sample(self, df: pd.DataFrame, text_column: str = "narrative_cleaned") -> pd.DataFrame:
        """
        Create a stratified sample from the cleaned dataframe.
        
        Args:
            df: Input dataframe with cleaned narratives
            text_column: Name of the text column
            
        Returns:
            Stratified sample dataframe
        """
        logger.info("Creating stratified sample...")
        
        # Check if product column exists
        if 'product' not in df.columns:
            raise ValueError("DataFrame must contain 'product' column for stratification")
        
        # Filter out rows with missing text
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].str.strip() != ""]
        
        # Get unique products
        products = df['product'].unique()
        logger.info(f"Found {len(products)} product categories")
        
        # Calculate sample size per product
        sample_per_product = self.config.sample_size // len(products)
        remaining = self.config.sample_size % len(products)
        
        sampled_dfs = []
        for i, product in enumerate(products):
            product_df = df[df['product'] == product]
            
            # Calculate sample size for this product
            n_samples = sample_per_product + (1 if i < remaining else 0)
            
            if len(product_df) >= n_samples:
                # Sample if enough instances
                sampled = product_df.sample(
                    n=min(n_samples, len(product_df)),
                    random_state=self.config.random_state
                )
            else:
                # Use all instances if not enough
                sampled = product_df.copy()
                logger.warning(f"Product '{product}' has only {len(product_df)} instances, using all")
            
            sampled_dfs.append(sampled)
        
        # Combine and shuffle
        stratified_sample = pd.concat(sampled_dfs, ignore_index=True)
        stratified_sample = stratified_sample.sample(
            frac=1, 
            random_state=self.config.random_state
        ).reset_index(drop=True)
        
        logger.info(f"Created stratified sample with {len(stratified_sample)} complaints")
        logger.info(f"Sample distribution:\n{stratified_sample['product'].value_counts()}")
        
        return stratified_sample
    
    def initialize_text_splitter(self):
        """Initialize the text splitter with configured parameters"""
        logger.info(f"Initializing text splitter with chunk_size={self.config.chunk_size}, "
                   f"chunk_overlap={self.config.chunk_overlap}")
        
        self.text_splitter = CustomTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
    
    def initialize_embedding_model(self):
        """Initialize the embedding model"""
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        
        try:
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            
            # Test the model
            test_embedding = self.embedding_model.encode(["test sentence"])
            logger.info(f"Embedding model loaded. Vector dimension: {test_embedding.shape[1]}")
        except ImportError:
            logger.error("sentence-transformers not installed. Please install it with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def chunk_texts(self, df: pd.DataFrame, text_column: str = "narrative_cleaned") -> Dict[str, List]:
        """
        Chunk the narratives and prepare metadata.
        
        Args:
            df: Input dataframe
            text_column: Name of the text column
            
        Returns:
            Dictionary with chunks and metadata
        """
        if self.text_splitter is None:
            self.initialize_text_splitter()
        
        logger.info("Chunking texts...")
        
        all_chunks = []
        all_metadata = []
        
        for idx, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Chunking")):
            text = getattr(row, text_column) if hasattr(row, text_column) else df.iloc[idx][text_column]
            complaint_id = getattr(row, 'complaint_id', None) if hasattr(row, 'complaint_id') else df.iloc[idx].get('complaint_id', idx)
            product = getattr(row, 'product', None) if hasattr(row, 'product') else df.iloc[idx].get('product', 'unknown')
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(str(text))
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                
                # Store metadata for each chunk
                metadata = {
                    'complaint_id': complaint_id,
                    'product': product,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'original_text_length': len(str(text))
                }
                all_metadata.append(metadata)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(df)} complaints")
        logger.info(f"Average chunks per complaint: {len(all_chunks) / len(df):.2f}")
        
        return {
            'chunks': all_chunks,
            'metadata': all_metadata
        }
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Create embeddings for text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Numpy array of embeddings
        """
        if self.embedding_model is None:
            self.initialize_embedding_model()
        
        logger.info(f"Creating embeddings for {len(chunks)} chunks...")
        
        # Create embeddings in batches to manage memory
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
            batch = chunks[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch, 
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        logger.info(f"Created embeddings with shape: {embeddings.shape}")
        
        return embeddings
    
    def create_vector_store(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Create and save FAISS vector store.
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries
        """
        logger.info("Creating FAISS vector store...")
        
        try:
            # Get embedding dimension
            dimension = embeddings.shape[1]
            
            # Create FAISS index (using inner product for cosine similarity since vectors are normalized)
            index = faiss.IndexFlatIP(dimension)
            
            # Add embeddings to index
            index.add(embeddings.astype('float32'))
            
            # Save the index
            vector_store_dir = Path(self.config.vector_store_path).parent
            vector_store_dir.mkdir(parents=True, exist_ok=True)
            
            faiss.write_index(index, self.config.vector_store_path)
            
            # Save metadata
            with open(self.config.metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"Vector store saved to: {self.config.vector_store_path}")
            logger.info(f"Metadata saved to: {self.config.metadata_path}")
            logger.info(f"Total vectors stored: {index.ntotal}")
            
            self.vector_store = index
            self.metadata = metadata
            
        except ImportError:
            logger.error("faiss-cpu not installed. Please install it with: pip install faiss-cpu")
            raise
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise
    
    def run_pipeline(self, df: pd.DataFrame, text_column: str = "narrative_cleaned") -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            df: Input dataframe with cleaned narratives
            text_column: Name of the text column
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting Task 2 Pipeline...")
        
        # Step 1: Create stratified sample
        sampled_df = self.create_stratified_sample(df, text_column)
        
        # Step 2: Initialize components
        self.initialize_text_splitter()
        self.initialize_embedding_model()
        
        # Step 3: Chunk texts
        chunked_data = self.chunk_texts(sampled_df, text_column)
        
        # Step 4: Create embeddings
        embeddings = self.create_embeddings(chunked_data['chunks'])
        
        # Step 5: Create vector store
        self.create_vector_store(embeddings, chunked_data['metadata'])
        
        # Prepare results summary
        results = {
            'sample_size': len(sampled_df),
            'chunks_created': len(chunked_data['chunks']),
            'embedding_dimension': embeddings.shape[1],
            'vector_store_path': self.config.vector_store_path,
            'metadata_path': self.config.metadata_path,
            'sample_distribution': sampled_df['product'].value_counts().to_dict()
        }
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results: {json.dumps(results, indent=2)}")
        
        return results
    
    def validate_vector_store(self, query_text: str = "bank account issue", k: int = 5):
        """
        Validate the vector store by performing a sample search.
        
        Args:
            query_text: Text to search for
            k: Number of results to return
        """
        if self.embedding_model is None:
            self.initialize_embedding_model()
        
        if self.vector_store is None:
            # Load existing vector store
            if not os.path.exists(self.config.vector_store_path):
                raise FileNotFoundError(f"Vector store not found at {self.config.vector_store_path}")
            
            self.vector_store = faiss.read_index(self.config.vector_store_path)
            
            # Load metadata
            with open(self.config.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query_text], normalize_embeddings=True)
        
        # Search
        distances, indices = self.vector_store.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # FAISS returns -1 for empty slots
                metadata = self.metadata[idx]
                results.append({
                    'rank': i + 1,
                    'similarity_score': float(distance),
                    'complaint_id': metadata['complaint_id'],
                    'product': metadata['product'],
                    'chunk_index': metadata['chunk_index']
                })
        
        return results