# tests/test_task3_fixed.py
"""
Fixed test script for Task 3 RAG Pipeline
"""
import os
import sys

print("=" * 60)
print("TASK 3 RAG PIPELINE TEST - FIXED VERSION")
print("=" * 60)

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
print(f"Test script location: {current_dir}")
print(f"Project root: {project_root}")

# Add project root to Python path
sys.path.insert(0, project_root)
print(f"\nPython path:")
for p in sys.path[:3]:
    print(f"  {p}")

# Try to import
print("\nTrying to import RAGPipeline...")
try:
    from src.rag_pipeline import RAGPipeline, create_test_questions
    print("✅ SUCCESS: Imported RAGPipeline from src.rag_pipeline")
except ImportError as e:
    print(f"❌ FAILED: Could not import from src.rag_pipeline: {e}")
    print("\nTrying to check if rag_pipeline.py exists...")
    
    # Check if the file exists
    rag_file_path = os.path.join(project_root, "src", "rag_pipeline.py")
    if os.path.exists(rag_file_path):
        print(f"✅ rag_pipeline.py exists at: {rag_file_path}")
        print("\nLet me check the file content...")
        try:
            with open(rag_file_path, 'r') as f:
                content = f.read(200)
                print(f"First 200 chars:\n{content}")
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print(f"❌ rag_pipeline.py not found at: {rag_file_path}")
        
        # List what's in src directory
        src_dir = os.path.join(project_root, "src")
        print(f"\nFiles in src directory:")
        for file in os.listdir(src_dir):
            print(f"  - {file}")

print("\n" + "=" * 60)
print("Checking vector store location...")
print("=" * 60)

# Check vector store
vector_store_path = os.path.join(project_root, "vector_store")
if os.path.exists(vector_store_path):
    print(f"✅ Vector store found at: {vector_store_path}")
    print("Contents of vector_store directory:")
    try:
        items = os.listdir(vector_store_path)
        for item in items[:10]:  # Show first 10 items
            item_path = os.path.join(vector_store_path, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
            else:
                print(f"  📄 {item}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more items")
    except Exception as e:
        print(f"Error listing directory: {e}")
else:
    print(f"❌ Vector store not found at: {vector_store_path}")
    
    # Look for vector store in other locations
    print("\nSearching for vector store in other locations...")
    possible_locations = [
        os.path.join(project_root, "vector_store", "chroma_db"),
        os.path.join(project_root, "data", "vector_store"),
        os.path.join(os.path.dirname(project_root), "vector_store"),
        "vector_store",
        "../vector_store"
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            print(f"✅ Found vector store at: {location}")
            vector_store_path = location
            break

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("1. Run this test first to check your environment")
print("2. If imports fail, we need to fix the rag_pipeline.py file")
print("3. If vector store not found, we need to locate it")