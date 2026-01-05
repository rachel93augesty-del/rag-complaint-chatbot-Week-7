# tests/test_simple_fix.py
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Current dir: {os.getcwd()}")

# Check if files exist
print("\n🔍 Checking file structure...")

chunker_path = os.path.join(PROJECT_ROOT, 'src', 'vectorization', 'chunker.py')
embedder_path = os.path.join(PROJECT_ROOT, 'src', 'vectorization', 'embedder.py')

print(f"chunker.py exists: {os.path.exists(chunker_path)}")
print(f"embedder.py exists: {os.path.exists(embedder_path)}")

# Try to import
print("\n🔧 Testing imports...")

try:
    from src.vectorization.chunker import TextChunker
    print("✅ Successfully imported TextChunker!")
    
    # Test it works
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    test_text = "This is a test complaint. It has multiple sentences."
    chunks = chunker.chunk_text(test_text, {'test': True})
    
    print(f"✅ Chunker works! Created {len(chunks)} chunks")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Let's check what's in the chunker file
    print("\n📄 Checking chunker.py content...")
    try:
        with open(chunker_path, 'r') as f:
            content = f.read()
            print(f"First 500 chars:\n{content[:500]}...")
    except:
        print("Could not read chunker.py")

# Test embedder
print("\n🔧 Testing embedder import...")
try:
    from src.vectorization.embedder import TextEmbedder
    print("✅ Successfully imported TextEmbedder!")
    
    # Note: This might fail if sentence-transformers not installed
    # That's OK for now
    try:
        embedder = TextEmbedder()
        print("✅ Embedder initialized!")
    except Exception as e:
        print(f"⚠️  Embedder init failed (may need package): {e}")
        
except Exception as e:
    print(f"❌ Error importing TextEmbedder: {e}")