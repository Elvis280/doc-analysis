
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Retrieval import get_vdb

def verify_intent_removal():
    print("Verifying intent removal from all chunks...")
    vdb = get_vdb()
    sources = vdb.get_all_sources()
    
    total_chunks = 0
    clean_chunks = 0
    intent_chunks = 0
    
    for source in sources:
        source_hash = source['source_hash']
        coll = vdb._load_collection(source_hash)
        if not coll: continue
        
        for meta in coll['metadata']:
            total_chunks += 1
            emb_text = meta.get("embedding_text", "")
            
            if emb_text.startswith("Intent:"):
                print(f"❌ FAIL: Intent found in chunk {meta['chunk_id']}")
                print(f"   Text: {emb_text[:50]}...")
                intent_chunks += 1
            else:
                clean_chunks += 1

    print("\n" + "="*30)
    print("VERIFICATION RESULTS")
    print("="*30)
    print(f"Total Chunks: {total_chunks}")
    print(f"Clean Chunks: {clean_chunks}")
    print(f"Intent Chunks: {intent_chunks}")
    
    if intent_chunks == 0:
        print("\n✅ SUCCESS: All chunks are clean (no intent prefix).")
    else:
        print(f"\n❌ FAILURE: Found {intent_chunks} chunks with intent prefix.")

if __name__ == "__main__":
    verify_intent_removal()
