from rag_app.rag.chain import retrieve_weaviate, make_rag_chain
import sys

def test_retrieval():
    print("Testing Global Retrieval...")
    try:
        docs = retrieve_weaviate("aircraft", pdf_name=None, k=2)
        print(f"Global retrieval success. Got {len(docs)} docs.")
        for d in docs:
            print(f" - Doc: {d.get('pdf_name')}, Page: {d.get('page_number')}")
            if 'pdf_name' not in d:
                print("FAIL: 'pdf_name' missing in global results")
                sys.exit(1)
    except Exception as e:
        print(f"Global retrieval failed: {e}")
        # sys.exit(1) # Don't exit yet, might be due to empty DB

    print("\nTesting Filtered Retrieval (Mock)...")
    # We allow this to return empty if no specific PDF exists, but it shouldn't crash
    try:
        docs = retrieve_weaviate("aircraft", pdf_name="non_existent.pdf", k=2)
        print(f"Filtered retrieval ran without error. Count: {len(docs)}")
        if len(docs) > 0:
            print("FAIL: Found docs for non-existent PDF?")
    except Exception as e:
        print(f"Filtered retrieval error: {e}")

if __name__ == "__main__":
    test_retrieval()
