import timeit
from env.retriever import HybridRetriever

text = "This is a sample text with multiple words, some punctuation, and numbers like 12345. It is designed to test the tokenization performance of the HybridRetriever class. The more words, the better."

def test_tokenize_list():
    HybridRetriever._tokenize_for_bm25(text)

def test_tokenize_set():
    HybridRetriever._tokenize_query_terms(text)

iterations = 100000

time_list = timeit.timeit(test_tokenize_list, number=iterations)
time_set = timeit.timeit(test_tokenize_set, number=iterations)

print(f"List tokenization: {time_list:.4f}s")
print(f"Set tokenization: {time_set:.4f}s")
