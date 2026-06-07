import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.context_tuner import ContextTunedPlanner, DemoCase  # noqa: E402

from env.corpus import Chunk  # noqa: E402


# Helper to mock torch since it might not be fully available
class MockTensor:
    def __init__(self, data, dtype=None):
        self.data = data
        self.dtype = dtype
    def __add__(self, other):
        return MockTensor(self.data)
    def float(self):
        return self
    def __mul__(self, other):
        return MockTensor(self.data)
    def __rmul__(self, other):
        return MockTensor(self.data)
    def sum(self):
        return MockTensor(1.0)
    def item(self):
        return 1.0
    def backward(self):
        pass
    def detach(self):
        return self
    def tolist(self):
        return [0.1] * len(self.data) if isinstance(self.data, list) else [0.1]
    def __matmul__(self, other):
        return MockTensor(self.data)
    def __truediv__(self, other):
        return MockTensor(self.data)
    def __gt__(self, other):
        return MockTensor(self.data)
    def __len__(self):
        return len(self.data) if isinstance(self.data, list) else 1

class MockF:
    @staticmethod
    def binary_cross_entropy_with_logits(logits, labels, pos_weight):
        return MockTensor(0.5)

class MockNN:
    @staticmethod
    def Parameter(t):
        return t

class MockOptim:
    class Adam:
        def __init__(self, params, lr):
            self.params = params
        def zero_grad(self):
            pass
        def step(self):
            pass

class MockTorch:
    float32 = "float32"
    nn = MockNN
    optim = MockOptim
    def tensor(self, data, dtype=None):
        return MockTensor(data, dtype=dtype)
    def manual_seed(self, seed):
        pass
    def rand_like(self, t):
        return MockTensor(t.data)

class MockRetriever:
    def hybrid_score(self, query, chunk):
        return 0.5
    def bm25_score(self, query, chunk):
        return 0.5
    def cross_encoder_score(self, query, chunk):
        return 0.5
    def embedding_similarity(self, q1, q2):
        return 0.5
    def keyword_overlap_score(self, query, chunk):
        return 0.5

def test_optimize_with_torch_no_chunks():
    tuner = ContextTunedPlanner(MockRetriever(), [], [])
    result = tuner._optimize_with_torch("query", [], [])
    assert len(result) == 10 # 10 is the number of features in _context_init

def test_optimize_with_torch_with_chunks(monkeypatch):
    import env.context_tuner as ct
    monkeypatch.setattr(ct, "torch", MockTorch())
    monkeypatch.setattr(ct, "F", MockF())

    tuner = ct.ContextTunedPlanner(MockRetriever(), [], [])
    tuner._train_steps = 1

    chunks = [
        Chunk(chunk_id="chunk1", domain="test", text="test text", tokens=10, keywords=["test"], relevance_tags=["tag"])
    ]
    demos = [
        DemoCase(name="demo1", query="test query", positive_chunk_ids=["chunk1"], expected_citations=["chunk1"], preferred_domains=["test"])
    ]

    result = tuner._optimize_with_torch("test query", chunks, demos)
    assert result == [0.1] * 10 # Since our mock tolist returns [0.1] * len(10 for init)
