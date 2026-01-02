import faiss, numpy as np
x = np.random.rand(1000, 768).astype("float32")
index = faiss.IndexFlatIP(768)
index.add(x)
D, I = index.search(x[:5], 10)
print(D.shape, I.shape)
