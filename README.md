## Build a real RAG Evaluation System with Python, LangChain, OpenAI, DeepEval, and uv.


We'll build a practical evaluation workflow around a PDF knowledge base. The system extracts text from a PDF, creates embeddings, stores the chunks in an in-memory vector store, retrieves relevant context for each question, generates an answer with an OpenAI model, and then scores the result across multiple RAG evaluation metrics.

The project uses a small question-and-answer dataset about penguins so we can clearly see what the RAG system retrieved, what the model answered, what the expected answer was, and how each metric scores the result. Instead of relying on vibes, we save structured JSONL outputs, evaluate each test case, calculate averages, measure pass rates, and compare experiments.

By the end, you'll have a portfolio-ready RAG evaluation project that shows how to test answer quality, retrieval quality, faithfulness, and factual correctness in a real AI engineering workflow.