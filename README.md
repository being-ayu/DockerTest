# RAG Leaderboard — NVIDIA Model Evaluation

A containerized Retrieval-Augmented Generation (RAG) service that evaluates and compares multiple LLMs using the same document, retrieval pipeline, prompts, and evaluation dataset.

The project is designed to answer an important question:

> **Which LLM performs better when used within the same RAG pipeline?**

Instead of evaluating models only based on the quality of their generated text, the system evaluates the complete RAG pipeline using multiple dimensions:

- Context Precision
- Context Recall
- Faithfulness
- Answer Relevance

The results are aggregated into an overall score and presented as a model leaderboard.

---

## 1. Project Objective

The objective of this project is to build a reproducible RAG evaluation pipeline that:

1. Loads a supplied document.
2. Splits the document into overlapping chunks.
3. Generates embeddings for each chunk.
4. Retrieves the most relevant chunks for a user question.
5. Generates an answer using multiple NVIDIA-hosted LLMs.
6. Evaluates each generated answer.
7. Aggregates evaluation metrics across multiple questions.
8. Produces a leaderboard comparing the models.
9. Exposes the functionality through a FastAPI service.
10. Runs entirely inside Docker.

The important principle is:

> **All models should be evaluated under the same retrieval and evaluation conditions so that the comparison is as fair as possible.**

---

# 2. High-Level Architecture

```text
                         ┌──────────────────────┐
                         │   input_document.txt │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Document Loading     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Chunking             │
                         │ 500 words            │
                         │ 75 word overlap      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ NVIDIA Embeddings    │
                         │                      │
                         │ nvidia/nv-embedqa... │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Vector Index         │
                         │ NumPy                │
                         └──────────┬───────────┘
                                    │
                                    │
                       User Question / Evaluation
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Query Embedding      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Similarity Search    │
                         │ Top-K = 5            │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
          ┌──────────────────┐            ┌──────────────────┐
          │ Llama 3.2 3B     │            │ GPT-OSS 20B      │
          └────────┬─────────┘            └────────┬─────────┘
                   │                               │
                   ▼                               ▼
          ┌──────────────────┐            ┌──────────────────┐
          │ Generated Answer │            │ Generated Answer │
          └────────┬─────────┘            └────────┬─────────┘
                   │                               │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │ Evaluation           │
                         │                      │
                         │ Context Precision    │
                         │ Context Recall       │
                         │ Faithfulness         │
                         │ Answer Relevance     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Overall Score        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Model Leaderboard    │
                         └──────────────────────┘