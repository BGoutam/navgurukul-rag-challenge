# Corpus — Place PDFs here

This directory is gitignored except for this README.

For the Navgurukul Challenge 1 submission, the corpus used is **10 PDFs / 2,088 pages** of NLP and ML literature:

## Originals (7 PDFs · ~156 pages, Medium articles + docs)

These are freely-accessible Medium and project-doc PDFs on transformers / attention / sentence embeddings:

- `4 Sentence Embedding Techniques One Should Know.pdf`
- `How Self Attention works in Transformer.pdf`
- `NLP Transformer Testing.pdf`
- `SentenceTransformers Documentation.pdf`
- `Transformers Part 1 — History of NLP before Transformers.pdf`
- `Transformers Part 2 — Architecture of Transformer.pdf`
- `Understanding Q,K,V In Transformer (Self Attention).pdf`

## Textbook augmentation (3 PDFs · ~1,932 pages)

These bring the corpus into spec (≥10 PDFs, each PDF ≥200 pages, total ≥2,000 pages):

| File | Author | Pages | Source URL |
|------|--------|-------|------------|
| `Jurafsky_Martin_SLP3.pdf` | Jurafsky & Martin | 626 | https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf |
| `Bishop_PRML.pdf` | Christopher Bishop | 758 | Microsoft Research free release (2021) |
| `Sutton_Barto_RL2020.pdf` | Sutton & Barto | 548 | http://incompleteideas.net/book/RLbook2020.pdf |

All three are freely-licensed for educational use.

## How to populate

For the demo, point the ingest script directly at any directory containing these files:

```bash
python scripts/ingest_pdfs.py path/to/your/pdfs/
```

Or copy them into `data/pdfs/` and run:

```bash
python scripts/ingest_pdfs.py data/pdfs/
```
