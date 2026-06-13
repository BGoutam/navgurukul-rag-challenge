# Ingestion — How to Run

Quick guide for getting from a fresh terminal to a populated vector DB.
Verified against the existing 10-PDF corpus at `../Bearpaw/Transformers/`.

---

## 1. One-time setup — Conda environment

```powershell
# Create a fresh, isolated env. Python 3.11 to match the Bearpaw stack.
conda create -n navgurukul-rag python=3.11 -y
conda activate navgurukul-rag

# Move into the project
cd C:\users\goutam\2025\claude-code\navgurukul-rag-challenge

# Install everything (pulls PyTorch + Sentence Transformers; ~2 GB first time)
pip install -r requirements.txt
```

The big one-time cost is PyTorch (Sentence Transformers depends on it). On a
laptop this is a 5-10 minute install. Subsequent envs reuse pip's wheel cache.

## 2. One-time setup — Tesseract for OCR fallback

Tesseract is a system binary (not a pip package). Only triggered when a
PDF page has < 50 words of native text (i.e. scanned pages). All of our
current 10 PDFs are digital, so OCR won't actually fire for this corpus —
but install it anyway so the pipeline is complete:

**Windows:**
1. Download installer from https://github.com/UB-Mannheim/tesseract/wiki
2. Run installer — accept default path `C:\Program Files\Tesseract-OCR\`
3. The ingest script auto-detects that path on Windows; no env var needed

**macOS:** `brew install tesseract`

**Linux:** `sudo apt-get install tesseract-ocr`

## 3. .env (optional for ingestion only)

`ANTHROPIC_API_KEY` is only needed for Phases 2-5 (retrieve + generate + guardrail + eval).
For Phase 1 ingestion you can skip the `.env` entirely — embeddings run locally.

When you do need it:

```powershell
copy .env.example .env
notepad .env
# paste: ANTHROPIC_API_KEY=sk-ant-...
```

## 4. Ingestion plan — what to ingest when

The corpus has 10 PDFs in two clusters:

| Cluster | Files | Pages | Ingest time |
|---------|-------|-------|-------------|
| **7 Medium / docs PDFs** (transformer concepts) | 7 | 156 | ~1 minute |
| **3 textbooks** (Jurafsky, Bishop, Sutton & Barto) | 3 | 1,932 | ~12-20 minutes |

### Recommended split

- **Now (before demo prep):** ingest only the 7 short PDFs. Total ingest time
  under a minute. You get a fully-working RAG corpus to test Phase 2 against,
  immediately.
- **During the live demo:** ingest the 3 textbooks. The page-by-page progress
  output is a more impressive visual ("watch the system process a 758-page
  Bishop PRML book in real time, OCR fallback ready if needed").

The system is idempotent — re-running the script after adding the textbooks
will skip the already-ingested 7 PDFs (SHA-256 match) and only process the
new ones. No `--force` flag needed.

## 5. Initial ingest — fast 7-PDF pass (do this now)

The Bearpaw `Transformers/` folder has the 7 short PDFs at the top level
and the 3 textbooks inside `Transformers/textbooks/`. To ingest *only* the
top-level 7, point the script at a path that excludes the subdirectory:

**Easiest:** copy the 7 PDFs into the repo's `data/pdfs/`:

```powershell
# From the navgurukul-rag-challenge directory:
copy ..\Bearpaw\Transformers\*.pdf data\pdfs\
python scripts/ingest_pdfs.py data\pdfs\
```

**Alternative (no copy):** ingest the whole `Transformers/` tree but only after
moving the textbooks aside, OR just ingest both clusters at once if you don't
mind the 12-20 minute wait.

```powershell
# Ingest everything at once (gets you all 10 PDFs in one shot)
python scripts/ingest_pdfs.py ..\Bearpaw\Transformers\
```

## 6. What you should see

For the 7-PDF run:

```
Ingesting from: data\pdfs
ChromaDB at:    C:\users\goutam\2025\claude-code\navgurukul-rag-challenge\data\chroma_db

  [ingest] 4 Sentence Embedding Techniques One Should Know.pdf .21/21
      (4.2s)
  [ingest] How Self Attention works in Transformer _ by mustafac _ Analytics Vidhya _ Medium.pdf .25/28 .28/28
      (5.1s)
  [ingest] NLP Transformer Testing....pdf .25/31 .31/31
      (5.4s)
  [ingest] SentenceTransformers Documentation....pdf .7/7
      (2.1s)
  [ingest] Transformers _ Part 1_ History of NLP before Transformers....pdf .21/21
      (4.0s)
  [ingest] Transformers _ Part 2_ Architecture of Transformer....pdf .20/20
      (4.1s)
  [ingest] Understanding Q,K,V In Transformer( Self Attention)....pdf .25/28 .28/28
      (5.0s)

────────────────────────────────────────────────────────────
  PDFs total:           7
  Ingested:             7
  Skipped (no change):  0
  Pages processed:      156
  Chunks created:       ~280-340 (depends on word density)
  Collection size:      ~280-340 vectors
  Wall time:            ~30-45s after model warm-up
  ChromaDB on disk:     ~5-8 MB
────────────────────────────────────────────────────────────
```

**Note on first-run latency:** the very first call downloads the
`all-MiniLM-L6-v2` model (~90 MB) from HuggingFace and caches it under
`~/.cache/huggingface/` (or `%USERPROFILE%\.cache\huggingface\` on Windows).
This adds ~30s the first time. After that the model loads from disk in 1-2s.

## 7. Verify the ingest worked

```powershell
python -c "import chromadb; c=chromadb.PersistentClient(path='data/chroma_db'); col=c.get_collection('navgurukul_corpus'); print(f'vectors: {col.count()}'); print(f'sample: {col.peek(2)}')"
```

You should see a count > 0 and a sample of two chunks with metadata
containing `pdf_name`, `page`, and `chunk_index`.

## 8. Adding the textbooks later (demo time)

When you're ready to show live ingestion:

```powershell
# Just point at the textbooks/ subdirectory
python scripts/ingest_pdfs.py ..\Bearpaw\Transformers\textbooks\
```

Each textbook prints a progress line as it ingests (Bishop will show
`.25/758 .50/758 ... .758/758`). The progress dots are the visual the
judges will see and appreciate.

If for some reason you want to start completely fresh:

```powershell
# Wipe the DB and re-ingest everything
rmdir /s /q data\chroma_db
python scripts/ingest_pdfs.py ..\Bearpaw\Transformers\
```

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: chromadb` | conda env not activated | `conda activate navgurukul-rag` |
| `tesseract is not installed` | Tesseract binary missing | install it (see §2). For digital PDFs the OCR path doesn't fire anyway. |
| `OSError: Sentence Transformers cache failed` | offline / proxy blocking HuggingFace | manually download `sentence-transformers/all-MiniLM-L6-v2` first |
| Chroma writes a `.sqlite3` lock error | another process holds the DB open | close other Python sessions; or delete `data/chroma_db/chroma.sqlite3-journal` |
| First page of every PDF returns nothing | scanned cover with no text and OCR not installed | install Tesseract; rerun with `--force` to re-process |
| Re-running ingests nothing | manifest match (good!) | this is correct behaviour; use `--force` to override |

## 10. What comes next

Once Phase 1 ingest is verified, we move to Phase 2 (retrieve + generate).
That's the chat-side code: a `retrieve(query)` function that pulls top-K
chunks from ChromaDB, and a `generate(query, chunks)` function that calls
Claude with a strict citation prompt. CLI test in Phase 2 will let you ask
a question from the terminal and see a cited answer before we wire the
web UI in Phase 3.
