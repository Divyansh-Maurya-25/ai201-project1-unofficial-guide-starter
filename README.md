# The Unofficial Guide — Project 1

A small Retrieval-Augmented Generation (RAG) system that answers questions about
the **USF Computer Science undergraduate program** using real student-sourced
knowledge — Reddit, Rate My Professors, GitHub study guides, and the official USF
program materials.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core stack; sentence-transformers is optional (see note)
echo "GROQ_API_KEY=your_real_key" > .env # get a free key at https://console.groq.com

python build_index.py                    # Stages 1-3: ingest -> chunk -> embed -> ChromaDB
python evaluate.py                        # run the 5 test questions end-to-end
python app.py                             # launch the Gradio chat UI at http://127.0.0.1:7860
```

---

## Domain

This system covers the **lived student experience of the USF Computer Science
program** — course difficulty, which professors to take, study strategies, the
easiest electives, the order programming languages are taught, and whether the
degree is worth it.

This knowledge is valuable because the official channels don't carry it. The USF
program page sells the degree (accreditation, salaries, career outlook) but never
tells you that COP 2510 exams under Dr. Small are open-book, that the realistic
goal in CDA is to "no-life" your way to a B, or that some single-professor courses
are quality coin-flips. That tacit advice is scattered across Reddit threads,
Rate My Professors reviews, a student-run GitHub study-guide repo, and the
4-year plan PDF. Consolidating it into one grounded, source-cited Q&A tool is the
point of this project.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | r/USF — "Program Design COP 3514" | Reddit thread | `documents/reddit_cop3514.txt` |
| 2 | r/USF — "COP 2510 Difficulty" (Dr. Small) | Reddit thread | `documents/reddit_cop2510.txt` |
| 3 | r/USF — "Easiest CS Electives" | Reddit thread | `documents/reddit_easiest_electives.txt` |
| 4 | r/USF — "How hard are these classes? Will working be too much?" | Reddit thread | `documents/reddit_course_load.txt` |
| 5 | r/USF — "Program quality and languages?" | Reddit thread | `documents/reddit_cs_experience.txt` |
| 6 | r/USF — "Need your opinion" (program experience) | Reddit thread | `documents/reddit_program_opinion.txt` |
| 7 | USF — Official BSCS program page | Web page (.txt) | `documents/usf_bscs_page.txt` |
| 8 | USF — BSCS 4-Year Plan of Study | PDF | `documents/usf_cs_plan.pdf` |
| 9 | GitHub — `aeckar/usf-cse-resources` README | GitHub (.txt) | `documents/github_cse_resources.txt` |
| 10 | GitHub — COP 3514 Exam 1 Review | GitHub (.txt) | `documents/github_cop3514_exam1_review.txt` |
| 11 | Rate My Professors — export of 8 USF professors | JSON dataset | `documents/dataset_rate-my-professors.json` |

The 11 source files expand to **17 ingested documents**, because the Rate My
Professors JSON is flattened into one document *per professor* (7 with usable
review text) and the multi-page PDF is read as a single document.

---

## Chunking Strategy

**Chunk size:** 800 characters (~150–200 words)

**Overlap:** 120 characters (~15%)

**Why these choices fit your documents:** The corpus is mostly short,
conversational units — individual Reddit comments, RMP reviews, and bullet-style
study-guide sections. 800 characters is large enough to keep a single comment or
review intact (so the reasoning behind a piece of advice stays attached to it)
while small enough that retrieval stays focused on one opinion instead of an
entire thread. The 120-character overlap keeps a key qualifier (a professor name,
a course code, "open book") from being orphaned at a boundary. Splitting prefers
natural boundaries — paragraph breaks first, then sentence/line breaks — and only
accepts a boundary in the back half of the window so chunks never become tiny
(`guide/chunk.py`).

**Preprocessing before chunking:** `.txt` files have their `Title/URL/Source`
header lifted into metadata and stripped from the body; the RMP JSON is rendered
into readable prose (rating summary + each review with course/grade); the PDF is
extracted page-by-page with `pdfplumber`. Additionally, each chunk is embedded
with a short **context header** (its document title, plus the professor's courses
for RMP entries) prepended to the body — this anchors a chunk to its parent topic
so a query matches a thread even when the body never repeats the title's wording.
The clean body (without the header) is kept for display and citation.

**Final chunk count:** 17 documents → **99 chunks** (chunk chars: min 188, max
796, avg ~619).

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` (384-dimensional sentence embeddings), stored
in a local **ChromaDB** persistent collection using cosine distance. The pipeline
prefers the `sentence-transformers` backend (named in the spec); if that isn't
installed it automatically falls back to ChromaDB's `DefaultEmbeddingFunction`,
which serves the *same* all-MiniLM-L6-v2 weights via ONNX with no PyTorch
dependency. This project was run with the ONNX path — identical model, ~1.5 GB
lighter install.

**Production tradeoff reflection:** all-MiniLM-L6-v2 is the right fit here — fast,
free, local, and strong on short-text semantic similarity. If I were deploying
for real users and cost weren't a constraint, I'd weigh a larger model (e.g.
OpenAI `text-embedding-3-large` or a BGE/E5 model):

- **Accuracy on domain-specific text:** a larger model captures more nuance,
  which matters when a query like "is the curve generous?" must match comments
  that never use the word "curve."
- **Context length:** MiniLM truncates around 256 tokens, so my ~800-char chunks
  sit near its ceiling; a longer-context model would let me embed bigger chunks
  without silently dropping the tail of a long review.
- **Latency & cost:** an API embedder adds per-token cost, network latency, and a
  hard dependency on a provider — a real tradeoff against the current zero-cost,
  offline-capable setup.
- **Multilingual support:** unnecessary for an English-only USF corpus, but would
  matter if expanded to international-student forums.

---

## Grounded Generation

**System prompt grounding instruction:** the model is given a strict system
prompt (`guide/generate.py`) that constrains it to the retrieved context:

> You are The Unofficial Guide to USF Computer Science. You answer questions for
> USF CS students using ONLY the numbered sources provided below.
> 1. Use ONLY information found in the sources. Do not use outside knowledge.
> 2. If the sources do not contain enough information to answer, say so plainly:
>    "The sources I have don't cover that." Do not guess or invent.
> 3. These are student opinions from Reddit, Rate My Professors, and study guides.
>    Attribute claims accordingly (e.g. "students say...") and note when sources disagree.
> 4. Cite the sources you used by their number, like [Source 2].
> 5. Be concise and concrete.

**Structural grounding (beyond the prompt):**

- **Numbered context blocks:** each retrieved chunk is injected as a
  `[Source N] <citation>\n<text>` block, and the model is told to cite the numbers
  it used — so every claim is traceable back to a specific document.
- **Relevance filtering:** retrieval drops any chunk with cosine distance above
  `MAX_DISTANCE` (1.15) before it ever reaches the model, so off-topic context
  can't leak in. If *no* chunk survives the filter, the system short-circuits and
  returns "The sources I have don't cover that" without calling the LLM at all.
- **Low temperature:** generation runs at `temperature=0.1` to keep answers
  faithful to the context rather than creative.

**How source attribution is surfaced in the response:** the model emits inline
`[Source N]` citations in its prose, and both the CLI (`evaluate.py`) and the
Gradio UI (`app.py`) print the full source list underneath each answer —
title, link, and cosine distance. The UI additionally shows a live "Retrieved
sources" panel with a snippet of each chunk, so the grounding is visible.

---

## Evaluation Report

Run with the Groq `llama-3.3-70b-versatile` backend over the 99-chunk index.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | How hard is COP 3514 and what should I focus on? | Moderate; lots of material/projects; focus on pointers, memory, linked lists; read the book; start projects early | "Large amount of information…focus on Pointers and Linked Lists…read the book and do practice problems…manage time for the weekly projects"; cites CS50 too | Relevant (5/5 on-topic) | Accurate |
| 2 | Which professor is best for CDA 3201? | Rangachar Kasturi (only prof reviewed for CDA 3201 in the data) | "Rangachar Kasturi is the best professor for CDA 3201…praise his lectures…one student waited a whole semester for him" | Relevant (top 3 = Kasturi) | Accurate |
| 3 | Is Dr. Small good for COP 2510, and are her exams open book? | Yes; programming exams open book (ZyBook/IDE), midterms not (hand-written) | "Dr. Small is a good professor…programming exams are open book…midterms are not open book, requiring code by hand" | Partially relevant (top 3 correct; ranks 4–5 pulled an unrelated Mass-Comm professor) | Accurate |
| 4 | What are the easiest CS electives at USF? | Software Systems Development, Software Engineering, Software Testing | "Software Testing (0 exams)…Software Systems Development and Software Engineering (needs a permit)" | Relevant after fix (Easiest-Electives thread retrieved at rank 3; see Failure Case Analysis) | Accurate |
| 5 | What languages does USF teach, in what order? | Python (formerly Java) → C → C++; electives add C#/Python | "Java, C, C++, Python, C#…intro was Java, 'might be Python now', then C, then C++, electives add C#/Python" | Relevant | Accurate |

**Overall retrieval quality:** Relevant (4/5 fully on-target, 1 partially)
**Overall response accuracy:** Accurate (5/5)

---

## Failure Case Analysis

**Question that failed:** "What are the easiest CS electives at USF?" (Q4) — this
failed on the *first* evaluation run, before I changed the chunking stage.

**What the system returned:** *"The sources I have don't cover that. They discuss
the overall experience of the CS program… but do not specifically mention the
easiest CS electives."* The correct source (`reddit_easiest_electives.txt`)
existed in the corpus but was **not retrieved** — the top 5 were dominated by the
general "program experience" threads.

**Root cause (tied to a specific pipeline stage):** the **embedding/retrieval
stage**. The body of the easiest-electives thread names specific courses
("Software Systems Development is stupid easy," "Software Testing has 0 exams")
but rarely uses the words "easiest" or "elective." Meanwhile the program-opinion
threads literally discuss coursework being "light" and "easy," so their chunks
embedded closer to the query "easiest CS electives" than the actually-correct
document did. The right answer lost on pure semantic similarity of the body text.

**What you would change to fix it (and did):** I added a **context header** to
each chunk before embedding — the document title (e.g. "Easiest CS Electives")
plus any associated courses — while keeping the clean body for display
(`guide/chunk.py`, `guide/embed_store.py`, `guide/retrieve.py`). After
re-indexing, the easiest-electives thread is retrieved at rank 3 and the system
answers correctly (Software Testing, Software Systems Development, Software
Engineering). A remaining, smaller issue: Q3 still pulls an unrelated
Mass-Communications professor into ranks 4–5 because "open book exams" is generic
phrasing shared across many reviews — a per-source or course-code metadata filter
would tighten that further.

---

## Spec Reflection

**One way the spec helped you during implementation:** Writing the Chunking and
Retrieval sections of `planning.md` first forced me to pin down concrete numbers
(800/120, top-k 5, cosine, distance cutoff 1.15) before writing code, so the
implementation became a matter of wiring those constants together in
`guide/config.py` rather than guessing. The Evaluation Plan was especially
useful: having 5 specific, checkable questions with expected answers written in
advance turned "does it work?" into a concrete pass/fail test, which is exactly
how I caught the Q4 retrieval failure.

**One way your implementation diverged from the spec, and why:** The original
chunking spec embedded each chunk's raw body only. During evaluation I diverged by
prepending a **context header** (title + courses) to the text that gets embedded,
because the Q4 failure showed that body-only embeddings lost to topically-similar
but less-relevant documents. I also split the stored data into an embedded
`embed_text` and a clean `display_text` so retrieval quality improved without
polluting the citations shown to users — a refinement the plan didn't anticipate
until I had real retrieval results to look at.

---

## AI Usage

**Instance 1 — Contextual-chunk fix for the Q4 retrieval failure**

- *What I gave the AI:* the failing evaluation output for "easiest CS electives,"
  the `chunk.py` / `embed_store.py` / `retrieve.py` source, and the observation
  that the correct document wasn't in the top-k.
- *What it produced:* a change that prepends a metadata-derived context header
  (document title + courses) to each chunk's embedded text, stores the clean body
  separately as `display_text`, and reads that clean text back at retrieval time.
- *What I changed or overrode:* I directed it to keep the *displayed* text clean
  (header used for embedding only) rather than the simpler approach of embedding
  and showing the header-prefixed text, so citations and source snippets stay
  readable. I re-ran `build_index.py` + `evaluate.py` to confirm the fix moved the
  electives thread into the top results.

**Instance 2 — Configuration and interface wiring**

- *What I gave the AI:* the `.env` file, the symptom that the Groq key wasn't
  being picked up, and a request for a demo-ready Gradio interface.
- *What it produced:* `load_dotenv(override=True)` centralized in
  `guide/config.py` (so a stale shell env var no longer shadowed the real key),
  and a rebuilt `app.py` using `gr.Blocks` with a streaming chat plus a live
  "Retrieved sources" panel showing each chunk's distance.
- *What I changed or overrode:* I had it add a compatibility shim for a
  gradio 4.44 + Python 3.9 schema bug that was crashing the server on startup
  (`argument of type 'bool' is not iterable`), and verified the app actually
  serves (HTTP 200 on `http://127.0.0.1:7860`) rather than trusting that it
  "should" launch.
