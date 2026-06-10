"""Stage 1 — Document ingestion.

Reads every file in documents/ and turns it into a list of plain-text
documents with metadata. Each source type is normalized into readable prose so
the downstream chunker can treat everything uniformly:

  - .txt   Reddit threads, GitHub guides, the USF page (already prose).
  - .json  Rate My Professors export -> one readable document per professor.
  - .pdf   USF 4-year plan of study -> extracted text.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .config import DOCUMENTS_DIR


def _read_txt(path: Path) -> list[dict]:
    """A .txt source is already prose. Lift the Title/URL/Source header into
    metadata and keep the body as the document text."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    title, url, source = path.stem, "", "unknown"
    body = raw

    if raw.startswith("Title:"):
        header, _, rest = raw.partition("---\n")
        for line in header.splitlines():
            if line.startswith("Title:"):
                title = line[len("Title:"):].strip()
            elif line.startswith("URL:"):
                url = line[len("URL:"):].strip()
            elif line.startswith("Source:"):
                source = line[len("Source:"):].strip()
        body = rest.strip() or raw

    return [{
        "text": body,
        "metadata": {"title": title, "url": url, "source": source, "file": path.name},
    }]


def _read_rmp_json(path: Path) -> list[dict]:
    """Flatten the Rate My Professors export. One document per professor keeps
    that professor's summary + all of their reviews together, which is exactly
    the unit a student asks about ("how is professor X for course Y")."""
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    docs: list[dict] = []

    for prof in data:
        name = prof.get("professorName", "Unknown")
        dept = prof.get("department", "Unknown")
        quality = prof.get("overallQualityRating")
        difficulty = prof.get("levelOfDifficulty")
        again = prof.get("wouldTakeAgainPercentage")
        n = prof.get("numberOfRatings")

        lines = [
            f"Professor {name} — {dept} department, University of South Florida.",
            f"Overall quality rating: {quality}/5. Level of difficulty: {difficulty}/5.",
            f"Would take again: {again}. Based on {n}.",
        ]

        for r in prof.get("reviews") or []:
            course = r.get("course", "")
            comment = (r.get("comment") or "").strip()
            if not comment:
                continue
            grade = r.get("grade") or "N/A"
            lines.append(
                f"\nReview for {course} (quality {r.get('qualityRating')}/5, "
                f"difficulty {r.get('difficultyRating')}/5, grade {grade}): {comment}"
            )

        # Skip professors with no usable review text at all.
        if len(lines) <= 3:
            continue

        courses = sorted({r.get("course", "") for r in (prof.get("reviews") or []) if r.get("course")})
        docs.append({
            "text": "\n".join(lines),
            "metadata": {
                "title": f"Rate My Professors — {name} ({dept})",
                "url": prof.get("professorUrl", ""),
                "source": "ratemyprofessors",
                "file": path.name,
                "professor": name,
                "courses": ", ".join(courses),
            },
        })
    return docs


def _read_pdf(path: Path) -> list[dict]:
    """Extract text from the plan-of-study PDF. Requires pdfplumber."""
    try:
        import pdfplumber  # noqa: WPS433 (optional dependency)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pdfplumber is required to ingest PDF documents. "
            "Install it with: pip install pdfplumber"
        ) from exc

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)

    return [{
        "text": "\n".join(pages),
        "metadata": {
            "title": "USF Computer Science B.S. — 4-Year Plan of Study",
            "url": "",
            "source": "usf.edu",
            "file": path.name,
        },
    }]


def load_documents(documents_dir: Path = DOCUMENTS_DIR) -> list[dict]:
    """Return every source document as {text, metadata}."""
    documents: list[dict] = []
    for path in sorted(documents_dir.iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".txt":
            documents.extend(_read_txt(path))
        elif suffix == ".json":
            documents.extend(_read_rmp_json(path))
        elif suffix == ".pdf":
            documents.extend(_read_pdf(path))
        # Silently ignore other file types (e.g. .gitkeep).
    return documents


def summarize(documents: Iterable[dict]) -> None:
    """Print a quick ingestion summary (used by build_index.py)."""
    docs = list(documents)
    by_source: dict[str, int] = {}
    for d in docs:
        by_source[d["metadata"]["source"]] = by_source.get(d["metadata"]["source"], 0) + 1
    print(f"Ingested {len(docs)} documents:")
    for source, count in sorted(by_source.items()):
        print(f"  - {source}: {count}")


if __name__ == "__main__":
    summarize(load_documents())
