"""Stage 1 runner: PDF -> pages.json + chunks.json, with statistics and validation.

Usage (from the project root):
    python -m src.process_pdf
"""
import hashlib
import json
import sys
import textwrap
from datetime import datetime, timezone

import numpy as np

from . import config
from .pdf_processing import chunk_document, extract_pages, page_label, validate_chunks


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_chunks(path=config.CHUNKS_JSON):
    """Load the chunk list saved by this stage (used by later stages / the notebook)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)["chunks"]


def print_stats(pages, chunks):
    lengths = np.array([c["char_count"] for c in chunks])
    with_text = [p for p in pages if p["text"]]
    print("\n=== PDF statistics ===")
    print(f"Source PDF                     : {config.SOURCE_PDF.name}")
    print(f"Total PDF pages                : {len(pages)}")
    print(f"Pages with extractable text    : {len(with_text)}")
    print(f"Total extracted characters     : {sum(p['char_count'] for p in pages):,}")
    print(f"Pages containing tables        : {[p['page'] for p in pages if p['n_tables']]}")

    flagged = [p for p in pages if set(p["flags"]) - {"has_tables"}]
    print("\n=== Empty / suspicious pages ===")
    if not flagged:
        print("none")
    for p in flagged:
        preview = p["text"][:80].replace("\n", " | ")
        print(f"page {p['page']:>2}: flags={p['flags']} chars={p['char_count']} "
              f"coverage={p['word_coverage']:.1%}  text='{preview}'")
    print(f"Min word coverage after cleaning: "
          f"{min(p['word_coverage'] for p in pages):.1%} "
          f"(page {min(pages, key=lambda p: p['word_coverage'])['page']})")

    print("\n=== Chunk statistics ===")
    print(f"chunk_size / overlap           : {config.CHUNK_SIZE} / {config.CHUNK_OVERLAP}")
    print(f"Total chunks                   : {len(chunks)}")
    print(f"Chunk length (chars) avg/median: {lengths.mean():.1f} / {np.median(lengths):.0f}")
    print(f"Chunk length (chars) min/max   : {lengths.min()} / {lengths.max()}")
    print(f"Chunks spanning >1 page        : {sum(len(c['pages']) > 1 for c in chunks)}")
    print(f"Chunks shorter than 300 chars  : {int((lengths < 300).sum())}")


def print_samples(chunks, ids):
    print("\n=== Sample chunks ===")
    for i in ids:
        c = chunks[i]
        print(f"\n--- chunk_id={c['chunk_id']} | {page_label(c)} | {c['char_count']} chars "
              f"| hash={c['chunk_hash']} ---")
        print(textwrap.indent(c["text"][:600] + (" [...]" if c["char_count"] > 600 else ""), "  "))


def main():
    if not config.SOURCE_PDF.exists():
        sys.exit(f"Source PDF not found at {config.SOURCE_PDF}. Place one complex PDF there.")

    pages = extract_pages(config.SOURCE_PDF, config.EXTRACT_TABLES, config.MIN_PAGE_CHARS)
    chunks, full_text = chunk_document(
        pages, config.SOURCE_PDF.name, config.CHUNK_SIZE, config.CHUNK_OVERLAP, config.SEPARATORS
    )

    problems = validate_chunks(chunks, full_text, len(pages), config.CHUNK_SIZE)

    meta = {
        "source_pdf": config.SOURCE_PDF.name,
        "source_sha256": file_sha256(config.SOURCE_PDF),
        "n_pages": len(pages),
        "n_chunks": len(chunks),
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "separators": config.SEPARATORS,
        "splitter": "langchain_text_splitters.RecursiveCharacterTextSplitter",
        "extract_tables": config.EXTRACT_TABLES,
        "page_numbering": "1-based PDF page index (printed page number = PDF page - 1)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PAGES_JSON, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "pages": pages}, f, ensure_ascii=False, indent=2)
    with open(config.CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "chunks": chunks}, f, ensure_ascii=False, indent=2)

    print_stats(pages, chunks)
    print_samples(chunks, sorted({0, len(chunks) // 3, len(chunks) // 2, len(chunks) - 1}))

    print("\n=== Validation ===")
    # re-load from disk to make sure what later stages read is what we checked
    reloaded = load_chunks()
    assert reloaded == chunks, "chunks.json does not round-trip"
    if problems:
        print("FAILED:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"OK: {len(chunks)} chunks; ids 0..{len(chunks) - 1} contiguous; text, page "
          f"metadata, offsets and full-document coverage verified; JSON round-trips.")
    print(f"Saved {config.PAGES_JSON.relative_to(config.PROJECT_ROOT)} and "
          f"{config.CHUNKS_JSON.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
