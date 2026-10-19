"""Stage 1: page-wise PDF extraction (PyMuPDF) and recursive character chunking.

Pipeline
    extract_pages()   -> one record per PDF page (cleaned text + diagnostics)
    chunk_document()  -> RecursiveCharacterTextSplitter chunks with page metadata
    validate_chunks() -> structural checks on the chunk list
"""
import hashlib
import re
import unicodedata
from collections import Counter

import pymupdf  # PyMuPDF (formerly imported as `fitz`)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Word 2007 exports bullets as Wingdings private-use glyphs; map them to "- ".
BULLET_GLYPHS = "•●▪"
QUOTE_MAP = str.maketrans({"“": '"', "”": '"', "„": '"', "‟": '"',
                           "‘": "'", "’": "'", "­": None})
# A line starting with one of these begins a new item and must not be glued
# onto the previous line: "- ", "a)", "(iv)", "3.", "R.14.2", "O.1".
NEW_ITEM_RE = re.compile(r"^(- |\(?[a-z]\)|\(?[ivx]+\)|\d+\.\s|[RO]\.\d)")


# ----------------------------------------------------------------------------
# Text normalisation
# ----------------------------------------------------------------------------
def clean_text(text):
    """Normalise one text block while keeping every word.

    - NFKC folds compatibility characters (ligatures, the Greek question mark
      that this PDF uses in place of ';').
    - Bullet glyphs become "- ", curly quotes become ASCII, soft hyphens vanish.
    - Words hyphenated across a line break are re-joined.
    - Wrapped lines are joined with a space, except where a line starts a new
      list item or clause, so enumerations stay on separate lines.
    """
    text = unicodedata.normalize("NFKC", text).translate(QUOTE_MAP)
    text = re.sub(f"[{BULLET_GLYPHS}]\\s*", "- ", text)
    # letter-spaced headings: "C O N T E N T S" -> "CONTENTS"
    text = re.sub(r"\b(?:[A-Z] ){3,}[A-Z]\b", lambda m: m.group(0).replace(" ", ""), text)
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)

    out = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if out and not NEW_ITEM_RE.match(line):
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)


def _table_to_markdown(table):
    """Render a PyMuPDF table as Markdown built from the raw cell text.

    Built from table.extract() rather than table.to_markdown() because the
    latter wraps bold cells in '**', which would be indistinguishable from
    this document's '**' footnote marker (= B.Tech Honours). Empty cells are
    kept so every value stays in its column.
    """
    rows = table.extract()
    if table.header.external:  # header row detected above the table body
        rows = [table.header.names] + rows
    cell = lambda c: re.sub(r"\s+", " ", unicodedata.normalize("NFKC", c or "")).strip()
    lines = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
        if i == 0:
            lines.append("|" + " --- |" * len(row))
    return "\n".join(lines).translate(QUOTE_MAP)


def _inside(rect, table_rects, min_frac=0.5):
    area = rect.get_area() or 1.0
    return any((rect & t).get_area() / area >= min_frac for t in table_rects)


def _word_coverage(raw, cleaned):
    """Fraction of alphanumeric tokens in the raw page text found in the cleaned text."""
    tok = lambda s: Counter(re.findall(r"[A-Za-z0-9]+", unicodedata.normalize("NFKC", s)))
    raw_c, clean_c = tok(raw), tok(cleaned)
    total = sum(raw_c.values())
    if total == 0:
        return 1.0
    return sum(min(n, clean_c[w]) for w, n in raw_c.items()) / total


# ----------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------
def extract_page(page, extract_tables=True):
    """Return (cleaned_text, n_tables, printed_page) for one PyMuPDF page.

    Text blocks and tables are interleaved in reading order (top-to-bottom,
    then left-to-right). Lines lying inside a detected table are replaced by
    that table's Markdown rendering, so cells stay in their rows. Filtering is
    per line, not per block, because PyMuPDF sometimes puts a table and the
    footnotes below it into one block.
    """
    tables = page.find_tables().tables if extract_tables else []
    table_rects = [pymupdf.Rect(t.bbox) for t in tables]

    items = []  # (y0, x0, kind, text)
    for block in page.get_text("dict", sort=True)["blocks"]:
        if block["type"] != 0:  # image block
            continue
        lines = [
            "".join(span["text"] for span in line["spans"])
            for line in block["lines"]
            if not (table_rects and _inside(pymupdf.Rect(line["bbox"]), table_rects))
        ]
        text = clean_text("\n".join(lines))
        if text:
            items.append((block["bbox"][1], block["bbox"][0], "text", text))
    for t, r in zip(tables, table_rects):
        items.append((r.y0, r.x0, "table", _table_to_markdown(t)))
    items.sort(key=lambda it: (round(it[0]), it[1]))

    # The printed page number (PDF page - 1 in this document, since the cover
    # is unnumbered) sits alone as the first or last block. Move it from the
    # text to metadata so it does not pollute chunks.
    printed_page = None
    for idx in (0, -1):
        if items and re.fullmatch(r"\d{1,3}", items[idx][3]) and int(items[idx][3]) == page.number:
            printed_page = int(items.pop(idx)[3])
            break

    # Choose the separator between consecutive items. A blank line ("\n\n") is
    # the splitter's preferred cut point, so only use it at real paragraph ends:
    #  - a table caption ("(for 2011-2012 batches)") stays glued to its table and
    #    a "*"-footnote to the table above it -- otherwise the near-identical
    #    batch tables would end up in chunks that do not say which batch;
    #  - PyMuPDF sometimes splits one sentence over two blocks ("Board of" /
    #    "Governors ..."); a block that does not end a sentence continues.
    text = ""
    for i, (_y, _x, kind, item) in enumerate(items):
        if i:
            prev_kind = items[i - 1][2]
            if prev_kind == "text" and kind == "table":
                text += "\n"
            elif prev_kind == "table" and item.startswith("*"):
                text += "\n"
            elif prev_kind == kind == "text":
                text += join_separator(text, item)
            else:
                text += "\n\n"
        text += item
    return text, len(tables), printed_page


def join_separator(prev, nxt):
    """Separator between two text runs: ' ' mid-sentence, '\\n' after an
    unterminated line (heading, list entry), '\\n\\n' after a finished sentence
    or after a table / table footnote."""
    if re.match(r"[|*]", prev.rsplit("\n", 1)[-1]):
        return "\n\n"
    if re.search(r"[a-z,]$", prev) and re.match(r"[a-z]", nxt):
        return " "
    if not re.search(r"[.:;?!]$", prev):
        return "\n"
    return "\n\n"


def extract_pages(pdf_path, extract_tables=True, min_chars=200):
    """Extract every page of the PDF.

    Returns a list of page records:
        {"page": 1-based PDF page, "printed_page", "text", "char_count",
         "n_tables", "n_images", "word_coverage", "flags": [...]}

    No page is dropped. Empty/suspicious pages are kept (so page numbers stay
    faithful) and flagged:
        empty           -- no extractable text at all
        low_text        -- fewer than `min_chars` characters
        image_low_text  -- has images but little text (may need OCR)
        has_tables      -- contains tables rendered as Markdown
        garbled         -- many unmapped/private-use glyphs remain
        content_loss    -- < 98% of the raw words survive cleaning
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:  # missing / corrupted file
        raise RuntimeError(f"Could not open PDF '{pdf_path}': {exc}") from exc

    pages = []
    for page in doc:
        pno = page.number + 1
        try:
            raw = page.get_text("text")
            text, n_tables, printed = extract_page(page, extract_tables)
        except Exception as exc:
            print(f"[warn] extraction failed on page {pno}: {exc}")
            raw, text, n_tables, printed = "", "", 0, None
        n_images = len(page.get_images())

        flags = []
        if not text:
            flags.append("empty")
        elif len(text) < min_chars:
            flags.append("low_text")
        if n_images and len(text) < min_chars:
            flags.append("image_low_text")
        if n_tables:
            flags.append("has_tables")
        odd = sum(1 for ch in text if ch == "�" or "" <= ch <= "")
        if text and odd / len(text) > 0.01:
            flags.append("garbled")
        # compare against the raw text after the same glyph/spacing fixes; the
        # printed page number was moved to metadata, so count it as kept
        coverage = _word_coverage(clean_text(raw), f"{text} {printed or ''}")
        if coverage < 0.98:
            flags.append("content_loss")

        pages.append({
            "page": pno,
            "printed_page": printed,
            "text": text,
            "char_count": len(text),
            "n_tables": n_tables,
            "n_images": n_images,
            "word_coverage": round(coverage, 4),
            "flags": flags,
        })
    doc.close()

    if not any(p["text"] for p in pages):
        raise RuntimeError(
            "No text could be extracted. The PDF is probably scanned; OCR would be "
            "required before this pipeline can be used."
        )
    return pages


# ----------------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------------
def chunk_document(pages, source, chunk_size=1000, chunk_overlap=200, separators=None):
    """Split the whole document with LangChain's RecursiveCharacterTextSplitter.

    Pages are concatenated so that a clause continuing onto the next page can
    stay in one chunk; the page boundary gets the same separator rule as
    blocks within a page (join_separator). Each chunk's character offsets in
    the concatenated text are mapped back to the page(s) it spans.

    chunk_id is the chunk's 0-based position in document order. It is
    deterministic for a fixed PDF + configuration and equals the row index
    in the FAISS index built later. chunk_hash (SHA-1 of the text) allows
    detecting whether a re-run changed chunk contents.
    """
    full_text, spans, pos = "", [], 0  # spans: (page, start, end) in full_text
    for p in pages:
        if full_text and p["text"]:
            sep = join_separator(full_text, p["text"])
            full_text += sep
            pos += len(sep)
        full_text += p["text"]
        spans.append((p["page"], pos, pos + len(p["text"])))
        pos += len(p["text"])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or ["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",   # a sentence keeps its full stop
        length_function=len,
        add_start_index=True,
        strip_whitespace=True,
    )

    chunks = []
    for doc in splitter.create_documents([full_text]):
        text, start = doc.page_content, doc.metadata["start_index"]
        end = start + len(text)
        page_list = [pg for pg, s, e in spans if s < end and e > start and e > s]
        chunks.append({
            "chunk_id": len(chunks),
            "source": source,
            "page_start": page_list[0],
            "page_end": page_list[-1],
            "pages": page_list,
            "text": text,
            "char_count": len(text),
            "start_index": start,
            "end_index": end,
            "chunk_hash": hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
        })
    return chunks, full_text


def validate_chunks(chunks, full_text, n_pages, chunk_size):
    """Assert the invariants later stages rely on. Returns a list of problems (empty = valid)."""
    problems = []
    required = {"chunk_id", "source", "page_start", "page_end", "pages", "text", "char_count"}
    for i, c in enumerate(chunks):
        missing = required - c.keys()
        if missing:
            problems.append(f"chunk {i}: missing keys {sorted(missing)}")
            continue
        if c["chunk_id"] != i:
            problems.append(f"chunk {i}: chunk_id {c['chunk_id']} != position")
        if not c["text"].strip():
            problems.append(f"chunk {i}: empty text")
        if c["char_count"] != len(c["text"]):
            problems.append(f"chunk {i}: char_count mismatch")
        if c["char_count"] > chunk_size:
            problems.append(f"chunk {i}: {c['char_count']} chars > chunk_size {chunk_size}")
        if not (1 <= c["page_start"] <= c["page_end"] <= n_pages) or not c["pages"]:
            problems.append(f"chunk {i}: invalid pages {c['pages']}")
        if full_text[c["start_index"]:c["end_index"]] != c["text"]:
            problems.append(f"chunk {i}: offsets do not reproduce the text")
    hashes = Counter(c["chunk_hash"] for c in chunks)
    dupes = [h for h, n in hashes.items() if n > 1]
    if dupes:
        problems.append(f"{len(dupes)} duplicated chunk text(s)")
    # every non-whitespace character of the document must be in some chunk
    covered = bytearray(len(full_text))
    for c in chunks:
        covered[c["start_index"]:c["end_index"]] = b"\x01" * c["char_count"]
    lost = sum(1 for ch, cov in zip(full_text, covered) if not cov and not ch.isspace())
    if lost:
        problems.append(f"{lost} non-whitespace characters not covered by any chunk")
    return problems


def page_label(chunk):
    """Human-readable page reference such as 'p. 4' or 'pp. 4-5' (PDF page numbers)."""
    if chunk["page_start"] == chunk["page_end"]:
        return f"p. {chunk['page_start']}"
    return f"pp. {chunk['page_start']}-{chunk['page_end']}"


# ----------------------------------------------------------------------------
# Legacy: overlapping word-window chunking from the first prototype. Not used
# by the pipeline any more (replaced by chunk_document); kept for reference.
# ----------------------------------------------------------------------------
def chunk_pages(pages, chunk_size=500, overlap=100):
    """Split the whole document into overlapping word windows."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words, word_pages = [], []
    for p in pages:
        for w in p["text"].split():
            words.append(w)
            word_pages.append(p["page"])

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        span_pages = sorted(set(word_pages[start:end]))
        chunks.append({
            "chunk_id": len(chunks),
            "text": " ".join(words[start:end]),
            "page_start": span_pages[0],
            "page_end": span_pages[-1],
            "pages": span_pages,
            "word_start": start,
            "word_end": end,
            "n_words": end - start,
        })
        if end == len(words):
            break
    return chunks
