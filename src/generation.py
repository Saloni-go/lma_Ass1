"""Answer generation with Gemini (google-genai SDK). Identical for both pipelines."""
import os
import re
import time

from .config import GEMINI_MODEL  # importing config also loads .env
from .pdf_processing import page_label

# One fixed configuration used for Answer 1 and Answer 2.
GENERATION_SETTINGS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_output_tokens": 8192,   # generous: 2.5-series "thinking" tokens count here too
    "seed": 42,
}

# The single prompt template used unchanged by Pipeline 1 and Pipeline 2;
# only the three context chunks differ between Answer 1 and Answer 2.
PROMPT_TEMPLATE = """You are answering a question about the IIT Madras B.Tech / B.Tech (Honours) Ordinances and Regulations, using excerpts retrieved from that document.

Rules:
- Use only the provided context. Do not use outside knowledge and do not invent facts, numbers or rules.
- Cite the supporting chunk and page for every claim, e.g. [Chunk 12, p. 5].
- Pay attention to conditions, exceptions and batch-specific variations stated in the context.
- If the context does not contain enough evidence to answer fully, say so explicitly and state what is missing.

Context:
{context}

Question:
{query}

Answer:"""


def format_context(context_chunks):
    return "\n\n".join(
        f"[Chunk {c['chunk_id']} | {page_label(c)}]\n{c['text']}" for c in context_chunks
    )


def build_prompt(query, context_chunks):
    assert len(context_chunks) == 3, "each pipeline must pass exactly 3 context chunks"
    return PROMPT_TEMPLATE.format(context=format_context(context_chunks), query=query)


def _use_vertex():
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")


def describe_backend():
    """One-line description of which Gemini backend get_client() will use."""
    if _use_vertex():
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            return "Vertex AI selected but GOOGLE_CLOUD_PROJECT missing"
        return f"Vertex AI (project={project}, location={os.environ.get('GOOGLE_CLOUD_LOCATION', 'global')})"
    if os.environ.get("GEMINI_API_KEY"):
        return "Gemini Developer API (API key)"
    return "no credentials configured (see README: Gemini credentials)"


def get_client():
    """Create a Gemini client, or return None if no credentials are configured.

    Two backends, both through the same google-genai SDK (so prompts and
    settings are identical):
      - Vertex AI (Google Cloud project, billed to its credits):
        GOOGLE_GENAI_USE_VERTEXAI=true, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION;
        authenticates with Application Default Credentials
        (`gcloud auth application-default login`).
      - Gemini Developer API (AI Studio key): GEMINI_API_KEY.
    """
    from google import genai
    if _use_vertex():
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            return None
        return genai.Client(vertexai=True, project=project,
                            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    key = os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=key) if key else None


def generate_answer(query, context_chunks, client, model=GEMINI_MODEL, retries=4):
    """Generate one answer. Returns (answer_text, latency_seconds).

    The latency covers the API call only (the successful attempt). Transient
    errors (rate limits / overload) are retried with exponential back-off.
    """
    from google.genai import types

    if client is None:
        raise RuntimeError(f"No Gemini credentials: {describe_backend()}; LLM generation cannot run.")
    prompt = build_prompt(query, context_chunks)
    config = types.GenerateContentConfig(**GENERATION_SETTINGS)

    for attempt in range(retries + 1):
        try:
            t0 = time.perf_counter()
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            latency = time.perf_counter() - t0
            text = (resp.text or "").strip()
            if not text:
                raise RuntimeError(f"empty response (finish reason: {resp.candidates[0].finish_reason})")
            return text, latency
        except Exception as exc:
            # 4xx errors other than 429 (bad key, unknown model, invalid request)
            # will not succeed on retry
            code = getattr(exc, "code", None)
            if attempt == retries or (isinstance(code, int) and 400 <= code < 500 and code != 429):
                raise
            wait = 2 ** attempt * 5
            # on 429 the API says how long to wait ("Please retry in 33.4s"); honour it
            hinted = re.search(r"retry in ([\d.]+)s", str(exc))
            if hinted:
                wait = max(wait, float(hinted.group(1)) + 2)
            print(f"[gemini] attempt {attempt + 1} failed ({str(exc)[:120]}...); retrying in {wait:.0f}s")
            time.sleep(wait)
