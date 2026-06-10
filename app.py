"""Milestone 5 — Query interface (Gradio).

    python app.py

A chat UI for The Unofficial Guide to USF Computer Science. Ask a question and
get a grounded answer on the left, with the exact source chunks the answer was
built from (and their cosine distances) shown on the right — so the retrieval
step is visible during a demo, not hidden.
"""
from __future__ import annotations

import gradio as gr

# --- Compatibility shim -----------------------------------------------------
# gradio 4.44 + gradio_client 1.3 (the newest pair that runs on Python 3.9) has
# a bug: schema introspection chokes on boolean JSON schemas ("argument of type
# 'bool' is not iterable"), which crashes the server's startup self-check and
# makes the page refuse to connect. Teach the two offending helpers to treat a
# bool schema as "Any" so launch succeeds.
import gradio_client.utils as _gcu

_orig_json_schema = _gcu._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema(schema, defs)


def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _gcu._orig_get_type(schema)


_gcu._json_schema_to_python_type = _safe_json_schema_to_python_type
if not hasattr(_gcu, "_orig_get_type"):
    _gcu._orig_get_type = _gcu.get_type
    _gcu.get_type = _safe_get_type
# ---------------------------------------------------------------------------

from guide.generate import generate
from guide.retrieve import retrieve

EXAMPLES = [
    "How hard is COP 3514 and what should I focus on?",
    "Is Dr. Small good for COP 2510, and are her exams open book?",
    "Which professor is best for CDA 3201?",
    "What are the easiest CS electives at USF?",
    "What languages does USF teach and in what order?",
    "Is the USF CS program worth it?",
]

INTRO = """
# 🐂 The Unofficial Guide to USF Computer Science

Ask about **course difficulty, professors, study strategies, electives, or careers.**
Every answer is grounded in real student sources — Reddit, Rate My Professors,
GitHub study guides, and the official USF program page — and cites the sources it used.
"""

SOURCES_PLACEHOLDER = (
    "*Retrieved sources will appear here after you ask a question.*\n\n"
    "Each card shows the source, its cosine **distance** (lower = closer match), "
    "and a snippet of the exact chunk fed to the model."
)


def _format_sources(chunks) -> str:
    """Render the retrieved chunks as a readable, demo-friendly panel."""
    if not chunks:
        return (
            "**No sufficiently relevant sources were found.**\n\n"
            "The guide will say it doesn't have the answer rather than guess."
        )
    lines = [f"**{len(chunks)} chunk(s) retrieved** (top-k, distance-filtered):\n"]
    for i, c in enumerate(chunks, 1):
        m = c.metadata
        title = m.get("title", m.get("file", "source"))
        url = m.get("url", "")
        heading = f"[{title}]({url})" if url else title
        snippet = " ".join(c.text.split())[:240]
        lines.append(
            f"**[{i}]** {heading}  \n"
            f"`distance {c.distance:.3f}` · *{m.get('source', 'unknown')}*  \n"
            f"> {snippet}…\n"
        )
    return "\n".join(lines)


def respond(message: str, history: list):
    """Run retrieval + grounded generation and stream the result into the chat."""
    message = (message or "").strip()
    if not message:
        yield history, "", SOURCES_PLACEHOLDER
        return

    history = history + [{"role": "user", "content": message}]

    # Show an immediate "working" state so the demo never looks frozen.
    history = history + [{"role": "assistant", "content": "_Retrieving sources…_"}]
    yield history, "", "*Searching the vector store…*"

    chunks = retrieve(message)
    sources_panel = _format_sources(chunks)

    if not chunks:
        history[-1]["content"] = "The sources I have don't cover that."
        yield history, "", sources_panel
        return

    history[-1]["content"] = "_Generating a grounded answer…_"
    yield history, "", sources_panel

    try:
        out = generate(message, chunks)
        answer = out["answer"]
        backend = out["backend"]
        cited = "  ·  ".join(f"[{s['n']}] {s['citation']}" for s in out["sources"])
        full = f"{answer}\n\n<sub>Answered via **{backend}** · sources: {cited}</sub>"
    except Exception as exc:  # keep the demo alive if the LLM call fails
        full = (
            "⚠️ **Couldn't reach the language model.**\n\n"
            "Retrieval worked (see the sources panel), but generation failed:\n\n"
            f"`{exc}`\n\n"
            "Check that a valid `GROQ_API_KEY` is set in your `.env`, or run a "
            "local Ollama model as the fallback."
        )

    history[-1]["content"] = full
    yield history, "", sources_panel


with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="yellow"),
               title="The Unofficial Guide to USF CS",
               analytics_enabled=False) as demo:
    gr.Markdown(INTRO)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                type="messages",
                height=460,
                label="Conversation",
                avatar_images=(None, None),
                show_copy_button=True,
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="e.g. How hard is COP 3514?",
                    label="Your question",
                    scale=8,
                    autofocus=True,
                )
                send = gr.Button("Ask", variant="primary", scale=1, min_width=80)
            gr.Examples(examples=EXAMPLES, inputs=msg, label="Try one of these")
            clear = gr.Button("Clear conversation", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("### 📚 Retrieved sources")
            sources = gr.Markdown(SOURCES_PLACEHOLDER)

    send.click(respond, [msg, chatbot], [chatbot, msg, sources])
    msg.submit(respond, [msg, chatbot], [chatbot, msg, sources])
    clear.click(lambda: ([], SOURCES_PLACEHOLDER), None, [chatbot, sources])


if __name__ == "__main__":
    # show_api=False keeps the demo clean (no "Use via API" link) and avoids a
    # noisy schema-introspection quirk in gradio 4.x on Python 3.9.
    demo.launch(show_api=False)
