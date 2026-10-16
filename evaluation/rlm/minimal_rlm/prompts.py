# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every word the models are shown. A short base prompt, plus one block per option.

The base is the RLM idea and nothing more: the document lives in a REPL, the root
writes code, `llm_query` hands text to a sub-model, `FINAL` ends the run. Each
option (ordering) adds only the lines that describe what it adds, so the
difference between two configs is visible in the prompt diff.
"""

from __future__ import annotations

from typing import Optional

BASE_PROMPT = """\
You answer a question about a long document that you cannot read directly. The document
is stored in a Python REPL. Each turn, reply with ONE ```python code block; its printed
output is shown to you in the next message (at most {output_chars} characters).

In the REPL:
  context                       the whole document, a string of {context_chars:,} characters
{tools}  llm_query(question, text)     a sub-model reads `text` and answers `question`. It sees
                                nothing else. One call reads at most {read_tokens:,} tokens
                                (about {read_chars:,} characters); a longer text is refused.
  FINAL(answer)                 end with your answer. Call it only on something you read.

Search `context` with Python (find, regular expressions, slicing), read the relevant parts
with llm_query, then call FINAL. Variables persist between turns. You have {max_steps} turns.
{ordering}{memory}
Question:
{question}
{answer_format}"""

CHUNKS_TOOL = """\
  chunks                        the document cut into {n_chunks} pieces of ~{chunk_chars} characters,
                                in the order described below. Each has .text, .start, .end;
                                llm_query(question, chunks[i:j]) reads a list of chunks.
"""

ORDERING_NOTES = {
    "bm25-query": (
        "\n`chunks` is sorted by BM25 keyword relevance to the question: chunks[0] matches best."
        "\nThe answer is usually near the top.\n"
    ),
    "bm25-graph": (
        "\n`chunks` is grouped by topic, before the question was known: chunks sharing names and"
        "\nrare words sit next to each other. Find a chunk mentioning the question's distinctive"
        "\nterms, then read it together with its neighbours.\n"
    ),
}

# The memory / cost block: only present when a config turns on factor choice, offloading
# or cost reporting, so every other config renders exactly the prompt it always did.
FACTOR_NOTE = (
    "\nMemory: one read keeps at most {budget:,} tokens of KV memory (B). llm_query(question, text, factor=F)"
    "\nwith F in {factors} reads up to B x F tokens, compressed F times: F=1 is exact, a higher F reads"
    "\nwider but keeps less detail. Without factor= a read uses F={default}.\n"
)
OFFLOAD_NOTE = (
    "\nA read may be larger than B x F (up to {cap:,} tokens): the memory beyond B is offloaded to CPU"
    "\nand fetched back when the read is answered.\n"
)
COST_NOTE = (
    "\nCost: every read reports what it cost: {gpu:g} per memory token kept on the GPU, {offloaded:g} per"
    "\ntoken offloaded to CPU. Answer correctly at the lowest total cost.\n"
)
COST_LINE = (
    "[cost: read {read:,} tokens at x{factor} -> {gpu:,} on GPU, {offloaded:,} offloaded; {cost:g} (total {total:g})]"
)

ANSWER_FORMAT = "Answer format: {answer_format}"
LIST_ANSWER_FORMAT = 'Answer format: {answer_format} followed by a list, e.g. ["first", "second", "third"]'

NO_CODE_BLOCK = "Your reply had no ```python code block. Reply with exactly one code block."
REPL_OUTPUT = "REPL output:\n{output}"
BEGIN = "Begin."

SUB_PROMPT = "Answer the question using ONLY the text below. Be concise and factual.\n\nQuestion: {question}"

CHARS_PER_TOKEN = 4  # only to tell the root roughly how many characters fit in one read


def system_prompt(
    *,
    question: str,
    context_chars: int,
    read_tokens: int,
    max_steps: int,
    output_chars: int,
    ordering: str = "none",
    n_chunks: int = 0,
    chunk_chars: int = 0,
    answer_format: Optional[str] = None,
    list_answer: bool = False,
    budget_tokens: Optional[int] = None,
    factors: tuple[int, ...] = (1,),
    default_factor: int = 1,
    offload_cap: Optional[int] = None,
    cost: Optional[tuple[float, float]] = None,
) -> str:
    """The root's system prompt for one question under one config.

    `offload_cap` is set when offloading is on; `cost` (gpu, offloaded) when costs are shown.
    """
    ordered = ordering != "none"
    memory = ""
    if budget_tokens is not None and len(factors) > 1:
        memory += FACTOR_NOTE.format(budget=budget_tokens, factors=list(factors), default=default_factor)
    if offload_cap is not None:
        memory += OFFLOAD_NOTE.format(cap=offload_cap)
    if cost is not None:
        memory += COST_NOTE.format(gpu=cost[0], offloaded=cost[1])
    if answer_format:
        fmt = (LIST_ANSWER_FORMAT if list_answer else ANSWER_FORMAT).format(answer_format=answer_format.strip())
    else:
        fmt = ""
    return BASE_PROMPT.format(
        output_chars=output_chars,
        context_chars=context_chars,
        tools=CHUNKS_TOOL.format(n_chunks=n_chunks, chunk_chars=chunk_chars) if ordered else "",
        read_tokens=read_tokens,
        read_chars=read_tokens * CHARS_PER_TOKEN,
        max_steps=max_steps,
        ordering=ORDERING_NOTES.get(ordering, ""),
        memory=memory,
        question=question.strip(),
        answer_format=fmt,
    )
