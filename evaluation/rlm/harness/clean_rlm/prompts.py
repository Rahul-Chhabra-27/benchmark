# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt templates and protocol constants for the clean RLM."""

from __future__ import annotations

import re

ROOT_SYSTEM_PROMPT = """You are solving a task over a LONG document that does NOT fit in your context window.
The document is stored in a Python REPL as a string variable named `context`
(length: {ctx_len} characters). You interact with it by writing Python code.

RULES:
- Reply with exactly ONE Python code block, fenced as ```python ... ```, and then
  STOP. Write NOTHING after the code block.
- CRITICAL: You CANNOT see the result of your code until the next message. NEVER
  write, guess, or simulate the REPL output yourself. NEVER state an answer you
  have not literally seen in a real REPL output message.
- The REPL is persistent: variables survive across turns. Like Python's
  interactive shell, the value of a bare final expression is echoed back to you.
- Useful tools available in the REPL:
    * `context` (str): the full document.
{llm_query_help}
    * `print(...)`: anything you print is shown back to you in the NEXT message
      (truncated to {obs_limit} chars), so print only what you need to see.{note_tool}
{read_strategy}
{search_terms}
- If a search or extraction returns ZERO matches or fewer than the task implies,
  that is USUALLY a signal your pattern is wrong — not that the answer is empty.
  Print a sample of the text around a likely keyword to see the actual format,
  then retry with a corrected pattern.
  - When the task states how many items exist, verify your extraction found that
    many before computing the final answer.
- Once (and only once) you have SEEN the information needed for the answer in a
  real REPL output, finish with a code block calling FINAL(...) on a VARIABLE or
  expression computed by your code. NEVER call FINAL with a literal guessed
  value — FINAL("42") with a made-up constant will be rejected.
- If, after genuinely searching, the document does NOT contain the answer, end
  with FINAL_NONE("short reason") instead of guessing. Abstaining is a legitimate
  ending and is always better than inventing a value. Do NOT abstain on your first
  empty search — try at least one different pattern or tactic first. An abstention
  backed by NO llm_query call will be rejected: you must have actually read some of
  the document before you can claim the answer is not in it.
- You have at most {max_steps} code turns. Be efficient.{budget_note}

{worked_session}

The user's task is:
{task}
{answer_format_note}"""

# The "how do I locate things" bullet and the worked session are slots rather than
# fixed text because they are the two places that DEMONSTRATE a locate tactic, and
# the demonstration is what the root copies. Commit ec2c8a2 measured this: told
# "FEWER, BIGGER slices" one line above a 700-char keyhole example, the root sent a
# mean of 1023 chars against a 32000-char cap, and a lane with a 131072-char cap
# emitted every call at exactly 1500 chars -- the width of the example. So a search
# arm that added an llm_query bullet but left `context.lower().find(term.lower())`
# demonstrated here would keep getting `.find`.
#
# The _FIND defaults below are byte-identical to the text they replaced, so every
# non-search arm renders exactly the prompt it always did.
SEARCH_TERMS_FIND = """\
- SEARCH TERMS: never search for the question itself. A query like "first who
  wants to be a millionaire winner uk" appears nowhere in any document as a
  literal string, so searching for it proves nothing. Pull out the DISTINCTIVE
  terms — proper nouns, names, titles, numbers, rare words — and search for each
  one SEPARATELY and case-insensitively, e.g.
  `idx = context.lower().find(term.lower())`. For that query you would try
  "millionaire", then "jackpot", then "winner". Try several terms before
  concluding anything is absent."""

WORKED_SESSION_FIND = """\
Example of a correct session (3 turns):
  You:  ```python
        idx = context.find("invoice total")
        print(context[max(0, idx-200):idx+500])
        ```
  REPL: ...The invoice total for March was $4,210 including tax...
  You:  ```python
{example_llm_query}
        print(ans)
        ```
  REPL: $4,210
  You:  ```python
        FINAL(ans)
        ```"""

SEARCH_TERMS_BM25 = """\
- FINDING THINGS: `search(...)` is a keyword index over the WHOLE document. Unlike
  `context.find(...)` it does NOT need a literal match and it does NOT stop at the
  first occurrence — it returns the {k} best-matching windows, ranked, from
  anywhere in the document. Pass the question itself, or the DISTINCTIVE terms in
  it — proper nouns, names, titles, numbers, rare words. `context.find(...)` still
  exists and is right for an exact string you already know, but it returns only
  the FIRST literal occurrence, which in a document this long is almost never the
  one you want."""

LLM_QUERY_HELP_SEARCH = """\
    * `search(query: str) -> list[Hit]`: keyword search over the whole document.
      Returns the {k} best-matching windows, ranked. Each Hit has `.start` and
      `.end` (character offsets into `context`), `.score`, and `.text` (exactly
      `context[.start:.end]`). Print the hits to see where they landed.
    * `llm_query(question: str, hits) -> str`: ask a sub-LLM about retrieved text.
      IMPORTANT: the sub-LLM CANNOT see `context` or any of your variables. Put the
      question FIRST and the HIT LIST second — `llm_query("What is X?", hits)` —
      and the windows' text is sent for you. Pass the hits themselves rather than
      text you joined yourself: that is what lets the harness know which part of
      the document was read. Capture the result: ans = llm_query(...) then
      print(ans)."""

READ_STRATEGY_SEARCH = """\
- Strategy: `search(...)` FIRST, then read. Print the hits, then hand the WHOLE hit
  list to `llm_query` in ONE call — reading all {k} windows together costs one call
  and is what search is for. Do NOT print the whole context.
- SECOND HOP: when the question needs a fact you must look up first — it names a
  film and asks about its director's birthplace — search AGAIN with what you just
  LEARNED, not a re-run of the original query. If your first read already answers
  the question, finish instead; an unnecessary second search wastes a turn."""

# The components of the search treatment that can be withheld one at a time.
SEARCH_ABLATABLE = frozenset({"locate", "example", "gate"})

EXAMPLE_LLM_QUERY_SEARCH = '        ans = llm_query("What was the March invoice total?", hits)'

WORKED_SESSION_SEARCH = """\
Example of a correct session (4 turns):
  You:  ```python
        hits = search("who directed the 1975 film about the shark")
        for h in hits:
            print(h)
        ```
  REPL: Hit(rank=1, start=812400, end=814400, score=17.40, '...Jaws (1975) was directed by Steven Spielberg...')
        Hit(rank=2, start=55200, end=57200, score=9.12, '...shark attacks reported off Amity that summer...')
  You:  ```python
        ans = llm_query("Who directed the 1975 shark film?", hits)
        print(ans)
        ```
  REPL: Steven Spielberg
  You:  ```python
        hits2 = search("Steven Spielberg born")
        ans2 = llm_query("Where was Steven Spielberg born?", hits2)
        print(ans2)
        ```
  REPL: Cincinnati, Ohio
  You:  ```python
        FINAL(ans2)
        ```"""

# --------------------------------------------------------------------------- #
# The ordered-chunk arms (--chunk-order)
# --------------------------------------------------------------------------- #
# The AGENTIC root handed the whole document as a list of chunks in a fixed order,
# and told what that order is. The structured arm read every chunk through a fixed
# map -> reduce, so the root never chose what to read; here the REPL loop chooses,
# and the order is what makes its first read a good one.
#
# A variable rather than search(): a frozen set behind search() returns the same
# windows for every query, and a1ff98f measured the root rephrasing and searching
# eleven times against an identical result. These slots also replace the ones that
# demonstrate `context.find` -- the root copies the demonstration (ec2c8a2), so an
# arm whose worked session still showed .find would keep using it.
#
# The help and the search-terms bullet are shared by both orders; only the order
# paragraph and the worked session differ, so the two arms differ in what they were
# told about the order and in nothing else the prompt says.
CHUNK_ORDERS = ("bm25-graph", "bm25-query")
# Bump for any change that can alter what an ordered-chunk root is shown or sent.
# Revision 6 changes no prompt text: FINAL also accepts a literal composed from several
# reads (composed_answer_grounded). The LOFT campaign ran revision 5.
CHUNK_ORDER_REVISION = 6
# One structured reader group, so one agentic read is the size of one mapped group.
DEFAULT_CHUNK_CALL_TOKENS = 8192
# The question a chunk read is priced with when the prompt states how many chunks fit
# in one call: longer than most questions the root writes, so the stated number holds.
CHUNK_QUESTION_ALLOWANCE = ("Which of these passages answers the question, and what exactly do they say? " * 4).strip()

CHUNKS_HELP = """\
    * `chunks` (list): the WHOLE document cut into {n_chunks} consecutive pieces of
      {chunk_chars} characters, then put in the order described below. Every
      character of `context` is in exactly one chunk. Each chunk has `.rank` (its
      position in the list, counting from 1), `.start` and `.end` (character
      offsets into `context`), `.score`, and `.text` (exactly `context[.start:.end]`).
      Printing a chunk object shows only a short prefix, NOT its full text.
      Read chunks by passing them to `llm_query`; use `.text` for keyword filters.
    * `llm_query(question: str, chunks[i:j]) -> str`: ask a sub-LLM about some
      chunks. IMPORTANT: the sub-LLM CANNOT see `context`, `chunks` or any of your
      variables. Put the question FIRST and a LIST OF CHUNKS second --
      `llm_query("What is X?", chunks[:5])` -- and their text is sent for you, in
      the order you list them. One call reads at most about {per_call} chunks
      ({call_tokens} tokens); a longer list is refused, never cut short, so split
      a long read into several calls. Capture the result: ans = llm_query(...)
      then print(ans)."""

CHUNK_ORDER_NOTES = {
    "bm25-query": """\
- HOW `chunks` IS ORDERED: it was SORTED, before you started, by BM25 keyword
  relevance to the question. `chunks[0]` is the best match and `.score` falls as
  you go down the list; a score of 0 means the chunk shares no distinctive word
  with the question at all. The answer is almost always near the TOP.
- Strategy: your FIRST code block calls `llm_query` over `chunks[:5]` and prints
  the returned answer. Do not spend a turn printing chunk objects or their ranks.
  If that answers the question, you are done: FINAL it straight away, with
  confidence; there is no need to check the rest of the document. Only if the top
  chunks do not answer it, read the next five chunks or filter `.text` for a
  distinctive term and read those hits. Do NOT print the whole context.""",
    "bm25-graph": """\
- HOW `chunks` IS ORDERED: it was ORGANIZED once, before your question was known,
  by a keyword-similarity graph: chunks that share distinctive words -- the same
  names, titles, places -- were put NEXT TO EACH OTHER. It is NOT ranked by your
  question (every `.score` is 0), but a chunk's neighbours in the list are usually
  about the same people and things as the chunk itself.
- Strategy: first FIND which chunks mention the question's distinctive terms,
  then read the best one TOGETHER WITH ITS NEIGHBOURS in one call --
  `chunks[max(0, i-2):i+6]` -- because related facts sit beside it. If that
  answers the question, FINAL it straight away, with confidence. If it does not,
  read another matching neighbourhood or try a different distinctive term before
  abstaining. Do NOT print the whole context.""",
}

SEARCH_TERMS_CHUNKS = """\
- SEARCH TERMS: never look for the question itself -- a whole question appears
  nowhere in a document as a literal string. Pull out the DISTINCTIVE terms --
  proper nouns, names, titles, numbers, rare words -- and look for each one
  SEPARATELY and case-insensitively across the chunks, e.g.
  `hits = [i for i, c in enumerate(chunks) if "millionaire" in c.text.lower()]`.
  `c.text.lower()` is all lower case, so write every term in lower case too:
  "steven spielberg" can match it, "Steven Spielberg" never will.
  A question that needs a fact you must LEARN first (it names a film and asks
  where its director was born) takes two reads: read for the first fact, then
  look for what you just learned and read again."""

WORKED_SESSION_RANKED = """\
Example of a correct session (3 turns) for "Where was the director of the 1975 shark film born?":
  You:  ```python
        ans = llm_query("Who directed the 1975 shark film?", chunks[:5])
        print(ans)
        ```
  REPL: Steven Spielberg
  You:  ```python
        hop2 = [c for c in chunks if "spielberg" in c.text.lower()][:8]
        ans2 = llm_query("Where was Steven Spielberg born?", hop2)
        print(ans2)
        ```
  REPL: Cincinnati, Ohio
  You:  ```python
        FINAL(ans2)
        ```"""

WORKED_SESSION_NEIGHBOURS = """\
Example of a correct session (3 turns) for "Where was the director of the 1975 shark film born?":
  You:  ```python
        hits = [i for i, c in enumerate(chunks) if "shark" in c.text.lower() and "1975" in c.text]
        print(hits)
        ```
  REPL: [41, 97]
  You:  ```python
        i = hits[0]
        ans = llm_query("Who directed the 1975 shark film, and where was that director born? "
                        "Reply with the birthplace only.", chunks[max(0, i-2):i+6])
        print(ans)
        ```
  REPL: Cincinnati, Ohio
  You:  ```python
        FINAL(ans)
        ```"""

CHUNK_WORKED_SESSIONS = {"bm25-query": WORKED_SESSION_RANKED, "bm25-graph": WORKED_SESSION_NEIGHBOURS}

# The bracketed list LOFT's multi-value scorer parses (see STRUCTURED_ANSWER_LIST_NOTE),
# phrased for FINAL: the agentic arm answers through the value it passes there.
RLM_ANSWER_LIST_NOTE = (
    "The answer is a LIST of separate items: pass FINAL(...) a bracketed, comma-separated "
    "list of quoted strings -- for example {example} -- with one item per distinct answer.\n"
)

# The only sub-call shape is one OpenAI-compatible chat request. Any selected
# document text is embedded in the user message before it is sent.
LLM_QUERY_HELP_DENSE = """\
    * `llm_query(prompt: str) -> str`: ask a sub-LLM about text. IMPORTANT: the
      sub-LLM CANNOT see `context` or any of your variables — it sees ONLY the
      prompt string you pass. You MUST embed the actual text snippet inside the
      prompt, e.g. llm_query("Answer X based on this text:\n" + context[i:j])
      (each call is truncated at {max_chars} characters, so pass a focused
      snippet, not the whole document). Capture the result:
      ans = llm_query(...) then print(ans)."""

EXAMPLE_LLM_QUERY_DENSE = (
    '        ans = llm_query("What was the March invoice total? Answer from this text:\n"'
    " + context[idx-200:idx+500])"
)

READ_STRATEGY_DEFAULT = """\
- Strategy: peek at structure first (e.g. `print(context[:2000])`,
  `print(len(context))`, regex search), then narrow down with string ops or
  chunked `llm_query` calls. Do NOT print the whole context."""

SUB_SYSTEM_PROMPT = (
    "You are a helpful sub-model. Answer the question using ONLY the text "
    "provided in the prompt. Be concise and factual."
)

# --------------------------------------------------------------------------- #
# The structured (map -> reduce) arm
# --------------------------------------------------------------------------- #
# NEW constants rather than edits to the text above: the dense help is pinned
# byte-for-byte by test_rlm_subcall_context, and every in-flight agentic arm
# renders it. A structured run must not be able to shift what those arms send.
#
# The map question is asked once PER CHUNK, so it has to be answerable from one
# passage alone -- "answer the question" is the wrong instruction for a chunk
# that holds one hop of a two-hop question. Asking for the relevant FACTS keeps a
# partially-relevant chunk useful instead of producing a confident wrong answer
# the reduce step would then have to overrule.
STRUCTURED_MAP_QUESTION = (
    "Question: {task}\n\n"
    "Read the passage above. Quote every fact in it that helps answer the question, "
    "including partial information (a name, date, or link to another entity) that "
    "would only answer part of the question. Do not guess, and do not use knowledge "
    "from outside the passage. If the passage contains nothing relevant, reply with "
    "exactly: {nothing}"
)
# Matched case-insensitively on a prefix, not compared for equality: a 4B model
# asked for an exact string returns it with trailing punctuation or a short
# apology attached often enough that equality would count those chunks as findings.
STRUCTURED_NOTHING = "NOTHING RELEVANT"

STRUCTURED_REDUCE_SYSTEM = (
    "You are answering a question from notes taken by a reader who saw the source "
    "passages. Use ONLY those notes. If they contradict each other, prefer the more "
    "specific one. If they do not answer the question, say so plainly rather than "
    "guessing."
)
STRUCTURED_REDUCE_TEMPLATE = (
    "Question: {task}\n\n"
    "Notes from {n_chunks} passage(s) of the document, in document order:\n{findings}\n\n"
    "Answer the question from these notes.{answer_format_note}"
)
# The ordered-group arms (organization.py). Their notes arrive in the order the
# reader groups were read, which for a graph order is NOT document order -- so the
# header says what the order is instead of claiming one it is not. Used by BOTH
# ordered arms, source order included, so the two differ in chunk order alone.
STRUCTURED_REDUCE_TEMPLATE_GROUPS = (
    "Question: {task}\n\n"
    "Notes from {n_chunks} passage(s) of the document, in reader-group order:\n{findings}\n\n"
    "Answer the question from these notes.{answer_format_note}"
)
# The agentic arms answer through FINAL(x), where x is a value the root computed --
# the protocol forces a bare value. A reduce step is an ordinary chat completion,
# and "be concise" is satisfied by a well-formed SENTENCE: measured on hotpotqa,
# 15 of 55 replies matched the gold as a subspan while restating the question
# around it ("Mike Leach currently coaches at Washington State"), against 2 for the
# agentic arm. That is the whole em gap, and it is answer shape rather than
# accuracy. So the shape is required here rather than offered, on the principle the
# harness already applies to answer_format: an arm must not be scored on a shape it
# was never asked for.
STRUCTURED_ANSWER_SHAPE = (
    "\nEnd your reply with a single final line beginning exactly `{answer_format}` and "
    "containing the answer ITSELF -- the bare value or list, with no sentence built "
    "around it, no restatement of the question, and no explanation after it."
)
# Revision 1's reduce wording, kept verbatim rather than paraphrased back into
# existence, so `--structured-answer-shape offer` restores exactly the treatment
# revision 2 replaced. Revision 2 shipped the shape requirement and the passage
# clustering in one commit; these two constants are what lets the first be measured
# apart from the second.
#
# ", in document order" stays in BOTH templates. Passages are emitted in document
# order at every granularity now (see _map_groups), and a header that lied about
# the order would put a third difference inside a two-way ablation.
STRUCTURED_REDUCE_TEMPLATE_OFFER = (
    "Question: {task}\n\n"
    "Notes from {n_chunks} passage(s) of the document, in document order:\n{findings}\n\n"
    "Answer the question from these notes. Be concise.{answer_format_note}"
)
STRUCTURED_ANSWER_SHAPE_OFFER = "\nFormat your answer as: {answer_format}"

# LOFT's multi-value subsets (qampari, quest) are scored through a parser that only
# splits a BRACKETED list. Their `answer_prefix` is the same bare "Final Answer: "
# every other subset ships, so nothing ever told the model the shape -- and a reply
# naming all five gold answers as prose ("Rival Schools, Gorilla Biscuits, ...")
# extracts as ONE string, intersects no gold, and scores coverage 0.0 while the
# harness's own loose check calls it correct. Measured on the oracle arm: 3 of 3
# right, 0.000 reported.
#
# So this is the shape instruction the arm was missing, not a scoring workaround.
# Empty by default, which reproduces every run already on disk byte for byte.
STRUCTURED_ANSWER_LIST_NOTE = (
    "\nThe answer is a LIST of separate items. Put it on that final line as a "
    "bracketed, comma-separated list of quoted strings -- for example {example} -- "
    "with no other text on the line. One item per distinct answer."
)

# The structured arm's three axes, in the shape of SEARCH_ABLATABLE above: a
# bundled treatment gets one knob per component. Every DEFAULT reproduces revision
# 2 byte-for-byte, so turning these on is the only way to change what the shipped
# arm sends.
#   map_question  task     the benchmark's task string, verbatim (revision 2)
#                 query    only the query inside it, for a benchmark that wraps one
#   granularity   passage  one call per contiguous passage (revision 2)
#                 window   one call per provided window (revision 1's granularity)
#                 all      ONE call over every window, as the agentic root sends
#   answer_shape  require  the answer shape is demanded (revision 2)
#                 offer    the answer shape is offered (revision 1)
STRUCTURED_MAP_QUESTIONS = ("task", "query")
STRUCTURED_MAP_GRANULARITIES = ("passage", "window", "all")
STRUCTURED_ANSWER_SHAPES = ("require", "offer")


def bare_query(task: str) -> str:
    """The question inside a benchmark's instruction wrapper, or the task unchanged.

    LOFT's `task` is not a question. It is a page of citation and formatting
    instructions with the real query on a final `query:` line, and interpolated
    verbatim into a PER-PASSAGE map prompt it derails a 4B sub-model: on nq:12 the
    model performed the citation ritual ("TITLE: Philadelphia 76ers | ID: 732") and
    then reported nothing relevant. Those instructions are aimed at a model that has
    read the whole corpus, so they are meaningful to the reduce step and noise to the
    map step -- which is why this is applied to the map question alone.

    Deliberately conservative: a task with no `query:` line comes back untouched, so
    a benchmark whose task IS the question is unaffected.
    """
    lines = str(task).splitlines()
    for index in range(len(lines) - 1, -1, -1):
        stripped = lines[index].lstrip()
        if stripped[:6].lower() == "query:":
            return "\n".join([stripped[6:], *lines[index + 1 :]]).strip()
    return str(task).strip()


# Sent when a model gives up without having read anything (see
# min_sub_calls_before_abstain). Names the specific mistake -- searching for the
# whole question -- because "search harder" alone left the root re-running the
# same doomed literal lookup with a different slice.
PREMATURE_ABSTAIN_NUDGE = (
    "REJECTED: you called FINAL_NONE without ever calling llm_query, so you have not "
    "actually read any of the document and cannot know the answer is absent. A "
    "natural-language question almost never appears verbatim in a document, so a "
    "failed search for the whole query proves nothing at all. Extract the DISTINCTIVE "
    "terms instead -- proper nouns, names, titles, numbers, rare words -- and search "
    "for each one SEPARATELY and case-insensitively, e.g. "
    "idx = context.lower().find(term.lower()). When a term hits, print the text around "
    "it to see the format, then pass that slice to llm_query. Only abstain once several "
    "different terms have genuinely failed."
)

# The same rejection for the ordered-chunk arms, without the `context.find` recipe:
# those arms were taught to read `chunks`, and a nudge demonstrating .find would pull
# the root back to the tactic their prompt replaced.
PREMATURE_ABSTAIN_NUDGE_CHUNKS = (
    "REJECTED: you called FINAL_NONE without ever calling llm_query, so you have not "
    "actually read any of the document and cannot know the answer is absent. Read the "
    "chunks most likely to hold the answer -- the top of `chunks` if it is sorted by "
    "relevance, or the chunks that mention the question's distinctive terms, e.g. "
    "[c for c in chunks if term.lower() in c.text.lower()] -- by passing them to "
    "llm_query(question, those_chunks). Only abstain once several reads have genuinely failed."
)

# How many chunk reads an ordered-chunk run makes before its FINAL_NONE is believed.
# The legacy bar -- one sub-call -- was set for a root that located a slice before
# reading it. Here one read of the top five chunks is 2% of a 238-chunk document, and
# on the revision-2 pilot all three musique query-arm rows abstained after exactly that
# one read, one of them with the gold in chunks[0]: the root had asked all three hops
# in one question, and the reader said the text did not answer it.
CHUNK_READS_BEFORE_ABSTAIN = 3

# The rejection for an ordered-chunk run that HAS read, but too little to claim absence.
PREMATURE_ABSTAIN_NUDGE_CHUNK_READS = (
    "REJECTED: you have read only {read} of the {total} chunks, so you cannot yet know the "
    "answer is absent. When a reader says the text does not answer, the question was often "
    "too big for one read: ask for ONE fact at a time -- the first thing you need to learn "
    "-- then look for what you learned. Read more before giving up: the next chunks of the "
    "list, or the chunks whose `.text` mentions a distinctive term from the question, e.g. "
    "[c for c in chunks if term.lower() in c.text.lower()]. Only abstain once several "
    "different reads have genuinely failed."
)

FOLD_MARKER = (
    "[MEMORY NOTICE] {n} earlier turn(s) were dropped to stay within your memory "
    "budget. They are gone from your context and were NOT saved anywhere — but the "
    "REPL is persistent, so any Python variable you assigned still exists. Re-query "
    "the REPL if you need anything from those turns. Continue from the recent output "
    "below."
)

NOTE_TOOL_HELP = (
    "\n    * `note(text: str)`: save a SHORT finding to a persistent scratchpad. "
    "The scratchpad is shown back to you on every turn, so notes never scroll "
    "out of view — save key facts, indices, and partial answers there."
)

SCRATCHPAD_TEMPLATE = "[SCRATCHPAD] Your saved notes so far (persistent — always visible):\n{notes}\n[END SCRATCHPAD]"

FOLD_SCRATCHPAD_TEMPLATE = (
    "[MEMORY NOTICE] {n} earlier turn(s) were dropped to stay within your memory "
    "budget. Raw output from them is gone, but the REPL is persistent (your "
    "variables still exist) and this scratchpad survived:\n"
    "{notes}\n[END SCRATCHPAD] Continue from the recent output below."
)


class _ExecTimeout(Exception):
    """Raised by the SIGALRM handler when model-generated code exceeds its budget."""


def _raise_exec_timeout(signum, frame):  # pragma: no cover - signal handler
    raise _ExecTimeout()


CODE_RE = re.compile(r"```(?:python|repl|py)?[ \t]*\n?(.*?)```", re.DOTALL)
# Fallback for a reply whose code block was cut off at max_tokens: an OPENING fence
# with no closing one. Without this the reply reads as "no code block", burns a nudge,
# and after three such replies the raw truncated text is returned as the answer.
UNTERMINATED_CODE_RE = re.compile(r"```(?:python|repl|py)?[ \t]*\n(.*)\Z", re.DOTALL)
# Spellings of the terminal calls the REPL accepts. Small models typo the case
# constantly: on the LOFT-1m smoke one example burned several of its 38 steps on
# `NameError: name 'Final' is not defined` before the repetition breaker cut it
# off. Accepting the variants is free -- but MISSING one is not, so the env, the
# AST literal-scan and the prose regexes below are all derived from these tuples.
# A spelling the env accepted while the scan missed it would walk a guessed
# literal straight past the grounding guard, which is a worse bug than the typo.
FINAL_NAMES = ("FINAL", "Final", "final")
FINAL_NONE_NAMES = ("FINAL_NONE", "Final_None", "FINAL_None", "final_none")
FINAL_VAR_NAMES = ("FINAL_VAR", "Final_Var", "FINAL_Var", "final_var")
_ALL_TERMINAL_NAMES = FINAL_NONE_NAMES + FINAL_VAR_NAMES + FINAL_NAMES  # longest-first

STMT_KEYWORDS = r"print|import|from|for|while|if|elif|try|except|finally|with|return|note|" + "|".join(
    _ALL_TERMINAL_NAMES
)
ONELINE_FIX_RE = re.compile(rf"(?<=[\)\w'\"])\s+(?=(?:{STMT_KEYWORDS})\b)")
_FINAL_ALT = "|".join(FINAL_NAMES)
_FINAL_NONE_ALT = "|".join(FINAL_NONE_NAMES)
_FINAL_VAR_ALT = "|".join(FINAL_VAR_NAMES)
TEXT_FINAL_RE = re.compile(rf"(?:{_FINAL_ALT})\(\s*(?:\"\"\"|'''|\"|')(.*?)(?:\"\"\"|'''|\"|')\s*\)", re.DOTALL)
TEXT_FINAL_VAR_RE = re.compile(rf"(?:{_FINAL_VAR_ALT})\(\s*[\"'](\w+)[\"']\s*\)")
# FINAL_NONE may be written bare or with a reason, in code or in prose.
TEXT_FINAL_NONE_RE = re.compile(
    rf"(?:{_FINAL_NONE_ALT})\(\s*(?:(?:\"\"\"|'''|\"|')(.*?)(?:\"\"\"|'''|\"|'))?\s*\)", re.DOTALL
)
# LOFT's answer line, written as a code cell. LOFT prompts close their few-shot answers
# with "Final Answer: [...]", and on the revision-2 pilot three graph-arm rows read the
# answer, then submitted `Final Answer: 2000` as Python. The one-line diagnosis
# ("rewrite it multi-line") cannot be acted on for a single statement, so each row
# resubmitted the same line until the repetition breaker replaced its answer. Only two
# words are matched: `answer: x` alone is a valid annotation and never reaches here.
PROSE_ANSWER_LINE_RE = re.compile(r"^\s*final\s+answer\s*:", re.IGNORECASE | re.MULTILINE)
PROSE_ANSWER_CELL_ERROR = (
    "[SYNTAX ERROR] `Final Answer: ...` is not Python, so nothing was recorded. Submit the "
    "answer by calling FINAL in a code block instead: FINAL(ans), with the variable that holds "
    'it. A quoted FINAL("...") is accepted only if that exact text appeared in the REPL output.'
)
# An ordered-chunk read of NO chunks: a keyword filter that matched nothing. Sent on, the
# reader says the text does not mention the answer -- about text it never saw -- and on
# the revision-3 pilot the root believed it and abstained. The cause each time was a
# capitalised term tested against `c.text.lower()`, which can never match.
EMPTY_CHUNK_READ_ERROR = (
    "llm_query refused: the list of chunks is EMPTY, so there is nothing to read -- your filter "
    "matched no chunk. `c.text.lower()` is all lower case, so a term with capitals in it never "
    "matches: write the term in lower case, or try a different distinctive term. Nothing was read."
)
# The ungrounded-FINAL rejection for the ordered-chunk arms, without the `context.find`
# recipe their prompt replaced. On the revision-4 pilot a root holding the reader's answer
# in `ans` typed FINAL("Al-Aqsa Mosque, None") -- words no output contained -- and was
# rejected into ungrounded_final; FINAL(ans) would have been accepted as it stood.
UNGROUNDED_FINAL_CHUNKS = (
    "REJECTED: FINAL({val!r}) is a literal that never appeared in any REPL output, so it looks "
    "like a guess. Pass FINAL the variable that holds a reader's answer -- e.g. FINAL(ans) -- or "
    "quote only text that appears word for word in the output above. If several different reads "
    'have genuinely failed, say so with FINAL_NONE("reason") instead.'
)

# Replies that mean an ordered reader call did not happen.
ORDERED_READER_FAILURES = ("[SUB-CALL LIMIT", "[TIME LIMIT")

# Seconds the exec watchdog is always given back after a sub-call, even if its
# timer expired mid-call. See the re-arm site in llm_query.
ALARM_REARM_FLOOR_S = 1.0

# Cap on ONE map finding pasted into the reduce prompt. A sub model told to quote
# the relevant facts occasionally returns the whole passage; k of those would put
# the entire payload back into the root's window, which is the one thing the RLM
# paradigm exists to avoid.
#
# A DEFAULT rather than a constant since the window geometry became an axis: at a
# 4000- or 8000-char window the map step reads more but the reduce step would still
# receive only 2000 chars of each finding, so the top of the window axis would
# measure a treatment this cap had already severed. Raise it with the window.
DEFAULT_REDUCE_FINDING_CHARS = 2000

# How many characters one token is worth, for the two places that must convert
# between the two units without a tokenizer: TokenCounter's no-tiktoken fallback
# and the adaptive observation limit. Shared so the two cannot silently disagree
# (they used 4 and 3 respectively, for no stated reason).
CHARS_PER_TOKEN_ESTIMATE = 4
