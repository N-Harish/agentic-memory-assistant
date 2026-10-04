import os
import json
import re
import threading
import time
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from mem0 import MemoryClient

from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend

from sentence_transformers import CrossEncoder


# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="Agentic Memory Assistant",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# ENV
# ============================================================
load_dotenv()

MEM0_API_KEY = os.getenv("MEM0_API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

if not MEM0_API_KEY:
    st.error("MEM0_API_KEY is not set")
    st.stop()

if not OLLAMA_API_KEY:
    st.error("OLLAMA_API_KEY is not set")
    st.stop()


# ============================================================
# CONFIG
# ============================================================
DEFAULT_USER_ID = "demo-user"

MEMORY_LIMIT = 10
MEMORY_READ_TOP_K = 5
MEMORY_WRITE_TOP_K = 10

MAIN_MODEL = "gpt-oss:20b"
MEMORY_REASONING_MODEL = "gpt-oss:20b"
SUMMARY_MODEL = "gpt-oss:20b"
ANYJEV_MODEL = "Qwen/Qwen3.5-4B"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"

STORE_THRESHOLD = 0.5

HISTORY_RECENT_MESSAGES = 6
SUMMARY_TRIGGER_MESSAGES = 12


# ============================================================
# SHARED RESOURCES (created once, survive Streamlit reruns)
# ============================================================
class ActivityLog:
    """Thread-safe log so the background write thread can report
    what it did, and the UI can display it."""

    def __init__(self, maxlen: int = 500):
        self._lines = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line)
        with self._lock:
            self._lines.append(line)

    def dump(self) -> str:
        with self._lock:
            return "\n".join(self._lines)

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()


class ConversationSummary:
    """Thread-safe rolling summary. Lives in st.session_state so the
    background thread can mutate it without touching Streamlit APIs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._summary = ""
        self._upto = 0          # number of messages already folded into summary
        self._busy = False

    def get_context(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Returns (summary, messages not yet summarized)."""
        with self._lock:
            pending = messages[self._upto:]
            return self._summary, [
                {"role": m["role"], "content": m["content"]} for m in pending
            ]

    def try_start(self) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            return True

    def finish(self) -> None:
        with self._lock:
            self._busy = False

    def update(self, summary: str, upto: int) -> None:
        with self._lock:
            self._summary = summary
            self._upto = upto

    def reset(self) -> None:
        with self._lock:
            self._summary = ""
            self._upto = 0

    @property
    def summary(self) -> str:
        with self._lock:
            return self._summary

    @property
    def upto(self) -> int:
        with self._lock:
            return self._upto


@st.cache_resource(show_spinner="Loading clients and models...")
def load_resources():
    import torch

    mem0 = MemoryClient(api_key=MEM0_API_KEY)

    ollama_kwargs = dict(
        base_url="https://ollama.com",
        client_kwargs={
            "headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
        },
    )

    main_llm = ChatOllama(
        model=MAIN_MODEL, temperature=0.2, **ollama_kwargs
    )

    memory_reasoning_llm = ChatOllama(
        model=MEMORY_REASONING_MODEL, temperature=0.0, **ollama_kwargs
    )

    summarizer_llm = ChatOllama(
        model=SUMMARY_MODEL, temperature=0.0, **ollama_kwargs
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(RERANKER_MODEL, device=device)

    decider = Decider(HFBackend(ANYJEV_MODEL))

    return {
        "mem0": mem0,
        "main_llm": main_llm,
        "memory_reasoning_llm": memory_reasoning_llm,
        "summarizer_llm": summarizer_llm,
        "reranker": reranker,
        "decider": decider,
        "model_lock": threading.Lock(),
        "log": ActivityLog(),
    }


R = load_resources()
mem0 = R["mem0"]
main_llm = R["main_llm"]
memory_reasoning_llm = R["memory_reasoning_llm"]
summarizer_llm = R["summarizer_llm"]
reranker = R["reranker"]
decider = R["decider"]
MODEL_LOCK = R["model_lock"]
LOG: ActivityLog = R["log"]


# ============================================================
# ANYJEV QUESTION
# ============================================================
SHOULD_STORE_QUESTION = Question.noul(
    """
    Does this conversational turn contain information about the user
    that would be useful to remember across future conversations?

    STORE user-specific facts, background, preferences, decisions,
    configurations, setup details, plans, goals, corrections,
    changes in state, changes in preferences, and newly established
    current information about the user.

    SKIP questions that do not reveal a useful user-specific fact,
    hypothetical or conditional scenarios, general knowledge,
    general discussion, task requests without new personal
    information, and small talk.

    When the user explicitly states or changes something about
    themselves, prefer STORE.

    When in doubt, STORE user-specific information that could
    affect future answers.
    """,
    name="should_store",
)


# ============================================================
# PROMPTS
# ============================================================

MAIN_SYSTEM_PROMPT = """
You are a helpful AI assistant.

You may use the user's long-term memories when they are relevant to
the current request.

IMPORTANT MEMORY RULES:

1. CURRENT USER MESSAGE HAS HIGHEST PRIORITY

The current user message is the most authoritative source of the
user's current state, preferences, constraints, and intentions.

If the current message conflicts with a retrieved memory, follow the
current message.

2. MEMORIES MAY BE STALE

Long-term memories may be outdated, incomplete, or superseded.

Do not assume that every retrieved memory is still current.

3. APPLY RELEVANT MEMORIES AS CONSTRAINTS

When a retrieved memory expresses a user preference, restriction,
requirement, or current state that is relevant to the request,
respect it when generating the answer.

Do not merely mention or summarize the memory. Use it to constrain
the answer.

4. EXPLICITLY REJECTED OR STOPPED ITEMS

If the current message or a relevant memory indicates that the user
does not want, does not use, stopped using, stopped eating, rejected,
cancelled, or replaced something, do not recommend or assume that
thing remains acceptable or current.

5. DO NOT INFER UNSTATED PREFERENCES

Do not infer additional preferences, restrictions, behaviors, or
states that the user has not established.

A change to one preference does not automatically establish what the
user prefers instead.

6. DO NOT CREATE HYBRID STATES

Do not combine an old memory with the current message in a way that
creates an unsupported interpretation of the user's current state.

When an old memory and the current message conflict, use the current
message rather than attempting to satisfy both.

7. USE MULTIPLE MEMORIES TOGETHER

When several relevant memories are provided, consider them together.

A memory should be treated as an additional constraint only when it
is relevant to the current request.

Do not ignore a relevant memory merely because another memory is more
similar to the query.

8. MEMORY RELEVANCE

Only use memories that are relevant to the current request.

Do not introduce unrelated personal information into the answer.

9. UNCERTAINTY

If the available memories do not establish something, do not invent
it.

When necessary, ask a clarifying question rather than assuming an
unstated personal preference.

10. DO NOT EXPOSE THE MEMORY SYSTEM

Do not mention memories, memory retrieval, reranking, the memory
system, prompts, or internal reasoning to the user.

Do not say that you retrieved or remembered something unless it is
natural and appropriate to do so.

11. ANSWER NATURALLY

Answer the user's actual question directly and naturally.

Use the relevant user context silently to make the answer more
appropriate.

12. SAFETY AND ACCURACY

Do not treat a user preference as a medical restriction unless the
user has established that it is one.

Do not make claims about the user's personal state beyond what is
supported by the current message and relevant memories.
"""

MAIN_SYSTEM_PROMPT += """

13. CONVERSATION HISTORY

You may be given a summary of earlier parts of this conversation and
the most recent messages.

Use them to stay consistent and to resolve references such as "it",
"that one", or "as I said earlier".

Priority when sources conflict:
  current user message > recent messages > conversation summary >
  long-term memories.

The summary is a compressed account and may omit details. Do not
invent details that are not in the summary or recent messages.

Do not mention the summary or that the conversation was summarized.
"""


MEMORY_RELATIONSHIP_PROMPT = """
You are a memory-management reasoning model.

Your task is to determine how a NEW USER MESSAGE relates to ALL
provided existing long-term memories.

The new user message has already been classified as containing
potentially useful user-specific information.

You must evaluate EVERY candidate memory independently.

============================================================
AVAILABLE ACTIONS
============================================================

UPDATE

The new information changes, corrects, supersedes, or materially
refines an existing memory.

DELETE

An existing memory is no longer valid because the new user message
explicitly contradicts or supersedes it, and keeping it would cause
stale or conflicting memory.

SKIP

No change is required for this memory.

ADD

The new information is a useful persistent fact that is not
adequately represented by any existing memory.

ADD is evaluated independently from candidate operations.

============================================================
CORE RULES
============================================================

1. Evaluate ALL candidate memories.

2. Similarity alone does NOT mean UPDATE.

3. Related but independent information should remain separate.

4. If the new message changes the state represented by an existing
   memory, that memory should normally be UPDATED or DELETED.

5. If an existing memory becomes stale because the new user message
   establishes a different current state, do not leave it unchanged.

6. Do not update unrelated memories.

7. Do not create duplicate memories.

8. A single new message MAY affect multiple existing memories.

9. However, do not modify multiple memories merely because they are
   semantically related.

10. Distinguish between:
    - current state
    - previous state
    - independent facts
    - independent preferences
    - behaviors
    - goals
    - configurations

11. When multiple memories describe different values of the same
    underlying current attribute, the newest explicit user
    statement takes precedence.

12. If one candidate already contains the newest current value,
    SKIP that candidate.

13. If another candidate contains an obsolete value of that same
    current attribute, DELETE that candidate.

14. UPDATE a candidate only when it is appropriate to transform
    that candidate into the new canonical current state.

15. Do not UPDATE an obsolete candidate into a duplicate when
    another candidate already contains the new current value.

16. Do not ADD information that is already adequately represented
    by an existing memory.

17. Do not infer unsupported personal information.

18. Do not invent facts.

19. Do not use the assistant's response as evidence.

20. Base decisions only on:
    - the new user message
    - the provided existing memories

21. Retrieval ranking is not evidence that a memory is true,
    current, or more important.

============================================================
STATE / ATTRIBUTE SEMANTICS
============================================================

Some memories represent a CURRENT STATE or ATTRIBUTE of the user.

For these memories, normally only the latest current value should
remain active.

This includes attributes where a new value normally replaces an
older value.

When the new user message establishes a different current value
for the same underlying attribute:

1. The new value takes precedence.
2. The older value is stale.
3. Do not keep both values as active current memories.
4. If an existing memory already represents the new value, SKIP it.
5. DELETE obsolete memories representing the previous value.
6. UPDATE a memory only when doing so is necessary to preserve the
   new canonical state.
7. Do not ADD a duplicate of an already existing current value.

Do not require explicit words such as "no longer", "stopped",
"replaced", or "changed" if the user's wording clearly establishes
a new current state.

============================================================
CURRENT STATE VS INDEPENDENT FACT
============================================================

Before modifying a candidate, determine whether it represents:

A. the same underlying current attribute or state as the new message

or

B. an independent fact, preference, behavior, goal, or configuration.

Only memories in category A should normally be affected by state
supersession.

A new message may therefore produce different actions for different
candidates.

One candidate may become obsolete while another independent
candidate remains valid.

============================================================
SUBSUMPTION
============================================================

If the new information is already adequately represented by an
existing memory:

- SKIP that existing memory.
- Do not ADD a duplicate.

If the new information replaces an older value:

- SKIP an existing memory that already contains the new value.
- DELETE older memories containing obsolete values.
- UPDATE only when an existing memory can be directly transformed
  into the canonical current representation without creating
  duplication.

============================================================
TEMPORAL PRIORITY
============================================================

When two memories represent different values of the same current
attribute, the newest explicit user statement takes precedence.

Do not preserve an older value as an active current state merely
because the older and newer values are not literal logical
contradictions.

The relevant question is whether they represent competing values
of the same underlying current attribute.

============================================================
RELATEDNESS IS NOT ENOUGH
============================================================

Do not modify a memory merely because it is semantically similar
to the new message.

Do not assume that a new state change invalidates every related
fact.

Determine whether the candidate actually represents the same
underlying state.

============================================================
MULTIPLE OPERATIONS
============================================================

Evaluate every candidate independently.

A single message may legitimately cause:

- SKIP for one candidate
- DELETE for another
- UPDATE for another
- and ADD for genuinely new independent information

Do not stop after finding the first related memory.

Do not modify candidates that remain independently valid.

============================================================
IMPORTANT
============================================================

The user's newest explicit statement is authoritative for the
current value of an attribute.

Do not invent historical states.

Do not invent additional user facts.

Do not use the assistant's response as evidence.

Do not use retrieval score as evidence of truth.

Use only the new user message and the provided candidate memories.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Do not use markdown.
Do not add commentary.
Do not use code fences.

{
  "operations": [
    {
      "action": "UPDATE | DELETE | SKIP",
      "memory_id": "existing candidate ID",
      "new_memory": "replacement memory text or null",
      "reason": "brief explanation"
    }
  ],
  "add_new_memory": false,
  "new_memory": null,
  "reason": "brief overall explanation"
}

============================================================
OUTPUT RULES
============================================================

For UPDATE:

- memory_id MUST be one of the provided candidate IDs.
- new_memory MUST contain the replacement memory.
- The replacement must represent the user's current state.
- Keep it concise.
- Do not invent information.

For DELETE:

- memory_id MUST be one of the provided candidate IDs.
- new_memory MUST be null.

For SKIP:

- memory_id MUST be one of the provided candidate IDs.
- new_memory MUST be null.

For ADD:

- Set add_new_memory=true.
- new_memory MUST contain the new persistent memory.
- The application will store the original user message, not the
  generated memory text.

Every candidate memory must appear exactly once in "operations".

If no existing memory needs modification and the new information is
already represented, use SKIP for all candidates and set
add_new_memory=false.

If the new information is independent of all candidates, use SKIP
for all candidates and set add_new_memory=true.

Do not update or delete a memory merely because it shares keywords
with the new message.
"""


SUMMARY_PROMPT = """
You maintain a rolling summary of a conversation between a user and an
AI assistant.

You will receive the PREVIOUS SUMMARY (possibly empty) and NEW MESSAGES.
Produce an UPDATED SUMMARY that merges them.

Rules:
- Preserve: topics discussed, questions asked, answers/recommendations
  given, decisions made, user-stated facts and preferences, constraints,
  open tasks, and unresolved questions.
- If the user changed or corrected something, keep only the newest value.
- Resolve references so the summary is understandable on its own.
- Be concise: at most ~250 words. Compress older material first.
- Do not invent anything. Use only the previous summary and new messages.
- Write plain prose or short bullet points. No preamble, no headings,
  no commentary. Output only the updated summary.
"""


# ============================================================
# HELPERS
# ============================================================
def llm_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content or ""


def extract_json(text: str) -> Dict[str, Any]:
    """Robustly extract a JSON object from an LLM response."""

    if not text:
        raise ValueError("Empty LLM response")

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)

    try:
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("Planner response is not a JSON object")
        return result
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON object in response:\n{text}")

    result = json.loads(text[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("Extracted planner response is not a JSON object")

    return result


def normalize_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": memory.get("id") or memory.get("memory_id"),
        "memory": memory.get("memory") or memory.get("text") or "",
        "score": memory.get("score"),
        "created_at": memory.get("created_at"),
        "updated_at": memory.get("updated_at"),
        "user_id": memory.get("user_id"),
    }


# ============================================================
# MEM0 SEARCH
# ============================================================
def search_memories(
    user_message: str,
    user_id: str,
    limit: int = MEMORY_LIMIT,
) -> List[Dict[str, Any]]:

    result = mem0.search(
        user_message,
        filters={"user_id": user_id},
        limit=limit,
    )

    if isinstance(result, dict):
        memories = result.get("results", [])
    elif isinstance(result, list):
        memories = result
    else:
        memories = []

    return [
        normalize_memory(m) for m in memories if isinstance(m, dict)
    ]


# ============================================================
# RERANK
# ============================================================
def rerank_memories(
    user_message: str,
    memories: List[Dict[str, Any]],
    top_k: int = MEMORY_READ_TOP_K,
) -> List[Dict[str, Any]]:

    if not memories:
        return []

    pairs = [(user_message, m.get("memory", "")) for m in memories]

    try:
        with MODEL_LOCK:
            scores = reranker.predict(pairs, show_progress_bar=False)

        reranked = []
        for memory, score in zip(memories, scores):
            item = dict(memory)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    except Exception as e:
        LOG.add(f"[RERANKER ERROR] {e}")
        return memories[:top_k]


# ============================================================
# ANYJEV
# ============================================================
def should_store_memory(user_message: str) -> Tuple[bool, float]:
    try:
        with MODEL_LOCK:
            result = decider.decide(
                {
                    "conversation": [
                        {"role": "user", "content": user_message}
                    ]
                },
                [SHOULD_STORE_QUESTION],
            )

        probability = float(result[0].p_true)
        should_store = probability >= STORE_THRESHOLD

        LOG.add(
            f"[ANYJEV] store={should_store} p_store={probability:.4f}"
        )
        return should_store, probability

    except Exception as e:
        LOG.add(f"[ANYJEV ERROR] {e}")
        return False, 0.0


# ============================================================
# PLANNER
# ============================================================
def plan_memory_operations(
    user_message: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:

    candidate_text = []
    for idx, memory in enumerate(candidates, start=1):
        candidate_text.append(
            f"""
Candidate {idx}
memory_id: {memory.get("id")}
memory: {memory.get("memory")}
"""
        )

    candidates_block = "\n".join(candidate_text)

    prompt = f"""
============================================================
NEW USER MESSAGE
============================================================

{user_message}

============================================================
EXISTING MEMORY CANDIDATES
============================================================

{candidates_block}

============================================================
TASK
============================================================

Evaluate EVERY candidate memory independently.

For each candidate, choose exactly one:

UPDATE
DELETE
SKIP

Then independently determine whether the new user information
requires ADD.

Return ONLY the required JSON object.
"""

    response = memory_reasoning_llm.invoke(
        [
            SystemMessage(content=MEMORY_RELATIONSHIP_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    return extract_json(llm_text(response.content))


def validate_memory_plan(
    plan: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:

    candidate_ids = {
        str(m.get("id")) for m in candidates if m.get("id") is not None
    }

    operations = plan.get("operations", [])
    if not isinstance(operations, list):
        operations = []

    validated_operations = []
    seen_ids = set()

    for operation in operations:
        if not isinstance(operation, dict):
            continue

        action = str(operation.get("action", "SKIP")).upper()
        memory_id = operation.get("memory_id")

        if memory_id is None:
            continue

        memory_id = str(memory_id)

        # Never let the LLM mutate a memory that wasn't a candidate.
        if memory_id not in candidate_ids:
            LOG.add(f"[PLANNER] Ignoring invalid memory ID: {memory_id}")
            continue

        if memory_id in seen_ids:
            LOG.add(f"[PLANNER] Duplicate operation ignored: {memory_id}")
            continue

        if action not in {"UPDATE", "DELETE", "SKIP"}:
            action = "SKIP"

        new_memory = operation.get("new_memory")

        if action != "UPDATE":
            new_memory = None
        elif not isinstance(new_memory, str) or not new_memory.strip():
            LOG.add(
                f"[PLANNER] Invalid UPDATE without new_memory for {memory_id}"
            )
            action = "SKIP"
            new_memory = None

        validated_operations.append(
            {
                "action": action,
                "memory_id": memory_id,
                "new_memory": new_memory,
                "reason": operation.get("reason", ""),
            }
        )
        seen_ids.add(memory_id)

    # Omitted candidates default to SKIP.
    for memory in candidates:
        memory_id = memory.get("id")
        if memory_id is None:
            continue
        memory_id = str(memory_id)

        if memory_id not in seen_ids:
            validated_operations.append(
                {
                    "action": "SKIP",
                    "memory_id": memory_id,
                    "new_memory": None,
                    "reason": "Candidate omitted by planner; defaulting to SKIP.",
                }
            )

    add_new_memory = bool(plan.get("add_new_memory", False))
    new_memory = plan.get("new_memory")

    if add_new_memory:
        if not isinstance(new_memory, str) or not new_memory.strip():
            LOG.add("[PLANNER] ADD requested without valid new_memory. Disabling ADD.")
            add_new_memory = False
            new_memory = None
    else:
        new_memory = None

    return {
        "operations": validated_operations,
        "add_new_memory": add_new_memory,
        "new_memory": new_memory,
        "reason": plan.get("reason", ""),
    }


def apply_memory_operations(
    plan: Dict[str, Any],
    user_message: str,
    user_id: str,
) -> None:

    for operation in plan.get("operations", []):
        action = operation.get("action")
        memory_id = operation.get("memory_id")
        reason = operation.get("reason", "")

        try:
            if action == "UPDATE":
                new_memory = operation.get("new_memory")
                LOG.add(f"[MEMORY] UPDATE {memory_id} -> {new_memory}  (reason: {reason})")
                mem0.update(memory_id=memory_id, text=new_memory)

            elif action == "DELETE":
                LOG.add(f"[MEMORY] DELETE {memory_id}  (reason: {reason})")
                mem0.delete(memory_id=memory_id)

            elif action == "SKIP":
                LOG.add(f"[MEMORY] SKIP {memory_id}")

        except Exception as e:
            LOG.add(f"[MEMORY ERROR] {action} {memory_id}: {e}")

    if plan.get("add_new_memory", False):
        try:
            LOG.add(
                f"[MEMORY] ADD planner_memory={plan.get('new_memory')}  "
                f"(reason: {plan.get('reason', '')})"
            )

            mem0.add(
                [{"role": "user", "content": user_message}],
                user_id=user_id,
            )

        except Exception as e:
            LOG.add(f"[MEMORY ADD ERROR] {e}")


# ============================================================
# WRITE PATH (runs in a background thread)
# ============================================================
def memory_write(user_message: str, user_id: str) -> None:
    try:
        LOG.add(f"--- write path: {user_message[:80]!r}")

        should_store, _ = should_store_memory(user_message)

        if not should_store:
            LOG.add("[WRITE PATH] SKIP (AnyJev)")
            return

        candidates = search_memories(
            user_message=user_message,
            user_id=user_id,
            limit=MEMORY_WRITE_TOP_K,
        )

        LOG.add(f"[WRITE PATH] retrieved={len(candidates)}")
        for m in candidates:
            LOG.add(f"    {m.get('id')} | score={m.get('score')} | {m.get('memory')}")

        if not candidates:
            LOG.add("[WRITE PATH] No candidates -> ADD")
            apply_memory_operations(
                plan={
                    "operations": [],
                    "add_new_memory": True,
                    "new_memory": user_message,
                    "reason": "No existing memory candidates were retrieved.",
                },
                user_message=user_message,
                user_id=user_id,
            )
            return

        LOG.add("[WRITE PATH] Running multi-memory planner...")

        raw_plan = plan_memory_operations(user_message, candidates)
        LOG.add("[PLANNER RAW] " + json.dumps(raw_plan, ensure_ascii=False))

        plan = validate_memory_plan(raw_plan, candidates)
        LOG.add("[PLANNER VALIDATED] " + json.dumps(plan, ensure_ascii=False))

        apply_memory_operations(plan, user_message, user_id)

    except Exception as e:
        LOG.add(f"[WRITE PATH ERROR] {e}")


def start_memory_write(user_message: str, user_id: str) -> None:
    threading.Thread(
        target=memory_write,
        args=(user_message, user_id),
        daemon=True,
    ).start()


# ============================================================
# CONVERSATION SUMMARIZER
# ============================================================

def format_transcript(messages: List[Dict[str, str]]) -> str:
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages
    )


def update_summary(
    previous_summary: str,
    new_messages: List[Dict[str, str]],
) -> str:
    prompt = f"""
PREVIOUS SUMMARY:
{previous_summary or "(empty)"}

NEW MESSAGES:
{format_transcript(new_messages)}

UPDATED SUMMARY:
"""
    response = summarizer_llm.invoke(
        [
            SystemMessage(content=SUMMARY_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    return llm_text(response.content).strip()


def maybe_summarize(
    state: ConversationSummary,
    messages: List[Dict[str, str]],
) -> None:
    """Fold older messages into the summary when the unsummarized
    backlog grows too large. Runs in a background thread."""

    if not state.try_start():
        return

    try:
        previous_summary, upto = state.summary, state.upto
        pending = messages[upto:]

        if len(pending) <= SUMMARY_TRIGGER_MESSAGES:
            return

        to_summarize = pending[:-HISTORY_RECENT_MESSAGES]
        if not to_summarize:
            return

        LOG.add(f"[SUMMARY] folding {len(to_summarize)} messages")
        new_summary = update_summary(previous_summary, to_summarize)

        if new_summary:
            state.update(new_summary, upto + len(to_summarize))
            LOG.add(f"[SUMMARY] updated ({len(new_summary)} chars)")

    except Exception as e:
        LOG.add(f"[SUMMARY ERROR] {e}")
    finally:
        state.finish()


def start_summarization(
    state: ConversationSummary,
    messages: List[Dict[str, Any]],
) -> None:
    snapshot = [{"role": m["role"], "content": m["content"]} for m in messages]
    threading.Thread(
        target=maybe_summarize,
        args=(state, snapshot),
        daemon=True,
    ).start()


# ============================================================
# READ PATH
# ============================================================
def read_memory_context(
    user_message: str,
    user_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (raw_retrieved, reranked_top_k)."""

    memories = search_memories(
        user_message=user_message,
        user_id=user_id,
        limit=MEMORY_LIMIT,
    )

    if not memories:
        return [], []

    reranked = rerank_memories(
        user_message=user_message,
        memories=memories,
        top_k=MEMORY_READ_TOP_K,
    )

    return memories, reranked


def build_messages(
    user_message: str,
    memories: List[Dict[str, Any]],
    summary: str = "",
    history: Optional[List[Dict[str, str]]] = None,
) -> list:
    history = history or []

    if memories:
        memory_context = "\n".join(
            f"{i}. {m.get('memory', '')}"
            for i, m in enumerate(memories, start=1)
        )
    else:
        memory_context = "No relevant long-term memories were retrieved."

    context_block = f"""
============================================================
SUMMARY OF EARLIER CONVERSATION
============================================================
{summary or "(none yet)"}

============================================================
RELEVANT LONG-TERM MEMORIES ABOUT THE USER
============================================================
{memory_context}
"""

    messages = [
        SystemMessage(content=MAIN_SYSTEM_PROMPT + "\n\n" + context_block)
    ]
    for m in history:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        messages.append(cls(content=m["content"]))
    messages.append(HumanMessage(content=user_message))
    return messages


def stream_answer(messages: list):
    """Yields text chunks only (skips reasoning/thinking tokens)."""
    try:
        for chunk in main_llm.stream(messages):
            text = llm_text(chunk.content)
            if text:
                yield text
    except Exception as e:
        LOG.add(f"[MAIN LLM ERROR] {e}")
        yield "\n\nSorry, I encountered an error while generating the response."


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []  # {role, content, memories?}

if "summary_state" not in st.session_state:
    st.session_state.summary_state = ConversationSummary()

summary_state: ConversationSummary = st.session_state.summary_state


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🧠 Memory")

    user_id = st.text_input("User ID", value=DEFAULT_USER_ID)
    show_used = st.toggle("Show memories used per answer", value=True)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        summary_state.reset()
        st.rerun()

    st.divider()
    st.subheader("Write-path activity")

    @st.fragment(run_every=2)
    def activity_panel():
        text = LOG.dump()
        st.code(text or "(no activity yet)", language="text", height=300)

    activity_panel()

    if st.button("Clear activity log", use_container_width=True):
        LOG.clear()
        st.rerun()

    st.divider()

    with st.expander("Conversation summary"):
        st.caption(
            f"{summary_state.upto} of {len(st.session_state.messages)} "
            f"messages summarized"
        )
        st.write(summary_state.summary or "(no summary yet)")

    with st.expander("Stored memories"):
        if st.button("Refresh", key="refresh_mem", use_container_width=True):
            pass  # a click triggers a rerun; list below re-fetches

        try:
            result = mem0.get_all(filters={"user_id": user_id})
            items = result.get("results", []) if isinstance(result, dict) else result
            items = [normalize_memory(m) for m in items if isinstance(m, dict)]

            if not items:
                st.caption("No memories stored.")

            for m in items:
                st.markdown(f"- {m['memory']}")
                st.caption(f"id: {m['id']}")

        except Exception as e:
            st.error(f"Could not load memories: {e}")


# ============================================================
# CHAT UI
# ============================================================

st.title("Agentic Memory Assistant")
st.caption("Answers use long-term memory; new facts are saved in the background.")


def render_memories(memories: List[Dict[str, Any]]) -> None:
    with st.expander(f"Memories used ({len(memories)})"):
        if not memories:
            st.caption("None retrieved.")
        for m in memories:
            st.markdown(f"- {m.get('memory', '')}")
            st.caption(
                f"mem0 score: {m.get('score')} · "
                f"rerank: {m.get('rerank_score')}"
            )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if (
            show_used
            and msg["role"] == "assistant"
            and msg.get("memories") is not None
        ):
            render_memories(msg["memories"])


prompt = st.chat_input("Say something...")


if prompt:
    summary_text, history = summary_state.get_context(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Retrieval still blocks, but it's before the first token
        with st.spinner("Retrieving memories..."):
            _, reranked = read_memory_context(prompt, user_id)

        messages = build_messages(prompt, reranked, summary_text, history)
        answer = st.write_stream(stream_answer(messages))

        if show_used:
            render_memories(reranked)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "memories": reranked}
    )

    start_memory_write(prompt, user_id)
    start_summarization(summary_state, st.session_state.messages)
