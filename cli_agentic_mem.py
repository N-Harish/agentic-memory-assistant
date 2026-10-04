import os
import asyncio
import json
import re
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from mem0 import MemoryClient

from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend

from sentence_transformers import CrossEncoder


# ============================================================
# ENV
# ============================================================

load_dotenv()

MEM0_API_KEY = os.getenv("MEM0_API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

if not MEM0_API_KEY:
    raise ValueError("MEM0_API_KEY is not set")

if not OLLAMA_API_KEY:
    raise ValueError("OLLAMA_API_KEY is not set")


# ============================================================
# CONFIG
# ============================================================

USER_ID = "demo-user"

# ------------------------------------------------------------
# Read path
# ------------------------------------------------------------

MEMORY_LIMIT = 10
MEMORY_READ_TOP_K = 5

# ------------------------------------------------------------
# Write path
# ------------------------------------------------------------

MEMORY_WRITE_TOP_K = 10

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

MAIN_MODEL = "gpt-oss:20b"
MEMORY_REASONING_MODEL = "gpt-oss:20b"

ANYJEV_MODEL = "Qwen/Qwen3.5-4B"

RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"

# ------------------------------------------------------------
# AnyJev threshold
# ------------------------------------------------------------

STORE_THRESHOLD = 0.5


# ============================================================
# CLIENTS
# ============================================================

mem0 = MemoryClient(
    api_key=MEM0_API_KEY
)


# ============================================================
# MAIN LLM
# ============================================================

main_llm = ChatOllama(
    model=MAIN_MODEL,
    base_url="https://ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {OLLAMA_API_KEY}"
        }
    },
    temperature=0.2,
)


# ============================================================
# MEMORY REASONING LLM
# ============================================================

memory_reasoning_llm = ChatOllama(
    model=MEMORY_REASONING_MODEL,
    base_url="https://ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {OLLAMA_API_KEY}"
        }
    },
    temperature=0.0,
)


# ============================================================
# RERANKER
# ============================================================

reranker = CrossEncoder(
    RERANKER_MODEL,
    device="cuda",
)


# ============================================================
# ANYJEV
# ============================================================

router_backend = HFBackend(
    ANYJEV_MODEL
)

decider = Decider(
    router_backend
)


# ============================================================
# STATE
# ============================================================

class MemoryState(TypedDict, total=False):

    user_id: str

    user_message: str
    assistant_message: str

    # --------------------------------------------------------
    # Read path
    # --------------------------------------------------------

    retrieved_memories: List[Dict[str, Any]]
    reranked_memories: List[Dict[str, Any]]
    filtered_memories: List[Dict[str, Any]]

    # --------------------------------------------------------
    # Write path
    # --------------------------------------------------------

    write_candidates: List[Dict[str, Any]]
    memory_operations: List[Dict[str, Any]]

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    answer: str

    # --------------------------------------------------------
    # AnyJev
    # --------------------------------------------------------

    should_store: bool
    store_probability: float

    # --------------------------------------------------------
    # Planner
    # --------------------------------------------------------

    add_new_memory: bool
    new_memory: str
    memory_reason: str


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
# MAIN SYSTEM PROMPT
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


# ============================================================
# MEMORY RELATIONSHIP / PLANNER PROMPT
# ============================================================

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


# ============================================================
# HELPERS
# ============================================================

def extract_json(text: str) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from an LLM response.
    """

    if not text:
        raise ValueError("Empty LLM response")

    text = text.strip()

    # Remove markdown fences if present.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # First attempt: direct parse.
    try:
        result = json.loads(text)

        if not isinstance(result, dict):
            raise ValueError(
                "Planner response is not a JSON object"
            )

        return result

    except json.JSONDecodeError:
        pass

    # Second attempt: find first JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"Could not find JSON object in response:\n{text}"
        )

    candidate = text[start:end + 1]

    result = json.loads(candidate)

    if not isinstance(result, dict):
        raise ValueError(
            "Extracted planner response is not a JSON object"
        )

    return result


# ============================================================
# NORMALIZE MEMORY
# ============================================================

def normalize_memory(
    memory: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Normalize Mem0 memory objects so downstream code can rely
    on a consistent shape.
    """

    return {
        "id": (
            memory.get("id")
            or memory.get("memory_id")
        ),

        "memory": (
            memory.get("memory")
            or memory.get("text")
            or ""
        ),

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
        filters={
            "user_id": user_id
        },
        limit=limit,
    )

    if isinstance(result, dict):

        memories = result.get(
            "results",
            []
        )

    elif isinstance(result, list):

        memories = result

    else:

        memories = []

    return [
        normalize_memory(memory)
        for memory in memories
        if isinstance(memory, dict)
    ]


# ============================================================
# RERANK MEMORIES
# ============================================================

def rerank_memories(
    user_message: str,
    memories: List[Dict[str, Any]],
    top_k: int = MEMORY_READ_TOP_K,
) -> List[Dict[str, Any]]:

    if not memories:
        return []

    pairs = [
        (
            user_message,
            memory.get(
                "memory",
                ""
            ),
        )
        for memory in memories
    ]

    try:

        scores = reranker.predict(
            pairs,
            show_progress_bar=False,
        )

        reranked = []

        for memory, rerank_score in zip(
            memories,
            scores,
        ):

            item = dict(memory)

            item["rerank_score"] = float(
                rerank_score
            )

            reranked.append(item)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]

    except Exception as e:

        print(
            f"[RERANKER ERROR] {e}"
        )

        return memories[:top_k]


# ============================================================
# ANYJEV STORE DECISION
# ============================================================

def should_store_memory(
    user_message: str,
) -> tuple[bool, float]:

    try:

        result = decider.decide(
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": user_message,
                    }
                ]
            },
            [
                SHOULD_STORE_QUESTION
            ],
        )

        decision = result[0]

        probability = float(
            decision.p_true
        )

        should_store = (
            probability >= STORE_THRESHOLD
        )

        print(
            f"[ANYJEV] "
            f"store={should_store} "
            f"p_store={probability:.4f}"
        )

        return (
            should_store,
            probability,
        )

    except Exception as e:

        print(
            f"[ANYJEV ERROR] {e}"
        )

        return False, 0.0


# ============================================================
# MEMORY PLANNER
# ============================================================

def plan_memory_operations(
    user_message: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:

    candidate_text = []

    for idx, memory in enumerate(
        candidates,
        start=1,
    ):

        candidate_text.append(
            f"""
Candidate {idx}
memory_id: {memory.get("id")}
memory: {memory.get("memory")}
"""
        )

    candidates_block = "\n".join(
        candidate_text
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The complete planner rules are already in the SYSTEM
    # message. Do not repeat MEMORY_RELATIONSHIP_PROMPT here.
    #
    # Also intentionally do NOT expose Mem0 retrieval scores
    # to the reasoning model.
    # --------------------------------------------------------

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
            SystemMessage(
                content=MEMORY_RELATIONSHIP_PROMPT
            ),
            HumanMessage(
                content=prompt
            ),
        ]
    )

    content = response.content

    if isinstance(
        content,
        list,
    ):

        content = "".join(
            part.get(
                "text",
                ""
            )
            if isinstance(part, dict)
            else str(part)
            for part in content
        )

    return extract_json(
        content
    )


# ============================================================
# VALIDATE MEMORY PLAN
# ============================================================

def validate_memory_plan(
    plan: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:

    candidate_ids = {
        str(memory.get("id"))
        for memory in candidates
        if memory.get("id") is not None
    }

    operations = plan.get(
        "operations",
        [],
    )

    if not isinstance(
        operations,
        list,
    ):

        operations = []

    validated_operations = []

    seen_ids = set()

    for operation in operations:

        if not isinstance(
            operation,
            dict,
        ):

            continue

        action = str(
            operation.get(
                "action",
                "SKIP",
            )
        ).upper()

        memory_id = operation.get(
            "memory_id"
        )

        if memory_id is None:
            continue

        memory_id = str(
            memory_id
        )

        # ----------------------------------------------------
        # Security / correctness:
        #
        # Never allow the LLM to mutate a memory that wasn't
        # actually supplied as a candidate.
        # ----------------------------------------------------

        if memory_id not in candidate_ids:

            print(
                f"[PLANNER] Ignoring invalid memory ID: "
                f"{memory_id}"
            )

            continue

        # ----------------------------------------------------
        # Prevent duplicate operations.
        # ----------------------------------------------------

        if memory_id in seen_ids:

            print(
                f"[PLANNER] Duplicate operation ignored: "
                f"{memory_id}"
            )

            continue

        # ----------------------------------------------------
        # Validate action.
        # ----------------------------------------------------

        if action not in {
            "UPDATE",
            "DELETE",
            "SKIP",
        }:

            action = "SKIP"

        new_memory = operation.get(
            "new_memory"
        )

        # ----------------------------------------------------
        # DELETE / SKIP cannot carry replacement text.
        # ----------------------------------------------------

        if action != "UPDATE":

            new_memory = None

        # ----------------------------------------------------
        # UPDATE requires replacement text.
        # ----------------------------------------------------

        elif (
            not isinstance(
                new_memory,
                str,
            )
            or not new_memory.strip()
        ):

            print(
                f"[PLANNER] Invalid UPDATE without "
                f"new_memory for {memory_id}"
            )

            action = "SKIP"

            new_memory = None

        validated_operations.append(
            {
                "action": action,
                "memory_id": memory_id,
                "new_memory": new_memory,
                "reason": operation.get(
                    "reason",
                    "",
                ),
            }
        )

        seen_ids.add(
            memory_id
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Any candidate omitted by the planner automatically becomes
    # SKIP.
    #
    # This guarantees every candidate has exactly one operation.
    # --------------------------------------------------------

    for memory in candidates:

        memory_id = memory.get(
            "id"
        )

        if memory_id is None:
            continue

        memory_id = str(
            memory_id
        )

        if memory_id not in seen_ids:

            validated_operations.append(
                {
                    "action": "SKIP",
                    "memory_id": memory_id,
                    "new_memory": None,
                    "reason": (
                        "Candidate omitted by planner; "
                        "defaulting to SKIP."
                    ),
                }
            )

    # --------------------------------------------------------
    # Validate ADD.
    # --------------------------------------------------------

    add_new_memory = bool(
        plan.get(
            "add_new_memory",
            False,
        )
    )

    new_memory = plan.get(
        "new_memory"
    )

    if add_new_memory:

        if (
            not isinstance(
                new_memory,
                str,
            )
            or not new_memory.strip()
        ):

            print(
                "[PLANNER] ADD requested without "
                "valid new_memory. Disabling ADD."
            )

            add_new_memory = False

            new_memory = None

    else:

        new_memory = None

    return {
        "operations": validated_operations,

        "add_new_memory": add_new_memory,

        "new_memory": new_memory,

        "reason": plan.get(
            "reason",
            "",
        ),
    }


# ============================================================
# APPLY MEMORY OPERATIONS
# ============================================================

def apply_memory_operations(
    plan: Dict[str, Any],
    user_message: str,
    user_id: str,
) -> None:

    operations = plan.get(
        "operations",
        [],
    )

    # ========================================================
    # UPDATE / DELETE / SKIP
    # ========================================================

    for operation in operations:

        action = operation.get(
            "action"
        )

        memory_id = operation.get(
            "memory_id"
        )

        reason = operation.get(
            "reason",
            "",
        )

        try:

            # ------------------------------------------------
            # UPDATE
            # ------------------------------------------------

            if action == "UPDATE":

                new_memory = operation.get(
                    "new_memory"
                )

                print(
                    f"[MEMORY] UPDATE "
                    f"{memory_id}"
                )

                print(
                    f"         {new_memory}"
                )

                print(
                    f"         reason={reason}"
                )

                mem0.update(
                    memory_id=memory_id,
                    text=new_memory,
                )

            # ------------------------------------------------
            # DELETE
            # ------------------------------------------------

            elif action == "DELETE":

                print(
                    f"[MEMORY] DELETE "
                    f"{memory_id}"
                )

                print(
                    f"         reason={reason}"
                )

                mem0.delete(
                    memory_id=memory_id
                )

            # ------------------------------------------------
            # SKIP
            # ------------------------------------------------

            elif action == "SKIP":

                print(
                    f"[MEMORY] SKIP "
                    f"{memory_id}"
                )

        except Exception as e:

            print(
                f"[MEMORY ERROR] "
                f"{action} "
                f"{memory_id}: {e}"
            )

    # ========================================================
    # ADD
    # ========================================================

    if plan.get(
        "add_new_memory",
        False,
    ):

        try:

            print(
                "[MEMORY] ADD"
            )

            print(
                f"         planner_memory="
                f"{plan.get('new_memory')}"
            )

            print(
                f"         reason="
                f"{plan.get('reason', '')}"
            )

            # ------------------------------------------------
            # IMPORTANT
            #
            # Store the ORIGINAL user message.
            #
            # The planner's new_memory is used only for
            # reasoning/validation.
            #
            # This prevents the planner from inventing or
            # subtly changing a personal fact.
            # ------------------------------------------------

            messages = [
                {
                    "role": "user",
                    "content": user_message,
                }
            ]

            mem0.add(
                messages,
                user_id=user_id,
            )

        except Exception as e:

            print(
                f"[MEMORY ADD ERROR] {e}"
            )


# ============================================================
# ASYNC WRITE PATH
# ============================================================

async def async_memory_write(
    state: MemoryState,
) -> None:

    user_message = state[
        "user_message"
    ]

    user_id = state[
        "user_id"
    ]

    try:

        # ====================================================
        # STEP 1: ANYJEV
        # ====================================================

        (
            should_store,
            probability,
        ) = should_store_memory(
            user_message
        )

        state[
            "should_store"
        ] = should_store

        state[
            "store_probability"
        ] = probability

        if not should_store:

            print(
                "[WRITE PATH] SKIP "
                "(AnyJev)"
            )

            return

        # ====================================================
        # STEP 2: RETRIEVE WRITE CANDIDATES
        # ====================================================

        candidates = search_memories(
            user_message=user_message,
            user_id=user_id,
            limit=MEMORY_WRITE_TOP_K,
        )

        state[
            "write_candidates"
        ] = candidates

        print(
            f"[WRITE PATH] "
            f"retrieved={len(candidates)}"
        )

        for memory in candidates:

            print(
                f"  {memory.get('id')} | "
                f"score={memory.get('score')} | "
                f"{memory.get('memory')}"
            )

        # ====================================================
        # STEP 3: NO CANDIDATES -> DIRECT ADD
        # ====================================================

        if not candidates:

            print(
                "[WRITE PATH] "
                "No candidates -> ADD"
            )

            plan = {
                "operations": [],

                "add_new_memory": True,

                "new_memory": user_message,

                "reason": (
                    "No existing memory candidates "
                    "were retrieved."
                ),
            }

            state[
                "memory_operations"
            ] = []

            state[
                "add_new_memory"
            ] = True

            state[
                "new_memory"
            ] = user_message

            state[
                "memory_reason"
            ] = plan[
                "reason"
            ]

            apply_memory_operations(
                plan=plan,
                user_message=user_message,
                user_id=user_id,
            )

            return

        # ====================================================
        # STEP 4: MULTI-MEMORY PLANNING
        # ====================================================

        print(
            "[WRITE PATH] "
            "Running multi-memory planner..."
        )

        raw_plan = plan_memory_operations(
            user_message=user_message,
            candidates=candidates,
        )

        print(
            "[PLANNER RAW]"
        )

        print(
            json.dumps(
                raw_plan,
                indent=2,
                ensure_ascii=False,
            )
        )

        # ====================================================
        # STEP 5: VALIDATE PLAN
        # ====================================================

        plan = validate_memory_plan(
            plan=raw_plan,
            candidates=candidates,
        )

        state[
            "memory_operations"
        ] = plan[
            "operations"
        ]

        state[
            "add_new_memory"
        ] = plan[
            "add_new_memory"
        ]

        state[
            "new_memory"
        ] = plan.get(
            "new_memory"
        )

        state[
            "memory_reason"
        ] = plan.get(
            "reason",
            "",
        )

        print(
            "[PLANNER VALIDATED]"
        )

        print(
            json.dumps(
                plan,
                indent=2,
                ensure_ascii=False,
            )
        )

        # ====================================================
        # STEP 6: APPLY
        # ====================================================

        apply_memory_operations(
            plan=plan,
            user_message=user_message,
            user_id=user_id,
        )

    except Exception as e:

        # ----------------------------------------------------
        # Memory must NEVER break the main assistant response.
        # ----------------------------------------------------

        print(
            f"[WRITE PATH ERROR] {e}"
        )


# ============================================================
# READ PATH
# ============================================================

def read_memory_context(
    user_message: str,
    user_id: str,
) -> List[Dict[str, Any]]:

    # ========================================================
    # STEP 1: MEM0 TOP-K
    # ========================================================

    memories = search_memories(
        user_message=user_message,
        user_id=user_id,
        limit=MEMORY_LIMIT,
    )

    print(
        f"[READ] Mem0 retrieved "
        f"{len(memories)} memories"
    )

    for memory in memories:

        print(
            f"  {memory.get('id')} | "
            f"score={memory.get('score')} | "
            f"{memory.get('memory')}"
        )

    if not memories:

        return []

    # ========================================================
    # STEP 2: CROSS-ENCODER RERANK
    # ========================================================

    reranked = rerank_memories(
        user_message=user_message,
        memories=memories,
        top_k=MEMORY_READ_TOP_K,
    )

    print(
        f"[READ] Reranked top "
        f"{len(reranked)}"
    )

    for memory in reranked:

        print(
            f"  {memory.get('id')} | "
            f"mem0={memory.get('score')} | "
            f"rerank={memory.get('rerank_score')} | "
            f"{memory.get('memory')}"
        )

    return reranked


# ============================================================
# MAIN TURN
# ============================================================

def run_turn(
    user_message: str,
    user_id: str = USER_ID,
) -> MemoryState:

    state: MemoryState = {
        "user_id": user_id,
        "user_message": user_message,
    }

    # ========================================================
    # READ PATH
    # ========================================================

    retrieved_memories = read_memory_context(
        user_message=user_message,
        user_id=user_id,
    )

    state[
        "retrieved_memories"
    ] = retrieved_memories

    state[
        "reranked_memories"
    ] = retrieved_memories

    # ========================================================
    # BUILD MEMORY CONTEXT
    # ========================================================

    if retrieved_memories:

        memory_lines = []

        for idx, memory in enumerate(
            retrieved_memories,
            start=1,
        ):

            memory_lines.append(
                f"{idx}. {memory.get('memory', '')}"
            )

        memory_context = "\n".join(
            memory_lines
        )

    else:

        memory_context = (
            "No relevant long-term memories "
            "were retrieved."
        )

    # ========================================================
    # MAIN LLM
    # ========================================================

    user_prompt = f"""
Relevant long-term memories about the user:

{memory_context}

============================================================
CURRENT USER MESSAGE
============================================================

{user_message}
"""

    try:

        response = main_llm.invoke(
            [
                SystemMessage(
                    content=MAIN_SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=user_prompt
                ),
            ]
        )

        answer = response.content

        if isinstance(
            answer,
            list,
        ):

            answer = "".join(
                part.get(
                    "text",
                    ""
                )
                if isinstance(part, dict)
                else str(part)
                for part in answer
            )

        state[
            "answer"
        ] = answer

        print(
            "\nASSISTANT:"
        )

        print(
            answer
        )

    except Exception as e:

        print(
            f"[MAIN LLM ERROR] {e}"
        )

        state[
            "answer"
        ] = (
            "Sorry, I encountered an error "
            "while generating the response."
        )

    return state


# ============================================================
# ASYNC TURN WRAPPER
# ============================================================

async def run_turn_async(
    user_message: str,
    user_id: str = USER_ID,
) -> MemoryState:

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # run_turn() performs the READ path and generates the
    # user-facing response synchronously.
    #
    # The memory write is then scheduled separately.
    # --------------------------------------------------------

    state = run_turn(
        user_message=user_message,
        user_id=user_id,
    )

    # --------------------------------------------------------
    # Start memory write in background.
    # --------------------------------------------------------

    asyncio.create_task(
        async_memory_write(
            state
        )
    )

    return state


# ============================================================
# CLI
# ============================================================

async def main():

    print(
        "Agentic Memory Assistant"
    )

    print(
        "Type 'exit' to quit."
    )

    print()

    while True:

        try:

            user_message = input(
                "\nUSER: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            break

        if not user_message:

            continue

        if user_message.lower() in {
            "exit",
            "quit",
        }:

            break

        # ====================================================
        # RUN TURN
        # ====================================================

        await run_turn_async(
            user_message=user_message,
            user_id=USER_ID,
        )

        # ====================================================
        # Give the event loop an opportunity to execute the
        # background write task.
        # ====================================================

        await asyncio.sleep(0)

    # ========================================================
    # Allow pending writes to finish before exiting.
    #
    # For a real service, replace this with a proper background
    # worker / queue rather than a fixed sleep.
    # ========================================================

    print(
        "\nWaiting for pending memory writes..."
    )

    await asyncio.sleep(
        3
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )