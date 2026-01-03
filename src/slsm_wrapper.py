# src/slsm_wrapper.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol
import json

Role = Literal["system", "user", "assistant"]

class LLM(Protocol):
    """Model-agnostic interface."""
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        ...

@dataclass
class SemanticState:
    # Keep it simple but extensible
    facts: List[Dict[str, Any]] = field(default_factory=list)
    assumptions: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    unknowns: List[Dict[str, Any]] = field(default_factory=list)
    edits: List[Dict[str, Any]] = field(default_factory=list)
    plan: Dict[str, Any] = field(default_factory=lambda: {"mode": "proceed", "reasons": [], "required_fixes": []})

    def to_compact_note(self, max_items: int = 6) -> str:
        """Render a very short memory note to minimize interference."""
        def take(xs): return xs[:max_items]

        lines = []
        if self.constraints:
            lines.append("Constraints:")
            for c in take(self.constraints):
                txt = c.get("text", "").strip()
                st = c.get("status", "")
                if txt:
                    lines.append(f"- [{st}] {txt}")
        if self.edits:
            lines.append("Latest edits:")
            for e in take(self.edits):
                lines.append(f"- turn {e.get('turn')}: {e.get('to','')}".strip())
        if self.facts:
            lines.append("User facts/preferences:")
            for f in take(self.facts):
                lines.append(f"- {f.get('text','')}".strip())

        # Required fixes only when needed
        rf = self.plan.get("required_fixes", []) if isinstance(self.plan, dict) else []
        if rf:
            lines.append("Required fixes before answering:")
            for x in take(rf):
                lines.append(f"- {str(x).strip()}")

        return "\n".join(lines).strip()

@dataclass
class SLSMConfig:
    controller_model_temp: float = 0.0
    controller_max_tokens: int = 1200

    # Injection policy: how much we perturb the final underlying prompt
    inject: Literal["never", "on_risk", "always"] = "on_risk"
    # Define what counts as "risk" that warrants injection
    risk_modes: tuple = ("verify", "clarify")

    # Keep the memory note small
    note_max_items: int = 6

CONTROLLER_SYSTEM = (
    "You are a strict semantic-state tracker for multi-turn conversations. "
    "Output ONLY valid JSON that matches the requested schema. No extra text."
)

def _controller_prompt(history: List[Dict[str, str]], new_msg: Dict[str, str], prev_state: Optional[Dict[str, Any]]) -> str:
    """
    Controller sees: truncated history + current new msg + previous state.
    It outputs updated SemanticState JSON.
    """
    prev_state_json = json.dumps(prev_state or {}, ensure_ascii=False)
    # Keep history in a compact text form to save tokens
    hist_txt = []
    for m in history:
        hist_txt.append(f"{m['role'].upper()}: {m['content']}")
    hist_txt = "\n\n".join(hist_txt)

    new_txt = f"{new_msg['role'].upper()}: {new_msg['content']}"

    schema = {
        "facts": [{"id":"F1","text":"...","support_turns":[1,2]}],
        "assumptions": [{"id":"A1","text":"...","status":"valid|contradicted|closed"}],
        "constraints": [{"id":"C1","text":"...","status":"satisfied|violated|closed"}],
        "unknowns": [{"id":"U1","text":"...","status":"open|closed"}],
        "edits": [{"turn": 3, "type":"revision|override", "from":"...", "to":"..."}],
        "plan": {"mode":"proceed|verify|clarify", "reasons":["..."], "required_fixes":["..."]}
    }

    return f"""
You will update a semantic state for a multi-turn conversation.

[PREVIOUS_STATE_JSON]
{prev_state_json}

[CONVERSATION_SO_FAR]
{hist_txt}

[NEW_MESSAGE]
{new_txt}

Your job:
1) Extract stable user facts/preferences as facts.
2) Track constraints explicitly stated in the conversation (format/style/forbidden words/requirements).
3) Track edits/revisions: if user changes a requirement, add an edit and update relevant facts/constraints.
4) Mark contradictions as assumption.status="contradicted" or constraint.status="violated".
5) Set plan.mode:
   - "verify" if any contradiction/violation exists that can be resolved using existing context.
   - "clarify" if a task-critical unknown remains open (but do NOT ask the user; instead add required_fixes that avoid guessing).
   - "proceed" otherwise.
6) required_fixes must be concrete, action-oriented instructions for the answering model.

Output JSON ONLY with this schema (no additional keys):
{json.dumps(schema, ensure_ascii=False)}
""".strip()

class SLSMController:
    """
    Tracks semantic state using a controller LLM (cheap, fixed).
    """
    def __init__(self, controller_llm: LLM, cfg: SLSMConfig):
        self.llm = controller_llm
        self.cfg = cfg

    def update(self, history: List[Dict[str, str]], new_msg: Dict[str, str], prev: Optional[SemanticState]) -> SemanticState:
        prompt = _controller_prompt(history=history, new_msg=new_msg, prev_state=(prev.__dict__ if prev else None))
        messages = [
            {"role": "system", "content": CONTROLLER_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        raw = self.llm.generate(messages, temp=self.cfg.controller_model_temp, max_tokens=self.cfg.controller_max_tokens)

        # Strict JSON parse
        try:
            data = json.loads(raw)
        except Exception as e:
            # Fail-closed: if controller fails, do not inject anything downstream
            return prev or SemanticState(plan={"mode":"proceed","reasons":[f"controller_json_parse_failed: {e}"],"required_fixes":[]})

        # Convert to dataclass while keeping unknown keys out
        st = SemanticState(
            facts=data.get("facts", []),
            assumptions=data.get("assumptions", []),
            constraints=data.get("constraints", []),
            unknowns=data.get("unknowns", []),
            edits=data.get("edits", []),
            plan=data.get("plan", {"mode":"proceed","reasons":[],"required_fixes":[]}),
        )
        return st

class SLSMWrapper:
    """
    Shadow-tracks state turn-by-turn, then minimally conditions the final answer generation.
    """
    def __init__(self, controller: SLSMController, cfg: SLSMConfig):
        self.controller = controller
        self.cfg = cfg

    def track_state(self, conversation: List[Dict[str, str]]) -> SemanticState:
        """
        conversation: full multi-turn list of messages.
        Returns final SemanticState after processing all turns.
        """
        st: Optional[SemanticState] = None
        history: List[Dict[str, str]] = []
        for msg in conversation:
            st = self.controller.update(history=history, new_msg=msg, prev=st)
            history.append(msg)
        return st or SemanticState()

    def build_final_messages(
        self,
        original_conversation: List[Dict[str, str]],
        state: SemanticState,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Returns messages to send to the underlying LLM for final response.
        Minimally perturbs the conversation by optionally adding a short memory note.
        """
        msgs: List[Dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})

        inject_note = False
        mode = (state.plan or {}).get("mode", "proceed")
        if self.cfg.inject == "always":
            inject_note = True
        elif self.cfg.inject == "on_risk" and mode in self.cfg.risk_modes:
            inject_note = True
        elif self.cfg.inject == "never":
            inject_note = False

        if inject_note:
            note = state.to_compact_note(max_items=self.cfg.note_max_items)
            if note:
                # Put note as a system message *after* system_prompt to guide without rewriting user text.
                # Alternative is as a first user message; this typically has more influence (and more interference).
                msgs.append({"role": "system", "content": f"[SLSM MEMORY NOTE]\n{note}\n\nFollow the constraints above."})

        # Append original conversation verbatim (no information loss)
        msgs.extend(original_conversation)
        return msgs

    def generate_last_turn(
        self,
        underlying_llm: LLM,
        original_conversation: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **gen_kwargs,
    ) -> str:
        """
        Main entry: track state, then generate final response from underlying model.
        """
        state = self.track_state(original_conversation)
        msgs = self.build_final_messages(original_conversation, state, system_prompt=system_prompt)
        return underlying_llm.generate(msgs, **gen_kwargs)
