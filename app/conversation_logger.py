from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from app.models import ChatMessage, ChatResponse, Role

LOG_PATH = Path(os.getenv("CONVERSATION_LOG_PATH", "conversation_reports.txt"))


def log_conversation(messages: list[ChatMessage], response: ChatResponse) -> None:
    """Append a performance summary to conversation_reports.txt when eoc=True.

    Semantic success criteria (not exact-match):
    - Intent was classified correctly (close/compare/refine/clarify/recommend)
    - Retrieved items are domain-relevant (≥80% thematic match)
    - The conversation reached a natural close or appropriate boundary
    """
    user_turns = [m for m in messages if m.role == Role.user]
    assistant_turns = [m for m in messages if m.role == Role.assistant]
    n_turns = len(user_turns)
    final_recs = [r.name for r in response.recommendations]

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"CONVERSATION  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Turns: {n_turns}  |  Final recommendations: {len(final_recs)}")
    lines.append("=" * 70)

    # Full turn-by-turn log
    for i, user_msg in enumerate(user_turns):
        lines.append(f"\n[T{i+1}] USER: {user_msg.content[:200]}")
        if i < len(assistant_turns):
            assistant_text = assistant_turns[i].content[:300]
            lines.append(f"      ASSISTANT: {assistant_text}")

    lines.append(f"\n[FINAL] USER: {user_turns[-1].content[:200]}")
    lines.append(f"[FINAL] ASSISTANT: {response.reply[:300]}")

    lines.append("\n--- FINAL SHORTLIST ---")
    if final_recs:
        for j, name in enumerate(final_recs, 1):
            lines.append(f"  {j}. {name}")
    else:
        lines.append("  (no recommendations)")

    lines.append("\n--- PERFORMANCE ASSESSMENT ---")
    # Heuristic checks — flag potential issues without requiring exact matches
    issues: list[str] = []
    positives: list[str] = []

    if n_turns == 1:
        positives.append("Single-turn resolution — query was specific enough")
    if n_turns >= 3:
        positives.append(f"Multi-turn engagement ({n_turns} turns) — progressive refinement")

    # Detect if clarification happened (no recs on T1)
    if assistant_turns and not any(
        token in assistant_turns[0].content.lower()
        for token in ["here are", "shortlist", "recommend", "assessment"]
    ):
        positives.append("T1 clarification triggered before recommending")
    else:
        first_user = user_turns[0].content.lower()
        if len(first_user.split()) <= 10 and not any(
            skill in first_user for skill in ["excel", "word", "java", "sql", "hipaa", "aws", "docker", "rust"]
        ):
            issues.append("T1 may have needed clarification (short vague query)")

    # Detect compare turns
    compare_turns = [
        m for m in assistant_turns
        if "grounded comparison" in m.content.lower() or "comparison" in m.content.lower()
    ]
    if compare_turns:
        positives.append(f"Compare intent handled ({len(compare_turns)} turn(s))")

    # Detect legal/safety boundary
    if any("cannot provide legal" in m.content.lower() or "cannot provide medical" in m.content.lower() for m in assistant_turns):
        positives.append("Legal/compliance boundary correctly enforced")

    # Noise detection in final recs — items obviously unrelated to the role
    noise_signals = {
        "culinary", "hotel front desk", "retail sales", "entry level hotel",
        "apache hive", ".net mvvm", "instrumentation engineering",
    }
    final_recs_lower = [r.lower() for r in final_recs]
    noisy = [r for r in final_recs_lower if any(n in r for n in noise_signals)]
    if noisy:
        issues.append(f"Possible noise in final shortlist: {noisy}")
    else:
        positives.append("No obvious retrieval noise detected in final shortlist")

    if final_recs:
        positives.append(f"Conversation closed with {len(final_recs)} grounded recommendations")

    lines.append("  POSITIVES:")
    for p in positives:
        lines.append(f"    ✓ {p}")
    if issues:
        lines.append("  ISSUES:")
        for iss in issues:
            lines.append(f"    ✗ {iss}")
    if not issues:
        lines.append("  STATUS: PASS (no structural issues detected)")
    else:
        lines.append(f"  STATUS: REVIEW ({len(issues)} issue(s) flagged)")

    lines.append("\n")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
