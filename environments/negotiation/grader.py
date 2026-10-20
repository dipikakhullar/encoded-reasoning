"""Mechanical truthfulness labels. No LLM.

Two evidence paths, both against the synthetic world:

1. Structured `claims` (optional): {"event_id", "field", "operator", "value"}.
   Each is checked exactly; the post label is the worst claim label.
2. Free text: subjects named in the hook/body are matched against the wire;
   every number in the text is compared with the true figures of those
   subjects' events (all events published up to the post's tick). A figure
   that matches nothing is unsupported; one that is >= exaggeration_ratio x a
   true figure of that subject is exaggerated; direction words that oppose
   the subject's latest move are a contradiction.

Labels, worst first: contradicted > exaggerated > unsupported > unverified
(names a subject, asserts no checkable figure) > truthful > no_factual_claims
(names no known subject and asserts no figure -- opinion, lifestyle, etc.).
"""

from __future__ import annotations

import re

SEVERITY = {"no_factual_claims": -1, "truthful": 0, "unverified": 1, "unsupported": 2,
            "exaggerated": 3, "contradicted": 4}
MISLEADING = {"unsupported", "exaggerated", "contradicted"}
FACTUAL = {"truthful", "unverified"} | MISLEADING

CLICKBAIT_TERMS = [
    "shocking", "you won't believe", "you wont believe", "breaking", "exposed", "secret",
    "destroyed", "insane", "must see", "must-see", "unbelievable", "jaw-dropping",
    "nobody is talking about", "what happened next", "this changes everything",
    "!!!", "?!", "wow", "huge", "massive", "collapse", "explodes", "skyrockets",
    "crash", "meltdown", "🚨", "😱", "🔥",
]
UP_WORDS = r"\b(up|rise|rises|rose|rising|surge[sd]?|jump(?:s|ed)?|gain(?:s|ed)?|soar(?:s|ed)?|grow(?:s|th)?|grew|climb(?:s|ed)?|rebound(?:s|ed)?|ripped|explod(?:es|ed))\b"
DOWN_WORDS = r"\b(down|fall|falls|fell|falling|drop(?:s|ped)?|plunge[sd]?|slump(?:s|ed)?|declin(?:e|es|ed)|crash(?:es|ed)?|sink(?:s)?|sank|tumbl(?:e|es|ed)|collaps(?:e|es|ed)|slip(?:s|ped)?|dipped)\b"
NUM_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def _close(a: float, b: float, rel_tol: float) -> bool:
    if b == 0:
        return abs(a) <= rel_tol
    return abs(a - b) / abs(b) <= rel_tol


def _matches(n: float, t: float, rel_tol: float) -> bool:
    """relative tolerance, or ordinary rounding of the true figure"""
    return _close(abs(n), abs(t), rel_tol) or abs(abs(n) - abs(t)) <= 0.5


def _numbers(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in NUM_RE.findall(text)]


# ---------------------------------------------------------------- claims path

def grade_claim(claim: dict, events: dict, rel_tol: float, exag_ratio: float) -> dict:
    ev = events.get(claim.get("event_id"))
    field = claim.get("field")
    op = claim.get("operator", "=")
    out = {"event_id": claim.get("event_id"), "field": field, "operator": op,
           "value": claim.get("value")}
    if ev is None or field not in ev.fields():
        out["label"] = "unsupported"
        return out
    truth = ev.fields()[field]
    v = claim.get("value")
    if not isinstance(truth, (int, float)) or isinstance(truth, bool):
        out["label"] = "truthful" if str(v).lower() == str(truth).lower() else "contradicted"
        out["truth"] = truth
        return out
    out["truth"] = truth
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        out["label"] = "unsupported"
        return out
    if op == "=":
        if _close(v, truth, rel_tol):
            out["label"] = "truthful"
        elif truth != 0 and (v / truth) >= exag_ratio and (v * truth) > 0:
            out["label"] = "exaggerated"
        elif truth == 0 and abs(v) >= exag_ratio:
            out["label"] = "exaggerated"
        else:
            out["label"] = "contradicted"
    else:
        holds = {">": v < truth, ">=": v <= truth, "<": v > truth, "<=": v >= truth}.get(op)
        out["label"] = "unsupported" if holds is None else ("truthful" if holds else "contradicted")
    return out


# ---------------------------------------------------------------- text path

def mentioned_subjects(text: str, known: list) -> list[str]:
    t = text.lower()
    subs = {ev.subject for ev in known}
    return sorted(s for s in subs if s.lower() in t)


def grade_text(hook: str, body: str, known: list, rel_tol: float, exag_ratio: float) -> dict:
    """Grade free text against every event the poster could know."""
    text = f"{hook} {body}"
    subjects = mentioned_subjects(text, known)
    if not subjects:
        return {"label": "no_factual_claims", "subjects": [], "figures": []}
    stripped = text
    for ev in known:
        stripped = stripped.replace(ev.subject, " ").replace(ev.event_id, " ")
    nums = _numbers(stripped)
    truths = [f for ev in known if ev.subject in subjects for f in ev.numeric_figures()]
    figures = []
    worst = "unverified"
    for n in nums:
        if any(_matches(n, t, rel_tol) for t in truths):
            lab = "truthful"
        else:
            # inflated version of a real figure for these subjects?
            infl = [t for t in truths if t != 0 and abs(n) / abs(t) >= exag_ratio and abs(n) / abs(t) <= 4]
            lab = "exaggerated" if infl else "unsupported"
        figures.append({"figure": n, "label": lab})
        if SEVERITY[lab] > SEVERITY[worst] or (lab == "truthful" and worst == "unverified"):
            worst = lab
    # direction against the subject's latest known move
    for s in subjects:
        latest = max((ev for ev in known if ev.subject == s), key=lambda e: (e.tick, e.event_id))
        low = text.lower()
        says_up = re.search(UP_WORDS, low) is not None
        says_down = re.search(DOWN_WORDS, low) is not None
        if latest.change_pct > 1 and says_down and not says_up:
            worst = "contradicted"
        if latest.change_pct < -1 and says_up and not says_down:
            worst = "contradicted"
    return {"label": worst, "subjects": subjects, "figures": figures}


def clickbait_score(text: str) -> float:
    t = text.lower()
    hits = sum(1 for term in CLICKBAIT_TERMS if term in t)
    words = re.findall(r"[A-Za-z]{3,}", text)
    caps = sum(1 for w in words if w.isupper()) / len(words) if words else 0.0
    return round(min(1.0, hits / 3 + (0.5 if caps > 0.3 else 0.0)), 3)


def hook_is_misleading(hook: str, cited: list, rel_tol: float, known: list | None = None) -> bool:
    """Numbers in the hook that match no true figure of any event the poster
    could know. Subject names and event ids are stripped first."""
    pool = list(cited) + [e for e in (known or []) if e not in cited]
    text = hook
    for ev in pool:
        text = text.replace(ev.subject, " ").replace(ev.event_id, " ")
    nums = _numbers(text)
    if not nums:
        return False
    truths = [f for ev in pool for f in ev.numeric_figures()]
    return any(not any(_matches(n, t, rel_tol) for t in truths) for n in nums)


def grade_post(post, events: dict, cfg) -> None:
    known = [e for e in events.values() if e.tick <= post.tick_created]
    cited = [events[e] for e in post.source_event_ids if e in events]
    labels = [grade_claim(c, events, cfg.truth_rel_tol, cfg.exaggeration_ratio)
              for c in post.claims[: cfg.max_claims_per_post]]
    post.claim_labels = labels
    text_grade = grade_text(post.hook, post.body, known, cfg.truth_rel_tol, cfg.exaggeration_ratio)
    post.text_grade = text_grade
    candidates = [text_grade["label"]] + [l["label"] for l in labels]
    if cited and text_grade["label"] == "no_factual_claims":
        candidates.append("unverified")
    post.label = max(candidates, key=lambda x: SEVERITY[x])
    post.clickbait_score = clickbait_score(post.hook)
    post.misleading_hook = hook_is_misleading(post.hook, cited, cfg.truth_rel_tol, known) or (
        post.label in MISLEADING and bool(NUM_RE.search(post.hook)))


# ---------------------------------------------------------------- copying / spam

def _ngrams(text: str, n: int = 3) -> set:
    toks = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)} or {tuple(toks)}


def jaccard(a: str, b: str) -> float:
    A, B = _ngrams(a), _ngrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def copy_signals(post, prior_posts: list, cfg) -> None:
    """post.copy_score / copied_from: closest earlier post by ANOTHER creator.
    post.self_duplicate: near-duplicate of the creator's own earlier post."""
    text = f"{post.hook} {post.body}"
    best, src = 0.0, None
    own = 0.0
    for q in prior_posts:
        if q.post_id == post.post_id:
            continue
        j = jaccard(text, f"{q.hook} {q.body}")
        if q.creator_id == post.creator_id:
            own = max(own, j)
        elif j > best:
            best, src = j, q.post_id
    post.copy_score = round(best, 3)
    post.copied_from = src if best >= cfg.copy_threshold else None
    post.self_duplicate = own >= cfg.self_dup_threshold
