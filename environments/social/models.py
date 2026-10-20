from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Post:
    post_id: str
    creator_id: str
    tick_created: int
    hook: str
    body: str
    claims: list[dict] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)
    # engagement, unique per (agent, post)
    impressions: int = 0
    full_views: int = 0
    likes: int = 0
    reshares: int = 0
    reports: int = 0
    viewers: set = field(default_factory=set)
    likers: set = field(default_factory=set)
    resharers: set = field(default_factory=set)
    reporters: set = field(default_factory=set)
    # moderation
    removed: bool = False
    removed_tick: int | None = None
    # grading (hidden from agents)
    label: str = "unverified"
    claim_labels: list[dict] = field(default_factory=list)
    text_grade: dict = field(default_factory=dict)
    misleading_hook: bool = False
    clickbait_score: float = 0.0
    copy_score: float = 0.0
    copied_from: str | None = None
    self_duplicate: bool = False
    judge: dict = field(default_factory=dict)   # filled by social.judge
    revenue: float = 0.0

    @property
    def live(self) -> bool:
        return not self.removed

    def public_stats(self) -> dict:
        return {"impressions": self.impressions, "full_views": self.full_views,
                "likes": self.likes, "reshares": self.reshares}

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("viewers", "likers", "resharers", "reporters"):
            d[k] = sorted(d[k])
        return d


@dataclass
class AgentState:
    agent_id: str
    balance: float = 0.0
    followers: set = field(default_factory=set)
    following: set = field(default_factory=set)
    posts: list[str] = field(default_factory=list)
    action_budget_remaining: int = 0
    history: list[dict] = field(default_factory=list)
    # moderation
    strikes: int = 0
    false_reports: int = 0
    suppressed_until: int = -1
    # bookkeeping
    viewed: set = field(default_factory=set)          # post ids opened, ever
    ever_followed: set = field(default_factory=set)   # pairs paid once
    notifications: list[dict] = field(default_factory=list)
    hidden: set = field(default_factory=set)          # accounts this agent reported (hide_reported_accounts)
    guideline_reads: int = 0
    inbox: list[dict] = field(default_factory=list)     # DMs received since the agent's last turn
    dm_threads: dict = field(default_factory=dict)      # correspondent -> chronological DMs both ways
    posting_disabled_tick: int | None = None            # report_blocks_posting
    removed_tick: int | None = None                     # report_removes_account

    def is_active(self, tick: int) -> bool:
        return self.removed_tick is None or tick <= self.removed_tick

    def is_suppressed(self, tick: int) -> bool:
        return tick <= self.suppressed_until

    def to_dict(self) -> dict:
        return {"agent_id": self.agent_id, "balance": round(self.balance, 4),
                "followers": sorted(self.followers), "following": sorted(self.following),
                "posts": list(self.posts), "strikes": self.strikes,
                "false_reports": self.false_reports,
                "suppressed_until": self.suppressed_until,
                "hidden": sorted(self.hidden), "guideline_reads": self.guideline_reads,
                "posting_disabled_tick": self.posting_disabled_tick, "removed_tick": self.removed_tick}
