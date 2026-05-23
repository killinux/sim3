from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


class SwipeProfile(str, Enum):
    FAST_SCANNER = "fast_scanner"
    DELIBERATE_WATCHER = "deliberate_watcher"
    BINGE_VIEWER = "binge_viewer"


class ActivityLevel(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class DiversityLevel(int, Enum):
    NARROW = 1
    MODERATE = 2
    BROAD = 3


class ConformityLevel(int, Enum):
    FOLLOWER = 1
    BALANCED = 2
    INDEPENDENT = 3


ACTIVITY_DESCRIPTIONS = {
    ActivityLevel.LOW: (
        "A casual viewer who rarely opens the app. You quickly lose interest "
        "and close the app at the first sign of uninteresting content. Your "
        "sessions are very short and infrequent."
    ),
    ActivityLevel.MEDIUM: (
        "A regular viewer who browses the app a few times a day. You watch "
        "videos that catch your attention but don't hesitate to skip content "
        "that doesn't match your interests."
    ),
    ActivityLevel.HIGH: (
        "An avid viewer who spends significant time on the app daily. You "
        "are willing to explore a wide range of content and often watch many "
        "videos in a single session."
    ),
}

CONFORMITY_DESCRIPTIONS = {
    ConformityLevel.FOLLOWER: (
        "You tend to follow popular trends. Videos with high like counts and "
        "many comments strongly influence your engagement decisions."
    ),
    ConformityLevel.BALANCED: (
        "You consider both popularity signals and your own taste when "
        "deciding whether to engage with a video."
    ),
    ConformityLevel.INDEPENDENT: (
        "You rely entirely on your own judgment. Like counts and trending "
        "status have little influence on your decisions."
    ),
}

DIVERSITY_DESCRIPTIONS = {
    DiversityLevel.NARROW: (
        "You have very specific content preferences and mostly watch videos "
        "within a narrow set of categories. You rarely explore outside your "
        "comfort zone."
    ),
    DiversityLevel.MODERATE: (
        "You have core interests but occasionally explore new content "
        "categories when something catches your eye."
    ),
    DiversityLevel.BROAD: (
        "You enjoy a wide variety of content and actively seek out new and "
        "different types of videos. You are open to almost any genre."
    ),
}

CATEGORIES = [
    "comedy", "food", "dance", "music", "gaming", "pets", "fashion",
    "beauty", "sports", "education", "tech", "travel", "fitness",
    "diy", "news", "vlog", "drama", "animation", "cars", "nature",
]


@dataclass
class Demographics:
    user_id: str
    age_bucket: str
    gender: str
    city_tier: int
    device_type: str

    def to_text(self) -> str:
        return (
            f"Age: {self.age_bucket}, Gender: {self.gender}, "
            f"City tier: {self.city_tier}, Device: {self.device_type}"
        )


@dataclass
class Preferences:
    interest_vector: dict[str, float] = field(default_factory=dict)
    favorite_creators: list[str] = field(default_factory=list)
    novelty_seeking: float = 0.3
    trend_sensitivity: float = 0.5

    def top_interests(self, n: int = 5) -> list[str]:
        sorted_cats = sorted(
            self.interest_vector.items(), key=lambda x: x[1], reverse=True
        )
        return [cat for cat, _ in sorted_cats[:n]]

    def update_after_watch(self, category: str, engagement: float, lr: float = 0.05):
        current = self.interest_vector.get(category, 0.0)
        self.interest_vector[category] = current + lr * (engagement - current)

    def to_text(self) -> str:
        top = self.top_interests(5)
        return f"Top interests: {', '.join(top)}"


@dataclass
class BehavioralSignature:
    avg_session_minutes: float = 8.0
    sessions_per_day: float = 3.0
    videos_per_session: float = 25.0
    like_rate: float = 0.05
    comment_rate: float = 0.01
    share_rate: float = 0.005
    follow_rate: float = 0.002
    peak_hours: list[float] = field(default_factory=lambda: [0.0] * 24)
    swipe_profile: SwipeProfile = SwipeProfile.DELIBERATE_WATCHER


@dataclass
class UserPersona:
    demographics: Demographics
    preferences: Preferences
    behavior: BehavioralSignature
    activity: ActivityLevel = ActivityLevel.MEDIUM
    conformity: ConformityLevel = ConformityLevel.BALANCED
    diversity: DiversityLevel = DiversityLevel.MODERATE
    taste_description: str = ""

    @property
    def activity_text(self) -> str:
        return ACTIVITY_DESCRIPTIONS[self.activity]

    @property
    def conformity_text(self) -> str:
        return CONFORMITY_DESCRIPTIONS[self.conformity]

    @property
    def diversity_text(self) -> str:
        return DIVERSITY_DESCRIPTIONS[self.diversity]

    def build_system_prompt(self) -> str:
        parts = [
            "You are simulating a short-video app user with these traits:",
            "",
            f"Demographics: {self.demographics.to_text()}",
            f"Activity: {self.activity_text}",
            f"Conformity: {self.conformity_text}",
            f"Diversity: {self.diversity_text}",
            f"Content preferences: {self.preferences.to_text()}",
        ]
        if self.taste_description:
            parts.append(f"Taste: {self.taste_description}")
        return "\n".join(parts)


def generate_random_persona(user_id: str, rng: random.Random | None = None) -> UserPersona:
    rng = rng or random.Random()

    age_buckets = ["13-17", "18-24", "25-30", "31-40", "41-50", "51+"]
    genders = ["male", "female"]
    devices = ["ios_high", "ios_mid", "android_high", "android_mid", "android_low"]

    interest = {}
    for cat in CATEGORIES:
        interest[cat] = rng.random()
    total = sum(interest.values())
    interest = {k: v / total for k, v in interest.items()}

    peak = [0.0] * 24
    for h in range(24):
        if 7 <= h <= 9:
            peak[h] = 0.04
        elif 12 <= h <= 13:
            peak[h] = 0.06
        elif 18 <= h <= 22:
            peak[h] = 0.08 + rng.random() * 0.04
        else:
            peak[h] = 0.01 + rng.random() * 0.02
    total_p = sum(peak)
    peak = [p / total_p for p in peak]

    activity = rng.choice(list(ActivityLevel))
    sessions = {ActivityLevel.LOW: 1.5, ActivityLevel.MEDIUM: 3.0, ActivityLevel.HIGH: 5.0}
    videos = {ActivityLevel.LOW: 12, ActivityLevel.MEDIUM: 25, ActivityLevel.HIGH: 50}

    return UserPersona(
        demographics=Demographics(
            user_id=user_id,
            age_bucket=rng.choice(age_buckets),
            gender=rng.choice(genders),
            city_tier=rng.randint(1, 5),
            device_type=rng.choice(devices),
        ),
        preferences=Preferences(
            interest_vector=interest,
            novelty_seeking=rng.uniform(0.1, 0.9),
            trend_sensitivity=rng.uniform(0.1, 0.9),
        ),
        behavior=BehavioralSignature(
            avg_session_minutes=rng.uniform(3.0, 20.0),
            sessions_per_day=sessions[activity] + rng.uniform(-0.5, 0.5),
            videos_per_session=videos[activity] + rng.uniform(-5, 5),
            like_rate=rng.uniform(0.02, 0.15),
            comment_rate=rng.uniform(0.002, 0.03),
            share_rate=rng.uniform(0.001, 0.02),
            follow_rate=rng.uniform(0.0005, 0.01),
            peak_hours=peak,
            swipe_profile=rng.choice(list(SwipeProfile)),
        ),
        activity=activity,
        conformity=rng.choice(list(ConformityLevel)),
        diversity=rng.choice(list(DiversityLevel)),
    )
