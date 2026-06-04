import random
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from engine.models.entity import Entity, EntityType
from engine.models.signal import Signal, Dimension, SignalType
from engine.models.trust import TrustEdge, EvidenceType
from engine.seed.scenarios import review_bombing_signals, coordinated_positive

random.seed(42)

NAMES = {
    "merchant": [
        "Alpine Kitchen",
        "Sakura Sushi",
        "Metro Mart",
        "Golden Spice",
        "Blue Harbor",
        "Sunset Cafe",
        "Prime Goods",
        "Fresh Fields",
        "Ocean Breeze",
        "Peak Performance",
    ],
    "user": [f"user_{i:03d}" for i in range(100)],
    "service": [
        "QuickFix Repairs",
        "SwiftDeliver",
        "CleanPro",
        "TechAssist",
        "PetCare Plus",
        "GreenThumb",
        "HomeBright",
        "AutoCare",
        "FitCoach",
        "TravelEase",
    ],
    "product": [
        "ErgoChair Pro",
        "SkyBuds Wireless",
        "AquaFilter X",
        "SolarPack 5000",
        "ChefMaster Blender",
        "ZenMat Yoga",
        "PureAir Purifier",
        "SmartLock V2",
        "NightOwl Lamp",
        "FrostKeep Cooler",
    ],
}


async def seed_database(db: AsyncSession) -> dict:
    entities = []

    for etype in ["merchant", "service", "product"]:
        for name in NAMES[etype][:10]:
            e = Entity(
                type=EntityType(etype),
                name=name,
                metadata_={"category": etype},
            )
            db.add(e)
            entities.append(e)

    for name in NAMES["user"][:100]:
        e = Entity(type=EntityType.user, name=name, metadata_={})
        db.add(e)
        entities.append(e)

    await db.flush()

    non_users = [e for e in entities if e.type != EntityType.user]
    users = [e for e in entities if e.type == EntityType.user]
    now = datetime.now(timezone.utc)
    signal_count = 0
    texts = [
        "Good quality overall",
        "Met my expectations",
        "Could be better",
        "Excellent service",
        "Average experience",
        "Very satisfied",
        "Not worth the price",
        "Will come back again",
        "Highly recommended",
        "Disappointing quality",
        "Great value for money",
        "",
    ]

    for entity in non_users:
        n = random.randint(30, 200)
        for _ in range(n):
            source = random.choice(users)
            dim = random.choice(list(Dimension))
            value = min(5.0, max(1.0, round(random.gauss(3.8, 0.9) * 2) / 2))
            s = Signal(
                entity_id=entity.id,
                source_id=source.id,
                dimension=dim,
                type=SignalType.review,
                value=value,
                text=random.choice(texts),
                created_at=now - timedelta(days=random.uniform(1, 365)),
            )
            db.add(s)
            signal_count += 1

    for attack in review_bombing_signals(0, 80, 15):
        s = Signal(
            entity_id=non_users[attack["entity_idx"]].id,
            source_id=users[attack["source_idx"]].id,
            dimension=Dimension(attack["dimension"]),
            type=SignalType(attack["type"]),
            value=attack["value"],
            text=attack["text"],
            created_at=now - timedelta(hours=attack["hours_ago"]),
        )
        db.add(s)
        signal_count += 1

    for attack in coordinated_positive(5, 60, 8):
        s = Signal(
            entity_id=non_users[attack["entity_idx"]].id,
            source_id=users[attack["source_idx"]].id,
            dimension=Dimension(attack["dimension"]),
            type=SignalType(attack["type"]),
            value=attack["value"],
            text=attack["text"],
            created_at=now - timedelta(hours=attack["hours_ago"]),
        )
        db.add(s)
        signal_count += 1

    trust_count = 0
    community_size = len(non_users) // 3
    for c in range(3):
        members = non_users[c * community_size : (c + 1) * community_size]
        for i in range(len(members)):
            for j in range(i + 1, min(i + 3, len(members))):
                edge = TrustEdge(
                    source_id=members[i].id,
                    target_id=members[j].id,
                    weight=random.uniform(0.5, 0.95),
                    category="general",
                    evidence_type=EvidenceType.transaction,
                )
                db.add(edge)
                trust_count += 1

    await db.commit()
    return {
        "entities": len(entities),
        "signals": signal_count,
        "trust_edges": trust_count,
    }
