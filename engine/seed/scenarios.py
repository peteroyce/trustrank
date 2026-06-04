import random


def review_bombing_signals(entity_idx, source_start, count=15):
    return [
        {
            "entity_idx": entity_idx,
            "source_idx": source_start + i,
            "dimension": "quality",
            "type": "review",
            "value": 1.0,
            "text": random.choice(
                [
                    "Terrible experience, complete waste of money",
                    "Awful product, do not buy this garbage",
                    "Worst purchase I have ever made in my life",
                    "Absolutely horrible, I want a refund immediately",
                ]
            ),
            "hours_ago": random.uniform(0, 2),
        }
        for i in range(count)
    ]


def coordinated_positive(entity_idx, source_start, count=8):
    variants = [
        "Amazing product, absolutely love it, highly recommend to everyone",
        "Amazing product, absolutely love this, highly recommend to all",
        "Amazing item, absolutely love it, highly recommend to everyone",
        "Amazing product, totally love it, highly recommend to everyone",
        "Amazing product, absolutely love it, strongly recommend to all",
        "Amazing goods, absolutely love it, highly recommend to everyone",
        "Amazing product, absolutely love it, highly recommend to anybody",
        "Amazing product, absolutely love it, really recommend to everyone",
    ]
    return [
        {
            "entity_idx": entity_idx,
            "source_idx": source_start + i,
            "dimension": "quality",
            "type": "review",
            "value": 5.0,
            "text": variants[i % len(variants)],
            "hours_ago": i * 0.5,
        }
        for i in range(count)
    ]
