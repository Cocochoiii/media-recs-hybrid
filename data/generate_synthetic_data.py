import argparse, os, random, csv, time
from datetime import datetime, timedelta

TITLES = ["Ocean Saga", "City Lights", "Forest Echo", "Quantum Leap", "Hidden Truths", "Neon Nights", "Silent Voices", "Solar Wind"]
DESCS  = ["An epic journey.", "A heartfelt drama.", "A quirky comedy.", "Sci-fi thriller.", "Nature documentary.", "Mystery puzzle.", "Slice of life.", "Space opera."]

def main(args):
    random.seed(42)
    os.makedirs("data", exist_ok=True)

    # Items
    items = []
    for i in range(args.items):
        items.append({
            "item_id": f"it_{i:05d}",
            "title": random.choice(TITLES),
            "desc": random.choice(DESCS),
            "genre": random.choice(["drama","comedy","sci-fi","documentary","mystery"]),
        })

    with open("data/items.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id","title","desc","genre"])
        w.writeheader(); w.writerows(items)

    # Users & interactions
    now = datetime.utcnow()
    inters = []
    for u in range(args.users):
        uid = f"u_{u:03d}"
        for _ in range(int(args.interactions/args.users)):
            it = f"it_{random.randint(0, args.items-1):05d}"
            ts = now - timedelta(days=random.randint(0, 120), minutes=random.randint(0, 60*24))
            inters.append({"user_id": uid, "item_id": it, "ts": int(ts.timestamp())})

    random.shuffle(inters)
    with open("data/interactions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["user_id","item_id","ts"])
        w.writeheader(); w.writerows(inters)

    print(f"Generated items={len(items)}, interactions={len(inters)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=500)
    ap.add_argument("--items", type=int, default=800)
    ap.add_argument("--interactions", type=int, default=8000)
    args = ap.parse_args()
    main(args)
