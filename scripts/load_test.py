import concurrent.futures, time, random, httpx

def hit(client, uid):
    r = client.get("http://127.0.0.1:8000/recommendations", params={"user_id": uid, "k": 10})
    return r.status_code

if __name__ == "__main__":
    users = [f"u_{i:03d}" for i in range(1,50)]
    with httpx.Client() as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
            futs = [ex.submit(hit, client, random.choice(users)) for _ in range(300)]
            ok = sum(1 for f in futs if f.result()==200)
            print("OK", ok, "of", len(futs))
