"""Pull mania 4K replays + beatmaps from osu! API v2.

Requires env vars:
  OSU_CLIENT_ID, OSU_CLIENT_SECRET

Examples:
  # 4K-variant top 20 players, up to 50 scores each
  python src/fetch_replays.py --top 20 --per-player 50

  # specific user IDs
  python src/fetch_replays.py --users 4650315 6447454 --per-player 30
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://osu.ppy.sh/api/v2"
OAUTH = "https://osu.ppy.sh/oauth/token"
OSU_FILE = "https://osu.ppy.sh/osu/{beatmap_id}"
USER_AGENT = "aiosu-fetcher/0.1"


def http_json(req: urllib.request.Request):
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def http_bytes(req: urllib.request.Request):
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def load_dotenv(path: Path):
    """Tiny KEY=VALUE loader. Skips comments/blank lines. Does not overwrite."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'").strip('"')
        os.environ.setdefault(k, v)


def get_token():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    cid = os.environ.get("OSU_CLIENT_ID")
    sec = os.environ.get("OSU_CLIENT_SECRET")
    if not cid or not sec:
        sys.exit("set OSU_CLIENT_ID and OSU_CLIENT_SECRET (in .env or env)")
    body = urllib.parse.urlencode({
        "client_id": cid,
        "client_secret": sec,
        "grant_type": "client_credentials",
        "scope": "public",
    }).encode()
    req = urllib.request.Request(OAUTH, data=body, method="POST",
                                 headers={"User-Agent": USER_AGENT})
    return http_json(req)["access_token"]


def auth_req(url: str, token: str):
    return urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    })


def top_players_4k(token: str, n: int):
    """Top N players on the 4K mania performance leaderboard."""
    players, page = [], 1
    while len(players) < n:
        url = f"{API}/rankings/mania/performance?variant=4k&page={page}"
        data = http_json(auth_req(url, token))
        rows = data.get("ranking", [])
        if not rows:
            break
        for row in rows:
            uid = row["user"]["id"]
            uname = row["user"]["username"]
            players.append((uid, uname))
            if len(players) >= n:
                break
        page += 1
        time.sleep(0.5)
    return players[:n]


def user_best(token: str, user_id: int, limit: int):
    """Top mania scores for a user (API caps single call at 100)."""
    out, offset = [], 0
    while len(out) < limit:
        take = min(100, limit - len(out))
        url = (f"{API}/users/{user_id}/scores/best"
               f"?mode=mania&limit={take}&offset={offset}")
        rows = http_json(auth_req(url, token))
        if not rows:
            break
        out.extend(rows)
        offset += len(rows)
        if len(rows) < take:
            break
        time.sleep(0.3)
    return out


def download_replay(token: str, score_id: int) -> bytes:
    """Try newer endpoint first, fall back to mode-scoped legacy path."""
    for url in (f"{API}/scores/{score_id}/download",
                f"{API}/scores/mania/{score_id}/download"):
        try:
            return http_bytes(auth_req(url, token))
        except urllib.error.HTTPError as e:
            if e.code in (404, 422):
                continue
            raise
    raise RuntimeError(f"no replay endpoint accepted score {score_id}")


def download_beatmap(beatmap_id: int) -> bytes:
    req = urllib.request.Request(OSU_FILE.format(beatmap_id=beatmap_id),
                                 headers={"User-Agent": USER_AGENT})
    return http_bytes(req)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=0,
                   help="pull top-N 4K mania players")
    p.add_argument("--users", nargs="*", type=int, default=[],
                   help="explicit user IDs")
    p.add_argument("--per-player", type=int, default=50,
                   help="max scores per player (API caps single call at 100)")
    p.add_argument("--replay-dir", default="data/replays")
    p.add_argument("--beatmap-dir", default="data/beatmaps")
    p.add_argument("--keys", type=int, default=4,
                   help="only keep maps with this key count")
    args = p.parse_args()

    if not args.top and not args.users:
        sys.exit("pass --top N or --users <ids>")

    rep_dir = Path(args.replay_dir); rep_dir.mkdir(parents=True, exist_ok=True)
    bm_dir = Path(args.beatmap_dir); bm_dir.mkdir(parents=True, exist_ok=True)

    print("authenticating...")
    token = get_token()

    targets = list(args.users)
    if args.top:
        print(f"fetching top {args.top} 4K players...")
        for uid, name in top_players_4k(token, args.top):
            print(f"  {uid:>10}  {name}")
            targets.append(uid)

    seen_beatmaps = {int(p.stem) for p in bm_dir.glob("*.osu")
                     if p.stem.isdigit()}
    seen_replays = {p.stem for p in rep_dir.glob("*.osr")}

    n_ok, n_skip, n_err = 0, 0, 0
    for uid in targets:
        print(f"\nuser {uid}:")
        try:
            scores = user_best(token, uid, args.per_player)
        except urllib.error.HTTPError as e:
            print(f"  fetch scores failed: {e}")
            continue
        for s in scores:
            sid = s.get("id") or s.get("best_id")
            bm = s.get("beatmap") or {}
            bid = bm.get("id")
            cs = bm.get("cs")
            replay_available = s.get("replay") or s.get("has_replay")

            if cs and int(cs) != args.keys:
                n_skip += 1
                continue
            if not replay_available:
                n_skip += 1
                continue
            if not sid or not bid:
                n_skip += 1
                continue

            rep_name = f"{uid}_{bid}_{sid}"
            if rep_name in seen_replays:
                n_skip += 1
                continue

            try:
                if bid not in seen_beatmaps:
                    bm_bytes = download_beatmap(bid)
                    if not bm_bytes.lstrip().startswith(b"osu file format"):
                        print(f"  bm {bid}: not a .osu (deleted?) — skip")
                        n_skip += 1
                        continue
                    (bm_dir / f"{bid}.osu").write_bytes(bm_bytes)
                    seen_beatmaps.add(bid)
                    time.sleep(0.3)

                rep_bytes = download_replay(token, sid)
                (rep_dir / f"{rep_name}.osr").write_bytes(rep_bytes)
                seen_replays.add(rep_name)
                n_ok += 1
                print(f"  + score {sid} on map {bid} "
                      f"({len(rep_bytes)//1024}KB)")
                time.sleep(2.0)
            except urllib.error.HTTPError as e:
                print(f"  score {sid}: HTTP {e.code} {e.reason}")
                n_err += 1
                if e.code == 429:
                    print("  (rate-limited, sleeping 30s)")
                    time.sleep(30)
            except Exception as e:
                print(f"  score {sid}: {e}")
                n_err += 1

    print(f"\ndone. ok={n_ok}  skipped={n_skip}  errors={n_err}")
    print(f"replays in {rep_dir}, maps in {bm_dir}")


if __name__ == "__main__":
    main()
