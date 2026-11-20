"""Reference pipeline for Shopping Sessions: plain Python, written to follow rules.md literally."""
import bisect
import datetime
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATA = Path(os.environ["DATA_DIR"])
OUT = Path(os.environ["OUTPUT_DIR"])

DAY_MS = 86400000
SESSION_GAP = 1800000
WEEK_MS = 7 * DAY_MS

U = pa.schema([("user_id", pa.string()), ("day", pa.date32()), ("sessions", pa.int64()), ("events", pa.int64()),
               ("revenue_cents", pa.int64()), ("top_category", pa.string())])
C = pa.schema([("campaign_id", pa.string()), ("channel", pa.string()), ("day", pa.date32()), ("sessions", pa.int64()),
               ("purchasing_sessions", pa.int64()), ("revenue_cents", pa.int64()), ("assisted_sessions", pa.int64())])

F1 = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:Z|([+-])([0-9]{2}):([0-9]{2}))")
F3 = re.compile(r"[0-9]+")
DIGITS = re.compile(r"[0-9]+")
EPOCH = datetime.datetime(1970, 1, 1)


def parse_ts(s):
    """Return epoch ms or None."""
    if s is None:
        return None
    if F3.fullmatch(s):
        return int(s)
    m = F1.fullmatch(s)
    if not m:
        return None
    y, mo, d, h, mi, se = (int(m.group(i)) for i in range(1, 7))
    if h > 23 or mi > 59 or se > 59:
        return None
    try:
        dt = datetime.datetime(y, mo, d, h, mi, se)
    except ValueError:
        return None
    ms = ((dt - EPOCH) // datetime.timedelta(milliseconds=1))
    if m.group(7):
        oh, om = int(m.group(8)), int(m.group(9))
        if oh > 23 or om > 59:
            return None
        off = (oh * 60 + om) * 60000
        ms -= off if m.group(7) == "+" else -off
    return ms


def split_line(line, n):
    """Split a CSV line into n fields per the dialect; None if malformed. Empty field -> None."""
    parts = line.split(",")
    if len(parts) != n:
        return None
    out = []
    for p in parts:
        if p.startswith('"'):
            if len(p) < 2 or not p.endswith('"') or '"' in p[1:-1]:
                return None
            p = p[1:-1]
        elif '"' in p:
            return None
        out.append(p if p != "" else None)
    return out


def read_file(path, n):
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [r for r in (split_line(l, n) for l in lines[1:]) if r is not None]


def read_rows(folder, n):
    """Yield field lists from every *.csv in folder (header row skipped)."""
    if not folder.is_dir():
        return
    for f in sorted(folder.glob("*.csv")):
        yield from read_file(f, n)


def main():
    try:
        man = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        ws = parse_ts(man["window_start"])
        we = parse_ts(man["window_end"])
    except Exception:
        ws = we = None
    if ws is None or we is None:
        write([], [])
        return

    products = {}
    for pid, cat in read_file(DATA / "products.csv", 2):
        if pid is not None:
            products[pid] = cat
    campaigns = {}
    for cid, ch in read_file(DATA / "campaigns.csv", 2):
        if cid is not None:
            campaigns[cid] = ch

    # links -> union-find
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in read_rows(DATA / "links", 2):
        if a is None or b is None:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    names = {}
    for x in list(parent):
        r = find(x)
        if r not in names or x < names[r]:
            names[r] = x

    def person(uid):
        if uid not in parent:
            return uid
        return names[find(uid)]

    # events: dedup on event_id keeping smallest ingest_ts
    best = {}
    for eid, uid, ts, ing, et, pid, price in read_rows(DATA / "events", 7):
        if eid is None:
            continue
        ing_ms = parse_ts(ing)
        key = ing_ms if ing_ms is not None else float("inf")
        cur = best.get(eid)
        if cur is None or key < cur[0]:
            best[eid] = (key, uid, ts, ing_ms, et, pid, price)

    per_person = defaultdict(list)  # person -> [(ts, price, purchase category or None)]
    for key, uid, ts, ing_ms, et, pid, price in best.values():
        if uid is None or pid is None or et not in ("view", "cart", "purchase"):
            continue
        t = parse_ts(ts)
        if t is None or not (ws <= t < we):
            continue
        p = 0
        if et == "purchase":
            if price is None or not DIGITS.fullmatch(price):
                continue
            p = int(price)
            if not (1 <= p <= 10000000):
                continue
        if ing_ms is not None and ing_ms - t > DAY_MS:
            continue
        cat = products.get(pid, "unknown") if et == "purchase" else None
        per_person[person(uid)].append((t, p, cat))

    # clicks
    seen = {}
    for clid, uid, cid, ts in read_rows(DATA / "clicks", 4):
        if clid is None:
            continue
        seen.setdefault(clid, (uid, cid, ts))
    clicks = defaultdict(list)  # person -> sorted [(ts, click_id, campaign)]
    for clid, (uid, cid, ts) in seen.items():
        if uid is None or cid is None or cid not in campaigns:
            continue
        t = parse_ts(ts)
        if t is None or not (ws - WEEK_MS <= t < we):
            continue
        clicks[person(uid)].append((t, clid, cid))
    for v in clicks.values():
        v.sort()

    user_rows = {}  # (person, day) -> [sessions, events, revenue, {cat: sum}]
    camp_rows = {}  # (campaign, day) -> [sessions, purchasing, revenue, assisted]

    def add_session(pname, start, evs):
        day = (EPOCH + datetime.timedelta(milliseconds=start)).date()
        rev = sum(e[1] for e in evs)
        cats = defaultdict(int)
        npurch = 0
        for e in evs:
            if e[2] is not None:
                cats[e[2]] += e[1]
                npurch += 1
        u = user_rows.setdefault((pname, day), [0, 0, 0, defaultdict(int)])
        u[0] += 1
        u[1] += len(evs)
        u[2] += rev
        for k, v in cats.items():
            u[3][k] += v
        cl = clicks.get(pname, [])
        lo = bisect.bisect_left(cl, (start - WEEK_MS,))
        hi = bisect.bisect_left(cl, (start + 1,))
        window = cl[lo:hi]
        if window:
            top_ts = window[-1][0]
            credited = min((c[1], c[2]) for c in window if c[0] == top_ts)[1]
        else:
            credited = "organic"
        r = camp_rows.setdefault((credited, day), [0, 0, 0, 0])
        r[0] += 1
        r[1] += 1 if npurch else 0
        r[2] += rev
        for cid in {c[2] for c in window}:
            camp_rows.setdefault((cid, day), [0, 0, 0, 0])[3] += 1

    for pname, evs in per_person.items():
        evs.sort(key=lambda e: e[0])
        cur = [evs[0]]
        prev = evs[0][0]
        for e in evs[1:]:
            if e[0] - prev > SESSION_GAP:
                add_session(pname, cur[0][0], cur)
                cur = []
            cur.append(e)
            prev = e[0]
        add_session(pname, cur[0][0], cur)

    urows = []
    for (pname, day), (s, n, rev, cats) in user_rows.items():
        top = None
        if cats:
            m = max(cats.values())
            top = min(k for k, v in cats.items() if v == m)
        urows.append((pname, day, s, n, rev, top))
    crows = []
    for (cid, day), (s, p, rev, a) in camp_rows.items():
        crows.append((cid, campaigns[cid] if cid != "organic" else "organic", day, s, p, rev, a))
    write(urows, crows)


def write(urows, crows):
    for name, schema, rows in (("user_daily", U, urows), ("campaign_daily", C, crows)):
        (OUT / name).mkdir(parents=True, exist_ok=True)
        cols = list(zip(*rows)) if rows else [[] for _ in schema]
        t = pa.table({f.name: pa.array(list(c), type=f.type) for f, c in zip(schema, cols)}, schema=schema)
        pq.write_table(t, OUT / name / "part-0.parquet")


main()
