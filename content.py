"""content.py BASE OTHER... : compare output tables as multisets of rows (row order and file layout ignored).
Prints one line per table and directory, and exits 1 if any table differs from BASE."""
import os, sys
import duckdb

COLS = {
    "user_daily": ["user_id", "day", "sessions", "events", "revenue_cents", "top_category"],
    "campaign_daily": ["campaign_id", "channel", "day", "sessions", "purchasing_sessions", "revenue_cents", "assisted_sessions"],
}
con = duckdb.connect()
con.execute("SET threads = 4"); con.execute("SET memory_limit = '9GB'"); con.execute("SET preserve_insertion_order = false")


def src(d, t):
    p = os.path.join(d, t)
    g = os.path.join(p, "**", "*.parquet") if os.path.isdir(p) else p + ".parquet"
    return "(select " + ",".join(COLS[t]) + " from read_parquet('" + g + "'))"


bad = 0
base = sys.argv[1]
for t in COLS:
    a = src(base, t)
    na = con.execute("select count(*) from " + a).fetchone()[0]
    print(f"{t} base {os.path.basename(base)}: rows {na}")
    for d in sys.argv[2:]:
        b = src(d, t)
        nb = con.execute("select count(*) from " + b).fetchone()[0]
        only_a = con.execute(f"select count(*) from (select * from {a} except all select * from {b})").fetchone()[0]
        only_b = con.execute(f"select count(*) from (select * from {b} except all select * from {a})").fetchone()[0]
        ok = na == nb and only_a == 0 and only_b == 0
        bad |= not ok
        print(f"{t} {os.path.basename(d)}: rows {nb} only_base {only_a} only_this {only_b} -> {'CONTENT SAME' if ok else 'CONTENT DIFFERENT'}")
        if not ok:
            for r in con.execute(f"select * from (select * from {a} except all select * from {b}) limit 5").fetchall():
                print("  base only:", r)
            for r in con.execute(f"select * from (select * from {b} except all select * from {a}) limit 5").fetchall():
                print("  this only:", r)
sys.exit(1 if bad else 0)
