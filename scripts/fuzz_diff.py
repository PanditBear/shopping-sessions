#!/usr/bin/env python3
"""Adversarial differential fuzzer: Python reference vs Rust binary.

Usage: fuzz_diff.py [--cases N] [--seed S] [--ncpu 1,8] [--jobs J] [--keep DIR] [--regen CASE_SEED]
Case i uses random.Random(seed*1_000_003+i); `--regen CS` regenerates one case into --keep/regen/input.
"""
import argparse, datetime, json, os, random, shutil, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY_SRC = (ROOT / "impl/py/pipeline.py").read_text(encoding="utf-8")
RS = os.environ.get("SHOP_RS", "C:/bxdata/shop/rs-target/release/shop.exe")
EPOCH = datetime.datetime(1970, 1, 1)
DAY = 86400000
WS = int((datetime.datetime(2028, 2, 28) - EPOCH).total_seconds()) * 1000   # window spans leap day
WE = WS + 14 * DAY
OOS = bool(os.environ.get("FUZZ_OOS"))  # out-of-spec: equal-ingest copies that differ, click copies that differ
BAD_INGEST = bool(os.environ.get("FUZZ_BAD_INGEST"))  # out-of-spec: ingest_ts is always valid F1


def iso(ms, off_min=None):
    if off_min is None:
        dt = EPOCH + datetime.timedelta(milliseconds=ms)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    dt = EPOCH + datetime.timedelta(milliseconds=ms + off_min * 60000)
    s = "+" if off_min >= 0 else "-"
    a = abs(off_min)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + "%s%02d:%02d" % (s, a // 60, a % 60)


BAD_TS = ["", "2028-03-01T00:00:00.5Z", "2028-03-01t00:00:00Z", "2028-03-01T00:00:00z", "2028-03-01 00:00:00Z",
          "2028-03-01T00:00:00", "2028-02-30T00:00:00Z", "2027-02-29T00:00:00Z", "2028-13-01T00:00:00Z",
          "2028-03-01T24:00:00Z", "2028-03-01T23:59:60Z", "2028-03-01T00:00:00+24:00", "2028-03-01T00:00:00+05:60",
          "2028-03-01T00:00:00+0200", "2028-03-01T00:00:00-24:00", "1.79e12", "-5", "1791071890000.0", "٢٥٠", "２５０",
          "2028-03-01T00:00:00-00:00", "2028-03-01T00:00:00+23:59", "2028-03-01T00:00:00-23:59", "2028-03-01T00:00:00+00:00",
          "2028-03-01T00:00:0Z", "028-03-01T00:00:00Z", "+5", " 2028-03-01T00:00:00Z", "2028-03-01T00:00:00Z ",
          "2028-03-01T00:00:00Z\r", "2028-03-01T00:00:00 Z", "2028-03-01T00:00:00+5:00", "0000-01-01T00:00:00Z",
          "9999-12-31T23:59:59Z", "9999-12-31T23:59:59-23:59", "99999999999999999999999999", "9" * 5000, "0" * 5000 + "5",
          "86400000", "0", "00000", "2028-00-10T00:00:00Z", "2028-01-00T00:00:00Z", "2100-02-29T00:00:00Z",
          "2000-02-29T00:00:00Z", "2028-02-29T23:59:59Z", "2028-04-31T00:00:00Z", "1969-12-31T23:59:59Z",
          "18446744073709551615", "18446744073709551616", "9223372036854775807", "9223372036854775808",
          "9223372036854775807000"]


def anchors(r):
    base = WS + r.choice([0, 1, DAY - 1800000, 5 * DAY, 13 * DAY + 23 * 3600000]) + r.randrange(0, 3) * 1000
    return [base, WS, WE - 1000, WS - 7 * DAY, WS + 7 * DAY, base + 3 * DAY]


def rand_ms(r, anc):
    k = r.random()
    a = r.choice(anc)
    if k < 0.15:
        return r.choice([WS, WS - 1, WE, WE - 1, WE - 1000, WS - 7 * DAY, WS - 7 * DAY - 1, WS - 7 * DAY + 1, WS + 7 * DAY,
                         WS - 7 * DAY - 1000, WE + 1000, WS - 1000])
    if k < 0.55:
        return a + r.choice([0, 0, 1, 1000, 1799000, 1800000, 1800001, 1801000, 60000, 900000, 3600000, DAY, 7 * DAY,
                             -7 * DAY, -7 * DAY - 1, -7 * DAY + 1, -1, -1000, -1800000, -1800001, -DAY])
    if k < 0.6:
        return r.randrange(0, 2 * 10 ** 13)
    return a + r.randrange(-3 * DAY, 3 * DAY)


def fmt_ts(r, ms):
    k = r.random()
    if k < 0.05:
        return r.choice(BAD_TS)
    if ms % 1000:
        return str(ms)
    if k < 0.45:
        return iso(ms)
    if k < 0.7:
        return iso(ms, r.choice([0, 60, -60, 330, -600, 720, 23 * 60 + 59, -(23 * 60 + 59), 1, -1, 30]))
    s = str(ms)
    if r.random() < 0.3:
        s = "0" * r.randrange(1, 8) + s
    return s


def fmt_field(r, v, qrate):
    if v is None:
        v = ""
    k = r.random()
    if k < qrate:
        return '"' + v + '"'
    if k < qrate + 0.01:
        return r.choice(['"' + v, v + '"', 'a"' + v, '"' + v + '"x', '""' + v, '"' + v + '"""', '"', '"""'])
    return v


USERS = ["u1", "u2", "u3", "U1", "u10", " u1", "u1 ", "u1\r", "é", "z", "中", "u\u00e9", "~", "a", "B", "", "\"x", "u1\"\""]
GOOD_USERS = ["u1", "u2", "u3", "U1", "u10", " u1", "u1 ", "é", "z", "中", "a", "B", "~", "u2\r"]


def line_join(r, fields):
    line = ",".join(fields)
    if r.random() < 0.02:
        line += ",x"
    if r.random() < 0.02 and "," in line:
        line = line.rsplit(",", 1)[0]
    return line


def write_files(r, folder, header, lines, crlf=False):
    nfiles = r.choice([0, 1, 1, 2, 3, 5])
    if nfiles == 0 and r.random() < 0.5:
        return
    folder.mkdir(parents=True, exist_ok=True)
    if nfiles == 0:
        return
    names = r.sample(["part-0", "part-1", "part-2", "part-10", "a", "B", "_x", ".h", "part-00", "Z", "part-1a", "é", "part-0 "], nfiles)
    files = {n + ".csv": [] for n in names}
    keys = list(files)
    for l in lines:
        files[r.choice(keys)].append(l)
    nl = "\r\n" if crlf else "\n"
    for n, ls in files.items():
        hdr = header if r.random() > 0.05 else "x,y"
        body = hdr + nl + nl.join(ls)
        if ls and r.random() < 0.7:
            body += nl
        if r.random() < 0.03:
            body = ""
        if r.random() < 0.05 and body:
            body += "\n\n"
        if r.random() < 0.03:
            body = "\ufeff" + body
        data = body.encode("utf-8")
        if OOS and r.random() < 0.08 and len(data) > 12:  # invalid UTF-8 is outside the spec
            i = r.randrange(len(data))
            data = data[:i] + r.choice([b"\xff", b"\xc3", b"\xe4\xb8", b"\x80", b"\xc0\x80", b"\xed\xa0\x80", b"\xf0\x9f\x98\x80", b"\x00"]) + data[i:]
        (folder / n).write_bytes(data)
    if r.random() < 0.05:
        (folder / "notes.txt").write_bytes(b"e1,u1,2028-03-01T00:00:00Z\n")
    if r.random() < 0.03:
        (folder / "dir.csv").mkdir(exist_ok=True)


def gen_case(case_seed, d):
    r = random.Random(case_seed)
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    ws, we = WS, WE
    man = json.dumps({"window_start": iso(ws), "window_end": iso(we)})
    if r.random() < 0.15:
        if r.random() < 0.5:
            ws = r.choice([WS, WS + 3 * DAY, WS - 5 * DAY, 0, 1800000 * 7])
            we = ws + r.choice([DAY, 14 * DAY, 3600000, 0, -DAY, 5 * DAY])
            ft = fmt_ts if OOS else (lambda r_, ms: r_.choice([iso(ms), str(ms), iso(ms, 60)]))
            man = json.dumps({"window_start": ft(r, ws), "window_end": ft(r, we)})
        elif OOS:
            man = r.choice(["", "{", "[]", "{}", '{"window_start": "2028-02-28T00:00:00Z"}', "null",
                            '{"window_start": 5, "window_end": 6}', '{"window_start": "x", "window_end": "y"}',
                            json.dumps({"window_start": iso(WS), "window_end": iso(WE), "extra": 1}),
                            json.dumps({"window_end": iso(WE), "window_start": iso(WS)}),
                            '{"window_start": "2028-02-28T00:00:00Z", "window_end": "2028-02-28T00:00:00Z"}',
                            '{"window_start": "1835395200000", "window_end": "1836604800000"}'])
    if OOS and r.random() < 0.2:  # BOM before JSON: the spec says nothing
        man = '\ufeff' + man
    if r.random() < 0.97:
        (d / "manifest.json").write_text(man, encoding="utf-8")
    anc = anchors(r)
    ncat = r.choice([1, 2, 3])
    cats = r.sample(["books", "toys", "Books", "a", "z", "é", "unknown", " books"], ncat)
    prods = ["p%d" % i for i in range(5)] + ["P1", " p1", "p1 "]
    pl = [(p, r.choice(cats)) for p in prods if r.random() < 0.8]
    cl = ["c_%d" % i for i in range(4)] + ["C_1", "c_1 "]
    cam = [(c, r.choice(["ads", "email", "social"])) for c in cl if r.random() < 0.8]
    qr = r.choice([0, 0.2, 0.5])
    crlf = r.random() < 0.05
    for name, hdr, rows in (("products.csv", "product_id,category", pl), ("campaigns.csv", "campaign_id,channel", cam)):
        nl = "\r\n" if crlf else "\n"
        body = hdr + nl + nl.join(",".join(fmt_field(r, v, qr * 0.5) for v in row) for row in rows)
        if r.random() < 0.8:
            body += nl
        if r.random() < 0.03:
            continue
        (d / name).write_bytes(body.encode())
    users = r.sample(GOOD_USERS, r.choice([1, 2, 3, 5]))
    allids = users + ["x%d" % i for i in range(r.choice([0, 3]))]
    links = []
    for _ in range(r.choice([0, 0, 1, 3, 6, 12])):
        a, b = r.choice(allids + USERS[:6]), r.choice(allids + USERS[:6])
        if r.random() < 0.05:
            a = ""
        links.append([a, b])
    if r.random() < 0.1:
        ch = r.sample(allids, len(allids))
        links += [[ch[i], ch[i + 1]] for i in range(len(ch) - 1)]
    lines = []
    for f in links:
        lines.append(line_join(r, [fmt_field(r, v, qr) for v in f]))
        if r.random() < 0.2:
            lines.append(r.choice(["", "a", ",", "a,b,c", '"a,b"', "\r", "u1,"]))
    write_files(r, d / "links", "user_id_a,user_id_b", lines, crlf and r.random() < 0.5)
    ev = []
    nev = r.choice([0, 1, 3, 8, 15, 30])
    base = rand_ms(r, anc)
    for i in range(nev):
        eid = "e%d" % (r.randrange(max(1, nev // 2 + 1)) if OOS else i)
        if r.random() < 0.02:
            eid = ""
        uid = r.choice(users) if r.random() > 0.05 else r.choice(USERS)
        if r.random() < 0.7:
            t = base + r.choice([0, 0, 1, 1000, 1800000, 1800001, 1800000, 3600000, -1800000, 60000, DAY])
        else:
            t = rand_ms(r, anc)
        if r.random() < 0.3:
            t -= t % 1000
        ing = t + r.choice([0, 1000, 600000, DAY, DAY + 1000, DAY - 1000, 5 * DAY, -1000, -DAY, 3600000])
        ing -= ing % 1000
        if ing < 0:
            ing = 0
        et = r.choice(["view", "view", "cart", "purchase", "purchase", "purchase", "Purchase", "cart ", "", "VIEW", "x"])
        pid = r.choice([p for p, _ in pl] + ["p1", "nope", "", "P1"]) if r.random() > 0.05 else ""
        price = r.choice(["0", "1", "10000000", "10000001", "00250", "+5", "-5", "1.5", "", "250", "1000\r", "12345", "500", "500", "7",
                          "٢٥٠", "１", "00000000000000000000000000000001", "99999999999999999999999", "9" * 5000, "1e3", " 5", "0x10", "100"])
        ingf = iso(ing) if (r.random() > 0.03 or not BAD_INGEST) else r.choice(["", "2028-03-01T00:00:00+00:00", "x"])
        fl = [eid, uid, fmt_ts(r, t), ingf, et, pid, price]
        ev.append(line_join(r, [fmt_field(r, v, qr) for v in fl]))
        seen_ing = {fl[3]}
        if r.random() < 0.3:
            for _ in range(r.choice([1, 2])):
                f2 = list(fl)
                if r.random() < 0.5:
                    f2[3] = iso(ing + r.choice([-1000, 0, 1000, 60000, -DAY * 2, DAY * 3]))
                if r.random() < 0.3:
                    f2[1] = r.choice(users)
                if r.random() < 0.3:
                    f2[2] = fmt_ts(r, t + r.choice([0, 1000, -1000, 7000]))
                if r.random() < 0.2:
                    f2[6] = r.choice(["1", "2", "0"])
                if r.random() < 0.2:
                    f2[4] = r.choice(["view", "purchase"])
                if r.random() < 0.15 and OOS:
                    f2[3] = fl[3]
                if not OOS and f2[3] in seen_ing:
                    f2 = list(fl)  # equal ingest_ts means identical copies
                seen_ing.add(f2[3])
                ev.append(line_join(r, [fmt_field(r, v, qr) for v in f2]))
    write_files(r, d / "events", "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents", ev, crlf and r.random() < 0.5)
    ck = []
    cids = [c for c, _ in cam] + ["c_1", "c_0", "zz", "C_1", ""]
    nck = r.choice([0, 1, 3, 8, 15])
    for i in range(nck):
        cid = "k%d" % (r.randrange(nck + 2) if OOS else i)
        if r.random() < 0.02:
            cid = ""
        uid = r.choice(users) if r.random() > 0.05 else r.choice(USERS)
        if r.random() < 0.5:
            t = rand_ms(r, anc)
        else:
            t = r.choice(anc) + r.choice([0, 0, 1, -1, -7 * DAY, -7 * DAY - 1, -7 * DAY + 1, 1000, -1000])
        fl = [cid, uid, r.choice(cids), fmt_ts(r, t)]
        ck.append(line_join(r, [fmt_field(r, v, qr) for v in fl]))
        if r.random() < 0.25:
            f2 = list(fl)
            if r.random() < 0.5 and OOS:
                f2[r.randrange(1, 4)] = r.choice(["u1", "c_1", "c_2", iso(t - t % 1000)])
            ck.append(line_join(r, [fmt_field(r, v, qr) for v in f2]))
        if r.random() < 0.3:
            f2 = list(fl)
            f2[0] = "k%d" % (r.randrange(nck + 2) if OOS else 100 + i)
            f2[2] = r.choice(cids)
            ck.append(line_join(r, [fmt_field(r, v, qr) for v in f2]))
    write_files(r, d / "clicks", "click_id,user_id,campaign_id,ts", ck, crlf and r.random() < 0.5)


def run_py(data, out):
    sys.set_int_max_str_digits(0)
    # in-process: pipeline.py runs main() at import and reads env at import time
    os.environ["DATA_DIR"] = str(data)
    os.environ["OUTPUT_DIR"] = str(out)
    exec(compile(PY_SRC, "pipeline.py", "exec"), {"__name__": "ref"})


def read_out(out):
    import pyarrow.parquet as pq
    res = {}
    for t in ("user_daily", "campaign_daily"):
        rows = []
        p = Path(out) / t
        for f in sorted(p.glob("*.parquet")) if p.is_dir() else []:
            if f.name.startswith((".", "_")):
                continue
            rows += [tuple(sorted(x.items())) for x in pq.read_table(f).to_pylist()]
        res[t] = sorted(rows, key=repr)
    return res


def run_case(args):
    seed, i, ncpus, keep = args
    cs = seed * 1_000_003 + i
    tmp = Path(tempfile.mkdtemp(prefix="fz"))
    try:
        data = tmp / "in"
        gen_case(cs, data)
        pyout = tmp / "py"
        pyout.mkdir()
        try:
            run_py(data, pyout)
            ref = read_out(pyout)
        except BaseException as e:
            ref = "PYERR " + repr(e)[:200]
        bad = []
        for n in ncpus:
            ro = tmp / ("rs%d" % n)
            sc = tmp / ("sc%d" % n)
            ro.mkdir()
            sc.mkdir()
            env = dict(os.environ, DATA_DIR=str(data), OUTPUT_DIR=str(ro), SCRATCH_DIR=str(sc), NCPU=str(n))
            try:
                p = subprocess.run([RS], env=env, capture_output=True, timeout=60)
                got = read_out(ro) if p.returncode == 0 else "RSERR %d %s" % (p.returncode, p.stderr[-200:])
            except Exception as e:
                got = "RSERR " + repr(e)[:200]
            if got != ref:
                bad.append(n)
        if bad:
            dst = Path(keep) / ("case%d" % cs)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(data, dst / "input")
            return (cs, bad, ref if isinstance(ref, str) else "")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ncpu", default="1,8")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--keep", default=str(Path(tempfile.gettempdir()) / "fuzz_fail"))
    ap.add_argument("--regen", type=int)
    a = ap.parse_args()
    if a.regen is not None:
        gen_case(a.regen, Path(a.keep) / "regen" / "input")
        return 0
    ncpus = [int(x) for x in a.ncpu.split(",")]
    Path(a.keep).mkdir(parents=True, exist_ok=True)
    fails = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for res in ex.map(run_case, [(a.seed, i, ncpus, a.keep) for i in range(a.cases)], chunksize=8):
            if res:
                fails.append(res)
                print("DIVERGE case seed %d ncpu %s %s" % res, flush=True)
    print("cases=%d divergences=%d (saved under %s)" % (a.cases, len(fails), a.keep))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
