"""fuzz_diff.py with generator-style IDs.

The published generator writes event ids as 'e' + base36(scramble(index, key)), user ids as
'u' + base36(scramble(person*512 + alias, key)) (shared devices: index 2^39 + k) and click ids as
'k' + base36(scramble(index, key)), with a random 40-bit key per dataset. This wrapper rewrites the
plain ids of fuzz_diff.py ("e5", "u1", "k3") into that form with a random key per case, and mixes in
near-misses that must take the exact slow path: a leading '0', upper case, a value of 2^40 or more,
a 9-char body, and an index far above the dataset size. Same options and output as fuzz_diff.py.
"""
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fuzz_diff  # noqa: E402

M = (1 << 40) - 1
C1 = 0x9E3779B97F4B
C2 = 0xD6E8FEB86659
B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def scr(v, key):
    m = ((v * C1) ^ key) & M
    m ^= m >> 17
    return (m * C2) & M


def b36(n):
    s = ""
    while True:
        n, d = divmod(n, 36)
        s = B36[d] + s
        if n == 0:
            return s


ST = {}
_orig_gen_case = fuzz_diff.gen_case
_orig_fmt = fuzz_diff.fmt_field


def gen_case(case_seed, d):
    g = random.Random(case_seed * 7 + 12345)
    ST.clear()
    ST["r"] = g
    ST["ekey"] = g.choice([0, g.getrandbits(40), g.getrandbits(40), g.getrandbits(40)])
    ST["ukey"] = g.getrandbits(40)
    ST["kkey"] = g.getrandbits(40)
    ST["off"] = g.choice([0, 0, 0, 3, 50])
    ST["miss"] = g.choice([0.0, 0.0, 0.05, 0.3, 0.9])
    gp = [g.randrange(1, 1000) for _ in range(3)]
    ST["users"] = {
        "1": gp[0] * 512 + g.randrange(4),
        "2": (gp[0] if g.random() < 0.6 else gp[1]) * 512 + 4 + g.randrange(4),
        "3": gp[2] * 512 + g.randrange(512),
        "10": (1 << 39) + g.randrange(1000),
    }
    ST["uout"] = {k: scr(v, ST["ukey"]) for k, v in ST["users"].items()}
    if g.random() < 0.4:
        # short bodies whose byte order differs from their numeric order ("u10" < "u2" < "u3z")
        small = [36, 2, 3 * 36 + 35, 37, 1296, 3]
        g.shuffle(small)
        for k, v in zip(["1", "2", "3", "10"], small):
            ST["uout"][k] = v
    return _orig_gen_case(case_seed, d)


def near_miss(pre, n, key, kind):
    body = b36(scr(n, key))
    if kind == 0:
        return pre + "0" + body
    if kind == 1:
        return (pre + body.upper()) if any(c.isalpha() for c in body) else (pre.upper() + body)
    if kind == 2:
        return pre + b36((1 << 40) + (n * 7919) % 1000003)
    if kind == 3:
        return pre + b36(36 ** 8 + n)
    return pre + b36(scr(n + (1 << 30), key))


def map_id(pre, n, key):
    g = ST["r"]
    if g.random() < ST["miss"]:
        return near_miss(pre, n, key, g.randrange(5))
    return pre + b36(scr(n, key))


EV = re.compile(r"^e(\d+)$")
CK = re.compile(r"^k(\d+)$")
US = re.compile(r"(?<![0-9A-Za-z_])([uU])(10|1|2|3)(?![0-9A-Za-z_])")


def fmt_field(r, v, qrate):
    if v and "r" in ST:
        m = EV.match(v)
        if m:
            v = map_id("e", int(m.group(1)) + ST["off"], ST["ekey"])
        else:
            m = CK.match(v)
            if m:
                v = map_id("k", int(m.group(1)), ST["kkey"])
            elif "u" in v or "U" in v:
                def sub(mm):
                    s = "u" + b36(ST["uout"][mm.group(2)])
                    return s.upper() if mm.group(1) == "U" else s
                v = US.sub(sub, v)
    return _orig_fmt(r, v, qrate)


fuzz_diff.gen_case = gen_case
fuzz_diff.fmt_field = fmt_field

if __name__ == "__main__":
    sys.exit(fuzz_diff.main())
