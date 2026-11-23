"""measure_poll.py CONTAINER_ID : sample the container's cgroup until it disappears, print one JSON line."""
import os, sys, time, json
cid = sys.argv[1]
cands = ["/sys/fs/cgroup/system.slice/docker-%s.scope" % cid, "/sys/fs/cgroup/docker/%s" % cid]
path = None
end = time.time() + 180
while path is None and time.time() < end:
    for c in cands:
        if os.path.isdir(c):
            path = c
            break
    else:
        time.sleep(0.0005)
r = {"found": path is not None, "user_us": 0, "sys_us": 0, "peak": 0, "anon_max": 0, "file_max": 0, "cur_max": 0, "n": 0}


def rd(f):
    with open(os.path.join(path, f)) as h:
        return h.read()


while path is not None:
    try:
        u = s = None
        for ln in rd("cpu.stat").splitlines():
            k, v = ln.split()
            if k == "user_usec":
                u = int(v)
            elif k == "system_usec":
                s = int(v)
        if u is not None:
            r["user_us"], r["sys_us"] = max(r["user_us"], u), max(r["sys_us"], s)
        try:
            r["peak"] = max(r["peak"], int(rd("memory.peak")))
        except (FileNotFoundError, OSError):
            pass
        cur = int(rd("memory.current"))
        r["cur_max"] = max(r["cur_max"], cur)
        for ln in rd("memory.stat").splitlines():
            k, v = ln.split()
            if k == "anon":
                r["anon_max"] = max(r["anon_max"], int(v))
            elif k == "file":
                r["file_max"] = max(r["file_max"], int(v))
        r["n"] += 1
    except (FileNotFoundError, ProcessLookupError, OSError):
        if not os.path.isdir(path):
            break
    time.sleep(0.002)
print(json.dumps(r))
