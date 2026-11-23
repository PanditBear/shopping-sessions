"""measure_sum.py res.tsv : medians per (size, label) from lines
size label n wall user sys peak_anon peak_cg peak_file outbytes nfiles exit"""
import sys, statistics as st
rows = {}
order = []
for ln in open(sys.argv[1]):
    f = ln.split()
    if len(f) < 12 or f[2] == "warm":
        continue
    key = (f[0], f[1])
    if key not in rows:
        rows[key] = []
        order.append(key)
    rows[key].append([float(x) for x in f[3:12]])
print("size label runs | wall_med wall_min wall_max | user_med sys_med | peak_anon_MB peak_cgroup_MB | out_bytes files exit_max")
for k in order:
    v = rows[k]
    c = lambda i: [x[i] for x in v]
    print(k[0], k[1], len(v), "| %.3f %.3f %.3f | %.2f %.2f | %.0f %.0f | %d %d %d" % (
        st.median(c(0)), min(c(0)), max(c(0)), st.median(c(1)), st.median(c(2)),
        st.median(c(3)) / 1048576, st.median(c(4)) / 1048576, st.median(c(6)), st.median(c(7)), max(c(8))))
