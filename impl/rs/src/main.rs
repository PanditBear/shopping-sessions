use ahash::{AHashMap, RandomState};
#[cfg(all(target_os = "linux", target_env = "musl", target_arch = "x86_64", target_feature = "avx2"))]
mod fastmem;
use memchr::memchr;
use rayon::prelude::*;
use std::fs::File;
use std::mem::MaybeUninit;
use std::path::Path;
use std::sync::Arc;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// mimalloc's arenas are already 2 MiB-aligned with MADV_HUGEPAGE, but by default freed memory is
// purged (MADV_DONTNEED in 64 KiB slices) about a second later. That splits the huge pages, and the
// memory is faulted and zeroed again when the next phase reuses it. Peak RSS is reached before the
// big frees, so keeping freed memory committed costs almost nothing. SHOP_PURGE=1 restores the default.
mod hpa {
    extern "C" {
        fn mi_option_set(option: i32, value: std::ffi::c_long);
    }
    const MI_OPTION_PURGE_DELAY: i32 = 15; // index in mimalloc v3's mi_option_t

    pub fn init() {
        if std::env::var_os("SHOP_PURGE").is_none() {
            unsafe { mi_option_set(MI_OPTION_PURGE_DELAY, -1) }
        }
    }

    // SHOP_THP=1: anonymous and huge-page memory of the process after each phase (Linux only)
    pub fn diag(label: &str) {
        if !cfg!(target_os = "linux") || std::env::var_os("SHOP_THP").is_none() {
            return;
        }
        let r = std::fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default();
        let f = |k: &str| {
            r.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0)
        };
        eprintln!("thp {label}: anon {} MB, anonhuge {} MB, rss {} MB", f("Anonymous:") >> 10, f("AnonHugePages:") >> 10, f("Rss:") >> 10);
    }
}

type B = &'static [u8];

const DAY: i64 = 86_400_000;
const WEEK: i64 = 7 * DAY;
const GAP: i64 = 1_800_000;
const CHUNK: usize = 4 << 20;
const EBB: u32 = 11;
const EB: usize = 1 << EBB; // event dedup buckets (by event_id)
const DDG: usize = 4; // event_id buckets per dedup task
const PBB: u32 = 10;
const PB: usize = 1 << PBB; // person buckets

fn hasher() -> RandomState {
    RandomState::with_seeds(0x243f6a8885a308d3, 0x13198a2e03707344, 0xa4093822299f31d0, 0x082efa98ec4e6c89)
}

#[inline(always)]
fn mix(k: u64) -> u64 {
    let k = (k ^ (k >> 31)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    k ^ (k >> 29)
}

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
static TIMING: AtomicBool = AtomicBool::new(false);
static PROF: [AtomicU64; 17] = [const { AtomicU64::new(0) }; 17];
const PROF_NAMES: [&str; 17] =
    ["ev.scan", "ev.bucket", "dd.table", "dd.collect", "dd.resolve", "dd.bucket", "ss.gather", "ss.sort", "ss.proc", "ck.scan", "ck.res", "ss.csort", "wr.ud", "ud.idx", "ud.int", "ud.revtop", "ud.write"];
#[inline]
fn now() -> Option<std::time::Instant> {
    if TIMING.load(Relaxed) {
        Some(std::time::Instant::now())
    } else {
        None
    }
}
#[inline]
fn pt(i: usize, t: &mut Option<std::time::Instant>) {
    if let Some(s) = t {
        let n = std::time::Instant::now();
        PROF[i].fetch_add((n - *s).as_nanos() as u64, Relaxed);
        *s = n;
    }
}

#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// Input prefetch distance in bytes for the event and click line kernels. SHOP_PFD overrides
/// (0 off). Otherwise 2048 on GenuineIntel CPUs, where it shortens the scan, and 0 elsewhere.
/// Read once.
fn pf_dist() -> usize {
    static D: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *D.get_or_init(|| {
        if let Some(v) = std::env::var("SHOP_PFD").ok().and_then(|s| s.parse().ok()) {
            return v;
        }
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: cpuid is available on every x86_64 CPU
            let r = unsafe { core::arch::x86_64::__cpuid(0) };
            if r.ebx == 0x756e_6547 && r.edx == 0x4965_6e69 && r.ecx == 0x6c65_746e {
                return 2048;
            }
        }
        0
    })
}

// ---------- id keys ----------
// An id of 1..=10 bytes drawn from [0-9A-Z_a-z] packs exactly into 6-bit codes (order-preserving,
// left aligned, zero padded): a value below 2^60. Any other id gets a 63-bit hash with bit 63 set,
// so the two key spaces never meet.

const fn mk_code() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 10 {
        t[b'0' as usize + i] = 1 + i as u8;
        i += 1;
    }
    i = 0;
    while i < 26 {
        t[b'A' as usize + i] = 11 + i as u8;
        t[b'a' as usize + i] = 38 + i as u8;
        i += 1;
    }
    t[b'_' as usize] = 37;
    t
}
static CODE: [u8; 256] = mk_code();
const fn mk_decode() -> [u8; 64] {
    let c = mk_code();
    let mut d = [0u8; 64];
    let mut i = 0;
    while i < 256 {
        if c[i] != 0 {
            d[c[i] as usize] = i as u8;
        }
        i += 1;
    }
    d
}
static DECODE: [u8; 64] = mk_decode();
const HBIT: u64 = 1 << 63;

const fn h8(x: u8) -> u64 {
    0x0101010101010101u64 * x as u64
}

#[inline(always)]
fn ld8(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

/// 6-bit codes (same values as CODE) of 8 bytes, packed with byte 0 most significant (48 bits),
/// and whether every byte is in [0-9A-Z_a-z].
#[inline(always)]
fn code8(w: u64) -> (u64, bool) {
    let hi = h8(0x80);
    let ge = |k: u8| w.wrapping_add(h8(0x80 - k)) & hi;
    let (g30, g3a, g41, g5b) = (ge(0x30), ge(0x3A), ge(0x41), ge(0x5B));
    let (g5f, g60, g61, g7b) = (ge(0x5F), ge(0x60), ge(0x61), ge(0x7B));
    let valid = (g30 & !g3a) | (g41 & !g5b) | (g5f & !g60) | (g61 & !g7b);
    let ok = (w & hi) == 0 && valid == hi;
    let off = h8(0x2F) + (g41 >> 7) * 7 + (g5f >> 7) * 4 + (g61 >> 7);
    let mut x = w.wrapping_sub(off).swap_bytes();
    x = ((x & 0xFF00FF00FF00FF00) >> 2) | (x & 0x00FF00FF00FF00FF);
    x = ((x & 0xFFFF0000FFFF0000) >> 4) | (x & 0x0000FFFF0000FFFF);
    x = ((x & 0xFFFFFFFF00000000) >> 8) | (x & 0x00000000FFFFFFFF);
    (x, ok)
}

#[inline(always)]
fn pack6(b: &[u8]) -> Option<u64> {
    let n = b.len();
    if (8..=10).contains(&n) {
        // first 8 bytes in one word, the 0-2 bytes after them by table
        let (k0, ok0) = code8(ld8(b, 0));
        let t = n - 8;
        let ca = CODE[b[8.min(n - 1)] as usize] as u64;
        let cb = CODE[b[n - 1] as usize] as u64;
        if !(ok0 & ((ca != 0) | (t < 1)) & ((cb != 0) | (t < 2))) {
            return None;
        }
        let ma = 0u64.wrapping_sub((t >= 1) as u64);
        let mb = 0u64.wrapping_sub((t >= 2) as u64);
        return Some((k0 << 12) | ((ca & ma) << 6) | (cb & mb));
    }
    pack6_slow(b)
}

#[inline(never)]
fn pack6_slow(b: &[u8]) -> Option<u64> {
    let n = b.len();
    if n == 0 || n > 10 {
        return None;
    }
    let mut k = 0u64;
    let mut bad = 0u8;
    for &c in b {
        let x = CODE[c as usize];
        bad |= (x == 0) as u8;
        k = (k << 6) | x as u64;
    }
    if bad != 0 {
        return None;
    }
    Some(k << (6 * (10 - n)))
}

fn decode6(k: u64, out: &mut Vec<u8>) {
    for i in 0..10 {
        let c = (k >> (54 - 6 * i)) & 63;
        if c == 0 {
            break;
        }
        out.push(DECODE[c as usize]);
    }
}

#[inline]
fn ukey(rs: &RandomState, b: B) -> u64 {
    match pack6(b) {
        Some(k) => k,
        None => rs.hash_one(b) | HBIT,
    }
}

// ---------- parsing helpers ----------

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn masks64(p: *const u8) -> (u64, u64, u64) {
    use core::arch::x86_64::*;
    unsafe {
        let c = _mm256_set1_epi8(b',' as i8);
        let q = _mm256_set1_epi8(b'"' as i8);
        let n = _mm256_set1_epi8(b'\n' as i8);
        let mut cm = 0u64;
        let mut qm = 0u64;
        let mut nm = 0u64;
        for i in 0..2 {
            let v = _mm256_loadu_si256(p.add(32 * i) as *const __m256i);
            cm |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, c)) as u32 as u64) << (32 * i);
            qm |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, q)) as u32 as u64) << (32 * i);
            nm |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, n)) as u32 as u64) << (32 * i);
        }
        (cm, qm, nm)
    }
}
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[inline(always)]
fn masks64(p: *const u8) -> (u64, u64, u64) {
    use core::arch::x86_64::*;
    unsafe {
        let c = _mm_set1_epi8(b',' as i8);
        let q = _mm_set1_epi8(b'"' as i8);
        let n = _mm_set1_epi8(b'\n' as i8);
        let mut cm = 0u64;
        let mut qm = 0u64;
        let mut nm = 0u64;
        for i in 0..4 {
            let v = _mm_loadu_si128(p.add(16 * i) as *const __m128i);
            cm |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, c)) as u32 as u64) << (16 * i);
            qm |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, q)) as u32 as u64) << (16 * i);
            nm |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, n)) as u32 as u64) << (16 * i);
        }
        (cm, qm, nm)
    }
}
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn masks64(p: *const u8) -> (u64, u64, u64) {
    let (mut cm, mut qm, mut nm) = (0u64, 0u64, 0u64);
    for i in 0..64 {
        let c = unsafe { *p.add(i) };
        cm |= ((c == b',') as u64) << i;
        qm |= ((c == b'"') as u64) << i;
        nm |= ((c == b'\n') as u64) << i;
    }
    (cm, qm, nm)
}

/// Quote check for the first fields of a line inside one 64-byte window. `q1`: the quote bits
/// before the field end `end`; `c`: the separator bits, with bit `end` set. Returns the quotes
/// that open a field and a mask that is nonzero unless every field before `end` is either free
/// of quotes or is '"', at least one byte, '"' with no other quote.
/// The borrow of `c - x` from each opening quote clears the separator that ends its field.
#[inline(always)]
fn quote_bad(q1: u64, c: u64) -> (u64, u64) {
    let x = q1 & ((c << 1) | 1);
    let e = !c.wrapping_sub(x) & c;
    (x, q1 ^ x ^ (e >> 1))
}

/// `quote_bad` on a 128-byte window.
#[inline(always)]
fn quote_bad128(q1: u128, c: u128) -> (u128, u128) {
    let x = q1 & ((c << 1) | 1);
    let e = !c.wrapping_sub(x) & c;
    (x, q1 ^ x ^ (e >> 1))
}

/// The `parse_clicks` line rule on 128 bytes, for a line with no newline in its first 64 bytes
/// (`ca`, `qa`: that window's masks): the newline l < 128 with exactly three commas e0 < e1 < e2
/// before it and only whole-field quotes; returns them with the opening-quote bit of each field,
/// or None. The caller guarantees 128 readable bytes at `p`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn line4_128(p: *const u8, ca: u64, qa: u64) -> Option<(u32, u32, u32, u32, [u64; 4])> {
    let (cw, qw, nw) = masks64(p.add(64));
    let c = (ca as u128) | ((cw as u128) << 64);
    let q = (qa as u128) | ((qw as u128) << 64);
    let n = (nw as u128) << 64;
    let l = n.trailing_zeros();
    let lm = n.wrapping_sub(1) & !n;
    let cl = c & lm;
    let e0 = cl.trailing_zeros();
    let c1 = cl & cl.wrapping_sub(1);
    let e1 = c1.trailing_zeros();
    let c2 = c1 & c1.wrapping_sub(1);
    let e2 = c2.trailing_zeros();
    let c3 = c2 & c2.wrapping_sub(1);
    let (x, bad) = quote_bad128(q & lm, c | n);
    if (nw == 0) | (c2 == 0) | (c3 != 0) | (bad != 0) {
        return None;
    }
    Some((l, e0, e1, e2, [(x & 1) as u64, ((x >> (e0 + 1)) & 1) as u64, ((x >> (e1 + 1)) & 1) as u64, ((x >> (e2 + 1)) & 1) as u64]))
}

/// The first-window part of the `parse_events_sol` classifier on 128 bytes, for a line with no
/// newline in its first 64 bytes (`ca`, `qa`: that window's masks): the first three commas
/// e0 < e1 < e2 < 128 with no newline before e2, the opening-quote bit of fields 0-2, and the
/// `quote_bad` mask of fields 0-2 folded to 64 bits; None if there are fewer than three commas
/// in the 128 bytes or a newline comes first. The caller guarantees 128 readable bytes at `p`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn first3_128(p: *const u8, ca: u64, qa: u64) -> Option<(u32, u32, u32, [u64; 3], u64)> {
    let (cw, qw, nw) = masks64(p.add(64));
    let c = (ca as u128) | ((cw as u128) << 64);
    let q = (qa as u128) | ((qw as u128) << 64);
    let e0 = c.trailing_zeros();
    let c1 = c & c.wrapping_sub(1);
    let e1 = c1.trailing_zeros();
    let c2 = c1 & c1.wrapping_sub(1);
    let e2 = c2.trailing_zeros();
    if (c2 == 0) | (e2 >= 64 + nw.trailing_zeros()) {
        return None;
    }
    let (x, bad) = quote_bad128(q & ((1u128 << e2) - 1), c);
    let xq = [(x & 1) as u64, ((x >> (e0 + 1)) & 1) as u64, ((x >> (e1 + 1)) & 1) as u64];
    Some((e0, e1, e2, xq, (bad as u64) | ((bad >> 64) as u64)))
}

/// Calls `f` with the unquoted fields of every well-formed line of `chunk` (N fields).
/// Same dialect as rules.md: only \n ends a line; quotes allowed only around a whole field.
#[inline(always)]
fn scan<const N: usize>(chunk: B, mut f: impl FnMut(&[B; N])) {
    let len = chunk.len();
    let mut commas = [0usize; N];
    let mut nc = 0usize;
    let mut ls = 0usize;
    let mut qcnt = 0u32;
    let mut pos = 0usize;
    let mut tmp = [0u8; 64];
    let mut emit = |ls: usize, le: usize, nc: usize, qcnt: u32, commas: &[usize; N]| {
        if nc != N - 1 {
            return;
        }
        let mut out: [B; N] = [&[]; N];
        let mut start = ls;
        let mut expect = 0u32;
        let base = chunk.as_ptr();
        for k in 0..N {
            let end = if k < N - 1 { commas[k] } else { le };
            let flen = end - start;
            // SAFETY: start <= end <= chunk.len(); the two reads happen only when flen >= 2
            let q = flen >= 2 && unsafe { (*base.add(start) == b'"') & (*base.add(end - 1) == b'"') };
            let q = q as usize;
            out[k] = unsafe { std::slice::from_raw_parts(base.add(start + q), flen - 2 * q) };
            expect += 2 * q as u32;
            start = end + 1;
        }
        if expect != qcnt {
            return;
        }
        f(&out);
    };
    while pos < len {
        let p = if pos + 64 <= len {
            unsafe { chunk.as_ptr().add(pos) }
        } else {
            tmp = [0u8; 64];
            tmp[..len - pos].copy_from_slice(&chunk[pos..]);
            tmp.as_ptr()
        };
        let (cm, qm, nm) = masks64(p);
        let mut s = cm | nm;
        let mut qrest = qm;
        while s != 0 {
            let bit = s.trailing_zeros();
            let at = pos + bit as usize;
            if (nm >> bit) & 1 != 0 {
                let low = (1u64 << bit) - 1;
                qcnt += (qrest & low).count_ones();
                qrest &= !low;
                emit(ls, at, nc, qcnt, &commas);
                ls = at + 1;
                nc = 0;
                qcnt = 0;
            } else {
                if nc < N - 1 {
                    commas[nc] = at;
                }
                nc += 1;
            }
            s &= s - 1;
        }
        qcnt += qrest.count_ones();
        pos += 64;
    }
    if ls < len {
        emit(ls, len, nc, qcnt, &commas);
    }
}

/// Same lines and fields as `scan::<7>` (same dialect, same order). Each line is classified from
/// two 64-byte windows: one at the line start (fields 0-2) and one just after the third comma
/// (fields 3-6). A line takes the fast path when the first window holds three commas with no
/// newline before the third, the second holds a newline with exactly three commas before it, and
/// every quote opens or closes a whole field of two or more bytes (checked on the masks: with X
/// the quotes at field starts and Y those at field ends, Y - X must touch no separator and no Y
/// bit). Every other line, and the last 128 bytes of the chunk, go to `scan` one line at a time.
/// The fast path of `scan7`: the fields of the line at `p` and its length with the newline, or
/// None if the line needs `scan`. The caller guarantees 128 readable bytes at `p`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
fn line7(p: *const u8) -> Option<([B; 7], usize)> {
    use core::arch::x86_64::_bzhi_u64;
    // SAFETY (bzhi): BMI2 is enabled for this build
    let bzhi = |x: u64, n: u32| unsafe { _bzhi_u64(x, n) };
    let blsr = |x: u64| x & x.wrapping_sub(1);
    let (ca, qa, na) = masks64(p);
    let e0 = ca.trailing_zeros();
    let c1 = blsr(ca);
    let e1 = c1.trailing_zeros();
    let c2 = blsr(c1);
    let e2 = c2.trailing_zeros();
    if (c2 != 0) & (bzhi(na, e2) == 0) {
        let pb = unsafe { p.add(e2 as usize + 1) };
        let (cb, qb, nb) = masks64(pb);
        let l = nb.trailing_zeros();
        let e3 = cb.trailing_zeros();
        let d1 = blsr(cb);
        let e4 = d1.trailing_zeros();
        let d2 = blsr(d1);
        let e5 = d2.trailing_zeros();
        let d3 = blsr(d2);
        let qa1 = bzhi(qa, e2);
        let xa = qa1 & ((ca << 1) | 1);
        let ya = qa1 & (ca >> 1);
        let ra = ya.wrapping_sub(xa);
        let bad_a = (qa1 & !(xa | ya)) | (xa & ya) | (ra & (bzhi(ca, e2 + 1) | ya));
        let sb = cb | nb;
        let qb1 = bzhi(qb, l);
        let xb = qb1 & ((cb << 1) | 1);
        let yb = qb1 & (sb >> 1);
        let rb = yb.wrapping_sub(xb);
        let bad_b = (qb1 & !(xb | yb)) | (xb & yb) | (rb & (bzhi(sb, l + 1) | yb));
        if (nb != 0) & (e5 < l) & (bzhi(d3, l) == 0) & ((bad_a | bad_b) == 0) {
            let fld = |p: *const u8, x: u64, s: u32, e: u32| {
                let q = ((x >> s) & 1) as u32;
                // SAFETY: s <= e <= 63 inside a window; a quoted field has e - s >= 2
                unsafe { std::slice::from_raw_parts(p.add((s + q) as usize), (e - s - 2 * q) as usize) }
            };
            let out: [B; 7] = [
                fld(p, xa, 0, e0),
                fld(p, xa, e0 + 1, e1),
                fld(p, xa, e1 + 1, e2),
                fld(pb, xb, 0, e3),
                fld(pb, xb, e3 + 1, e4),
                fld(pb, xb, e4 + 1, e5),
                fld(pb, xb, e5 + 1, l),
            ];
            return Some((out, (e2 + l + 2) as usize));
        }
    }
    None
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
fn scan7(chunk: B, mut f: impl FnMut(&[B; 7])) {
    let len = chunk.len();
    let base = chunk.as_ptr();
    let mut pos = 0usize;
    // lines from the slow path wait in buf, so f has one call site and its state never
    // reaches a non-inlined function
    let mut buf: Vec<[B; 7]> = Vec::new();
    let mut bi = 0usize;
    loop {
        let line: [B; 7] = if bi < buf.len() {
            bi += 1;
            buf[bi - 1]
        } else if pos + 128 <= len {
            // SAFETY: pos + 128 <= len, and the second window starts at most 64 bytes in
            match line7(unsafe { base.add(pos) }) {
                Some((l, n)) => {
                    #[cfg(test)]
                    SCAN7_FAST.with(|c| c.set(c.get() + 1));
                    pos += n;
                    l
                }
                None => {
                    buf.clear();
                    bi = 0;
                    pos = scan7_line(chunk, pos, &mut buf);
                    continue;
                }
            }
        } else if pos < len {
            buf.clear();
            bi = 0;
            scan::<7>(&chunk[pos..], |l| buf.push(*l));
            pos = len;
            continue;
        } else {
            break;
        };
        f(&line);
    }
}

#[cfg(test)]
thread_local! {
    static SCAN7_FAST: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The slow path of `scan7`: the one line at `pos` through `scan`; returns the next line start.
#[cold]
#[inline(never)]
fn scan7_line(chunk: B, pos: usize, buf: &mut Vec<[B; 7]>) -> usize {
    let end = match memchr(b'\n', &chunk[pos..]) {
        Some(i) => pos + i + 1,
        None => chunk.len(),
    };
    scan::<7>(&chunk[pos..end], |l| buf.push(*l));
    end
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
#[inline(always)]
fn scan7(chunk: B, f: impl FnMut(&[B; 7])) {
    scan::<7>(chunk, f)
}

/// Same lines and fields as `scan::<4>` (same dialect, same order). A line takes the fast path
/// when one 64-byte window at its start holds its newline with exactly three commas before it and
/// every quote opens or closes a whole field of two or more bytes (the mask rule of `line7`).
/// Every other line, and the last 64 bytes of the chunk, go to `scan` one line at a time.
/// The fast path of `scan4`: the fields of the line at `p` and its length with the newline, or
/// None if the line needs `scan`. The caller guarantees 64 readable bytes at `p`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
fn line4(p: *const u8) -> Option<([B; 4], usize)> {
    use core::arch::x86_64::_bzhi_u64;
    // SAFETY (bzhi): BMI2 is enabled for this build
    let bzhi = |x: u64, n: u32| unsafe { _bzhi_u64(x, n) };
    let blsr = |x: u64| x & x.wrapping_sub(1);
    let (ca, qa, na) = masks64(p);
    let l = na.trailing_zeros();
    let cl = bzhi(ca, l);
    let e0 = cl.trailing_zeros();
    let c1 = blsr(cl);
    let e1 = c1.trailing_zeros();
    let c2 = blsr(c1);
    let e2 = c2.trailing_zeros();
    let c3 = blsr(c2);
    let sb = ca | na;
    let q1 = bzhi(qa, l);
    let x = q1 & ((ca << 1) | 1);
    let y = q1 & (sb >> 1);
    let r = y.wrapping_sub(x);
    let bad = (q1 & !(x | y)) | (x & y) | (r & (bzhi(sb, l + 1) | y));
    if (na != 0) & (c2 != 0) & (c3 == 0) & (bad == 0) {
        let fld = |s: u32, e: u32| {
            let q = ((x >> s) & 1) as u32;
            // SAFETY: s <= e <= 63 inside the window; a quoted field has e - s >= 2
            unsafe { std::slice::from_raw_parts(p.add((s + q) as usize), (e - s - 2 * q) as usize) }
        };
        let out: [B; 4] = [fld(0, e0), fld(e0 + 1, e1), fld(e1 + 1, e2), fld(e2 + 1, l)];
        return Some((out, (l + 1) as usize));
    }
    None
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
fn scan4(chunk: B, mut f: impl FnMut(&[B; 4])) {
    let len = chunk.len();
    let base = chunk.as_ptr();
    let mut pos = 0usize;
    // lines from the slow path wait in buf, so f has one call site
    let mut buf: Vec<[B; 4]> = Vec::new();
    let mut bi = 0usize;
    loop {
        let line: [B; 4] = if bi < buf.len() {
            bi += 1;
            buf[bi - 1]
        } else if pos + 64 <= len {
            // SAFETY: pos + 64 <= len
            match line4(unsafe { base.add(pos) }) {
                Some((l, n)) => {
                    #[cfg(test)]
                    SCAN7_FAST.with(|c| c.set(c.get() + 1));
                    pos += n;
                    l
                }
                None => {
                    buf.clear();
                    bi = 0;
                    pos = scan4_line(chunk, pos, &mut buf);
                    continue;
                }
            }
        } else if pos < len {
            buf.clear();
            bi = 0;
            scan::<4>(&chunk[pos..], |l| buf.push(*l));
            pos = len;
            continue;
        } else {
            break;
        };
        f(&line);
    }
}

/// The slow path of `scan4`: the one line at `pos` through `scan`; returns the next line start.
#[cold]
#[inline(never)]
fn scan4_line(chunk: B, pos: usize, buf: &mut Vec<[B; 4]>) -> usize {
    let end = match memchr(b'\n', &chunk[pos..]) {
        Some(i) => pos + i + 1,
        None => chunk.len(),
    };
    scan::<4>(&chunk[pos..end], |l| buf.push(*l));
    end
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
#[inline(always)]
fn scan4(chunk: B, f: impl FnMut(&[B; 4])) {
    scan::<4>(chunk, f)
}

#[inline]
fn d2(b: &[u8]) -> Option<i64> {
    let (x, y) = (b[0].wrapping_sub(b'0'), b[1].wrapping_sub(b'0'));
    if x > 9 || y > 9 {
        None
    } else {
        Some((x * 10 + y) as i64)
    }
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// direct-mapped cache of validated dates: the tag is the 10 date bytes, the value is the day number
#[derive(Clone, Copy)]
struct TsEnt {
    p8: u64,
    tag2: u32,
    days: i32,
}
struct TsCache {
    ents: Vec<TsEnt>,
    sh: u32,
}
impl TsCache {
    fn new() -> Self {
        Self::with_bits(10)
    }
    fn with_bits(bits: u32) -> Self {
        // tag2 == 0 never matches (a lookup tag always has bit 16 set)
        TsCache { ents: vec![TsEnt { p8: 0, tag2: 0, days: 0 }; 1 << bits], sh: 64 - bits }
    }
}

#[inline(always)]
fn dig8(w: u64) -> bool {
    ((w.wrapping_sub(h8(0x30)) | w.wrapping_add(h8(0x46))) & h8(0x80)) == 0
}
/// value of 8 ASCII digits (byte 0 most significant); caller checks dig8
#[inline(always)]
fn parse8(w: u64) -> u64 {
    let mut v = w.wrapping_sub(h8(0x30));
    v = v.wrapping_mul(10).wrapping_add(v >> 8);
    ((v & 0x000000FF000000FF)
        .wrapping_mul(100 + (1000000u64 << 32))
        .wrapping_add(((v >> 16) & 0x000000FF000000FF).wrapping_mul(1 + (10000u64 << 32))))
        >> 32
}
fn parse_ts(b: &[u8]) -> Option<i64> {
    parse_ts_c(b, &mut TsCache::with_bits(1))
}
/// Day number of a validated "YYYY-MM-DD" prefix: the cache-miss path of `ts_iso`.
#[cold]
#[inline(never)]
fn ts_days(b: &[u8]) -> Option<i64> {
    if b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let y = d2(&b[0..2])? * 100 + d2(&b[2..4])?;
    let mo = d2(&b[5..7])?;
    let d = d2(&b[8..10])?;
    if !(1..=12).contains(&mo) || d < 1 {
        return None;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let dim = match mo {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if leap {
                29
            } else {
                28
            }
        }
    };
    if d > dim {
        return None;
    }
    Some(days_from_civil(y, mo, d))
}

/// `parse_ts_c` for "YYYY-MM-DDTHH:MM:SSZ" (n == 20) or "...+HH:MM" (n == 25); the caller
/// checks the length and byte 19.
#[inline(always)]
fn ts_iso(b: &[u8], c: &mut TsCache) -> Option<i64> {
    let n = b.len();
    if b[10] != b'T' {
        return None;
    }
    let t = ld8(b, 11);
    let t2 = t ^ 0x00000A00000A0000;
    if (t & 0x0000FF0000FF0000) != 0x00003A00003A0000 || !dig8(t2) {
        return None;
    }
    let v = t2.wrapping_sub(h8(0x30));
    let v = v.wrapping_mul(10).wrapping_add(v >> 8);
    let (h, mi, s) = ((v & 0xFF) as i64, ((v >> 24) & 0xFF) as i64, ((v >> 48) & 0xFF) as i64);
    if h > 23 || mi > 59 || s > 59 {
        return None;
    }
    let days;
    let p8 = ld8(b, 0);
    let tag2 = u16::from_le_bytes([b[8], b[9]]) as u32 | 0x1_0000;
    let hi = ((p8 ^ ((tag2 as u64) << 40)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> c.sh) as usize;
    let ent = &mut c.ents[hi];
    if ent.p8 == p8 && ent.tag2 == tag2 {
        days = ent.days as i64;
    } else {
        days = ts_days(b)?;
        if days == days as i32 as i64 {
            *ent = TsEnt { p8, tag2, days: days as i32 };
        }
    }
    let mut off = 0i64;
    if n == 25 {
        if b[22] != b':' {
            return None;
        }
        let oh = d2(&b[20..22])?;
        let om = d2(&b[23..25])?;
        if oh > 23 || om > 59 {
            return None;
        }
        off = (oh * 60 + om) * 60_000;
        if b[19] == b'-' {
            off = -off;
        }
    }
    Some(days * DAY + ((h * 60 + mi) * 60 + s) * 1000 - off)
}

/// `parse_ts_c` for an all-digit value of 9..=16 bytes; the caller checks the length.
#[inline(always)]
fn ts_num(b: &[u8]) -> Option<i64> {
    let n = b.len();
    let sh = 8 * (16 - n) as u32;
    let hw = (ld8(b, 0) << sh) | (h8(0x30) & ((1u64 << sh) - 1));
    let lw = ld8(b, n - 8);
    if !(dig8(hw) & dig8(lw)) {
        return None;
    }
    Some((parse8(hw) * 100_000_000 + parse8(lw)) as i64)
}

/// Same result as `parse_ts_c`, with the two common shapes inlined.
#[inline(always)]
fn ts_fast(b: &[u8], c: &mut TsCache) -> Option<i64> {
    let n = b.len();
    if n == 20 && b[19] == b'Z' {
        ts_iso(b, c)
    } else if n == 13 {
        ts_num(b)
    } else {
        parse_ts_c(b, c)
    }
}

/// 1 for "view" or "cart", 2 for "purchase", 0 otherwise.
#[inline(always)]
fn et_code(b: &[u8]) -> u32 {
    match b.len() {
        4 => {
            let w = u32::from_le_bytes(b[..4].try_into().unwrap());
            ((w == u32::from_le_bytes(*b"view")) | (w == u32::from_le_bytes(*b"cart"))) as u32
        }
        8 => 2 * (ld8(b, 0) == u64::from_le_bytes(*b"purchase")) as u32,
        _ => 0,
    }
}

#[inline]
fn parse_ts_c(b: &[u8], c: &mut TsCache) -> Option<i64> {
    let n = b.len();
    if n == 0 {
        return None;
    }
    if (n == 20 && b[19] == b'Z') || (n == 25 && (b[19] == b'+' || b[19] == b'-')) {
        return ts_iso(b, c);
    }
    if (9..=16).contains(&n) {
        return ts_num(b);
    }
    if n <= 18 {
        let mut v: u64 = 0;
        for &c in b {
            let x = c.wrapping_sub(b'0');
            if x > 9 {
                return None;
            }
            v = v * 10 + x as u64;
        }
        return Some(v as i64);
    }
    let mut v: u64 = 0;
    for &c in b {
        let x = c.wrapping_sub(b'0');
        if x > 9 {
            return None;
        }
        v = v.checked_mul(10)?.checked_add(x as u64)?;
    }
    if v > i64::MAX as u64 {
        return None;
    }
    Some(v as i64)
}

#[inline]
fn parse_price(b: &[u8]) -> Option<u32> {
    if b.is_empty() {
        return None;
    }
    let mut v: u64 = 0;
    for &c in b {
        let x = c.wrapping_sub(b'0');
        if x > 9 {
            return None;
        }
        v = v * 10 + x as u64;
        if v > 10_000_000 {
            // keep scanning for non-digits is unnecessary: over the limit is bad either way
            return None;
        }
    }
    if v >= 1 {
        Some(v as u32)
    } else {
        None
    }
}

fn chunks(data: B, sz: usize) -> Vec<B> {
    let start = match memchr(b'\n', data) {
        Some(i) => i + 1,
        None => return vec![],
    };
    let mut v = vec![];
    let mut a = start;
    while a < data.len() {
        let mut b = (a + sz).min(data.len());
        if b < data.len() {
            b = match memchr(b'\n', &data[b - 1..]) {
                Some(i) => b - 1 + i + 1,
                None => data.len(),
            };
        }
        v.push(&data[a..b]);
        a = b;
    }
    v
}

#[cfg(target_os = "linux")]
static INPUT_MAPS: std::sync::Mutex<Vec<&'static memmap2::Mmap>> = std::sync::Mutex::new(Vec::new());

/// Drop the page-table entries of the input mappings (MADV_DONTNEED) on all cores once the output is
/// written, so the single-threaded process teardown has about 1.4M fewer 4K PTEs to zap at 60M (GHA
/// 44886103884: exit -0.016 s, wall -0.05 s). The mappings are private, read-only and file-backed, so a
/// later read of an input byte would just fault the same page back in from the page cache.
/// SHOP_DROPIN=off skips it.
#[cfg(target_os = "linux")]
fn drop_inputs() {
    let maps: Vec<&'static memmap2::Mmap> = INPUT_MAPS.lock().unwrap().clone();
    let mut parts: Vec<(&'static memmap2::Mmap, usize, usize)> = vec![];
    const CH: usize = 64 << 20;
    for m in maps {
        let mut o = 0;
        while o < m.len() {
            let n = CH.min(m.len() - o);
            parts.push((m, o, n));
            o += n;
        }
    }
    parts.par_iter().for_each(|&(m, o, n)| {
        let _ = unsafe { m.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, o, n) };
    });
}
#[cfg(not(target_os = "linux"))]
fn drop_inputs() {}

fn leak_file(p: &Path) -> Option<B> {
    let f = File::open(p).ok()?;
    let len = f.metadata().ok()?.len();
    if len == 0 {
        return None;
    }
    let m = unsafe { memmap2::Mmap::map(&f).ok()? };
    #[cfg(target_os = "linux")]
    {
        let _ = m.advise(memmap2::Advice::Sequential);
    }
    let m: &'static memmap2::Mmap = Box::leak(Box::new(m));
    #[cfg(target_os = "linux")]
    INPUT_MAPS.lock().unwrap().push(m);
    if std::env::var_os("SHOP_TOUCH").is_some() {
        // measurement aid: fault every page in before the timed phases
        let mut acc = 0u8;
        let mut i = 0;
        while i < m.len() {
            acc ^= m[i];
            i += 4096;
        }
        std::hint::black_box(acc);
    }
    Some(&m[..])
}

fn csv_files(dir: &Path) -> Vec<B> {
    csv_files_in(dir, true)
}

/// `par: false` maps the files on the calling thread only. Code inside a rayon::join branch that
/// must not wait on the pool uses it: a pool thread that waits for a stolen job steals other work
/// meanwhile and only comes back when that work is done.
fn csv_files_in(dir: &Path, par: bool) -> Vec<B> {
    let mut v = vec![];
    if let Ok(rd) = std::fs::read_dir(dir) {
        let mut paths: Vec<_> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "csv").unwrap_or(false) && p.is_file())
            .collect();
        paths.sort();
        let r: Vec<Option<B>> =
            if par { paths.par_iter().map(|p| leak_file(p)).collect() } else { paths.iter().map(|p| leak_file(p)).collect() };
        v.extend(r.into_iter().flatten());
    }
    v
}

/// Event chunk size. The event-id dedup walks every (bucket, chunk) slice, so on large inputs
/// small chunks cost more in slice starts and cache misses than they gain in balance: aim for
/// about 300 chunks in all, never below CHUNK.
fn ev_chunk(files: &[B]) -> usize {
    if let Some(mb) = std::env::var("SHOP_CHUNK").ok().and_then(|s| s.parse::<usize>().ok()) {
        return mb << 20;
    }
    let total: usize = files.iter().map(|f| f.len()).sum();
    (total / 300).next_multiple_of(1 << 20).clamp(CHUNK, 8 * CHUNK)
}

fn all_chunks(files: &[B], sz: usize) -> Vec<B> {
    files.iter().flat_map(|f| chunks(f, sz)).collect()
}

#[inline]
fn pack(b: B) -> u64 {
    (b.as_ptr() as u64) | ((b.len() as u64) << 48)
}
#[inline]
fn unpack(r: u64) -> B {
    unsafe { std::slice::from_raw_parts((r & ((1u64 << 48) - 1)) as *const u8, (r >> 48) as usize) }
}

/// Lookup table for a canonical "<prefix><digits>" id (no leading zero, at most 9 digits).
#[inline]
fn canon_num(b: &[u8], prefix: &[u8]) -> Option<usize> {
    let pl = prefix.len();
    let n = b.len();
    if n <= pl || n > pl + 9 || &b[..pl] != prefix || b[pl] == b'0' {
        return None;
    }
    let mut v = 0usize;
    for &c in &b[pl..] {
        let x = c.wrapping_sub(b'0');
        if x > 9 {
            return None;
        }
        v = v * 10 + x as usize;
    }
    Some(v)
}

struct IdLookup {
    prefix: &'static [u8],
    arr: Vec<u32>,
    map: AHashMap<B, u32>,
}
impl IdLookup {
    fn new(prefix: &'static [u8], items: &[(B, u32)]) -> Self {
        let mut arr: Vec<u32> = vec![];
        let mut map = AHashMap::new();
        for &(id, v) in items {
            match canon_num(id, prefix) {
                Some(n) if n < (1 << 24) => {
                    if arr.len() <= n {
                        arr.resize(n + 1, u32::MAX);
                    }
                    if arr[n] == u32::MAX {
                        arr[n] = v;
                    }
                }
                _ => {
                    map.entry(id).or_insert(v);
                }
            }
        }
        IdLookup { prefix, arr, map }
    }
    #[inline]
    fn get(&self, b: B) -> Option<u32> {
        match canon_num(b, self.prefix) {
            Some(n) if n < (1 << 24) => self.arr.get(n).copied().filter(|&v| v != u32::MAX),
            Some(_) => self.map.get(b).copied(),
            None => {
                if self.map.is_empty() {
                    None
                } else {
                    self.map.get(b).copied()
                }
            }
        }
    }
    /// Same result as `get`; `lo` is the lowest readable address at or before `b` (a one-byte
    /// prefix and up to 8 digits are read as one word ending at the end of `b`).
    #[inline(always)]
    fn get_fast(&self, b: B, lo: usize) -> Option<u32> {
        let n = b.len();
        let p = b.as_ptr() as usize;
        if self.prefix.len() == 1 && (2..=9).contains(&n) && p + n >= lo + 8 && b[0] == self.prefix[0] && b[1] != b'0' {
            // SAFETY: [p + n - 8, p + n) lies inside the chunk that starts at lo
            let w = unsafe { std::ptr::read_unaligned((p + n - 8) as *const u64) };
            let d = (n - 1) as u32;
            let low = (u64::MAX >> (8 * d - 1)) >> 1;
            let w2 = (w & !low) | (h8(b'0') & low);
            if dig8(w2) {
                let v = parse8(w2) as usize;
                if v < (1 << 24) {
                    return self.arr.get(v).copied().filter(|&x| x != u32::MAX);
                }
            }
        }
        self.get(b)
    }
    /// `get_fast` for a two-byte prefix: same result as `get`; `lo` is the lowest readable address
    /// at or before `b` (up to 8 digits are read as one word ending at the end of `b`).
    #[inline(always)]
    fn get_fast2(&self, b: B, lo: usize) -> Option<u32> {
        let n = b.len();
        let p = b.as_ptr() as usize;
        if self.prefix.len() == 2 && (3..=10).contains(&n) && p + n >= lo + 8 && b[0] == self.prefix[0] && b[1] == self.prefix[1] && b[2] != b'0' {
            // SAFETY: [p + n - 8, p + n) lies inside the chunk that starts at lo
            let w = unsafe { std::ptr::read_unaligned((p + n - 8) as *const u64) };
            let d = (n - 2) as u32;
            let low = (u64::MAX >> (8 * d - 1)) >> 1;
            let w2 = (w & !low) | (h8(b'0') & low);
            if dig8(w2) {
                let v = parse8(w2) as usize;
                if v < (1 << 24) {
                    return self.arr.get(v).copied().filter(|&x| x != u32::MAX);
                }
            }
        }
        self.get(b)
    }
}

// ---------- identity table: id key -> person key ----------

struct IdTab {
    s: Vec<[u64; 2]>,
    shift: u32,
    mask: usize,
    /// keys whose person key is not the key itself
    mov: usize,
}
impl IdTab {
    fn with_cap(n: usize) -> Self {
        let cap = (n + n / 2).max(16).next_power_of_two();
        Self::with_slots(cap)
    }
    fn with_slots(cap: usize) -> Self {
        IdTab { s: vec![[0u64; 2]; cap], shift: 64 - cap.trailing_zeros(), mask: cap - 1, mov: 0 }
    }
    #[inline(always)]
    fn slot(&self, k: u64) -> usize {
        (mix(k) >> self.shift) as usize
    }
    #[inline(always)]
    fn pf(&self, k: u64) {
        prefetch(unsafe { self.s.as_ptr().add(self.slot(k)) });
    }
    /// returns existing value or inserts v
    #[inline]
    fn get_or_insert(&mut self, k: u64, v: u64) -> (u64, bool) {
        let mut i = self.slot(k);
        loop {
            let e = self.s[i];
            if e[0] == k {
                return (e[1], false);
            }
            if e[0] == 0 {
                self.s[i] = [k, v];
                return (v, true);
            }
            i = (i + 1) & self.mask;
        }
    }
    #[inline(always)]
    fn pf_slot(&self, i: usize) {
        prefetch(unsafe { self.s.as_ptr().add(i) });
    }
    #[inline(always)]
    fn get(&self, k: u64) -> u64 {
        self.get_from(k, self.slot(k))
    }
    /// `get` with the home slot `i = slot(k)` already computed
    #[inline(always)]
    fn get_from(&self, k: u64, mut i: usize) -> u64 {
        loop {
            let e = unsafe { *self.s.get_unchecked(i) };
            if e[0] == k {
                return e[1];
            }
            if e[0] == 0 {
                return k;
            }
            i = (i + 1) & self.mask;
        }
    }
}

/// Blocked bloom filter (three bits in one u64 word per key) over the identity table's movers:
/// keys whose person key is not the key itself. It has no false negatives, so a key that misses it
/// maps to itself and needs no table lookup.
struct MoverF {
    w: Vec<u64>,
    mask: usize,
    /// movers of generator-form user IDs by generator person, when the user_id key is known
    g: Option<GpTab>,
}
impl MoverF {
    fn build(tab: &IdTab) -> Self {
        Self::build_with(tab, UKEYC.get().copied().flatten())
    }
    fn build_with(tab: &IdTab, ukey: Option<u64>) -> Self {
        let nw = (tab.s.len() / 16).max(1);
        let gn = if ukey.is_some() { 2 * tab.mov + (1 << 16) } else { 0 };
        // Without atomics: each table chunk lists its movers' filter hashes and person-table
        // entries grouped by destination range (512 KB of filter words or of person words), then
        // each range is filled by one task with plain writes that stay in cache.
        const WR: u32 = 16;
        const GR: u32 = 15;
        let (nwr, ngr) = ((nw + (1 << WR) - 1) >> WR, (gn + (1 << GR) - 1) >> GR);
        let parts: Vec<((Vec<u64>, Vec<u32>), (Vec<[u64; 2]>, Vec<u32>))> = tab
            .s
            .par_chunks(1 << 15)
            .map(|c| {
                let (mut h, mut g) = (Vec::new(), Vec::new());
                for e in c {
                    if e[0] != 0 && e[0] != e[1] {
                        h.push(mix(e[0]));
                        if let Some(key) = ukey {
                            if let Some(v) = gen_val(e[0], UPRE) {
                                let idx = gen_idx(v, key);
                                if idx & 511 < GPA && ((idx >> 9) as usize) < gn {
                                    // person key, then generator person * 16 + alias
                                    g.push([e[1], (idx >> 9) << 4 | (idx & 511)]);
                                }
                            }
                        }
                    }
                }
                (bucketize(&h, nwr, |&x| (x as usize & (nw - 1)) >> WR), bucketize(&g, ngr, |x| (x[1] >> (4 + GR)) as usize))
            })
            .collect();
        let mut w = vec![0u64; nw];
        let mut ge = vec![0u64; 2 * gn];
        rayon::join(
            || {
                w.par_chunks_mut(1 << WR).enumerate().for_each(|(r, ws)| {
                    for ((hv, ho), _) in &parts {
                        for &h in &hv[ho[r] as usize..ho[r + 1] as usize] {
                            ws[h as usize & (nw - 1) & ((1 << WR) - 1)] |= Self::bits(h);
                        }
                    }
                })
            },
            || {
                ge.par_chunks_mut(2 << GR).enumerate().for_each(|(r, gs)| {
                    for (_, (gv, go)) in &parts {
                        for x in &gv[go[r] as usize..go[r + 1] as usize] {
                            let l = 2 * ((x[1] >> 4) as usize & ((1 << GR) - 1));
                            let (p, m) = (gs[l], gs[l + 1]);
                            if p == 0 {
                                gs[l] = x[0];
                            }
                            let mixed = ((p != 0 && p != x[0]) as u64) << 63;
                            gs[l + 1] = m | mixed | (1 << (x[1] & 15));
                        }
                    }
                })
            },
        );
        let g = ukey.map(|key| GpTab { key, n: gn, e: ge });
        MoverF { w, mask: nw - 1, g }
    }
    /// person key of user key k
    #[inline(always)]
    fn person(&self, k: u64, tab: &IdTab) -> u64 {
        if let Some(p) = self.g.as_ref().and_then(|g| g.get(k)) {
            return p;
        }
        if self.hit(k) {
            tab.get(k)
        } else {
            k
        }
    }
    #[inline(always)]
    fn bits(h: u64) -> u64 {
        (1u64 << ((h >> 20) & 63)) | (1u64 << ((h >> 26) & 63)) | (1u64 << ((h >> 32) & 63))
    }
    #[inline(always)]
    fn pf(&self, k: u64) {
        prefetch(unsafe { self.w.as_ptr().add(mix(k) as usize & self.mask) });
    }
    #[inline(always)]
    fn hit(&self, k: u64) -> bool {
        let h = mix(k);
        let b = Self::bits(h);
        (unsafe { *self.w.get_unchecked(h as usize & self.mask) }) & b == b
    }
}

/// Movers among generator-form user IDs ('u' + base36(scramble(person * 512 + alias, key))),
/// indexed by generator person: two words per person, the person key of its movers and a mask of
/// the aliases (below GPA) that are movers, bit 63 set when its movers do not share one person
/// key. Lines in index order come person by person, so lookups walk it in order. Exact: canonical
/// IDs and indices are in one-to-one correspondence for any key, so a set bit is that key's own
/// mover entry and a clear bit means the key is not a mover; persons at or above n, aliases at or
/// above GPA, mixed persons and other IDs take the filter and table.
struct GpTab {
    key: u64,
    n: usize,
    e: Vec<u64>,
}
const GPA: u64 = 16;
/// pack6 code of 'u'
const UPRE: u64 = 58;
impl GpTab {
    /// person key of user key k, or None when the filter and table must answer
    #[inline(always)]
    fn get(&self, k: u64) -> Option<u64> {
        let v = gen_val(k, UPRE)?;
        let idx = gen_idx(v, self.key);
        let (gp, a) = ((idx >> 9) as usize, idx & 511);
        if a >= GPA || gp >= self.n {
            return None;
        }
        // SAFETY: gp < n and e holds 2n words
        let (p, m) = unsafe { (*self.e.get_unchecked(2 * gp), *self.e.get_unchecked(2 * gp + 1)) };
        if m >> 63 != 0 {
            return None;
        }
        Some(if (m >> a) & 1 != 0 { p } else { k })
    }
}

static UKEYC: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();

const PFD: usize = 12;
/// prefetch distance of the user -> person passes (mover filter, then table), sized for DRAM latency
const RPD: usize = 32;

// ---------- records ----------

/// Dedup record of one event line: packed event_id, ingest key (`ing32`) and the line's index in
/// its chunk's record list.
#[derive(Clone, Copy)]
struct DRec {
    eid: u64,
    ing: u32,
    loc: u32,
}

/// Dense dedup record: the generator index of the event_id, the ingest key and the line's record.
#[derive(Clone, Copy)]
struct DR<E> {
    i: u32,
    ing: u32,
    e: E,
}

/// One 64-byte cache line.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Line([u64; 8]);

/// records staged per bucket before they go to the slots as whole cache lines (WK records of 24 or
/// 32 bytes are 3 or 4 lines)
const WK: usize = 8;

thread_local! {
    /// a parser thread's staging lines, reused from chunk to chunk
    static WCP: std::cell::Cell<Vec<MaybeUninit<Line>>> = const { std::cell::Cell::new(Vec::new()) };
}

/// A chunk's dense dedup records, written once by the parser: a line whose event_id decodes to an
/// index below the bound goes to the `cap` slots of its bucket (index >> sh) in `r`, or to `ov`
/// once those are full. A record's position is its slot (b * cap + n), or ndb * cap plus its place
/// in `ov` (bucket order after `finish`); `dside` holds the exact ingest of the records whose key
/// is inexact, by position. Slot records are staged per bucket in `wc` (WK at a time, in L2) and
/// copied to `r` with streaming stores once WK are there, so the scattered slots are written as
/// whole lines and never read in; cap is a multiple of WK and `r` is line-aligned, so every copy
/// covers whole lines.
struct Dz<E> {
    on: bool,
    key: u64,
    lim: u64,
    sh: u32,
    ndb: usize,
    cap: usize,
    r: Vec<MaybeUninit<Line>>,
    wc: Vec<MaybeUninit<Line>>,
    _e: std::marker::PhantomData<E>,
    dn: Vec<u32>,
    ov: Vec<DR<E>>,
    ovb: Vec<u32>,
    ovoff: Vec<u32>,
    dside: Vec<(u32, i64)>,
}

impl<E: Ev> Dz<E> {
    /// takes no record: every line keeps the event_id hash path
    fn off() -> Self {
        Dz { on: false, key: 0, lim: 0, sh: 0, ndb: 0, cap: 0, r: vec![], wc: vec![], _e: std::marker::PhantomData, dn: vec![], ov: vec![], ovb: vec![], ovoff: vec![], dside: vec![] }
    }
    /// for a chunk of `len` bytes: room for the expected bucket size plus a margin (the rest
    /// overflows)
    fn new(cfg: Option<IdxCfg>, len: usize) -> Self {
        let Some(c) = cfg else { return Self::off() };
        let lines = len as f64 * c.lpb;
        let mean = lines * (1u64 << c.sh) as f64 / ((c.lim - 64) as f64 / 1.3).max(1.0) * 1.04;
        let cap = ((mean + 1.5 * mean.sqrt() + 1.0) as usize).min(lines as usize + 16).next_multiple_of(WK);
        let sz = std::mem::size_of::<DR<E>>();
        let nl = (c.ndb * cap * sz).div_ceil(64);
        let mut r = Vec::with_capacity(nl);
        // SAFETY: MaybeUninit needs no initialization
        unsafe { r.set_len(nl) };
        let wl = (c.ndb * WK * sz).div_ceil(64);
        let mut wc = WCP.with(|p| p.take());
        if wc.len() < wl {
            wc = Vec::with_capacity(wl);
            // SAFETY: as above
            unsafe { wc.set_len(wl) };
        }
        Dz { on: true, key: c.key, lim: c.lim, sh: c.sh, ndb: c.ndb, cap, r, wc, _e: std::marker::PhantomData, dn: vec![0; c.ndb], ov: vec![], ovb: vec![], ovoff: vec![], dside: vec![] }
    }
    /// the dense index of a packed event_id, if it has one
    #[inline(always)]
    fn idx(&self, k: u64) -> Option<u64> {
        if !self.on {
            return None;
        }
        match gen_val(k, 42) {
            Some(v) => {
                let i = gen_idx(v, self.key);
                if i < self.lim {
                    Some(i)
                } else {
                    None
                }
            }
            None => None,
        }
    }
    /// `idx(pack6(id))` for an ID of `n` bytes at `p` (None also when the ID does not pack)
    /// SAFETY: the 9 bytes from `p` are readable
    #[inline(always)]
    unsafe fn idx_txt(&self, p: *const u8, n: usize) -> Option<u64> {
        if !self.on {
            return None;
        }
        match gen_val_txt(p, n, b'e') {
            Some(v) => {
                let i = gen_idx(v, self.key);
                if i < self.lim {
                    Some(i)
                } else {
                    None
                }
            }
            None => None,
        }
    }
    /// the dense index of an event id whose base-36 value `val` is already decoded and valid
    #[inline(always)]
    fn idx_val(&self, val: u64) -> Option<u64> {
        let i = gen_idx(val, self.key);
        if self.on & (i < self.lim) {
            Some(i)
        } else {
            None
        }
    }
    #[inline(always)]
    fn put(&mut self, i: u64, ing: u32, e: E, exact: bool, ingest: i64) {
        let b = (i >> self.sh) as usize;
        let x = DR { i: i as u32, ing, e };
        // SAFETY: i < lim, so b < ndb; a slot below cap lies in b's stretch of r
        let n = unsafe { *self.dn.get_unchecked(b) } as usize;
        let pos = if n < self.cap {
            // SAFETY: b < ndb, so b's WK staging records lie in wc; n + 1 - WK is a multiple of
            // WK below cap, so the copied slots lie in b's stretch of r
            unsafe {
                let w = (self.wc.as_mut_ptr() as *mut DR<E>).add(b * WK);
                w.add(n % WK).write(x);
                *self.dn.get_unchecked_mut(b) = n as u32 + 1;
                if (n + 1) % WK == 0 {
                    let d = (self.r.as_mut_ptr() as *mut DR<E>).add(b * self.cap + n + 1 - WK);
                    stream_lines(w as *const Line, d as *mut Line, WK * std::mem::size_of::<DR<E>>() / 64);
                }
            }
            b * self.cap + n
        } else {
            self.ov_put(b, x)
        };
        if !exact {
            self.dside.push((pos as u32, ingest));
        }
    }
    #[cold]
    #[inline(never)]
    fn ov_put(&mut self, b: usize, x: DR<E>) -> usize {
        self.ov.push(x);
        self.ovb.push(b as u32);
        self.ndb * self.cap + self.ov.len() - 1
    }
    /// copies the staged tail of every bucket to its slots, orders `ov` by bucket (keeping the
    /// line order) and sorts `dside` by position
    fn finish(&mut self) {
        if self.on {
            for b in 0..self.ndb {
                let n = self.dn[b] as usize;
                let t = n % WK;
                // SAFETY: the t records staged last for b are its slots n - t .. n
                unsafe {
                    let w = (self.wc.as_ptr() as *const DR<E>).add(b * WK);
                    let d = (self.r.as_mut_ptr() as *mut DR<E>).add(b * self.cap + n - t);
                    std::ptr::copy_nonoverlapping(w, d, t);
                }
            }
            sfence();
            WCP.with(|p| p.set(std::mem::take(&mut self.wc)));
        }
        if !self.ov.is_empty() {
            let base = self.ndb * self.cap;
            let mut off = vec![0u32; self.ndb + 1];
            for &b in &self.ovb {
                off[b as usize + 1] += 1;
            }
            for i in 0..self.ndb {
                off[i + 1] += off[i];
            }
            let mut w = off.clone();
            let mut perm = vec![0u32; self.ov.len()];
            let mut ov = self.ov.clone();
            for (j, &b) in self.ovb.iter().enumerate() {
                let q = w[b as usize];
                ov[q as usize] = self.ov[j];
                perm[j] = q;
                w[b as usize] += 1;
            }
            for x in self.dside.iter_mut() {
                if x.0 as usize >= base {
                    x.0 = (base + perm[x.0 as usize - base] as usize) as u32;
                }
            }
            self.ov = ov;
            self.ovoff = off;
            self.ovb = vec![];
        }
        self.dside.sort_unstable_by_key(|x| x.0);
    }
    fn len(&self) -> usize {
        self.dn.iter().map(|&n| n as usize).sum::<usize>() + self.ov.len()
    }
    /// bucket b's records: its slots and the position of the first, then its overflow records and
    /// the position of the first
    #[inline]
    fn seg(&self, b: usize) -> (&[DR<E>], usize, &[DR<E>], usize) {
        if b >= self.dn.len() {
            return (&[], 0, &[], 0);
        }
        // SAFETY: put wrote the first dn[b] slots of b's stretch
        let s = unsafe { std::slice::from_raw_parts((self.r.as_ptr() as *const DR<E>).add(b * self.cap), self.dn[b] as usize) };
        let (o, p) = if self.ovoff.is_empty() {
            (&self.ov[..0], 0)
        } else {
            let (a, z) = (self.ovoff[b] as usize, self.ovoff[b + 1] as usize);
            (&self.ov[a..z], self.ndb * self.cap + a)
        };
        (s, b * self.cap, o, p)
    }
    fn slots(&self) -> usize {
        self.ndb * self.cap
    }
}

/// Copies `n` whole lines from `s` to `d` with streaming stores (no read of the target lines).
#[inline(always)]
unsafe fn stream_lines(s: *const Line, d: *mut Line, n: usize) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use std::arch::x86_64::*;
        for k in 0..n {
            let a = _mm256_load_si256(s.add(k) as *const __m256i);
            let b = _mm256_load_si256((s.add(k) as *const __m256i).add(1));
            _mm256_stream_si256(d.add(k) as *mut __m256i, a);
            _mm256_stream_si256((d.add(k) as *mut __m256i).add(1), b);
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    std::ptr::copy_nonoverlapping(s, d, n);
}

/// Orders earlier streaming stores before later stores.
#[inline(always)]
fn sfence() {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: sse is part of x86_64
    unsafe {
        std::arch::x86_64::_mm_sfence()
    };
}

const IOFF: i64 = 1 << 31;

/// Order-preserving 32-bit key of an ingest time (ms): whole seconds from `base_s`, offset by
/// 2^31 and clamped. `exact` is false when the key does not give the time back exactly (clamped,
/// or not whole seconds); such records keep their full value on the side.
#[inline(always)]
fn ing32(ingest: i64, base_s: i64) -> (u32, bool) {
    let v = ingest.div_euclid(1000).saturating_sub(base_s).saturating_add(IOFF);
    let exact = ingest.rem_euclid(1000) == 0 && v > 0 && v < u32::MAX as i64;
    (v.clamp(0, u32::MAX as i64) as u32, exact)
}

/// Raw output pointer shared by tasks that write disjoint ranges.
struct SPtr<T>(*mut T);
unsafe impl<T> Send for SPtr<T> {}
unsafe impl<T> Sync for SPtr<T> {}
impl<T> SPtr<T> {
    #[inline(always)]
    fn at(&self, i: usize) -> *mut T {
        unsafe { self.0.add(i) }
    }
}

/// One event line after parsing: user or person key (0 marks a dropped line), ts relative to
/// the window start, and for a purchase its price (never 0) and category index.
trait Tp: Copy + Send + Sync {
    fn rts(&self) -> u64;
    /// (price, category) of a purchase
    fn pc(&self) -> Option<(u64, u32)>;
}

trait Ev: Tp {
    /// the keyless record kept in em / evs (the key lives in RunD / Run)
    type S: Tp;
    fn nil() -> Self;
    fn mk(key: u64, rts: u64, price: u32, cat: u32) -> Self;
    fn key(&self) -> u64;
    fn set_key(&mut self, k: u64);
    fn slim(&self) -> Self::S;
}

/// 8-byte keyless record of EvL.
#[derive(Clone, Copy)]
#[repr(C)]
struct EvS {
    ts: u32,
    meta: u32,
}

impl Tp for EvS {
    #[inline(always)]
    fn rts(&self) -> u64 {
        self.ts as u64
    }
    #[inline(always)]
    fn pc(&self) -> Option<(u64, u32)> {
        let p = self.meta & 0xFF_FFFF;
        if p != 0 { Some((p as u64, self.meta >> 24)) } else { None }
    }
}

/// 16-byte keyless record of EvW.
#[derive(Clone, Copy)]
#[repr(C)]
struct EvSW {
    ts: u64,
    meta: u64,
}

impl Tp for EvSW {
    #[inline(always)]
    fn rts(&self) -> u64 {
        self.ts
    }
    #[inline(always)]
    fn pc(&self) -> Option<(u64, u32)> {
        let p = self.meta & 0xFF_FFFF;
        if p != 0 { Some((p, (self.meta >> 24) as u32)) } else { None }
    }
}

/// 16-byte record: needs a window of at most 2^32 ms and at most 256 categories.
#[derive(Clone, Copy)]
struct EvL {
    key: u64,
    ts: u32,
    meta: u32,
}

impl Tp for EvL {
    #[inline(always)]
    fn rts(&self) -> u64 {
        self.ts as u64
    }
    #[inline(always)]
    fn pc(&self) -> Option<(u64, u32)> {
        let p = self.meta & 0xFF_FFFF;
        if p != 0 { Some((p as u64, self.meta >> 24)) } else { None }
    }
}

impl Ev for EvL {
    type S = EvS;
    #[inline(always)]
    fn slim(&self) -> EvS {
        EvS { ts: self.ts, meta: self.meta }
    }
    #[inline(always)]
    fn nil() -> Self {
        EvL { key: 0, ts: 0, meta: 0 }
    }
    #[inline(always)]
    fn mk(key: u64, rts: u64, price: u32, cat: u32) -> Self {
        EvL { key, ts: rts as u32, meta: price | (cat << 24) }
    }
    #[inline(always)]
    fn key(&self) -> u64 {
        self.key
    }
    #[inline(always)]
    fn set_key(&mut self, k: u64) {
        self.key = k;
    }
}

/// 24-byte record for any window and category count.
#[derive(Clone, Copy)]
struct EvW {
    key: u64,
    ts: u64,
    meta: u64,
}

impl Tp for EvW {
    #[inline(always)]
    fn rts(&self) -> u64 {
        self.ts
    }
    #[inline(always)]
    fn pc(&self) -> Option<(u64, u32)> {
        let p = self.meta & 0xFF_FFFF;
        if p != 0 { Some((p, (self.meta >> 24) as u32)) } else { None }
    }
}

impl Ev for EvW {
    type S = EvSW;
    #[inline(always)]
    fn slim(&self) -> EvSW {
        EvSW { ts: self.ts, meta: self.meta }
    }
    #[inline(always)]
    fn nil() -> Self {
        EvW { key: 0, ts: 0, meta: 0 }
    }
    #[inline(always)]
    fn mk(key: u64, rts: u64, price: u32, cat: u32) -> Self {
        EvW { key, ts: rts, meta: price as u64 | ((cat as u64) << 24) }
    }
    #[inline(always)]
    fn key(&self) -> u64 {
        self.key
    }
    #[inline(always)]
    fn set_key(&mut self, k: u64) {
        self.key = k;
    }
}

#[derive(Clone, Copy)]
struct ClickRec {
    key: u64,
    ts: i64,
    cid: u64,
    camp: u32,
}

/// One click line of `scan4`: its record when the ids are not empty, the campaign is known and
/// the time parses and lies in [ws - WEEK, we). `lo` is the chunk start (for `get_fast2`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn click_line(f: &[B; 4], camp: &IdLookup, rs: &RandomState, ws: i64, we: i64, lo: usize, tc: &mut TsCache, out: &mut Vec<ClickRec>) {
    let [cid, uid, cmp, ts] = *f;
    if cid.is_empty() || uid.is_empty() {
        return;
    }
    let Some(ci) = camp.get_fast2(cmp, lo) else { return };
    let Some(t) = ts_fast(ts, tc) else { return };
    if t < ws - WEEK || t >= we {
        return;
    }
    out.push(ClickRec { key: ukey(rs, uid), ts: t, cid: pack(cid), camp: ci });
}

/// The click records of one chunk, in line order: `scan4` with `click_line` on every line.
#[cfg_attr(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"), allow(dead_code))]
fn parse_clicks_old(c: B, camp: &IdLookup, rs: &RandomState, ws: i64, we: i64, out: &mut Vec<ClickRec>) {
    let mut tc = TsCache::new();
    let lo = c.as_ptr() as usize;
    scan4(c, |f| click_line(f, camp, rs, ws, we, lo, &mut tc, out));
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
fn parse_clicks(c: B, camp: &IdLookup, rs: &RandomState, ws: i64, we: i64, out: &mut Vec<ClickRec>) {
    parse_clicks_old(c, camp, rs, ws, we, out)
}

/// Same records in the same order as `parse_clicks_old`. A line whose 64-byte window at its start
/// holds its newline, exactly three commas before it and only whole-field quotes (`quote_bad`)
/// is parsed in place: fields as pointer and length, F1 times through the day cache and `hms`,
/// 13-digit times through `ts13`, 25-byte times through `ts25`; any None there goes to the
/// general parser. Every other line goes to `scan4_line` and `click_line`, the last 64 bytes of
/// the chunk to `scan::<4>`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(never)]
fn parse_clicks(c: B, camp: &IdLookup, rs: &RandomState, ws: i64, we: i64, out: &mut Vec<ClickRec>) {
    use core::arch::x86_64::_bzhi_u64;
    let mut tc = TsCache::new();
    let mut dc = DayCache::new();
    let lo = c.as_ptr() as usize;
    // SAFETY (bzhi): BMI2 is enabled for this build
    let bzhi = |x: u64, n: u32| unsafe { _bzhi_u64(x, n) };
    let blsr = |x: u64| x & x.wrapping_sub(1);
    // SAFETY (sl): only called on fields found inside the chunk
    let sl = |p: *const u8, n: usize| -> B { unsafe { std::slice::from_raw_parts(p, n) } };
    let tlo = ws - WEEK;
    let len = c.len();
    let base = c.as_ptr();
    let mut pos = 0usize;
    let mut sbuf: Vec<[B; 4]> = Vec::new();
    let pfd = pf_dist();
    'ln: while pos + 64 <= len {
        if pfd != 0 {
            // a prefetch never faults, so the address may lie past the chunk
            prefetch(base.wrapping_add(pos + pfd));
        }
        // SAFETY: pos + 64 <= len
        let p = unsafe { base.add(pos) };
        let (ca, qa, na) = masks64(p);
        let l = na.trailing_zeros();
        let cl = bzhi(ca, l);
        let e0 = cl.trailing_zeros();
        let c1 = blsr(cl);
        let e1 = c1.trailing_zeros();
        let c2 = blsr(c1);
        let e2 = c2.trailing_zeros();
        let c3 = blsr(c2);
        let (x, bad) = quote_bad(bzhi(qa, l), ca | na);
        'slow: {
            // l: the newline; e0..e2: the commas; xq: whether each field opens with a quote;
            // win: the readable bytes at p
            let (l, e0, e1, e2, xq, win) = if (na != 0) & (c2 != 0) & (c3 == 0) & (bad == 0) {
                (l, e0, e1, e2, [x & 1, (x >> (e0 + 1)) & 1, (x >> (e1 + 1)) & 1, (x >> (e2 + 1)) & 1], 64)
            } else if (na == 0) & (pos + 128 <= len) {
                // a line longer than the window (long ids): the same rule on 128 bytes
                // SAFETY: pos + 128 <= len
                match unsafe { line4_128(p, ca, qa) } {
                    Some((l, e0, e1, e2, xq)) => (l, e0, e1, e2, xq, 128),
                    None => break 'slow,
                }
            } else {
                break 'slow;
            };
            #[cfg(test)]
            SCAN7_FAST.with(|c| c.set(c.get() + 1));
            pos += (l + 1) as usize;
            let fld = |q: u64, s: u32, e: u32| {
                let q = q as u32;
                // SAFETY: s <= e < win inside the window; a quoted field has e - s >= 2
                (unsafe { p.add((s + q) as usize) }, (e - s - 2 * q) as usize)
            };
            let (pc, nc) = fld(xq[0], 0, e0);
            let (pu, nu) = fld(xq[1], e0 + 1, e1);
            if (nc == 0) | (nu == 0) {
                continue 'ln;
            }
            let (pm, nm) = fld(xq[2], e1 + 1, e2);
            let Some(ci) = camp.get_fast2(sl(pm, nm), lo) else { continue 'ln };
            let (pt, nt) = fld(xq[3], e2 + 1, l);
            // SAFETY (ldp, day_get, hms, ts25): a 20- or 25-byte field lies inside the window.
            // ts13: the time is the fourth field and both ids are not empty, so the 3 bytes
            // before it and the 16 bytes from pt - 3 lie inside the window.
            let f = if nt == 20 && unsafe { *pt.add(19) } == b'Z' {
                let (ta, tb) = unsafe { (ldp(pt), ldp(pt.add(3))) };
                match unsafe { (day_get(ta, tb, pt, &mut dc), hms(pt.add(11))) } {
                    (Some(d), Some(h)) => Some((d + h) * 1000),
                    _ => None,
                }
            } else if nt == 13 {
                unsafe { ts13(pt) }
            } else if nt == 25 {
                unsafe { ts25(pt, D0_A, D0_B, D0_S, &mut dc) }
            } else {
                None
            };
            let Some(t) = f.or_else(|| parse_ts_c(sl(pt, nt), &mut tc)) else { continue 'ln };
            if (t < tlo) | (t >= we) {
                continue 'ln;
            }
            let uid = sl(pu, nu);
            // SAFETY (pack6x1): e0 + 17 < win puts the 16 bytes from pu inside the window
            let key = match if e0 + 17 < win { unsafe { pack6x1(pu, nu) } } else { (0, false) } {
                (k, true) => k,
                _ => ukey(rs, uid),
            };
            out.push(ClickRec { key, ts: t, cid: pack(sl(pc, nc)), camp: ci });
            continue 'ln;
        }
        sbuf.clear();
        pos = scan4_line(c, pos, &mut sbuf);
        for f in &sbuf {
            click_line(f, camp, rs, ws, we, lo, &mut tc, out);
        }
    }
    if pos < len {
        scan::<4>(&c[pos..], |f| click_line(f, camp, rs, ws, we, lo, &mut tc, out));
    }
}

struct UD {
    key: u64,
    day: i32,
    sessions: u32,
    events: u32,
    rev: u64,
    top: u32,
}

/// Scatter `v` into an exact-size vec ordered by bucket `bk(x)`; returns (vec, offsets[nb+1]).
fn bucketize<T: Copy>(v: &[T], nb: usize, bk: impl Fn(&T) -> usize) -> (Vec<T>, Vec<u32>) {
    let mut cnt = vec![0u32; nb + 1];
    for x in v {
        cnt[bk(x) + 1] += 1;
    }
    for i in 0..nb {
        cnt[i + 1] += cnt[i];
    }
    let mut w = cnt.clone();
    let mut out: Vec<T> = Vec::with_capacity(v.len());
    {
        let sp: &mut [MaybeUninit<T>] = out.spare_capacity_mut();
        for x in v {
            let b = bk(x);
            sp[w[b] as usize].write(*x);
            w[b] += 1;
        }
    }
    unsafe { out.set_len(v.len()) };
    (out, cnt)
}

/// `bucketize` when the bucket counts are already known: `cnt[b + 1]` = the number of items in
/// bucket `b` (`cnt[0]` = 0); returns (vec ordered by bucket, offsets[nb+1]).
fn scatter_counted<T: Copy>(v: &[T], mut cnt: Vec<u32>, bk: impl Fn(&T) -> usize) -> (Vec<T>, Vec<u32>) {
    let nb = cnt.len() - 1;
    for i in 0..nb {
        cnt[i + 1] += cnt[i];
    }
    debug_assert_eq!(cnt[nb] as usize, v.len());
    let mut w = cnt.clone();
    let mut out: Vec<T> = Vec::with_capacity(v.len());
    {
        let sp: &mut [MaybeUninit<T>] = out.spare_capacity_mut();
        for x in v {
            let b = bk(x);
            sp[w[b] as usize].write(*x);
            w[b] += 1;
        }
    }
    assert_eq!(cnt[nb] as usize, v.len());
    unsafe { out.set_len(v.len()) };
    (out, cnt)
}

#[inline(always)]
fn pbk(key: u64) -> usize {
    (mix(key) >> 20) as usize & (PB - 1)
}

// ---------- per-bucket processing ----------

struct DayAcc {
    key: u64,
    day: i64,
    sessions: u32,
    events: u32,
    rev: u64,
    cats: Vec<(u32, u64)>,
}

fn flush_day(d: &mut DayAcc, out: &mut Vec<UD>) {
    if d.sessions == 0 {
        return;
    }
    let mut top = u32::MAX;
    let mut best = 0u64;
    for &(c, s) in &d.cats {
        if s > best || (s == best && c < top) {
            best = s;
            top = c;
        }
    }
    out.push(UD { key: d.key, day: d.day as i32, sessions: d.sessions, events: d.events, rev: d.rev, top });
    d.sessions = 0;
    d.events = 0;
    d.rev = 0;
    d.cats.clear();
}

struct Win {
    cnt: Vec<u32>,
    pos: Vec<u32>,
    active: Vec<u32>,
}
impl Win {
    #[inline]
    fn add(&mut self, c: u32) {
        let ci = c as usize;
        if self.cnt[ci] == 0 {
            self.pos[ci] = self.active.len() as u32;
            self.active.push(c);
        }
        self.cnt[ci] += 1;
    }
    #[inline]
    fn remove(&mut self, c: u32) {
        let ci = c as usize;
        self.cnt[ci] -= 1;
        if self.cnt[ci] == 0 {
            let p = self.pos[ci] as usize;
            let last = *self.active.last().unwrap();
            self.active.swap_remove(p);
            if last != c {
                self.pos[last as usize] = p as u32;
            }
        }
    }
}

type CampMap = AHashMap<u64, [u64; 4]>;

/// One dense (campaign, day) cell: sessions, purchases (u32: each is at most the number of kept
/// lines, which is below u32::MAX) and revenue.
#[derive(Clone, Copy, Default)]
struct CaCell {
    s: u32,
    p: u32,
    r: u64,
}

/// (campaign, day) accumulator: dense when the window is small, a map otherwise. The touch count
/// lives in its own u32 array (`tch`, same index as `dense`): it is bumped for every active campaign
/// at every session start, so its array is kept small enough to stay in L2.
struct CampAcc {
    dense: Vec<CaCell>,
    tch: Vec<u32>,
    d0: i64,
    nd: usize,
    map: CampMap,
}
impl CampAcc {
    fn new(ncamp1: usize, d0: i64, nd: usize) -> Self {
        let n = if nd > 0 && nd * ncamp1 <= (1 << 22) { nd * ncamp1 } else { 0 };
        CampAcc { dense: vec![CaCell::default(); n], tch: vec![0; n], d0, nd, map: CampMap::new() }
    }
    /// Dense index of (c, d), if the cell is dense.
    #[inline(always)]
    fn di(&self, d: i64) -> Option<usize> {
        let di = d - self.d0;
        if !self.dense.is_empty() && di >= 0 && (di as usize) < self.nd {
            Some(di as usize)
        } else {
            None
        }
    }
    /// Count one session attributed to campaign c that started on day d.
    #[inline(always)]
    fn sess(&mut self, c: u32, d: i64, purch: bool, rev: u64) {
        match self.di(d) {
            Some(di) => {
                let e = &mut self.dense[c as usize * self.nd + di];
                e.s += 1;
                e.p += purch as u32;
                e.r += rev;
            }
            None => {
                let e = self.map.entry(((c as u64) << 32) | (d as u32 as u64)).or_insert([0; 4]);
                e[0] += 1;
                e[1] += purch as u64;
                e[2] += rev;
            }
        }
    }
    /// One touch for every campaign in `act` on day d.
    #[inline(always)]
    fn touch(&mut self, act: &[u32], d: i64) {
        match self.di(d) {
            Some(di) => {
                let nd = self.nd;
                let t = &mut self.tch[..];
                for &c in act {
                    t[c as usize * nd + di] += 1;
                }
            }
            None => {
                for &c in act {
                    self.map.entry(((c as u64) << 32) | (d as u32 as u64)).or_insert([0; 4])[3] += 1;
                }
            }
        }
    }
    fn merge(&mut self, o: &CampAcc) {
        for (a, b) in self.dense.iter_mut().zip(o.dense.iter()) {
            a.s += b.s;
            a.p += b.p;
            a.r += b.r;
        }
        for (a, b) in self.tch.iter_mut().zip(o.tch.iter()) {
            *a += *b;
        }
        for (&k, v) in &o.map {
            let e = self.map.entry(k).or_insert([0; 4]);
            for i in 0..4 {
                e[i] += v[i];
            }
        }
    }
    fn rows(&self) -> Vec<(u32, i32, [u64; 4])> {
        let mut rows: Vec<(u32, i32, [u64; 4])> = vec![];
        for (i, (v, &t)) in self.dense.iter().zip(self.tch.iter()).enumerate() {
            if v.s != 0 || t != 0 {
                rows.push(((i / self.nd) as u32, (self.d0 + (i % self.nd) as i64) as i32, [v.s as u64, v.p as u64, v.r, t as u64]));
            }
        }
        for (&k, &v) in &self.map {
            rows.push(((k >> 32) as u32, k as u32 as i32, v));
        }
        rows.sort_by_key(|r| (r.0, r.1));
        rows
    }
}

#[inline(always)]
fn k128<E: Ev>(e: &E) -> u128 {
    ((e.key() as u128) << 64) | e.rts() as u128
}

fn sort_small<E: Ev>(v: &mut [E]) {
    if v.len() <= 24 {
        for i in 1..v.len() {
            let x = v[i];
            let kx = k128(&x);
            let mut j = i;
            while j > 0 && k128(&v[j - 1]) > kx {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[j] = x;
        }
    } else {
        v.sort_unstable_by_key(k128);
    }
}

/// Bits of the in-bucket person digit.
const NBK: u32 = 13;

/// In-bucket person digit: hash bits disjoint from the bucket bits (`pbk`). Within a bucket,
/// events and clicks are both ordered by (digit, key): every person's records stay contiguous
/// and both lists can be merged in one walk. Persons spread evenly over the digits, so most
/// digit ranges hold a single person.
#[inline(always)]
fn dg(key: u64) -> usize {
    (mix(key) >> 36) as usize & ((1 << NBK) - 1)
}

/// Counting-sort pass of `v` into `out` on `dg`. Afterwards cnt[d] is the end of range d.
fn hk_scatter<T: Copy>(v: &[T], out: &mut Vec<T>, cnt: &mut Vec<u32>, key: impl Fn(&T) -> u64) {
    let n = v.len();
    let nb = 1usize << NBK;
    cnt.clear();
    cnt.resize(nb + 1, 0);
    for e in v {
        cnt[dg(key(e)) + 1] += 1;
    }
    for i in 0..nb {
        cnt[i + 1] += cnt[i];
    }
    out.clear();
    out.reserve(n);
    {
        let sp = out.spare_capacity_mut();
        for e in v {
            let d = dg(key(e));
            sp[cnt[d] as usize].write(*e);
            cnt[d] += 1;
        }
    }
    unsafe { out.set_len(n) };
}

#[inline(always)]
fn isort_by<T: Copy>(v: &mut [T], less: impl Fn(&T, &T) -> bool) {
    for i in 1..v.len() {
        let x = v[i];
        let mut j = i;
        while j > 0 && less(&x, &v[j - 1]) {
            v[j] = v[j - 1];
            j -= 1;
        }
        v[j] = x;
    }
}

thread_local! {
    static TS_KEYS: std::cell::RefCell<(Vec<u64>, Vec<u64>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Bits of the quantized ts in `sort_evs_ts`.
const TSB: u32 = 22;

/// Fast path of `sort_evs` for input in random ts order: one pass packs
/// dg(key) | ts >> q (TSB bits) | index into u64s and counts three digits, two 11-bit LSD passes
/// order the u64s by quantized ts, and one counting scatter by dg moves the records. Each dg
/// range is then ts-ordered up to the quantum: a one-key range gets an insertion fixup on rts,
/// a multi-key range a stable split by key and the same fixup. Returns false when the fields do
/// not fit in 64 bits.
fn sort_evs_ts<E: Ev>(evs: &mut Vec<E>, tmp: &mut Vec<E>, h: &mut Vec<u32>, span: i64) -> bool {
    if !sort_ts_into(&[&evs[..]], evs.len(), tmp, h, span) {
        return false;
    }
    std::mem::swap(evs, tmp);
    true
}

/// `sort_evs_ts` over the concatenation of the non-empty slices `sl` (n records in all), written
/// into `out`: the index field holds slice number << ob | offset, which orders like the index in
/// the concatenation, so the result is the one `sort_evs_ts` gives on the concatenated records.
/// Returns false (and touches nothing) when the fields do not fit in 64 bits.
fn sort_ts_into<E: Ev>(sl: &[&[E]], n: usize, out: &mut Vec<E>, h: &mut Vec<u32>, span: i64) -> bool {
    if span < 0 || n == 0 {
        return false;
    }
    let sb = 64 - (span as u64).leading_zeros();
    let tb = if sl.len() > 1 { 64 - ((sl.len() - 1) as u64).leading_zeros() } else { 0 };
    let ml = sl.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
    let ob = (64 - ((ml - 1) as u64).leading_zeros()).max(1);
    let ib = tb + ob;
    if ib + TSB + NBK > 64 {
        return false;
    }
    let om = (1u64 << ob) - 1;
    let bp: Vec<*const E> = sl.iter().map(|s| s.as_ptr()).collect();
    let q = sb.saturating_sub(TSB);
    let ds = ib + TSB;
    let im = (1u64 << ib) - 1;
    TS_KEYS.with(|c| {
        let (ka, kb) = &mut *c.borrow_mut();
        h.clear();
        h.resize(2048 * 2 + (1 << NBK), 0);
        let h = &mut h[..];
        ka.clear();
        ka.reserve(n);
        kb.clear();
        kb.reserve(n);
        // SAFETY: every slot of ka is written below; kb is fully written by a pass before it
        // is read (a skipped pass leaves src = ka).
        unsafe {
            ka.set_len(n);
            kb.set_len(n);
        }
        let mut i = 0usize;
        for (si, s) in sl.iter().enumerate() {
            let sx = (si as u64) << ob;
            for (j, e) in s.iter().enumerate() {
                let t = e.rts() >> q;
                let d = dg(e.key()) as u64;
                // SAFETY: i < n = ka.len() (the slices hold n records); the indexes are masked
                // below the histogram sizes.
                unsafe {
                    *ka.get_unchecked_mut(i) = (d << ds) | (t << ib) | sx | j as u64;
                    *h.get_unchecked_mut((t & 2047) as usize) += 1;
                    *h.get_unchecked_mut(2048 + ((t >> 11) & 2047) as usize) += 1;
                    *h.get_unchecked_mut(4096 + d as usize) += 1;
                }
                i += 1;
            }
        }
        debug_assert_eq!(i, n);
        let (mut src, mut dst) = (&mut ka[..], &mut kb[..]);
        for (off, sh) in [(0usize, ib), (2048, ib + 11)] {
            let hj = &mut h[off..off + 2048];
            if hj.iter().any(|&c| c as usize == n) {
                continue;
            }
            let mut s = 0u32;
            for x in hj.iter_mut() {
                let c = *x;
                *x = s;
                s += c;
            }
            for &k in src.iter() {
                let d = ((k >> sh) & 2047) as usize;
                // SAFETY: d < 2048; the prefix sums keep every position below n.
                unsafe {
                    let p = hj.get_unchecked_mut(d);
                    *dst.get_unchecked_mut(*p as usize) = k;
                    *p += 1;
                }
            }
            std::mem::swap(&mut src, &mut dst);
        }
        let hj = &mut h[4096..4096 + (1 << NBK)];
        let mut s = 0u32;
        for x in hj.iter_mut() {
            let c = *x;
            *x = s;
            s += c;
        }
        out.clear();
        out.reserve(n);
        // SAFETY: the scatter below writes each of the n slots exactly once.
        unsafe { out.set_len(n) };
        for &k in src.iter() {
            let d = (k >> ds) as usize;
            let x = k & im;
            // SAFETY: x >> ob < sl.len() and x & om < that slice's length; d < 2^NBK;
            // positions stay below n.
            unsafe {
                let e = *bp.get_unchecked((x >> ob) as usize).add((x & om) as usize);
                let p = hj.get_unchecked_mut(d);
                *out.get_unchecked_mut(*p as usize) = e;
                *p += 1;
            }
        }
        let mut s = 0usize;
        for &e in hj.iter() {
            let e = e as usize;
            if e - s > 1 {
                let r = &mut out[s..e];
                let k0 = r[0].key();
                let mut mk = false;
                let mut ok = true;
                for w in 1..r.len() {
                    mk |= r[w].key() != k0;
                    ok &= r[w].rts() >= r[w - 1].rts();
                }
                if mk {
                    fix_multi(r);
                } else if !ok {
                    isort_by(r, |a, b| a.rts() < b.rts());
                }
            }
            s = e;
        }
    });
    true
}

/// Range whose records are nearly ts-ordered but hold several keys: stable split by key rank,
/// then an insertion fixup on rts per key. More than 16 keys or 4096 records: `sort_small`.
fn fix_multi<E: Ev>(r: &mut [E]) {
    let m = r.len();
    let mut uk = [0u64; 16];
    let mut nk = 0usize;
    for e in r.iter() {
        let k = e.key();
        if !uk[..nk].contains(&k) {
            if nk == 16 || m > 4096 {
                sort_small(r);
                return;
            }
            uk[nk] = k;
            nk += 1;
        }
    }
    let uk = &mut uk[..nk];
    uk.sort_unstable();
    let slot = |k: u64| uk.iter().position(|&x| x == k).unwrap_or(0);
    let mut c = [0usize; 17];
    for e in r.iter() {
        c[slot(e.key()) + 1] += 1;
    }
    for i in 0..nk {
        c[i + 1] += c[i];
    }
    let mut buf = [const { MaybeUninit::<E>::uninit() }; 4096];
    let mut pos = c;
    for e in r.iter() {
        let s = slot(e.key());
        buf[pos[s]].write(*e);
        pos[s] += 1;
    }
    for (o, b) in r.iter_mut().zip(&buf[..m]) {
        // SAFETY: the split wrote each of the first m slots exactly once (the counts sum to m).
        *o = unsafe { b.assume_init() };
    }
    for i in 0..nk {
        let g = &mut r[c[i]..c[i + 1]];
        if g.windows(2).any(|w| w[1].rts() < w[0].rts()) {
            isort_by(g, |a, b| a.rts() < b.rts());
        }
    }
}

/// Sort by (dg(key), key, ts); equal (key, ts) pairs may end up in any order. `span` bounds
/// every rts. A counting pass on the digit, then per range: a range shorter than RADIX_MIN with
/// few keys is ordered as packed u64s (`pack_sort`), a long one-key range with `radix_rts`.
fn sort_evs<E: Ev>(evs: &mut Vec<E>, tmp: &mut Vec<E>, cnt: &mut Vec<u32>, span: i64) {
    let n = evs.len();
    if n < 256 {
        evs.sort_unstable_by_key(|e| (dg(e.key()), k128(e)));
        return;
    }
    if sort_evs_ts(evs, tmp, cnt, span) {
        return;
    }
    hk_scatter(evs, tmp, cnt, |e| e.key());
    let fast = span >= 0 && span < (1i64 << 55);
    let sb = 64 - (span as u64).leading_zeros();
    let mut ks = vec![0u64; RADIX_MIN];
    let mut st = 0usize;
    for d in 0..1usize << NBK {
        let en = cnt[d] as usize;
        let m = en - st;
        let r = &tmp[st..en];
        if m == 1 {
            evs[st] = r[0];
        } else if m > 1 {
            let k0 = r[0].key();
            if fast && m < RADIX_MIN && pack_sort(r, &mut evs[st..en], &mut ks, sb) {
            } else if fast && m >= RADIX_MIN && r.iter().all(|e| e.key() == k0) {
                radix_rts(&mut tmp[st..en], &mut evs[st..en]);
            } else {
                evs[st..en].copy_from_slice(r);
                sort_small(&mut evs[st..en]);
            }
        }
        st = en;
    }
}

/// `sort_evs` of the concatenation of the non-empty slices `sl`, into `evs`. The fast path reads
/// the slices in place; otherwise they are copied into `evs` and sorted there.
fn sort_evs_parts<E: Ev>(sl: &[&[E]], evs: &mut Vec<E>, tmp: &mut Vec<E>, cnt: &mut Vec<u32>, span: i64) {
    let n: usize = sl.iter().map(|s| s.len()).sum();
    if n >= 256 && sort_ts_into(sl, n, evs, cnt, span) {
        return;
    }
    evs.clear();
    for s in sl {
        evs.extend_from_slice(s);
    }
    sort_evs(evs, tmp, cnt, span);
}

/// Orders `r` (2 <= len < RADIX_MIN, every rts below 2^sb) into `out` by (key, rts) when it holds
/// at most 16 keys: (key rank, rts, index) packed into u64s are sorted, then gathered. Returns
/// false and leaves `out` alone for more keys or when the fields do not fit in 64 bits.
#[inline(always)]
fn pack_sort<E: Ev>(r: &[E], out: &mut [E], ks: &mut [u64], sb: u32) -> bool {
    let m = r.len();
    let ib = 64 - ((m - 1) as u64).leading_zeros();
    let sh = sb + ib;
    if sh > 60 {
        return false;
    }
    let k0 = r[0].key();
    let mut uk = [0u64; 16];
    uk[0] = k0;
    let mut nk = 1usize;
    let ks = &mut ks[..m];
    for (i, e) in r.iter().enumerate() {
        let k = e.key();
        let mut s = 0usize;
        if k != k0 {
            s = 1;
            while s < nk && uk[s] != k {
                s += 1;
            }
            if s == nk {
                if nk == 16 {
                    return false;
                }
                uk[nk] = k;
                nk += 1;
            }
        }
        ks[i] = ((s as u64) << sh) | (e.rts() << ib) | i as u64;
    }
    if nk > 1 {
        // slot (order of first appearance) -> rank of its key
        let mut rk = [0u64; 16];
        for s in 0..nk {
            rk[s] = uk[..nk].iter().filter(|&&x| x < uk[s]).count() as u64;
        }
        let lo = (1u64 << sh) - 1;
        for k in ks.iter_mut() {
            *k = (rk[(*k >> sh) as usize] << sh) | (*k & lo);
        }
    }
    ks.sort_unstable();
    let im = (1u64 << ib) - 1;
    for (o, &k) in out.iter_mut().zip(ks.iter()) {
        *o = r[(k & im) as usize];
    }
    true
}

/// Smallest one-person range that `sort_evs` orders with `radix_rts`.
const RADIX_MIN: usize = 2048;

/// Stable LSD radix sort of `a` by rts (11-bit digits) into `b`; `a` is scratch.
fn radix_rts<E: Ev>(a: &mut [E], b: &mut [E]) {
    let n = a.len();
    let mx = a.iter().map(|e| e.rts()).max().unwrap_or(0);
    let np = ((64 - mx.leading_zeros()) as usize).div_ceil(11).max(1);
    let mut h = vec![[0u32; 2048]; np];
    for e in a.iter() {
        let t = e.rts();
        for (j, hj) in h.iter_mut().enumerate() {
            hj[((t >> (11 * j)) & 2047) as usize] += 1;
        }
    }
    let (mut src, mut dst) = (a, b);
    let mut in_b = false;
    for (j, hj) in h.iter_mut().enumerate() {
        if hj.iter().any(|&c| c as usize == n) {
            continue;
        }
        let mut s = 0u32;
        for x in hj.iter_mut() {
            let c = *x;
            *x = s;
            s += c;
        }
        for e in src.iter() {
            let d = ((e.rts() >> (11 * j)) & 2047) as usize;
            dst[hj[d] as usize] = *e;
            hj[d] += 1;
        }
        std::mem::swap(&mut src, &mut dst);
        in_b = !in_b;
    }
    if !in_b {
        dst.copy_from_slice(src);
    }
}

#[inline(always)]
fn click_less(a: &ClickRec, b: &ClickRec) -> bool {
    (a.key, a.ts) < (b.key, b.ts)
}

/// Last touch among clicks of one person with one ts that do not all share a campaign: the
/// campaign of the click with the byte-wise smallest click_id. Copies of a click are equal here.
#[inline(never)]
fn min_cid_camp(g: &[ClickRec]) -> u32 {
    let mut w = &g[0];
    for c in &g[1..] {
        if unpack(c.cid) < unpack(w.cid) {
            w = c;
        }
    }
    w.camp
}

/// Sort clicks by (dg(key), key, ts): a counting pass on the digit, then a small sort per range.
/// Clicks of one key with equal ts stay in no particular order: the sessionizer reads them as a
/// set, and its last-touch choice orders a tie by click_id bytes itself (`min_cid_camp`), so the
/// sort never touches the click_id bytes.
fn sort_clicks(cl: &mut Vec<ClickRec>, tmp: &mut Vec<ClickRec>, cnt: &mut Vec<u32>) {
    let n = cl.len();
    if n < 256 {
        cl.sort_unstable_by_key(|c| (dg(c.key), c.key, c.ts));
        return;
    }
    if ck_rdx() && sort_clicks_rdx(cl, tmp, cnt) {
        return;
    }
    hk_scatter(cl, tmp, cnt, |c| c.key);
    let mut pk: Vec<u64> = vec![];
    // walk the non-empty digit ranges only: a bucket holds fewer clicks than there are digits
    let mut st = 0usize;
    while st < n {
        let en = cnt[dg(tmp[st].key)] as usize;
        let m = en - st;
        if m == 1 {
            cl[st] = tmp[st];
        } else if m > 32 && sort_one_clicker(&tmp[st..en], &mut cl[st..en], &mut pk) {
        } else if m > 1 {
            let o = &mut cl[st..en];
            o.copy_from_slice(&tmp[st..en]);
            if m <= 32 {
                isort_by(o, click_less);
            } else {
                o.sort_unstable_by_key(|c| (c.key, c.ts));
            }
        }
        st = en;
    }
}

fn ck_rdx() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("SHOP_CKRDX").map(|v| v != "0").unwrap_or(true))
}

thread_local! {
    static CK_KEYS: std::cell::RefCell<(Vec<u64>, Vec<u64>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Bucket-wide click sort without per-range sorts: (dg, ts quantum, index) packed into u64s are
/// LSD radix sorted (two 11-bit ts passes, one digit pass), the records are gathered once, then
/// each digit range that is not yet in (key, ts) order (several keys, or two clicks in one
/// quantum) is sorted. Returns false (and leaves `cl` alone) when the fields do not fit.
fn sort_clicks_rdx(cl: &mut Vec<ClickRec>, tmp: &mut Vec<ClickRec>, cnt: &mut Vec<u32>) -> bool {
    let n = cl.len();
    let ib = 64 - ((n - 1) as u64).leading_zeros();
    if ib + 22 + NBK > 64 {
        return false;
    }
    let (mut lo, mut hi) = (i64::MAX, i64::MIN);
    for c in cl.iter() {
        lo = lo.min(c.ts);
        hi = hi.max(c.ts);
    }
    let Some(span) = hi.checked_sub(lo) else { return false };
    let q = (64 - (span as u64).leading_zeros()).saturating_sub(22);
    let (sht, shd) = (ib, ib + 22);
    const H1: usize = 2048;
    const HD: usize = 4096;
    cnt.clear();
    cnt.resize(HD + (1 << NBK), 0);
    CK_KEYS.with(|kk| {
        let (ka, kb) = &mut *kk.borrow_mut();
        ka.clear();
        ka.reserve(n);
        kb.clear();
        kb.resize(n, 0);
        for (j, c) in cl.iter().enumerate() {
            let t = (c.ts.wrapping_sub(lo) as u64) >> q;
            let d = dg(c.key);
            ka.push(((d as u64) << shd) | (t << sht) | j as u64);
            cnt[(t & 2047) as usize] += 1;
            cnt[H1 + (t >> 11) as usize] += 1;
            cnt[HD + d] += 1;
        }
        let (mut src, mut dst) = (&mut ka[..], &mut kb[..]);
        for (base, len, sh, mask) in [(0usize, H1, sht, 2047u64), (H1, H1, sht + 11, 2047), (HD, 1usize << NBK, shd, (1u64 << NBK) - 1)] {
            let h = &mut cnt[base..base + len];
            if base != HD && h[((src[0] >> sh) & mask) as usize] as usize == n {
                continue;
            }
            let mut s = 0u32;
            for x in h.iter_mut() {
                let c = *x;
                *x = s;
                s += c;
            }
            for &k in src.iter() {
                let b = ((k >> sh) & mask) as usize;
                dst[h[b] as usize] = k;
                h[b] += 1;
            }
            std::mem::swap(&mut src, &mut dst);
        }
        let im = (1u64 << ib) - 1;
        tmp.clear();
        tmp.reserve(n);
        {
            let sp = tmp.spare_capacity_mut();
            for (o, &k) in sp.iter_mut().zip(src.iter()) {
                o.write(cl[(k & im) as usize]);
            }
        }
        unsafe { tmp.set_len(n) };
    });
    std::mem::swap(cl, tmp);
    // after the digit pass cnt[HD + d] is the end of range d
    let mut st = 0usize;
    while st < n {
        let en = cnt[HD + dg(cl[st].key)] as usize;
        let r = &mut cl[st..en];
        let k0 = r[0].key;
        if r.len() > 1 && !(r.iter().all(|c| c.key == k0) && isort_one(r)) {
            if r.len() <= 32 {
                isort_by(r, click_less);
            } else {
                r.sort_unstable_by_key(|c| (c.key, c.ts));
            }
        }
        st = en;
    }
    true
}

/// Insertion sort of one clicker's clicks (already in ts quantum order) by ts.
/// Gives up after about two moves per click and returns false, leaving `r` a permutation.
fn isort_one(r: &mut [ClickRec]) -> bool {
    #[inline(always)]
    fn less(a: &ClickRec, b: &ClickRec) -> bool {
        a.ts < b.ts
    }
    let mut budget = 2 * r.len() + 16;
    for i in 1..r.len() {
        if !less(&r[i], &r[i - 1]) {
            continue;
        }
        let x = r[i];
        let mut j = i;
        while j > 0 && less(&x, &r[j - 1]) {
            if budget == 0 {
                r[j] = x;
                return false;
            }
            budget -= 1;
            r[j] = r[j - 1];
            j -= 1;
        }
        r[j] = x;
    }
    true
}

/// A long digit range is nearly always one heavy clicker. When every click in `src` has the same key,
/// sort (ts, index) pairs packed into u64s and gather into `out`. Returns false (and leaves `out` alone) for a range with several keys.
fn sort_one_clicker(src: &[ClickRec], out: &mut [ClickRec], pk: &mut Vec<u64>) -> bool {
    let m = src.len();
    let k0 = src[0].key;
    let (mut lo, mut hi) = (i64::MAX, i64::MIN);
    for c in src {
        if c.key != k0 {
            return false;
        }
        lo = lo.min(c.ts);
        hi = hi.max(c.ts);
    }
    if m >= 1 << 20 || hi.wrapping_sub(lo) as u64 >= 1 << 43 {
        return false;
    }
    pk.clear();
    pk.extend(src.iter().enumerate().map(|(j, c)| ((c.ts.wrapping_sub(lo) as u64) << 20) | j as u64));
    pk.sort_unstable();
    for (o, &u) in out.iter_mut().zip(pk.iter()) {
        *o = src[(u & 0xF_FFFF) as usize];
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn process_bucket<E: Ev>(
    sl: &[&[E]],
    evs: &mut Vec<E>,
    tmp: &mut Vec<E>,
    cnt: &mut Vec<u32>,
    clicks: &mut Vec<ClickRec>,
    ctmp: &mut Vec<ClickRec>,
    win: &mut Win,
    organic: u32,
    uds: &mut Vec<UD>,
    cm: &mut CampAcc,
    base: i64,
    span: i64,
) {
    let mut tm = now();
    sort_evs_parts(sl, evs, tmp, cnt, span);
    pt(7, &mut tm);
    sort_clicks(clicks, ctmp, cnt);
    pt(11, &mut tm);
    let mut ci = 0;
    let mut i = 0;
    let mut day = DayAcc { key: 0, day: 0, sessions: 0, events: 0, rev: 0, cats: vec![] };
    while i < evs.len() {
        let pid = evs[i].key();
        let pd = dg(pid);
        let mut j = i;
        while j < evs.len() && evs[j].key() == pid {
            j += 1;
        }
        while ci < clicks.len() && (dg(clicks[ci].key), clicks[ci].key) < (pd, pid) {
            ci += 1;
        }
        let mut cj = ci;
        while cj < clicks.len() && clicks[cj].key == pid {
            cj += 1;
        }
        let cs = &clicks[ci..cj];
        let (mut lo, mut hi) = (0usize, 0usize);
        day.key = pid;
        day.day = 0;
        let mut prev = 0i64;
        let mut first = true;
        let mut s_camp = 0u32;
        let mut s_day = 0i64;
        let mut s_rev = 0u64;
        let mut s_purch = false;
        let mut in_sess = false;
        for e in &evs[i..j] {
            let et = e.rts() as i64 + base;
            if first || et - prev > GAP {
                if in_sess {
                    cm.sess(s_camp, s_day, s_purch, s_rev);
                }
                first = false;
                in_sess = true;
                let s = et;
                let d = s.div_euclid(DAY);
                s_day = d;
                s_rev = 0;
                s_purch = false;
                while hi < cs.len() && cs[hi].ts <= s {
                    win.add(cs[hi].camp);
                    hi += 1;
                }
                while lo < hi && cs[lo].ts < s - WEEK {
                    win.remove(cs[lo].camp);
                    lo += 1;
                }
                s_camp = if hi > lo {
                    // the clicks at the latest ts; ties are not in click_id order
                    let t = cs[hi - 1].ts;
                    let c0 = cs[hi - 1].camp;
                    let mut g = hi - 1;
                    let mut mixed = false;
                    while g > lo && cs[g - 1].ts == t {
                        g -= 1;
                        mixed |= cs[g].camp != c0;
                    }
                    if mixed {
                        min_cid_camp(&cs[g..hi])
                    } else {
                        c0
                    }
                } else {
                    organic
                };
                cm.touch(&win.active, d);
                if day.day != d {
                    flush_day(&mut day, uds);
                    day.day = d;
                }
                day.sessions += 1;
            }
            prev = et;
            day.events += 1;
            if let Some((p, ecat)) = e.pc() {
                day.rev += p;
                s_rev += p;
                s_purch = true;
                match day.cats.iter_mut().find(|x| x.0 == ecat) {
                    Some(x) => x.1 += p,
                    None => day.cats.push((ecat, p)),
                }
            }
        }
        if in_sess {
            cm.sess(s_camp, s_day, s_purch, s_rev);
        }
        flush_day(&mut day, uds);
        for &c in &win.active {
            win.cnt[c as usize] = 0;
        }
        win.active.clear();
        ci = cj;
        i = j;
    }
    pt(8, &mut tm);
}

// ---------- parquet output ----------

use parquet::basic::Compression;
use parquet::data_type::{ByteArray, ByteArrayType, Int32Type, Int64Type};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;

/// Output page codec. Fewer output bytes means less writeback when the container exits, and the
/// output is small next to the input, so a fast codec costs little. SHOP_CODEC=none|lz4|zstd[:level].
fn codec() -> Compression {
    let v = std::env::var("SHOP_CODEC").unwrap_or_default();
    let (name, lvl) = v.split_once(':').unwrap_or((v.as_str(), "1"));
    match name {
        "none" => Compression::UNCOMPRESSED,
        "lz4" => Compression::LZ4_RAW,
        _ => Compression::ZSTD(parquet::basic::ZstdLevel::try_new(lvl.parse().unwrap_or(1)).unwrap_or_default()),
    }
}

fn props() -> Arc<WriterProperties> {
    // campaign_daily only (user_daily has its own encoder below). Dictionary encoding on the
    // low-cardinality columns makes the output smaller for no measurable write time, and fewer bytes
    // means less writeback when the container exits.
    Arc::new(
        WriterProperties::builder()
            .set_compression(codec())
            .set_dictionary_enabled(true)
            .set_statistics_enabled(EnabledStatistics::None)
            .set_write_batch_size(65536)
            .build(),
    )
}

fn ba(b: &[u8]) -> ByteArray {
    ByteArray::from(b.to_vec())
}

struct Names {
    map: AHashMap<u64, u64>,
}

// ---------- user_daily: a small direct parquet encoder ----------
//
// user_daily is most of the output, and a general writer spends its time hashing values into
// dictionaries and copying names. The rows are known here: the rows of one person are adjacent, so
// the user_id dictionary is the persons in order and its indices are runs; day, sessions and events
// are small ranges, so their dictionaries are the whole range and an index is `value - min`; the
// category dictionary is the category list. Format: data page v1, RLE/bit-packed hybrid indices,
// one row group per at most 2^20 rows, uncompressed pages (SHOP_CODEC=zstd[:level] compresses them).

fn uvar(o: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        o.push(v as u8 | 0x80);
        v >>= 7;
    }
    o.push(v as u8);
}

fn zz(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// Thrift compact protocol writer, only the parts parquet metadata needs.
struct Tc {
    b: Vec<u8>,
    last: i16,
    st: Vec<i16>,
}
impl Tc {
    fn new() -> Self {
        Tc { b: Vec::with_capacity(256), last: 0, st: vec![] }
    }
    fn fh(&mut self, id: i16, ty: u8) {
        let d = id - self.last;
        if d > 0 && d <= 15 {
            self.b.push(((d as u8) << 4) | ty);
        } else {
            self.b.push(ty);
            uvar(&mut self.b, zz(id as i64));
        }
        self.last = id;
    }
    fn i32(&mut self, id: i16, v: i32) {
        self.fh(id, 5);
        uvar(&mut self.b, zz(v as i64));
    }
    fn i64(&mut self, id: i16, v: i64) {
        self.fh(id, 6);
        uvar(&mut self.b, zz(v));
    }
    fn bin(&mut self, id: i16, s: &[u8]) {
        self.fh(id, 8);
        uvar(&mut self.b, s.len() as u64);
        self.b.extend_from_slice(s);
    }
    /// struct-valued field
    fn sfield(&mut self, id: i16) {
        self.fh(id, 12);
        self.sbeg();
    }
    /// start of a struct (a list element, or after `sfield`)
    fn sbeg(&mut self) {
        self.st.push(self.last);
        self.last = 0;
    }
    fn send(&mut self) {
        self.b.push(0);
        self.last = self.st.pop().unwrap_or(0);
    }
    fn list(&mut self, id: i16, ety: u8, n: usize) {
        self.fh(id, 9);
        if n < 15 {
            self.b.push(((n as u8) << 4) | ety);
        } else {
            self.b.push(0xF0 | ety);
            uvar(&mut self.b, n as u64);
        }
    }
}

/// Bit width of the largest value `m` (at least 1).
fn bwid(m: u64) -> u32 {
    (64 - m.leading_zeros()).max(1)
}

/// One bit-packed run of the RLE/bit-packed hybrid, for exactly `n` values given by `put`.
struct Bp {
    o: Vec<u8>,
    acc: u64,
    nb: u32,
    w: u32,
    n: usize,
}
impl Bp {
    /// `pre` bytes go first (the bit width byte of dictionary indices, or nothing).
    fn new(w: u32, n: usize, pre: &[u8]) -> Self {
        let mut o = Vec::with_capacity(pre.len() + 10 + (n + 7) / 8 * w as usize);
        o.extend_from_slice(pre);
        uvar(&mut o, (((n + 7) / 8) as u64) << 1 | 1);
        Bp { o, acc: 0, nb: 0, w, n: 0 }
    }
    #[inline(always)]
    fn put(&mut self, v: u32) {
        self.acc |= (v as u64) << self.nb;
        self.nb += self.w;
        while self.nb >= 8 {
            self.o.push(self.acc as u8);
            self.acc >>= 8;
            self.nb -= 8;
        }
        self.n += 1;
    }
    fn finish(mut self) -> Vec<u8> {
        while self.n % 8 != 0 {
            self.put(0);
        }
        self.o
    }
}

/// Metadata of one written column chunk.
struct Cm {
    ty: i32,
    name: &'static str,
    encs: Vec<i32>,
    nv: i64,
    unc: i64,
    comp: i64,
    dpo: i64,
    dico: Option<i64>,
    // compress this chunk's pages (when the file has a compressor)
    z: bool,
}

fn cm(ty: i32, name: &'static str) -> Cm {
    Cm { ty, name, encs: vec![0, 3], nv: 0, unc: 0, comp: 0, dpo: 0, dico: None, z: true }
}

struct Pqf {
    out: Vec<u8>,
    z: Option<zstd::bulk::Compressor<'static>>,
    zb: Vec<u8>,
    codec: i32,
}
impl Pqf {
    fn new(cap: usize) -> Self {
        // zstd (level 1) on the chunks that shrink a lot for little work; SHOP_CODEC=none|zstd[:level]
        let v = std::env::var("SHOP_CODEC").unwrap_or_default();
        let (name, lvl) = v.split_once(':').unwrap_or((v.as_str(), "1"));
        let z = if name == "none" { None } else { Some(zstd::bulk::Compressor::new(lvl.parse().unwrap_or(1)).unwrap()) };
        let codec = if z.is_some() { 6 } else { 0 };
        let mut out = Vec::with_capacity(cap);
        out.extend_from_slice(b"PAR1");
        Pqf { out, z, zb: vec![], codec }
    }
    /// One page: `dict` selects a dictionary page (PLAIN) over a data page v1 with value encoding `enc`.
    fn page(&mut self, cm: &mut Cm, dict: bool, nv: usize, enc: i32, body: &[u8]) {
        let comp: &[u8] = match &mut self.z {
            Some(z) if cm.z => {
                self.zb.clear();
                self.zb.reserve(zstd::zstd_safe::compress_bound(body.len()));
                z.compress_to_buffer(body, &mut self.zb).unwrap();
                &self.zb
            }
            _ => body,
        };
        let mut h = Tc::new();
        h.i32(1, if dict { 2 } else { 0 });
        h.i32(2, body.len() as i32);
        h.i32(3, comp.len() as i32);
        if dict {
            h.sfield(7);
            h.i32(1, nv as i32);
            h.i32(2, 0);
        } else {
            h.sfield(5);
            h.i32(1, nv as i32);
            h.i32(2, enc);
            h.i32(3, 3);
            h.i32(4, 3);
        }
        h.send();
        h.b.push(0);
        let off = self.out.len() as i64;
        if dict {
            cm.dico = Some(off);
        } else {
            cm.dpo = off;
            cm.nv = nv as i64;
            if !cm.encs.contains(&enc) {
                cm.encs.push(enc);
            }
        }
        cm.unc += (h.b.len() + body.len()) as i64;
        cm.comp += (h.b.len() + comp.len()) as i64;
        self.out.extend_from_slice(&h.b);
        self.out.extend_from_slice(comp);
    }
}

/// An integer column as a range dictionary (`lo..=hi`) with bit-packed `v - lo` indices when the
/// range is small, else PLAIN. `c.ty` 1 = INT32, 2 = INT64.
fn int_col(f: &mut Pqf, c: &mut Cm, rows: &[&[UD]], n: usize, get: impl Fn(&UD) -> i64, lo: i64, hi: i64) {
    let int32 = c.ty == 1;
    if n > 0 && (hi - lo) < (1 << 16) {
        let mut d: Vec<u8> = Vec::with_capacity(((hi - lo + 1) * 8) as usize);
        for v in lo..=hi {
            if int32 {
                d.extend_from_slice(&(v as i32).to_le_bytes());
            } else {
                d.extend_from_slice(&v.to_le_bytes());
            }
        }
        f.page(c, true, (hi - lo + 1) as usize, 0, &d);
        let w = bwid((hi - lo) as u64);
        let mut bp = Bp::new(w, n, &[w as u8]);
        for g in rows {
            for r in g.iter() {
                bp.put((get(r) - lo) as u32);
            }
        }
        let b = bp.finish();
        f.page(c, false, n, 8, &b);
    } else {
        let mut b: Vec<u8> = Vec::with_capacity(n * 8);
        for g in rows {
            for r in g.iter() {
                if int32 {
                    b.extend_from_slice(&(get(r) as i32).to_le_bytes());
                } else {
                    b.extend_from_slice(&get(r).to_le_bytes());
                }
            }
        }
        f.page(c, false, n, 0, &b);
    }
}

/// Writes one row group of `rows` (a list of row runs, n rows in all); returns its column chunks.
fn ud_row_group(f: &mut Pqf, rows: &[&[UD]], n: usize, cats: &[B], names: &Names, rev32: bool) -> Vec<Cm> {
    let mut t3 = now();
    // pass 1: ranges and counts
    let (mut dlo, mut dhi) = (i64::MAX, i64::MIN);
    let (mut slo, mut shi, mut elo, mut ehi) = (u64::MAX, 0u64, u64::MAX, 0u64);
    let mut persons = 0usize;
    let mut nn = 0usize;
    let mut last = None;
    for g in rows {
        for r in g.iter() {
            dlo = dlo.min(r.day as i64);
            dhi = dhi.max(r.day as i64);
            slo = slo.min(r.sessions as u64);
            shi = shi.max(r.sessions as u64);
            elo = elo.min(r.events as u64);
            ehi = ehi.max(r.events as u64);
            nn += (r.top != u32::MAX) as usize;
            if last != Some(r.key) {
                persons += 1;
                last = Some(r.key);
            }
        }
    }
    let mut cols = vec![
        cm(6, "user_id"),
        cm(1, "day"),
        cm(2, "sessions"),
        cm(2, "events"),
        cm(if rev32 { 1 } else { 2 }, "revenue_cents"),
        cm(6, "top_category"),
    ];
    // which chunks to compress, bit i = column i. Default revenue_cents only: mostly zeros, so zstd
    // shrinks it about 7x for almost no work; the other chunks are already dictionary/bit-packed
    let zm: u32 = std::env::var("SHOP_ZCOLS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    for (i, c) in cols.iter_mut().enumerate() {
        c.z = (zm >> i) & 1 == 1;
    }
    // user_id: dictionary of the persons in row order, indices as one RLE run per person
    {
        let w = bwid(persons.saturating_sub(1) as u64);
        let vb = ((w + 7) / 8) as usize;
        let mut dict: Vec<u8> = Vec::with_capacity(persons * 14);
        let mut idx: Vec<u8> = Vec::with_capacity(1 + persons * (3 + vb));
        idx.push(w as u8);
        let mut k = 0u64;
        let mut run = 0u64;
        let mut last = None;
        for g in rows {
            for r in g.iter() {
                if last != Some(r.key) {
                    if run > 0 {
                        uvar(&mut idx, run << 1);
                        idx.extend_from_slice(&(k - 1).to_le_bytes()[..vb]);
                    }
                    last = Some(r.key);
                    k += 1;
                    run = 0;
                    if r.key & HBIT == 0 {
                        // the codes are left aligned and end at the first zero code
                        let kk = r.key;
                        let ln = if kk == 0 { 0 } else { 10 - (kk.trailing_zeros() as usize) / 6 };
                        let mut cb = [0u8; 14];
                        cb[..4].copy_from_slice(&(ln as u32).to_le_bytes());
                        for i in 0..10 {
                            cb[4 + i] = DECODE[((kk >> (54 - 6 * i)) & 63) as usize];
                        }
                        dict.extend_from_slice(&cb[..4 + ln]);
                    } else {
                        let s = unpack(*names.map.get(&r.key).expect("name for hashed key"));
                        dict.extend_from_slice(&(s.len() as u32).to_le_bytes());
                        dict.extend_from_slice(s);
                    }
                }
                run += 1;
            }
        }
        if run > 0 {
            uvar(&mut idx, run << 1);
            idx.extend_from_slice(&(k - 1).to_le_bytes()[..vb]);
        }
        f.page(&mut cols[0], true, persons, 0, &dict);
        f.page(&mut cols[0], false, n, 8, &idx);
    }
    pt(13, &mut t3);
    int_col(f, &mut cols[1], rows, n, |r| r.day as i64, dlo, dhi);
    int_col(f, &mut cols[2], rows, n, |r| r.sessions as i64, slo as i64, shi as i64);
    int_col(f, &mut cols[3], rows, n, |r| r.events as i64, elo as i64, ehi as i64);
    pt(14, &mut t3);
    // revenue: PLAIN (wide range, mostly zeros), INT32 when every value of the file fits
    {
        let mut b: Vec<u8> = Vec::with_capacity(n * 8);
        for g in rows {
            for r in g.iter() {
                if rev32 {
                    b.extend_from_slice(&(r.rev as i32).to_le_bytes());
                } else {
                    b.extend_from_slice(&(r.rev as i64).to_le_bytes());
                }
            }
        }
        f.page(&mut cols[4], false, n, 0, &b);
    }
    // top_category: dictionary = the category list; definition levels, then the non-null indices
    {
        let mut dict: Vec<u8> = Vec::with_capacity(cats.iter().map(|c| c.len() + 4).sum());
        for c in cats {
            dict.extend_from_slice(&(c.len() as u32).to_le_bytes());
            dict.extend_from_slice(c);
        }
        f.page(&mut cols[5], true, cats.len(), 0, &dict);
        let mut defs = Bp::new(1, n, &[0, 0, 0, 0]);
        let w = bwid(cats.len().saturating_sub(1) as u64);
        let mut vals = Bp::new(w, nn, &[w as u8]);
        for g in rows {
            for r in g.iter() {
                if r.top == u32::MAX {
                    defs.put(0);
                } else {
                    defs.put(1);
                    vals.put(r.top);
                }
            }
        }
        let mut b = defs.finish();
        let dl = (b.len() - 4) as u32;
        b[..4].copy_from_slice(&dl.to_le_bytes());
        b.extend_from_slice(&vals.finish());
        f.page(&mut cols[5], false, n, 8, &b);
    }
    pt(15, &mut t3);
    cols
}

fn write_user_daily(path: &Path, groups: &[&Vec<UD>], cats: &[B], names: &Names, rev32: bool) {
    let mut tm = now();
    const RG: usize = 1 << 20;
    let total: usize = groups.iter().map(|g| g.len()).sum();
    let mut f = Pqf::new(total * 6 + 4096);
    // split the groups into row groups of at most RG rows
    let mut rgs: Vec<(Vec<Cm>, usize)> = vec![];
    let mut cur: Vec<&[UD]> = vec![];
    let mut n = 0usize;
    for g in groups {
        let mut s: &[UD] = g;
        while !s.is_empty() {
            let k = s.len().min(RG - n);
            cur.push(&s[..k]);
            n += k;
            s = &s[k..];
            if n == RG {
                rgs.push((ud_row_group(&mut f, &cur, n, cats, names, rev32), n));
                cur.clear();
                n = 0;
            }
        }
    }
    if n > 0 {
        rgs.push((ud_row_group(&mut f, &cur, n, cats, names, rev32), n));
    }
    // footer (FileMetaData)
    let mut t = Tc::new();
    t.i32(1, 1);
    t.list(2, 12, 7);
    t.sbeg();
    t.bin(4, b"schema");
    t.i32(5, 6);
    t.send();
    // (physical type, repetition, name, converted type or -1, logical type union field)
    let schema: [(i32, i32, &str, i32, i16); 6] = [
        (6, 0, "user_id", 0, 1),
        (1, 0, "day", 6, 6),
        (2, 0, "sessions", -1, 0),
        (2, 0, "events", -1, 0),
        (if rev32 { 1 } else { 2 }, 0, "revenue_cents", -1, 0),
        (6, 1, "top_category", 0, 1),
    ];
    for (ty, rep, name, conv, lt) in schema {
        t.sbeg();
        t.i32(1, ty);
        t.i32(3, rep);
        t.bin(4, name.as_bytes());
        if conv >= 0 {
            t.i32(6, conv);
            t.sfield(10);
            t.sfield(lt);
            t.send();
            t.send();
        }
        t.send();
    }
    t.i64(3, total as i64);
    t.list(4, 12, rgs.len());
    for (cols, n) in &rgs {
        t.sbeg();
        t.list(1, 12, cols.len());
        for c in cols {
            t.sbeg();
            t.i64(2, c.dico.unwrap_or(c.dpo));
            t.sfield(3);
            t.i32(1, c.ty);
            t.list(2, 5, c.encs.len());
            for &e in &c.encs {
                uvar(&mut t.b, zz(e as i64));
            }
            t.list(3, 8, 1);
            uvar(&mut t.b, c.name.len() as u64);
            t.b.extend_from_slice(c.name.as_bytes());
            t.i32(4, if c.z { f.codec } else { 0 });
            t.i64(5, c.nv);
            t.i64(6, c.unc);
            t.i64(7, c.comp);
            t.i64(9, c.dpo);
            if let Some(d) = c.dico {
                t.i64(11, d);
            }
            t.send();
            t.send();
        }
        t.i64(2, cols.iter().map(|c| c.unc).sum());
        t.i64(3, *n as i64);
        t.i64(5, cols[0].dico.unwrap_or(cols[0].dpo));
        t.i64(6, cols.iter().map(|c| c.comp).sum());
        t.send();
    }
    t.bin(6, b"shop");
    t.b.push(0);
    f.out.extend_from_slice(&t.b);
    f.out.extend_from_slice(&(t.b.len() as u32).to_le_bytes());
    f.out.extend_from_slice(b"PAR1");
    pt(12, &mut tm);
    std::fs::write(path, &f.out).unwrap();
    out_kick(path);
    pt(16, &mut tm);
}

fn write_campaign_daily(path: &Path, rows: &[(u32, i32, [u64; 4])], names: &[B], channels: &[B]) {
    let schema = Arc::new(
        parse_message_type(
            "message campaign_daily { required binary campaign_id (STRING); required binary channel (STRING); required int32 day (DATE); required int64 sessions; required int64 purchasing_sessions; required int64 revenue_cents; required int64 assisted_sessions; }",
        )
        .unwrap(),
    );
    let f = File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(f, schema, props()).unwrap();
    {
        let mut rg = w.next_row_group().unwrap();
        let v: Vec<ByteArray> = rows.iter().map(|r| ba(names[r.0 as usize])).collect();
        let mut c = rg.next_column().unwrap().unwrap();
        c.typed::<ByteArrayType>().write_batch(&v, None, None).unwrap();
        c.close().unwrap();
        let v: Vec<ByteArray> = rows.iter().map(|r| ba(channels[r.0 as usize])).collect();
        let mut c = rg.next_column().unwrap().unwrap();
        c.typed::<ByteArrayType>().write_batch(&v, None, None).unwrap();
        c.close().unwrap();
        let v: Vec<i32> = rows.iter().map(|r| r.1).collect();
        let mut c = rg.next_column().unwrap().unwrap();
        c.typed::<Int32Type>().write_batch(&v, None, None).unwrap();
        c.close().unwrap();
        for k in 0..4 {
            let v: Vec<i64> = rows.iter().map(|r| r.2[k] as i64).collect();
            let mut c = rg.next_column().unwrap().unwrap();
            c.typed::<Int64Type>().write_batch(&v, None, None).unwrap();
            c.close().unwrap();
        }
        rg.close().unwrap();
    }
    w.close().unwrap();
    out_kick(path);
}

/// Start writeback of a finished output file now (SYNC_FILE_RANGE_WRITE, does not wait). The
/// container teardown flushes the output filesystem, and that flush waits for every dirty output
/// page (GHA 20M: about 4 ms per output MB after the process ends); started here, the disk writes
/// overlap the rest of the output phase and the exit. SHOP_OKICK=off skips it.
fn out_kick(path: &Path) {
    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn sync_file_range(fd: i32, off: i64, n: i64, flags: u32) -> i32;
        }
        if std::env::var("SHOP_OKICK").map_or(false, |v| v == "off") {
            return;
        }
        use std::os::fd::AsRawFd;
        if let Ok(f) = std::fs::OpenOptions::new().write(true).open(path) {
            // SAFETY: plain syscall on an open descriptor; 2 = SYNC_FILE_RANGE_WRITE, whole file
            unsafe { sync_file_range(f.as_raw_fd(), 0, 0, 2) };
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = path;
}

// ---------- main ----------

fn manifest_ts(txt: &str, key: &str) -> Option<i64> {
    let i = txt.find(key)?;
    let rest = &txt[i + key.len()..];
    let q1 = rest.find('"')?; // closing quote of key
    let rest = &rest[q1 + 1..];
    let q2 = rest.find('"')?; // opening quote of value
    let rest = &rest[q2 + 1..];
    let q3 = rest.find('"')?;
    parse_ts(rest[..q3].as_bytes())
}

struct EvCtx<'a> {
    prod: &'a IdLookup,
    unknown_cat: u32,
    ws: i64,
    we: i64,
    ibase: i64,
    rs: &'a RandomState,
}

/// Parses one chunk of event lines. Every line with an event_id gets a record in `m` (key 0 when
/// the line is invalid) and a dedup entry: packed event_id in `d`, others in `fb`; ingest times
/// with an inexact key go to `side`. User ids with no packed key are noted in `fbn`.
#[inline(never)]
fn parse_events<E: Ev>(c: B, cx: &EvCtx, m: &mut Vec<E>, d: &mut Vec<DRec>, fb: &mut Fb, fbn: &mut Vec<(u64, u64)>, side: &mut Vec<(u32, i64)>) {
    let mut tci = TsCache::new();
    let mut tct = TsCache::new();
    let c0 = c.as_ptr() as usize;
    scan7(c, |&[eid, uid, ts, ing, et, pr, price]| {
        {
            if eid.is_empty() {
                return;
            }
            let ingest = ts_fast(ing, &mut tci).unwrap_or(i64::MAX);
            let mut r = E::nil();
            'v: {
                if uid.is_empty() || pr.is_empty() {
                    break 'v;
                }
                let Some(t) = ts_fast(ts, &mut tct) else { break 'v };
                if t < cx.ws || t >= cx.we {
                    break 'v;
                }
                let purch = match et_code(et) {
                    1 => false,
                    2 => true,
                    _ => break 'v,
                };
                let (mut pv, mut cat) = (0u32, 0u32);
                if purch {
                    match parse_price(price) {
                        Some(p) => pv = p,
                        None => break 'v,
                    }
                }
                if ingest.wrapping_sub(t) > DAY {
                    break 'v;
                }
                // the category only matters for a purchase
                if purch {
                    cat = cx.prod.get_fast(pr, c0).unwrap_or(cx.unknown_cat);
                }
                let key = match pack6(uid) {
                    Some(k) => k,
                    None => {
                        let k = cx.rs.hash_one(uid) | HBIT;
                        fbn.push((k, pack(uid)));
                        k
                    }
                };
                r = E::mk(key, (t - cx.ws) as u64, pv, cat);
            }
            let loc = m.len() as u32;
            m.push(r);
            let (ik, exact) = ing32(ingest, cx.ibase);
            if !exact {
                side.push((loc, ingest));
            }
            match pack6(eid) {
                Some(k) => d.push(DRec { eid: k, ing: ik, loc }),
                None => fb_put(cx.rs, d, fb, eid, ingest, ik, loc),
            }
        }
    });
}

/// The body of `parse_events` for one line, as a plain function (the slow lines of
/// `parse_events_sol` go through it).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn ev_line<E: Ev>(f: &[B; 7], cx: &EvCtx, c0: usize, tci: &mut TsCache, tct: &mut TsCache, m: &mut Vec<E>, d: &mut Vec<DRec>, fb: &mut Fb, fbn: &mut Vec<(u64, u64)>, side: &mut Vec<(u32, i64)>, dz: &mut Dz<E>) {
    let &[eid, uid, ts, ing, et, pr, price] = f;
    if eid.is_empty() {
        return;
    }
    let ingest = ts_fast(ing, tci).unwrap_or(i64::MAX);
    let mut r = E::nil();
    'v: {
        if uid.is_empty() || pr.is_empty() {
            break 'v;
        }
        let Some(t) = ts_fast(ts, tct) else { break 'v };
        if t < cx.ws || t >= cx.we {
            break 'v;
        }
        let purch = match et_code(et) {
            1 => false,
            2 => true,
            _ => break 'v,
        };
        let mut pv = 0u32;
        let mut cat = 0u32;
        if purch {
            match parse_price(price) {
                Some(p) => pv = p,
                None => break 'v,
            }
            cat = cx.prod.get_fast(pr, c0).unwrap_or(cx.unknown_cat);
        }
        if ingest.wrapping_sub(t) > DAY {
            break 'v;
        }
        r = E::mk(ev_key(uid, cx.rs, fbn), (t - cx.ws) as u64, pv, cat);
    }
    let (ik, exact) = ing32(ingest, cx.ibase);
    let k6 = pack6(eid);
    if let Some(i) = k6.and_then(|k| dz.idx(k)) {
        dz.put(i, ik, r, exact, ingest);
        return;
    }
    let loc = m.len() as u32;
    m.push(r);
    if !exact {
        side.push((loc, ingest));
    }
    match k6 {
        Some(k) => d.push(DRec { eid: k, ing: ik, loc }),
        None => fb_put(cx.rs, d, fb, eid, ingest, ik, loc),
    }
}

/// An event_id that pack6 rejects. A Hex id (idkey-spec section 2) goes whole into `fb.x`: its prefix code
/// and 128-bit value are the exact key of the Hex dedup table (`dx_table`), no byte compare. Any other id
/// goes to the hash buckets with key hash | HBIT (pack6 keys keep bits 63..60 clear), and its bytes stay in
/// `fb.v` (in loc order) for the exact compare on key matches.
#[inline(never)]
fn fb_put(rs: &RandomState, d: &mut Vec<DRec>, fb: &mut Fb, eid: B, ingest: i64, ik: u32, loc: u32) {
    // hex_full checks n == 33 and the prefix first, so other ids pay two compares
    if let Some((_, full)) = hex_full(eid) {
        if loc <= DX_LOC {
            fb.x.push(DX::new(full, CODE[eid[0] as usize], ik, loc));
            return;
        }
    }
    fb_put_other(rs, d, fb, eid, ingest, ik, loc);
}

/// An event_id that is neither Short nor (storable) Hex: hash bucket record plus its bytes in `fb.v`.
#[inline(never)]
fn fb_put_other(rs: &RandomState, d: &mut Vec<DRec>, fb: &mut Fb, eid: B, ingest: i64, ik: u32, loc: u32) {
    d.push(DRec { eid: rs.hash_one(eid) | HBIT, ing: ik, loc });
    fb.v.push((pack(eid), ingest, loc));
}

/// Event ids that pack6 rejects: Other ids `v` (packed bytes, ingest, loc; loc order), Hex ids `x` (loc order).
#[derive(PartialEq, Default)]
struct Fb {
    v: Vec<(u64, i64, u32)>,
    x: Vec<DX>,
}

/// One line with a Hex event_id: the 32 hex digits as hi:lo, the ingest key and lp = loc | prefix code << 26
/// (the code is 1..=63, so lp >> 26 != 0).
#[derive(Clone, Copy, PartialEq, Debug)]
struct DX {
    hi: u64,
    lo: u64,
    ing: u32,
    lp: u32,
}
/// largest loc a DX can hold (26 bits)
const DX_LOC: u32 = (1 << 26) - 1;
impl DX {
    /// the DX of a Hex id with digit value `full` and prefix code `c0` (non-zero), line `loc` <= DX_LOC
    #[inline(always)]
    fn new(full: u128, c0: u8, ik: u32, loc: u32) -> DX {
        DX { hi: (full >> 64) as u64, lo: full as u64, ing: ik, lp: loc | (c0 as u32) << 26 }
    }
}

/// hash of a Hex event id: the top EBB bits pick its bucket, bits 8.. its first table slot
#[inline(always)]
fn dx_hash(r: &DX) -> u64 {
    mix(r.hi.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ r.lo ^ ((r.lp >> 26) as u64) << 58)
}

/// The Hex event ids of dedup bucket b, every chunk in order and each chunk's in loc order. Slot: [hi, lo,
/// ingest key << 32 | global index of the current winner, prefix code] (code != 0 marks a full slot); the
/// key (code, hi, lo) is the whole id, so equal keys are copies. On a match `better(g, ing, og, oing)` says
/// whether the new line g beats the slot's line og; the loser goes to `lose`. Returns the table length used
/// (the winners are the full slots of `tb[..len]`).
#[inline(always)]
fn dx_table<E>(ev_out: &[EvChunk<E>], mbase: &[u32], b: usize, tb: &mut Vec<[u64; 4]>, better: &impl Fn(u32, u32, u32, u32) -> bool, mut lose: impl FnMut(u32)) -> usize {
    let total: usize = ev_out.iter().map(|c| (c.dxoff[b + 1] - c.dxoff[b]) as usize).sum();
    if total == 0 {
        return 0;
    }
    let cap = (total * 2).max(16).next_power_of_two();
    if tb.len() < cap {
        tb.resize(cap, [0; 4]);
    }
    let t = &mut tb[..cap];
    t.fill([0; 4]);
    let mask = cap - 1;
    for (ci, c) in ev_out.iter().enumerate() {
        let mb = mbase[ci];
        for r in &c.dx[c.dxoff[b] as usize..c.dxoff[b + 1] as usize] {
            let gi = mb + (r.lp & DX_LOC);
            let pre = (r.lp >> 26) as u64;
            let pk = ((r.ing as u64) << 32) | gi as u64;
            let mut p = (dx_hash(r) >> 8) as usize & mask;
            loop {
                // SAFETY: p is masked to cap
                let o = unsafe { t.get_unchecked_mut(p) };
                if o[3] == 0 {
                    *o = [r.hi, r.lo, pk, pre];
                    break;
                }
                if o[0] == r.hi && o[1] == r.lo && o[3] == pre {
                    let ok = (o[2] >> 32) as u32;
                    let og = o[2] as u32;
                    if better(gi, r.ing, og, ok) {
                        o[2] = pk;
                        lose(og);
                    } else {
                        lose(gi);
                    }
                    break;
                }
                p = (p + 1) & mask;
            }
        }
    }
    cap
}

/// Person key of a valid line's user id: the packed id, or else a hash with HBIT set (noted in
/// `fbn` with the id's location). The one place that maps a user id to its record key.
#[inline(always)]
fn ev_key(uid: B, rs: &RandomState, fbn: &mut Vec<(u64, u64)>) -> u64 {
    match pack6(uid) {
        Some(k) => k,
        None => ev_key_slow(uid, rs, fbn),
    }
}
#[cold]
#[inline(never)]
fn ev_key_slow(uid: B, rs: &RandomState, fbn: &mut Vec<(u64, u64)>) -> u64 {
    let k = rs.hash_one(uid) | HBIT;
    fbn.push((k, pack(uid)));
    k
}

/// `pack6` of two ids at once: 16 bytes are read at each pointer (`na`, `nb` are the lengths).
/// Returns (key a, a ok, key b, b ok); a key is only meaningful when its flag is set.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn pack6x2(pa: *const u8, na: usize, pb: *const u8, nb: usize) -> (u64, bool, u64, bool) {
    use core::arch::x86_64::*;
    let x = _mm256_set_m128i(_mm_loadu_si128(pb as *const __m128i), _mm_loadu_si128(pa as *const __m128i));
    let l4 = _mm256_set1_epi8(0x0F);
    let hn = _mm256_and_si256(_mm256_srli_epi16(x, 4), l4);
    let ln = _mm256_and_si256(x, l4);
    // valid bytes [0-9A-Z_a-z] by nibbles: high nibble 3 -> class 1, 4|6 -> 2, 5 -> 4, 7 -> 8;
    // the low-nibble table holds the classes in which that low nibble is valid
    let hi_t = _mm256_broadcastsi128_si256(_mm_setr_epi8(0, 0, 0, 1, 2, 4, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0));
    let lo_t = _mm256_broadcastsi128_si256(_mm_setr_epi8(0x0D, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0E, 0x02, 0x02, 0x02, 0x02, 0x06));
    let cls = _mm256_and_si256(_mm256_shuffle_epi8(hi_t, hn), _mm256_shuffle_epi8(lo_t, ln));
    let invalid = _mm256_cmpeq_epi8(cls, _mm256_setzero_si256());
    // CODE values: byte minus a per-high-nibble offset ('_' 4 less than the other 0x5_ bytes)
    let off_t = _mm256_broadcastsi128_si256(_mm_setr_epi8(0, 0, 0, 0x2F, 0x36, 0x36, 0x3B, 0x3B, 0, 0, 0, 0, 0, 0, 0, 0));
    let us = _mm256_and_si256(_mm256_cmpeq_epi8(x, _mm256_set1_epi8(0x5F)), _mm256_set1_epi8(4));
    let code = _mm256_sub_epi8(_mm256_sub_epi8(x, _mm256_shuffle_epi8(off_t, hn)), us);
    let iota = _mm256_broadcastsi128_si256(_mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
    let nv = _mm256_set_m128i(_mm_set1_epi8(nb.min(16) as i8), _mm_set1_epi8(na.min(16) as i8));
    let inf = _mm256_cmpgt_epi8(nv, iota);
    let bad = _mm256_movemask_epi8(_mm256_and_si256(invalid, inf)) as u32;
    let c = _mm256_and_si256(code, inf);
    // pairs -> 12 bits, quads -> 24 bits: dwords 0..2 of a lane hold codes 0-3, 4-7, 8-11
    let w = _mm256_maddubs_epi16(c, _mm256_set1_epi16(0x0140));
    let dw = _mm256_madd_epi16(w, _mm256_set1_epi32(0x0001_1000));
    let lo = _mm256_castsi256_si128(dw);
    let hi = _mm256_extracti128_si256(dw, 1);
    let k3 = |q: u64, d2: u32| ((q & 0xFFFF_FFFF) << 36) | ((q >> 32) << 12) | (d2 as u64 >> 12);
    let ka = k3(_mm_cvtsi128_si64(lo) as u64, _mm_extract_epi32(lo, 2) as u32);
    let kb = k3(_mm_cvtsi128_si64(hi) as u64, _mm_extract_epi32(hi, 2) as u32);
    let oka = (bad & 0xFFFF) == 0 && na.wrapping_sub(1) < 10;
    let okb = (bad >> 16) == 0 && nb.wrapping_sub(1) < 10;
    (ka, oka, kb, okb)
}

//// `pack6x2` for one id: 16 bytes are read at `pa` (`na` is the length). Returns (key, ok); the
/// key equals `pack6` of the id when ok is set, and ok is clear when `pack6` gives None.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn pack6x1(pa: *const u8, na: usize) -> (u64, bool) {
    use core::arch::x86_64::*;
    let x = _mm_loadu_si128(pa as *const __m128i);
    let l4 = _mm_set1_epi8(0x0F);
    let hn = _mm_and_si128(_mm_srli_epi16(x, 4), l4);
    let ln = _mm_and_si128(x, l4);
    let hi_t = _mm_setr_epi8(0, 0, 0, 1, 2, 4, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0);
    let lo_t = _mm_setr_epi8(0x0D, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0E, 0x02, 0x02, 0x02, 0x02, 0x06);
    let cls = _mm_and_si128(_mm_shuffle_epi8(hi_t, hn), _mm_shuffle_epi8(lo_t, ln));
    let invalid = _mm_cmpeq_epi8(cls, _mm_setzero_si128());
    let off_t = _mm_setr_epi8(0, 0, 0, 0x2F, 0x36, 0x36, 0x3B, 0x3B, 0, 0, 0, 0, 0, 0, 0, 0);
    let us = _mm_and_si128(_mm_cmpeq_epi8(x, _mm_set1_epi8(0x5F)), _mm_set1_epi8(4));
    let code = _mm_sub_epi8(_mm_sub_epi8(x, _mm_shuffle_epi8(off_t, hn)), us);
    let iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let inf = _mm_cmpgt_epi8(_mm_set1_epi8(na.min(16) as i8), iota);
    let bad = _mm_movemask_epi8(_mm_and_si128(invalid, inf)) as u32;
    let c = _mm_and_si128(code, inf);
    let w = _mm_maddubs_epi16(c, _mm_set1_epi16(0x0140));
    let dw = _mm_madd_epi16(w, _mm_set1_epi32(0x0001_1000));
    let q = _mm_cvtsi128_si64(dw) as u64;
    let d2 = _mm_extract_epi32(dw, 2) as u32;
    let k = ((q & 0xFFFF_FFFF) << 36) | ((q >> 32) << 12) | (d2 as u64 >> 12);
    (k, bad == 0 && na.wrapping_sub(1) < 10)
}

// ---------- id classes (Short / Hex / Other) ----------
// Short: pack6 ids (1..=10 bytes of [0-9A-Z_a-z]). Hex: 33 bytes, a CODE prefix byte and 32
// lowercase hex digits. Other: the rest. A Hex key is HBIT | 1<<62 | prefix code << 56 | the top
// 56 bits of the digits; the full 128-bit value goes to a side list (see idk_ref).

/// Key of a Hex id from its prefix code `c0` (CODE value, non-zero) and digit value `f`.
#[inline(always)]
fn hex_key(c0: u8, f: u128) -> u64 {
    HBIT | (1u64 << 62) | ((c0 as u64) << 56) | (f >> 72) as u64
}

/// Scalar reference of the id classes: (key, full, cls), cls 0 Short (key = pack6, full 0),
/// 1 Hex (key = hex_key, full = the 32 digits as a number), 2 Other (key and full 0).
fn idk_ref(b: &[u8]) -> (u64, u128, u8) {
    if let Some(k) = pack6(b) {
        return (k, 0, 0);
    }
    if b.len() == 33 && CODE[b[0] as usize] != 0 && b[1..].iter().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')) {
        let mut f = 0u128;
        for &c in &b[1..] {
            f = (f << 4) | (if c <= b'9' { c - b'0' } else { c - b'a' + 10 }) as u128;
        }
        return (hex_key(CODE[b[0] as usize], f), f, 1);
    }
    (0, 0, 2)
}

/// The 32 bytes at `p` read as lowercase hex digits: (value, ok); ok iff every byte is [0-9a-f].
/// The value (big-endian digit order) is only meaningful when ok.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn hex32(p: *const u8) -> (u128, bool) {
    use core::arch::x86_64::*;
    let x = _mm256_loadu_si256(p as *const __m256i);
    let d = _mm256_sub_epi8(x, _mm256_set1_epi8(b'0' as i8));
    let l = _mm256_sub_epi8(x, _mm256_set1_epi8(b'a' as i8));
    let isd = _mm256_cmpeq_epi8(_mm256_min_epu8(d, _mm256_set1_epi8(9)), d);
    let isl = _mm256_cmpeq_epi8(_mm256_min_epu8(l, _mm256_set1_epi8(5)), l);
    let ok = _mm256_movemask_epi8(_mm256_or_si256(isd, isl)) == -1;
    // digit values: '0'..'9' -> 0..9, 'a'..'f' -> 10..15 ('a' - '0' - 10 = 39)
    let v = _mm256_sub_epi8(d, _mm256_and_si256(isl, _mm256_set1_epi8(39)));
    // 16 * first + second per byte pair, then the 8 pair bytes of each lane in reverse order
    let w = _mm256_maddubs_epi16(v, _mm256_set1_epi16(0x0110));
    let r = _mm256_shuffle_epi8(
        w,
        _mm256_setr_epi8(14, 12, 10, 8, 6, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1),
    );
    let hi = _mm256_extract_epi64(r, 0) as u64;
    let lo = _mm256_extract_epi64(r, 2) as u64;
    (((hi as u128) << 64) | lo as u128, ok)
}

/// The full value of a Hex id given as a slice, or None when it is not Hex. Reads only `b`.
#[inline(always)]
fn hex_full(b: &[u8]) -> Option<(u64, u128)> {
    if b.len() != 33 || CODE[b[0] as usize] == 0 {
        return None;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: b[1..33] is 32 readable bytes
        let (f, ok) = unsafe { hex32(b.as_ptr().add(1)) };
        ok.then(|| (hex_key(CODE[b[0] as usize], f), f))
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let (k, f, c) = idk_ref(b);
        (c == 1).then_some((k, f))
    }
}

/// Class, key and full value of one id of `n >= 1` bytes at `p` (as idk_ref). Reads 16 bytes
/// at p, and 33 when n == 33; the caller guarantees they are readable.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn idk_x(p: *const u8, n: usize) -> (u64, u128, u8) {
    let (k, ok) = pack6x1(p, n);
    if ok {
        return (k, 0, 0);
    }
    idk_hex(p, n)
}

/// `idk_x` of an id whose `pack6` is None: the hex32 key (class 1) or class 2.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn idk_hex(p: *const u8, n: usize) -> (u64, u128, u8) {
    if n == 33 {
        let c0 = CODE[*p as usize];
        let (f, okh) = hex32(p.add(1));
        if okh & (c0 != 0) {
            return (hex_key(c0, f), f, 1);
        }
    }
    (0, 0, 2)
}

/// `idk_x` of two ids (event_id and user_id), same contract per id.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn idk_x2(pa: *const u8, na: usize, pb: *const u8, nb: usize) -> ((u64, u128, u8), (u64, u128, u8)) {
    let (ka, oka, kb, okb) = pack6x2(pa, na, pb, nb);
    let one = |p: *const u8, n: usize, k: u64, ok: bool| {
        if ok {
            return (k, 0u128, 0u8);
        }
        if n == 33 {
            let c0 = CODE[*p as usize];
            let (f, okh) = hex32(p.add(1));
            if okh & (c0 != 0) {
                return (hex_key(c0, f), f, 1);
            }
        }
        (0, 0, 2)
    };
    (one(pa, na, ka, oka), one(pb, nb, kb, okb))
}

/// `pack6x1` stays for the event-id miss path; this is the hot-path fused pass.
/// Per `n & 15`: the shuffle that right-aligns an event id's body (bytes 1..n) in bytes 0..8 of a
/// lane (0x80 = zero byte), and the identity shuffle that keeps the first `n` bytes of a user id.
const fn shuf_tabs() -> ([[u8; 16]; 16], [[u8; 16]; 16]) {
    let mut e = [[0x80u8; 16]; 16];
    let mut u = [[0x80u8; 16]; 16];
    let mut n = 0usize;
    while n < 16 {
        let mut j = 0usize;
        while j < 16 {
            if j < n {
                u[n][j] = j as u8;
            }
            if n >= 2 && n <= 9 && j < 8 && j + n >= 9 {
                e[n][j] = (j + n - 8) as u8;
            }
            j += 1;
        }
        n += 1;
    }
    (e, u)
}
static SHUF_E: [[u8; 16]; 16] = shuf_tabs().0;
static SHUF_U: [[u8; 16]; 16] = shuf_tabs().1;

/// One 256-bit pass over an event id (low lane) and a user id (high lane); 16 bytes are read at
/// each pointer. Returns (event ok, event base-36 value, user key, user ok):
/// the first two equal `gen_val_txt(pe, ne, b'e')` (ok iff it is Some), the last two equal
/// `pack6x2`'s user half.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn fuse_eu(pe: *const u8, ne: usize, pu: *const u8, nu: usize) -> (bool, u64, u64, bool) {
    use core::arch::x86_64::*;
    let x = _mm256_set_m128i(_mm_loadu_si128(pu as *const __m128i), _mm_loadu_si128(pe as *const __m128i));
    // SAFETY: `& 15` keeps both table indices in range (lengths outside the table only matter
    // through the scalar length checks below, which reject them)
    let m = _mm256_set_m128i(
        _mm_loadu_si128(SHUF_U.as_ptr().add(nu & 15) as *const __m128i),
        _mm_loadu_si128(SHUF_E.as_ptr().add(ne & 15) as *const __m128i),
    );
    // event body right-aligned at bytes 0..8, user id's first nu bytes kept; other bytes zero
    let x = _mm256_shuffle_epi8(x, m);
    let l4 = _mm256_set1_epi8(0x0F);
    let hn = _mm256_and_si256(_mm256_srli_epi16(x, 4), l4);
    let ln = _mm256_and_si256(x, l4);
    // low lane: [0-9a-z] (hi nibble 3 -> class 1, 6 -> 2, 7 -> 4); high lane: [0-9A-Z_a-z] as pack6x2
    let hi_t = _mm256_set_m128i(
        _mm_setr_epi8(0, 0, 0, 1, 2, 4, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0),
        _mm_setr_epi8(0, 0, 0, 1, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0),
    );
    let lo_t = _mm256_set_m128i(
        _mm_setr_epi8(0x0D, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0E, 0x02, 0x02, 0x02, 0x02, 0x06),
        _mm_setr_epi8(5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 2, 2, 2, 2, 2),
    );
    let cls = _mm256_and_si256(_mm256_shuffle_epi8(hi_t, hn), _mm256_shuffle_epi8(lo_t, ln));
    let invalid = _mm256_cmpeq_epi8(cls, _mm256_setzero_si256());
    // CODE values: byte minus a per-high-nibble offset (event: '0'-'9' -> 0..9, 'a'-'z' -> 10..35;
    // user: '_' 4 less than the other 0x5_ bytes)
    let off_t = _mm256_set_m128i(
        _mm_setr_epi8(0, 0, 0, 0x2F, 0x36, 0x36, 0x3B, 0x3B, 0, 0, 0, 0, 0, 0, 0, 0),
        _mm_setr_epi8(0, 0, 0, 0x30, 0, 0, 0x57, 0x57, 0, 0, 0, 0, 0, 0, 0, 0),
    );
    let us = _mm256_and_si256(_mm256_cmpeq_epi8(x, _mm256_set1_epi8(0x5F)), _mm256_set1_epi8(4));
    let c = _mm256_sub_epi8(_mm256_sub_epi8(x, _mm256_shuffle_epi8(off_t, hn)), us);
    // padding bytes (m < 0) are zero here and code 0; they only need to be ignored in `invalid`
    let bad = (_mm256_movemask_epi8(invalid) as u32) & !(_mm256_movemask_epi8(m) as u32);
    let w = _mm256_maddubs_epi16(c, _mm256_set_m128i(_mm_set1_epi16(0x0140), _mm_set1_epi16(0x0124)));
    let dw = _mm256_madd_epi16(w, _mm256_set_m128i(_mm_set1_epi32(0x0001_1000), _mm_set1_epi32(0x0001_0510)));
    let hi = _mm256_extracti128_si256(dw, 1);
    let k3 = |q: u64, d2: u32| ((q & 0xFFFF_FFFF) << 36) | ((q >> 32) << 12) | (d2 as u64 >> 12);
    let ku = k3(_mm_cvtsi128_si64(hi) as u64, _mm_extract_epi32(hi, 2) as u32);
    let q = _mm_cvtsi128_si64(_mm256_castsi256_si128(dw)) as u64;
    let val = (q & 0xFFFF_FFFF) * 1_679_616 + (q >> 32);
    let oku = (bad >> 16) == 0 && nu.wrapping_sub(1) < 10;
    // SAFETY: 16 bytes at pe are readable, so are bytes 0 and 1
    let oke = ((bad & 0xFFFF) == 0) & (ne.wrapping_sub(2) < 8) & (core::ptr::read_volatile(pe) == b'e') & (core::ptr::read_volatile(pe.add(1)) != b'0') & (val <= GM);
    (oke, val, ku, oku)
}

// Bytes 0..8 and 3..11 of "2000-01-01T", and that day's start in seconds: the key every
/// `DayCache` slot starts with (a match on it is a correct hit).
const D0_A: u64 = u64::from_le_bytes(*b"2000-01-");
const D0_B: u64 = u64::from_le_bytes(*b"0-01-01T");
const D0_S: i64 = 10_957 * 86_400;

#[derive(Clone, Copy)]
struct DayEnt {
    a: u64,
    b: u64,
    ds: i64,
}
/// Day starts (seconds) of "YYYY-MM-DDT" prefixes, keyed by bytes 0..8 and 3..11; only valid
/// dates followed by 'T' are stored.
struct DayCache {
    ents: Box<[DayEnt; 256]>,
}
impl DayCache {
    fn new() -> Self {
        DayCache { ents: Box::new([DayEnt { a: D0_A, b: D0_B, ds: D0_S }; 256]) }
    }
}

/// Load 8 bytes at `p` (little-endian).
#[inline(always)]
unsafe fn ldp(p: *const u8) -> u64 {
    u64::from_le((p as *const u64).read_unaligned())
}

/// Day start in seconds of the "YYYY-MM-DDT" at `p`, given `a` = bytes 0..8 and `b` = bytes 3..11.
#[inline(always)]
unsafe fn day_get(a: u64, b: u64, p: *const u8, c: &mut DayCache) -> Option<i64> {
    let e = &c.ents[(b.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as usize];
    if (e.a == a) & (e.b == b) {
        Some(e.ds)
    } else {
        day_miss(a, b, p, c)
    }
}

#[cold]
#[inline(never)]
unsafe fn day_miss(a: u64, b: u64, p: *const u8, c: &mut DayCache) -> Option<i64> {
    let s = std::slice::from_raw_parts(p, 11);
    if s[10] != b'T' {
        return None;
    }
    let ds = ts_days(s)? * 86_400;
    c.ents[(b.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as usize] = DayEnt { a, b, ds };
    Some(ds)
}

/// Seconds of the day of the 8 bytes "HH:MM:SS" at `p`. One range test per byte (digits with
/// first digits up to 2, 5, 5, and the two colons), then the hour test.
#[inline(always)]
unsafe fn hms(p: *const u8) -> Option<i64> {
    const LO: u64 = u64::from_le_bytes(*b"00:00:00");
    const HI: u64 = u64::from_le_bytes([0x7F - b'2', 0x7F - b'9', 0x7F - b':', 0x7F - b'5', 0x7F - b'9', 0x7F - b':', 0x7F - b'5', 0x7F - b'9']);
    let t = ldp(p);
    let v = t.wrapping_sub(LO);
    if ((v | t.wrapping_add(HI)) & h8(0x80)) != 0 {
        return None;
    }
    let v = v.wrapping_mul(10).wrapping_add(v >> 8);
    let h = (v & 0xFF) as i64;
    if h > 23 {
        return None;
    }
    Some((h * 60 + ((v >> 24) & 0xFF) as i64) * 60 + ((v >> 48) & 0xFF) as i64)
}

/// `hms` of the 8 bytes at `a` and of the 8 bytes at `b` in one vector: (seconds, ok) for each,
/// with ok exactly when `hms` gives Some (then the seconds are its value). Per byte, the digit
/// value (colons xored to 0) must not exceed 2, 9, 0, 5, 9, 0, 5, 9; the hour test is the sum
/// test (minutes and seconds are at most 59 then).
#[inline(always)]
unsafe fn hms2(a: *const u8, b: *const u8) -> (i64, bool, i64, bool) {
    use core::arch::x86_64::*;
    let t = _mm_set_epi64x(ldp(b) as i64, ldp(a) as i64);
    let v = _mm_xor_si128(_mm_sub_epi8(t, _mm_set1_epi8(b'0' as i8)), _mm_setr_epi8(0, 0, 10, 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 10, 0, 0));
    let over = _mm_subs_epu8(v, _mm_setr_epi8(2, 9, 0, 5, 9, 0, 5, 9, 2, 9, 0, 5, 9, 0, 5, 9));
    let z = _mm_movemask_epi8(_mm_cmpeq_epi8(over, _mm_setzero_si128())) as u32;
    // words: HH, 10*M1, M2, SS (colon bytes weigh 0); then HH*3600 + M1*600 and M2*60 + SS
    let w = _mm_maddubs_epi16(v, _mm_setr_epi8(10, 1, 0, 10, 1, 0, 10, 1, 10, 1, 0, 10, 1, 0, 10, 1));
    let d = _mm_madd_epi16(w, _mm_setr_epi16(3600, 60, 60, 1, 3600, 60, 60, 1));
    let s = _mm_add_epi32(d, _mm_srli_epi64(d, 32));
    let sa = _mm_cvtsi128_si32(s) as u32;
    let sb = _mm_extract_epi32(s, 2) as u32;
    (sa as i64, ((z & 0xFF) == 0xFF) & (sa < 86_400), sb as i64, (z >> 8 == 0xFF) & (sb < 86_400))
}

/// `ts_num` of the 13 bytes at `p`, read as one 16-byte vector from `p - 3` (those 3 bytes are
/// masked off, so they act as leading zeros): digit test, then 2-, 4- and 8-digit sums.
#[inline(always)]
unsafe fn ts13(p: *const u8) -> Option<i64> {
    use core::arch::x86_64::*;
    let keep = _mm_setr_epi8(0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let raw = _mm_loadu_si128(p.sub(3) as *const __m128i);
    let v = _mm_and_si128(_mm_add_epi8(raw, _mm_set1_epi8(-(b'0' as i8))), keep);
    let over = _mm_subs_epu8(v, _mm_set1_epi8(9));
    if _mm_testz_si128(over, over) == 0 {
        return None;
    }
    let a = _mm_maddubs_epi16(v, _mm_setr_epi8(10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1));
    let b = _mm_madd_epi16(a, _mm_setr_epi16(100, 1, 100, 1, 100, 1, 100, 1));
    let c = _mm_packus_epi32(b, b);
    let d = _mm_madd_epi16(c, _mm_setr_epi16(10000, 1, 10000, 1, 10000, 1, 10000, 1));
    let x = _mm_cvtsi128_si64(d) as u64;
    Some(((x & 0xFFFF_FFFF) * 100_000_000 + (x >> 32)) as i64)
}

/// Milliseconds of the 25 bytes "YYYY-MM-DDTHH:MM:SS+HH:MM" (or '-') at `p`, the same value as
/// `ts_iso`; None means "use the general parser". `(ra, rb, rd)` is a valid date key and its day
/// start. The 8 bytes "SS+HH:MM" get one range test per byte (',' sits between '+' and '-').
#[inline(always)]
unsafe fn ts25(p: *const u8, ra: u64, rb: u64, rd: i64, c: &mut DayCache) -> Option<i64> {
    let h = hms(p.add(11));
    ts25h(p, ra, rb, rd, c, h.unwrap_or(0), h.is_some())
}

/// `ts25`, given `hms(p + 11)` as (seconds, ok).
#[inline(always)]
unsafe fn ts25h(p: *const u8, ra: u64, rb: u64, rd: i64, c: &mut DayCache, hs: i64, hok: bool) -> Option<i64> {
    const LO: u64 = u64::from_le_bytes(*b"00+00:00");
    const HI: u64 = u64::from_le_bytes([0x7F - b'5', 0x7F - b'9', 0x7F - b'-', 0x7F - b'9', 0x7F - b'9', 0x7F - b':', 0x7F - b'5', 0x7F - b'9']);
    let w = ldp(p.add(17));
    let v = w.wrapping_sub(LO);
    let sg = ((w >> 16) & 0xFF) as u8;
    if ((v | w.wrapping_add(HI)) & h8(0x80)) != 0 || sg == b',' {
        return None;
    }
    let u = v.wrapping_mul(10).wrapping_add(v >> 8);
    let (oh, om) = (((u >> 24) & 0xFF) as i64, ((u >> 48) & 0xFF) as i64);
    if oh > 23 {
        return None;
    }
    let (a, b) = (ldp(p), ldp(p.add(3)));
    let d = if (a == ra) & (b == rb) { rd } else { day_get(a, b, p, c)? };
    if !hok {
        return None;
    }
    let h = hs;
    let off = (oh * 60 + om) * 60_000;
    Some((d + h) * 1000 - if sg == b'-' { -off } else { off })
}

/// Same records, dedup entries and side lists, in the same order, as `parse_events`, except
/// that the category of a record with no price is 0 (no product lookup for views and carts;
/// `Ev::pc` never reads it). One fused pass: the `scan7` line classifier with the fields kept
/// as pointer and length, both ids packed in one vector, ISO times straight to seconds.
/// `DZ` must equal `dz.on`: without dense ids the event id is only needed as its pack6 key, so
/// both ids take one `pack6x2` pass and the base-36 value is never formed.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(never)]
fn parse_events_sol<E: Ev, const DZ: bool>(c: B, cx: &EvCtx, m: &mut Vec<E>, d: &mut Vec<DRec>, fb: &mut Fb, fbn: &mut Vec<(u64, u64)>, side: &mut Vec<(u32, i64)>, dz: &mut Dz<E>) {
    use core::arch::x86_64::_bzhi_u64;
    let mut tci = TsCache::new();
    let mut tct = TsCache::new();
    let mut dci = DayCache::new();
    let mut dct = DayCache::new();
    let c0 = c.as_ptr() as usize;
    // SAFETY (bzhi): BMI2 is enabled for this build
    let bzhi = |x: u64, n: u32| unsafe { _bzhi_u64(x, n) };
    let blsr = |x: u64| x & x.wrapping_sub(1);
    // SAFETY (sl): only called on fields found inside the chunk
    let sl = |p: *const u8, n: usize| -> B { unsafe { std::slice::from_raw_parts(p, n) } };
    let (ws, we, ibase) = (cx.ws, cx.we, cx.ibase);
    let (imax_k, imax_x) = ing32(i64::MAX, ibase);
    let len = c.len();
    let base = c.as_ptr();
    let mut pos = 0usize;
    let mut sbuf: Vec<[B; 7]> = Vec::new();
    let pfd = pf_dist();
    'ln: while pos + 128 <= len {
        if pfd != 0 {
            // a prefetch never faults, so the address may lie past the chunk
            prefetch(base.wrapping_add(pos + pfd));
            prefetch(base.wrapping_add(pos + pfd + 64));
        }
        // SAFETY: pos + 128 <= len, and the second window starts at most 64 bytes in
        let p = unsafe { base.add(pos) };
        let (ca, qa, na) = masks64(p);
        let e0 = ca.trailing_zeros();
        let c1 = blsr(ca);
        let e1 = c1.trailing_zeros();
        let c2 = blsr(c1);
        let e2 = c2.trailing_zeros();
        'slow: {
            // e0..e2: the first three commas; xq: whether fields 0-2 open with a quote
            let (e0, e1, e2, xq, bad_a) = if (c2 != 0) & (bzhi(na, e2) == 0) {
                let (xa, bad_a) = quote_bad(bzhi(qa, e2), ca);
                (e0, e1, e2, [xa & 1, (xa >> (e0 + 1)) & 1, (xa >> (e1 + 1)) & 1], bad_a)
            } else if (na == 0) & (pos + 192 <= len) {
                // fields 0-2 run past the first window (long ids): the same rule on 128 bytes
                // SAFETY: pos + 192 <= len
                match unsafe { first3_128(p, ca, qa) } {
                    Some(v) => v,
                    None => break 'slow,
                }
            } else {
                break 'slow;
            };
            // SAFETY: e2 < 128, and 192 bytes at p are readable whenever e2 >= 64
            let pb = unsafe { p.add(e2 as usize + 1) };
            let (cb, qb, nb) = masks64(pb);
            let l = nb.trailing_zeros();
            let e3 = cb.trailing_zeros();
            let d1 = blsr(cb);
            let e4 = d1.trailing_zeros();
            let d2 = blsr(d1);
            let e5 = d2.trailing_zeros();
            let d3 = blsr(d2);
            let (xb, bad_b) = quote_bad(bzhi(qb, l), cb | nb);
            if !((nb != 0) & (e5 < l) & (bzhi(d3, l) == 0) & ((bad_a | bad_b) == 0)) {
                break 'slow;
            }
            pos += (e2 + l + 2) as usize;
            let fq = |p: *const u8, q: u64, s: u32, e: u32| {
                let q = q as u32;
                // SAFETY: s <= e < 128 inside the classified bytes; a quoted field has e - s >= 2
                (unsafe { p.add((s + q) as usize) }, (e - s - 2 * q) as usize)
            };
            let fld = |p: *const u8, x: u64, s: u32, e: u32| fq(p, (x >> s) & 1, s, e);
            let (pe, ne) = fq(p, xq[0], 0, e0);
            if ne == 0 {
                continue 'ln;
            }
            let (pu, nu) = fq(p, xq[1], e0 + 1, e1);
            let (pt, nt) = fq(p, xq[2], e1 + 1, e2);
            let (pi, ni) = fld(pb, xb, 0, e3);
            let (pet, net) = fld(pb, xb, e3 + 1, e4);
            let (pp, np) = fld(pb, xb, e4 + 1, e5);
            let (ppr, npr) = fld(pb, xb, e5 + 1, l);
            // SAFETY: both ids start before byte 128 (e1 < e2 < 128), so 16 bytes from each stay
            // inside the readable bytes at p (128 when e2 < 64, else 192)
            let (oke, val, ku, oku, ke6, oke6) = if DZ {
                let (oke, val, ku, oku) = unsafe { fuse_eu(pe, ne, pu, nu) };
                (oke, val, ku, oku, 0, false)
            } else {
                let (ke6, oke6, ku, oku) = unsafe { pack6x2(pe, ne, pu, nu) };
                (false, 0, ku, oku, ke6, oke6)
            };
            // SAFETY (ldp, hms): a 20-byte field lies inside its window. (ra, rb, rd): the
            // ingest's date key and day start when it is a valid F1 date, else 2000-01-01's.
            let (mut ra, mut rb, mut rd) = (D0_A, D0_B, D0_S);
            // SAFETY: pi + 19 and pt + 19 lie inside window B (readable, see above)
            let (his, hio, hts, hto) = unsafe { hms2(pi.add(11), pt.add(11)) };
            let iso_i = if ni == 20 && unsafe { *pi.add(19) } == b'Z' {
                let (ia, ib) = unsafe { (ldp(pi), ldp(pi.add(3))) };
                let (dd, hs) = unsafe { (day_get(ia, ib, pi, &mut dci), if hio { Some(his) } else { None }) };
                Some(match dd {
                    Some(d) => {
                        (ra, rb, rd) = (ia, ib, d);
                        hs.map(|h| d + h)
                    }
                    None => None,
                })
            } else {
                None
            };
            let (ingest, ik, iex) = match iso_i {
                Some(Some(s)) => {
                    let v = s - ibase + IOFF;
                    (s * 1000, v.clamp(0, u32::MAX as i64) as u32, (v > 0) & (v < u32::MAX as i64))
                }
                Some(None) => (i64::MAX, imax_k, imax_x),
                None => {
                    let ingest = ts_fast(sl(pi, ni), &mut tci).unwrap_or(i64::MAX);
                    let (k, x) = ing32(ingest, ibase);
                    (ingest, k, x)
                }
            };
            let mut r = E::nil();
            'v: {
                if (nu == 0) | (np == 0) {
                    break 'v;
                }
                let t = if nt == 20 && unsafe { *pt.add(19) } == b'Z' {
                    // same date bytes as the ingest: same day start
                    let (ta, tb) = unsafe { (ldp(pt), ldp(pt.add(3))) };
                    let d = if (ta == ra) & (tb == rb) {
                        rd
                    } else {
                        match unsafe { day_get(ta, tb, pt, &mut dct) } {
                            Some(d) => d,
                            None => break 'v,
                        }
                    };
                    if !hto {
                        break 'v;
                    }
                    (d + hts) * 1000
                } else if nt == 13 {
                    // SAFETY: ts is the third field and the ids are not empty here, so the 3
                    // bytes before it and the 16 bytes from pt - 3 lie inside the window
                    match unsafe { ts13(pt) } {
                        Some(t) => t,
                        None => break 'v,
                    }
                } else {
                    // SAFETY (ts25): a 25-byte field lies inside its window
                    let f = if nt == 25 { unsafe { ts25h(pt, ra, rb, rd, &mut dct, hts, hto) } } else { None };
                    match f.or_else(|| parse_ts_c(sl(pt, nt), &mut tct)) {
                        Some(t) => t,
                        None => break 'v,
                    }
                };
                if (t < ws) | (t >= we) | (ingest.wrapping_sub(t) > DAY) {
                    break 'v;
                }
                let mut pv = 0u32;
                let mut cat = 0u32;
                match et_code(sl(pet, net)) {
                    1 => {}
                    2 => {
                        match parse_price(sl(ppr, npr)) {
                            Some(v) => pv = v,
                            None => break 'v,
                        }
                        cat = cx.prod.get_fast(sl(pp, np), c0).unwrap_or(cx.unknown_cat);
                    }
                    _ => break 'v,
                }
                let key = if oku { ku } else { ev_key_slow(sl(pu, nu), cx.rs, fbn) };
                r = E::mk(key, (t - ws) as u64, pv, cat);
            }
            if DZ && oke {
                if let Some(i) = dz.idx_val(val) {
                    dz.put(i, ik, r, iex, ingest);
                    continue 'ln;
                }
            }
            let loc = m.len() as u32;
            m.push(r);
            if !iex {
                side.push((loc, ingest));
            }
            // SAFETY: 16 bytes from pe are readable (see pack6x2's callers above), and when ne == 33
            // idk_x reads the 33 id bytes themselves
            let (ke, full, cls) = if DZ {
                unsafe { idk_x(pe, ne) }
            } else if oke6 {
                (ke6, 0, 0)
            } else {
                unsafe { idk_hex(pe, ne) }
            };
            if cls == 0 {
                d.push(DRec { eid: ke, ing: ik, loc });
            } else if cls == 1 && loc <= DX_LOC {
                fb.x.push(DX::new(full, (ke >> 56) as u8 & 63, ik, loc));
            } else {
                fb_put_other(cx.rs, d, fb, sl(pe, ne), ingest, ik, loc);
            }
            continue 'ln;
        }
        sbuf.clear();
        pos = scan7_line(c, pos, &mut sbuf);
        for f in &sbuf {
            ev_line(f, cx, c0, &mut tci, &mut tct, m, d, fb, fbn, side, dz);
        }
    }
    if pos < len {
        scan::<7>(&c[pos..], |f| ev_line(f, cx, c0, &mut tci, &mut tct, m, d, fb, fbn, side, dz));
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
fn parse_events_sol<E: Ev, const DZ: bool>(c: B, cx: &EvCtx, m: &mut Vec<E>, d: &mut Vec<DRec>, fb: &mut Fb, fbn: &mut Vec<(u64, u64)>, side: &mut Vec<(u32, i64)>, dz: &mut Dz<E>) {
    let mut tci = TsCache::new();
    let mut tct = TsCache::new();
    let c0 = c.as_ptr() as usize;
    scan::<7>(c, |f| ev_line(f, cx, c0, &mut tci, &mut tct, m, d, fb, fbn, side, dz));
}

/// Output buffers of one event-parse call.
struct EvBufs<E> {
    m: Vec<E>,
    d: Vec<DRec>,
    fb: Fb,
    fbn: Vec<(u64, u64)>,
    side: Vec<(u32, i64)>,
}
impl<E: Ev> EvBufs<E> {
    fn new() -> Self {
        EvBufs { m: Vec::with_capacity(CHUNK / 32), d: Vec::with_capacity(CHUNK / 32), fb: Fb::default(), fbn: vec![], side: vec![] }
    }
    fn clear(&mut self) {
        self.m.clear();
        self.d.clear();
        self.fb.v.clear();
        self.fb.x.clear();
        self.fbn.clear();
        self.side.clear();
    }
    /// the first difference from `o` (records compared by key, ts and `pc`), if any
    fn diff(&self, o: &Self) -> Option<String> {
        if self.m.len() != o.m.len() {
            return Some(format!("m len {} vs {}", self.m.len(), o.m.len()));
        }
        for (i, (a, b)) in self.m.iter().zip(&o.m).enumerate() {
            if a.key() != b.key() || a.rts() != b.rts() || a.pc() != b.pc() {
                return Some(format!("m[{i}]: key {:x}/{:x} rts {}/{} pc {:?}/{:?}", a.key(), b.key(), a.rts(), b.rts(), a.pc(), b.pc()));
            }
        }
        let dk = |v: &[DRec]| v.iter().map(|r| (r.eid, r.ing, r.loc)).collect::<Vec<_>>();
        if dk(&self.d) != dk(&o.d) {
            return Some(format!("d differs ({} vs {})", self.d.len(), o.d.len()));
        }
        if self.fb != o.fb {
            return Some(format!("fb differs ({}+{} vs {}+{})", self.fb.v.len(), self.fb.x.len(), o.fb.v.len(), o.fb.x.len()));
        }
        if self.fbn != o.fbn {
            return Some(format!("fbn differs ({} vs {})", self.fbn.len(), o.fbn.len()));
        }
        if self.side != o.side {
            return Some(format!("side differs ({} vs {})", self.side.len(), o.side.len()));
        }
        None
    }
}

/// measurement aid (SHOP_SOLBENCH): `parse_events` and `parse_events_sol`, serial, alternating
/// which runs first per chunk; prints ns per record for each and checks the outputs agree.
fn sol_bench<E: Ev>(ev_chunks: &[B], cx: &EvCtx) {
    let (mut a, mut b) = (EvBufs::<E>::new(), EvBufs::<E>::new());
    let reps: usize = std::env::var("SHOP_SOLREPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let (mut best_o, mut best_n) = (f64::MAX, f64::MAX);
    for rep in 0..reps {
        let (mut to, mut tn, mut n, mut bad) = (0f64, 0f64, 0usize, 0usize);
        for (i, &c) in ev_chunks.iter().enumerate() {
            a.clear();
            b.clear();
            let old = |a: &mut EvBufs<E>| {
                let t = std::time::Instant::now();
                parse_events(c, cx, &mut a.m, &mut a.d, &mut a.fb, &mut a.fbn, &mut a.side);
                t.elapsed().as_secs_f64()
            };
            let new = |b: &mut EvBufs<E>| {
                let t = std::time::Instant::now();
                parse_events_sol::<E, false>(c, cx, &mut b.m, &mut b.d, &mut b.fb, &mut b.fbn, &mut b.side, &mut Dz::off());
                t.elapsed().as_secs_f64()
            };
            if i % 2 == 0 {
                to += old(&mut a);
                tn += new(&mut b);
            } else {
                tn += new(&mut b);
                to += old(&mut a);
            }
            n += a.m.len();
            if let Some(e) = a.diff(&b) {
                if bad == 0 {
                    eprintln!("solbench MISMATCH chunk {i}: {e}");
                }
                bad += 1;
            }
        }
        let (no, nn) = (to * 1e9 / n.max(1) as f64, tn * 1e9 / n.max(1) as f64);
        best_o = best_o.min(no);
        best_n = best_n.min(nn);
        eprintln!("solbench rep{rep} old {no:.2} ns/rec new {nn:.2} ns/rec ratio {:.3} recs {n} chunks {} mismatched {bad}", no / nn, ev_chunks.len());
    }
    eprintln!("solbench best old {best_o:.2} ns/rec new {best_n:.2} ns/rec speedup {:.3}", best_o / best_n);
}

struct EvChunk<E> {
    m: Vec<E>,
    d: Vec<DRec>,
    doff: Vec<u32>,
    dz: Dz<E>,
    fb: Fb,
    /// the Hex event ids (`Fb::x`) grouped by `dx_hash` bucket, with offsets like `d` / `doff`
    dx: Vec<DX>,
    dxoff: Vec<u32>,
    fbn: Vec<(u64, u64)>,
    side: Vec<(u32, i64)>,
}

#[allow(clippy::too_many_arguments)]
fn ev_after<E: Ev>(
    mut ev_out: Vec<EvChunk<E>>,
    tab: &IdTab,
    mf: &MoverF,
    click_out: &[(Vec<ClickRec>, Vec<u32>)],
    ws: i64,
    we: i64,
    ncamp: usize,
    d0: i64,
    nd: usize,
    tick: &dyn Fn(&str),
) -> Vec<(Vec<Vec<UD>>, CampAcc)> {
    if let Some(cfg) = idx_cfg() {
        return ev_after_idx(ev_out, tab, mf, click_out, ws, we, ncamp, d0, nd, tick, cfg);
    }
    // dedup on event_id: every line has a global index (chunk base + its index in the chunk);
    // each event_id's losers are marked in the dead bitmap. Each task takes DDG consecutive
    // event_id buckets, one small (L2-sized) table at a time.
    let mut mbase: Vec<u32> = Vec::with_capacity(ev_out.len() + 1);
    let mut acc = 0usize;
    for c in &ev_out {
        mbase.push(acc as u32);
        acc += c.m.len();
    }
    assert!(acc < u32::MAX as usize);
    mbase.push(acc as u32);
    let dead: Vec<AtomicU64> = (0..acc / 64 + 1).map(|_| AtomicU64::new(0)).collect();
    let kill = |g: u32| {
        dead[(g >> 6) as usize].fetch_or(1u64 << (g & 63), Relaxed);
    };
    {
        let side_any = ev_out.iter().any(|c| !c.side.is_empty());
        let ibase = ws.div_euclid(1000);
        // exact ingest of line g with key k (only needed on key ties when some key is inexact)
        let exact = |g: u32, k: u32| -> i64 {
            let c = mbase.partition_point(|&b| b <= g) - 1;
            let loc = g - mbase[c];
            let sd = &ev_out[c].side;
            match sd.binary_search_by_key(&loc, |x| x.0) {
                Ok(i) => sd[i].1,
                Err(_) => (k as i64 - IOFF + ibase) * 1000,
            }
        };
        // the bytes of line g's event_id when pack6 rejected it (its record sits in `fb`, loc order)
        let fbb = |g: u32| -> B {
            let c = mbase.partition_point(|&b| b <= g) - 1;
            let loc = g - mbase[c];
            let f = &ev_out[c].fb.v;
            unpack(f[f.binary_search_by_key(&loc, |x| x.2).unwrap()].0)
        };
        // does line g (ingest key k) beat line og (key ok) for the same event_id: smaller ingest, then exact ingest
        let better = |g: u32, k: u32, og: u32, ok: u32| k < ok || (k == ok && side_any && exact(g, k) < exact(og, ok));
        let (ndb, dsh) = idx_cfg().map_or((0, 0), |c| (c.ndb, c.sh));
        rayon::join(
            || {
                rayon::join(
                    || {
                        // dense buckets: one slot per index, a full slot is a duplicate event_id
                        const G: usize = 4;
                        let span = 1usize << dsh;
                        let mask = span as u64 - 1;
                        (0..ndb.div_ceil(G)).into_par_iter().for_each(|g| {
                            let mut tm = now();
                            let mut t: Vec<u64> = vec![u64::MAX; span];
                            for b in g * G..((g + 1) * G).min(ndb) {
                                let mut any = false;
                                for (ci, c) in ev_out.iter().enumerate() {
                                    let mb = mbase[ci];
                                    for r in &c.d[c.doff[b] as usize..c.doff[b + 1] as usize] {
                                        any = true;
                                        let gi = mb + r.loc;
                                        let pk = ((r.ing as u64) << 32) | gi as u64;
                                        // SAFETY: the slot index is masked to span
                                        let o = unsafe { t.get_unchecked_mut((r.eid & mask) as usize) };
                                        if *o == u64::MAX {
                                            *o = pk;
                                            continue;
                                        }
                                        let ok = (*o >> 32) as u32;
                                        let og = *o as u32;
                                        let win = r.ing < ok || (r.ing == ok && side_any && exact(gi, r.ing) < exact(og, ok));
                                        if win {
                                            *o = pk;
                                            kill(og);
                                        } else {
                                            kill(gi);
                                        }
                                    }
                                }
                                if any {
                                    t.fill(u64::MAX);
                                }
                            }
                            pt(2, &mut tm);
                        })
                    },
                    || {
                (0..EB / DDG).into_par_iter().for_each(|g| {
                    let mut tm = now();
                    let totals: [usize; DDG] = std::array::from_fn(|i| {
                        ev_out.iter().map(|c| (c.doff[ndb + g * DDG + i + 1] - c.doff[ndb + g * DDG + i]) as usize).sum()
                    });
                    let maxcap = (totals.iter().max().unwrap() * 2).max(16).next_power_of_two();
                    // slot: (event_id, ingest key << 32 | global index of the current winner)
                    let mut tbuf: Vec<[u64; 2]> = vec![[0, 0]; maxcap];
                    let mut xbuf: Vec<[u64; 4]> = Vec::new();
                    for (i, &total) in totals.iter().enumerate() {
                        let b = ndb + g * DDG + i;
                        let cap = (total * 2).max(16).next_power_of_two();
                        let mask = cap - 1;
                        let t = &mut tbuf[..cap];
                        if i > 0 {
                            t.fill([0, 0]);
                        }
                        for (ci, c) in ev_out.iter().enumerate() {
                            let mb = mbase[ci];
                            for r in &c.d[c.doff[b] as usize..c.doff[b + 1] as usize] {
                                let gi = mb + r.loc;
                                let pk = ((r.ing as u64) << 32) | gi as u64;
                                let mut p = (mix(r.eid) >> 8) as usize & mask;
                                loop {
                                    let o = unsafe { t.get_unchecked_mut(p) };
                                    if o[0] == 0 {
                                        *o = [r.eid, pk];
                                        break;
                                    }
                                    if o[0] == r.eid {
                                        if r.eid & HBIT != 0 && fbb(gi) != fbb(o[1] as u32) {
                                            p = (p + 1) & mask;
                                            continue;
                                        }
                                        let ok = (o[1] >> 32) as u32;
                                        let og = o[1] as u32;
                                        let win = r.ing < ok || (r.ing == ok && side_any && exact(gi, r.ing) < exact(og, ok));
                                        if win {
                                            o[1] = pk;
                                            kill(og);
                                        } else {
                                            kill(gi);
                                        }
                                        break;
                                    }
                                    p = (p + 1) & mask;
                                }
                            }
                        }
                        dx_table(&ev_out, &mbase, g * DDG + i, &mut xbuf, &better, &kill);
                    }
                    pt(2, &mut tm);
                })
                    },
                );
            },
            || {},
        );
        if TIMING.load(Relaxed) {
            let dense: usize = ev_out.iter().map(|c| c.doff[ndb] as usize).sum();
            let all: usize = ev_out.iter().map(|c| c.d.len() + c.fb.v.len() + c.dx.len()).sum();
            tick(&format!("idx dense records {dense} of {all} ({:.5})", dense as f64 / all.max(1) as f64));
        }
    }
    let parts = resolve_parts(ev_out, &mbase, &dead, tab, mf);
    drop(dead);
    tick("dedup done");

    let offs: Vec<&[u32]> = parts.iter().map(|x| &x.1[..]).collect();
    let cost = pb_cost(&offs, click_out);
    lpt_run(&cost, || SsW::new(ncamp, d0, nd), |p, w: &mut SsW<E>| {
        let sl = w.gather(p, &parts, click_out);
        let mut uds = Vec::new();
        process_bucket(&sl, &mut w.evs, &mut w.tmp, &mut w.cnt, &mut w.cl, &mut w.ctmp, &mut w.win, ncamp as u32, &mut uds, &mut w.cm, ws, we.wrapping_sub(ws));
        w.out.push(uds);
    })
    .into_iter()
    .map(|w| (w.out, w.cm))
    .collect()
}

/// Scratch and accumulators of one sessionize worker.
struct SsW<E: Ev> {
    cm: CampAcc,
    win: Win,
    out: Vec<Vec<UD>>,
    evs: Vec<E>,
    tmp: Vec<E>,
    evs_s: Vec<E::S>,
    tmp_s: Vec<E::S>,
    runs: Vec<Run>,
    srt: Vec<SRun>,
    cnt: Vec<u32>,
    cl: Vec<ClickRec>,
    ctmp: Vec<ClickRec>,
}

impl<E: Ev> SsW<E> {
    fn new(ncamp: usize, d0: i64, nd: usize) -> Self {
        SsW {
            cm: CampAcc::new(ncamp + 1, d0, nd),
            win: Win { cnt: vec![0; ncamp + 1], pos: vec![0; ncamp + 1], active: vec![] },
            out: vec![],
            evs: vec![],
            tmp: vec![],
            evs_s: vec![],
            tmp_s: vec![],
            runs: vec![],
            srt: vec![],
            cnt: vec![],
            cl: vec![],
            ctmp: vec![],
        }
    }
    /// Collects person bucket p's clicks from every chunk; returns its non-empty line slices in
    /// chunk order (the sort reads them in place).
    fn gather<'a>(&mut self, p: usize, parts: &'a [(Vec<E>, Vec<u32>)], click_out: &[(Vec<ClickRec>, Vec<u32>)]) -> Vec<&'a [E]> {
        let mut tm = now();
        self.cl.clear();
        for (v, o) in click_out.iter() {
            self.cl.extend_from_slice(&v[o[p] as usize..o[p + 1] as usize]);
        }
        let sl: Vec<&[E]> = parts.iter().map(|(v, o)| &v[o[p] as usize..o[p + 1] as usize]).filter(|s| !s.is_empty()).collect();
        pt(6, &mut tm);
        sl
    }

    /// Dense path: collects person bucket p's clicks and run descriptors, orders the runs by
    /// (digit, key, part, offset) (the order process_runs used to rebuild from the gathered
    /// lines), then copies each run's lines into evs in that order. Runs of one person end up
    /// adjacent and are joined into one Run (sorted if every piece is sorted and the seams are).
    fn gather_rd(&mut self, p: usize, dparts: &[RdPart<E::S>], rest: &(Vec<E>, Vec<u32>), click_out: &[(Vec<ClickRec>, Vec<u32>)]) {
        let mut tm = now();
        self.cl.clear();
        for (v, o) in click_out.iter() {
            self.cl.extend_from_slice(&v[o[p] as usize..o[p + 1] as usize]);
        }
        let srt = &mut self.srt;
        srt.clear();
        for (t, (_, dv, o, _)) in dparts.iter().enumerate() {
            for r in &dv[o[p] as usize..o[p + 1] as usize] {
                srt.push(SRun { d: dg(r.k) as u32, lk: r.lk, k: r.k, src: ((t as u64) << 32) | r.off as u64 });
            }
        }
        {
            // the few lines that took the hash path: runs found here
            let (rv, ro) = rest;
            let (lo, hi) = (ro[p] as usize, ro[p + 1] as usize);
            let t = (dparts.len() as u64) << 32;
            let mut i = lo;
            while i < hi {
                let k = rv[i].key();
                let mut ok = true;
                let mut j = i + 1;
                while j < hi && rv[j].key() == k {
                    ok &= rv[j].rts() >= rv[j - 1].rts();
                    j += 1;
                }
                srt.push(SRun { d: dg(k) as u32, lk: RunD::lk(j - i, ok), k, src: t | i as u64 });
                i = j;
            }
        }
        pt(6, &mut tm);
        srt.sort_unstable_by_key(|r| (r.d, r.k, r.src));
        pt(7, &mut tm);
        let nd = dparts.len();
        let src = |r: &SRun| -> *const u8 {
            let t = (r.src >> 32) as usize;
            let off = r.src as u32 as usize;
            if t < nd {
                dparts[t].0.as_ptr().wrapping_add(off) as *const u8
            } else {
                rest.0.as_ptr().wrapping_add(off) as *const u8
            }
        };
        let evs = &mut self.evs_s;
        let runs = &mut self.runs;
        evs.clear();
        runs.clear();
        for i in 0..srt.len() {
            if let Some(nx) = srt.get(i + RD_AHEAD) {
                let q = src(nx);
                prefetch(q);
                prefetch(q.wrapping_add(64));
            }
            let r = &srt[i];
            let t = (r.src >> 32) as usize;
            let off = r.src as u32 as usize;
            let n = (r.lk >> 1) as usize;
            let ok = r.lk & 1 != 0;
            let s = evs.len();
            if t < nd {
                let v = &dparts[t].0[off..off + n];
                match runs.last_mut() {
                    Some(last) if last.k == r.k => {
                        last.ok = last.ok && ok && v[0].rts() >= evs[s - 1].rts();
                        last.l += n as u32;
                    }
                    _ => runs.push(Run { d: r.d, k: r.k, s: s as u32, l: n as u32, ok }),
                }
                evs.extend_from_slice(v);
            } else {
                let v = &rest.0[off..off + n];
                match runs.last_mut() {
                    Some(last) if last.k == r.k => {
                        last.ok = last.ok && ok && v[0].rts() >= evs[s - 1].rts();
                        last.l += n as u32;
                    }
                    _ => runs.push(Run { d: r.d, k: r.k, s: s as u32, l: n as u32, ok }),
                }
                evs.extend(v.iter().map(|e| e.slim()));
            }
        }
        pt(6, &mut tm);
    }
}

/// Run descriptor of the dense dedup walk: person key, offset of the run in its task's lines,
/// length << 1 | (run sorted by ts).
#[derive(Clone, Copy)]
struct RunD {
    k: u64,
    off: u32,
    lk: u32,
}

impl RunD {
    #[inline(always)]
    fn lk(len: usize, ok: bool) -> u32 {
        debug_assert!(len < 1 << 31);
        ((len as u32) << 1) | ok as u32
    }
    #[inline(always)]
    fn new(k: u64, off: usize, len: usize, ok: bool) -> Self {
        RunD { k, off: off as u32, lk: Self::lk(len, ok) }
    }
}

/// A descriptor in gather_rd: digit, person key, part << 32 | offset (its sort key) and lk.
#[derive(Clone, Copy)]
struct SRun {
    d: u32,
    lk: u32,
    k: u64,
    src: u64,
}

/// How many descriptors ahead gather_rd prefetches a run's first lines.
const RD_AHEAD: usize = 3;

/// One dense dedup task's output: its kept lines in index order, its run descriptors grouped by
/// person bucket, their bucket offsets, and the bucket offsets of the lines they cover.
type RdPart<S> = (Vec<S>, Vec<RunD>, Vec<u32>, Vec<u32>);

/// Groups run descriptors by person bucket (order kept inside a bucket). Returns the grouped
/// descriptors, their bucket offsets, and prefix sums of the lines per bucket.
fn rd_scatter(ds: &[RunD]) -> (Vec<RunD>, Vec<u32>, Vec<u32>) {
    let mut cnt = vec![0u32; PB + 1];
    let mut eo = vec![0u32; PB + 1];
    for r in ds {
        let p = pbk(r.k);
        cnt[p + 1] += 1;
        eo[p + 1] += r.lk >> 1;
    }
    for p in 0..PB {
        cnt[p + 1] += cnt[p];
        eo[p + 1] += eo[p];
    }
    let mut out: Vec<RunD> = Vec::with_capacity(ds.len());
    {
        let mut w = cnt.clone();
        let sp = out.spare_capacity_mut();
        for r in ds {
            let p = pbk(r.k);
            sp[w[p] as usize].write(*r);
            w[p] += 1;
        }
    }
    unsafe { out.set_len(ds.len()) };
    (out, cnt, eo)
}

/// Estimated sessionize work of each person bucket: its lines plus twice its clicks (a click
/// costs about two lines: radix sort plus window upkeep).
fn pb_cost(offs: &[&[u32]], click_out: &[(Vec<ClickRec>, Vec<u32>)]) -> Vec<u64> {
    const BLK: usize = 64;
    let mut cost = vec![0u64; PB];
    cost.par_chunks_mut(BLK).enumerate().for_each(|(b, c)| {
        let lo = b * BLK;
        for o in offs.iter() {
            for (i, x) in c.iter_mut().enumerate() {
                *x += (o[lo + i + 1] - o[lo + i]) as u64;
            }
        }
        for (_, o) in click_out.iter() {
            for (i, x) in c.iter_mut().enumerate() {
                *x += 2 * (o[lo + i + 1] - o[lo + i]) as u64;
            }
        }
    });
    cost
}

/// Runs `f(p, state)` for every partition p (cost[p] = its estimated work) on one worker per pool
/// thread. The workers pull partitions biggest first from a shared counter (LPT list scheduling):
/// big partitions start early and the small ones fill the end, so the workers finish nearly
/// together instead of one static group running alone. Returns every worker's state.
fn lpt_run<S: Send>(cost: &[u64], init: impl Fn() -> S + Sync, f: impl Fn(usize, &mut S) + Sync) -> Vec<S> {
    let mut ord: Vec<u32> = (0..cost.len() as u32).collect();
    ord.sort_unstable_by_key(|&p| (std::cmp::Reverse(cost[p as usize]), p));
    let next = AtomicUsize::new(0);
    let nt = rayon::current_num_threads().clamp(1, ord.len().max(1));
    let t0 = now();
    let r: Vec<(S, f64)> = (0..nt)
        .into_par_iter()
        .with_min_len(1)
        .with_max_len(1)
        .map(|_| {
            let mut s = init();
            loop {
                let i = next.fetch_add(1, Relaxed);
                if i >= ord.len() {
                    break;
                }
                f(ord[i] as usize, &mut s);
            }
            (s, t0.map_or(0.0, |t| t.elapsed().as_secs_f64()))
        })
        .collect();
    if t0.is_some() {
        let mut e: Vec<f64> = r.iter().map(|x| x.1).collect();
        e.sort_by(f64::total_cmp);
        let tot: u64 = cost.iter().sum();
        eprintln!("lpt parts={} workers={nt} cost max={} total={tot} finish first={:.3} last={:.3}", cost.len(), cost.iter().max().unwrap_or(&0), e[0], e[e.len() - 1]);
    }
    r.into_iter().map(|x| x.0).collect()
}

/// Maps every item on the pool like an in-order map, but the workers pull the items biggest cost
/// first from a shared counter (LPT list scheduling), so the big items start early and the small
/// ones fill the end. The results come back in item order, so the output does not depend on the
/// schedule.
fn lpt_map<I: Send, T: Send>(items: Vec<I>, cost: &[u64], f: impl Fn(usize, I) -> T + Sync) -> Vec<T> {
    let n = items.len();
    let mut ord: Vec<u32> = (0..n as u32).collect();
    ord.sort_unstable_by_key(|&p| (std::cmp::Reverse(cost[p as usize]), p));
    let slots: Vec<std::sync::Mutex<Option<I>>> = items.into_iter().map(|x| std::sync::Mutex::new(Some(x))).collect();
    let next = AtomicUsize::new(0);
    let nt = rayon::current_num_threads().clamp(1, n.max(1));
    let r: Vec<Vec<(u32, T)>> = (0..nt)
        .into_par_iter()
        .with_min_len(1)
        .with_max_len(1)
        .map(|_| {
            let mut v = Vec::new();
            loop {
                let i = next.fetch_add(1, Relaxed);
                if i >= n {
                    break;
                }
                let p = ord[i];
                let x = slots[p as usize].lock().unwrap().take().unwrap();
                v.push((p, f(p as usize, x)));
            }
            v
        })
        .collect();
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    for v in r {
        for (p, t) in v {
            out[p as usize] = Some(t);
        }
    }
    out.into_iter().map(|x| x.unwrap()).collect()
}

// ---------- dense-index pipeline after the parse ----------
// With dense event indices, the winners of each dense bucket are emitted in index order, which
// is the generator's order: a person's events then mostly arrive as one run already sorted by
// ts. The sessionizer sorts only the persons that do not (process_runs). Exact for any input:
// the order of the lines only decides the speed.

/// Kept lines of the noidx path: user key -> person key (empty keys and dedup losers are dropped),
/// grouped by person bucket. The sessionizer gathers a bucket from every part.
/// Lines whose key misses the mover filter map to themselves: each chunk groups them by person
/// bucket at once (a cache-sized scatter). Lines that hit it need a table lookup; a lookup per
/// line would miss the cache (the table is about 16 bytes per linked ID), so each chunk lists
/// those lines grouped by the top bits of their table slot (about 1 MB of the table per region),
/// and one task per region answers that region's lines of every chunk with its probes in cache,
/// then groups them by person bucket as one more part. Exact: every hit key gets `tab.get` as
/// before and every miss maps to itself as before; only the order of lines inside a person
/// bucket changes, and the sessionizer orders each person's lines by ts (equal ts share a session).
fn resolve_parts<E: Ev>(ev_out: Vec<EvChunk<E>>, mbase: &[u32], dead: &[AtomicU64], tab: &IdTab, mf: &MoverF) -> Vec<(Vec<E>, Vec<u32>)> {
    let rb = (tab.s.len() >> 16).clamp(1, 512).trailing_zeros();
    let nr = 1usize << rb;
    let reg = move |k: u64| -> usize {
        if rb == 0 {
            0
        } else {
            (mix(k) >> (64 - rb)) as usize
        }
    };
    // (filter misses by person bucket, their offsets, filter hits split by region: one list per
    // region, filled in line order during the filter pass, so no second pass over the hits)
    // chunks and regions start largest first (lpt_map), results stay in chunk / region order
    let ccost: Vec<u64> = ev_out.iter().map(|c| c.m.len() as u64).collect();
    let a: Vec<(Vec<E>, Vec<u32>, Vec<Vec<E>>)> = lpt_map(ev_out, &ccost, |ci, c: EvChunk<E>| {
        let mut tm = now();
        let mb = mbase[ci] as usize;
        let m = c.m;
        let n = m.len();
        let mut cnt = vec![0u32; PB + 1];
        // Without data-dependent branches: a live line (key set, not a dedup loser) that misses
        // the filter sets its bit in `km` and counts in its person bucket; a live line that hits
        // it lists its index in `hi`. Every other line adds 0 to a count and lists nothing. m is
        // only read.
        let mut km = vec![0u64; n.div_ceil(64)];
        let mut hi: Vec<u32> = Vec::with_capacity(n);
        {
            let hs = hi.spare_capacity_mut();
            let mut nh = 0usize;
            for i in 0..n {
                if i + RPD < n {
                    mf.pf(m[i + RPD].key());
                }
                let g = mb + i;
                let k = m[i].key();
                let live = (k != 0) & (dead[g >> 6].load(Relaxed) & (1u64 << (g & 63)) == 0);
                let hit = mf.hit(k);
                let keep = live & !hit;
                cnt[pbk(k) + 1] += keep as u32;
                km[i >> 6] |= (keep as u64) << (i & 63);
                // SAFETY: nh <= i < n = hs.len()
                unsafe { hs.get_unchecked_mut(nh) }.write(i as u32);
                nh += (live & hit) as usize;
            }
            // SAFETY: the first nh slots were written
            unsafe { hi.set_len(nh) };
        }
        let mut h: Vec<Vec<E>> = (0..nr).map(|_| Vec::with_capacity(hi.len() / nr + hi.len() / (4 * nr) + 8)).collect();
        for &i in &hi {
            let e = m[i as usize];
            // SAFETY: reg(k) < nr = h.len()
            unsafe { h.get_unchecked_mut(reg(e.key())) }.push(e);
        }
        drop(hi);
        pt(4, &mut tm);
        for p in 0..PB {
            cnt[p + 1] += cnt[p];
        }
        let tot = cnt[PB] as usize;
        let mut out: Vec<E> = Vec::with_capacity(tot);
        {
            let mut w = cnt.clone();
            let sp = out.spare_capacity_mut();
            for (wi, &b0) in km.iter().enumerate() {
                let mut b = b0;
                while b != 0 {
                    let i = wi * 64 + b.trailing_zeros() as usize;
                    b &= b - 1;
                    let e = m[i];
                    let p = pbk(e.key());
                    sp[w[p] as usize].write(e);
                    w[p] += 1;
                }
            }
        }
        unsafe { out.set_len(tot) };
        pt(5, &mut tm);
        (out, cnt, h)
    });
    // regroup the hit lists by region (chunk order kept inside a region)
    let mut a0: Vec<(Vec<E>, Vec<u32>)> = Vec::with_capacity(a.len() + nr);
    let mut regs: Vec<Vec<Vec<E>>> = (0..nr).map(|_| Vec::with_capacity(a.len())).collect();
    for (out, cnt, h) in a {
        a0.push((out, cnt));
        for (r, v) in h.into_iter().enumerate() {
            regs[r].push(v);
        }
    }
    let rcost: Vec<u64> = regs.iter().map(|rl| rl.iter().map(|s| s.len() as u64).sum()).collect();
    let hp: Vec<(Vec<E>, Vec<u32>)> = lpt_map(regs, &rcost, |_, mut rl: Vec<Vec<E>>| {
        const PD: usize = 16;
        let mut tm = now();
        let mut cnt = vec![0u32; PB + 1];
        // person keys set in place: same lookups as a line-by-line pass, with this region of
        // the table in cache. The region's hit lists (one per chunk) are one stream for the
        // prefetch: a second cursor runs PD lines ahead across list ends, computes each home slot
        // once, prefetches it and parks it in a ring for the lookup.
        {
            let ls: Vec<(*mut E, usize)> = rl.iter_mut().filter(|s| !s.is_empty()).map(|s| (s.as_mut_ptr(), s.len())).collect();
            let tot: usize = ls.iter().map(|x| x.1).sum();
            let mut ring = [0usize; 32];
            let mut pc = (0usize, 0usize);
            let adv = |pc: &mut (usize, usize)| -> usize {
                let (p, n) = ls[pc.0];
                // SAFETY: pc.1 < n, the length of the list at p
                let k = unsafe { (*p.add(pc.1)).key() };
                pc.1 += 1;
                if pc.1 == n {
                    pc.0 += 1;
                    pc.1 = 0;
                }
                let sl = tab.slot(k);
                tab.pf_slot(sl);
                sl
            };
            for t in 0..tot.min(PD) {
                ring[t] = adv(&mut pc);
            }
            let mut t = 0usize;
            for &(p, n) in &ls {
                for j in 0..n {
                    if t + PD < tot {
                        ring[(t + PD) & 31] = adv(&mut pc);
                    }
                    // SAFETY: j < n; the look-ahead cursor only reads lines after this one
                    let e = unsafe { &mut *p.add(j) };
                    let pk = tab.get_from(e.key(), ring[t & 31]);
                    e.set_key(pk);
                    cnt[pbk(pk) + 1] += 1;
                    t += 1;
                }
            }
        }
        pt(4, &mut tm);
        for p in 0..PB {
            cnt[p + 1] += cnt[p];
        }
        let tot = cnt[PB] as usize;
        let mut out: Vec<E> = Vec::with_capacity(tot);
        {
            let mut w = cnt.clone();
            let sp = out.spare_capacity_mut();
            for s in &rl {
                for e in s {
                    let b = pbk(e.key());
                    sp[w[b] as usize].write(*e);
                    w[b] += 1;
                }
            }
        }
        unsafe { out.set_len(tot) };
        drop(rl);
        pt(5, &mut tm);
        (out, cnt)
    });
    a0.extend(hp);
    a0
}

/// Last few (user key, person key) pairs: lines in index order mostly repeat a user.
struct PkCache {
    k: [u64; 4],
    p: [u64; 4],
    n: usize,
}

#[inline(always)]
fn pk_of(c: &mut PkCache, k: u64, tab: &IdTab, mf: &MoverF) -> u64 {
    for i in 0..4 {
        if c.k[i] == k {
            return c.p[i];
        }
    }
    let p = mf.person(k, tab);
    c.k[c.n & 3] = k;
    c.p[c.n & 3] = p;
    c.n += 1;
    p
}

/// Groups kept lines (person keys set) by person bucket, keeping their order inside a bucket.
fn pb_scatter<E: Ev>(em: &[E]) -> (Vec<E>, Vec<u32>) {
    let mut cnt = vec![0u32; PB + 1];
    for e in em {
        cnt[pbk(e.key()) + 1] += 1;
    }
    for p in 0..PB {
        cnt[p + 1] += cnt[p];
    }
    let tot = cnt[PB] as usize;
    let mut out: Vec<E> = Vec::with_capacity(tot);
    {
        let mut w = cnt.clone();
        let sp = out.spare_capacity_mut();
        for e in em {
            let p = pbk(e.key());
            sp[w[p] as usize].write(*e);
            w[p] += 1;
        }
    }
    unsafe { out.set_len(tot) };
    (out, cnt)
}

#[allow(clippy::too_many_arguments)]
fn ev_after_idx<E: Ev>(
    ev_out: Vec<EvChunk<E>>,
    tab: &IdTab,
    mf: &MoverF,
    click_out: &[(Vec<ClickRec>, Vec<u32>)],
    ws: i64,
    we: i64,
    ncamp: usize,
    d0: i64,
    nd: usize,
    tick: &dyn Fn(&str),
    cfg: IdxCfg,
) -> Vec<(Vec<Vec<UD>>, CampAcc)> {
    let _ = we;
    let ndb = cfg.ndb;
    let mut mbase: Vec<u32> = Vec::with_capacity(ev_out.len() + 1);
    let mut acc = 0usize;
    for c in &ev_out {
        mbase.push(acc as u32);
        acc += c.m.len();
    }
    assert!(acc < u32::MAX as usize);
    mbase.push(acc as u32);
    if TIMING.load(Relaxed) {
        let dense: usize = ev_out.iter().map(|c| c.dz.len()).sum();
        let ov: usize = ev_out.iter().map(|c| c.dz.ov.len()).sum();
        let slots: usize = ev_out.iter().map(|c| c.dz.slots()).sum();
        let all: usize = dense + ev_out.iter().map(|c| c.d.len() + c.fb.v.len() + c.dx.len()).sum::<usize>();
        tick(&format!("idx dense records {dense} of {all} ({:.5}) overflow {ov} slots {slots}", dense as f64 / all.max(1) as f64));
    }
    let (dparts, rest): (Vec<RdPart<E::S>>, (Vec<E>, Vec<u32>)) = {
        let side_any = ev_out.iter().any(|c| !c.side.is_empty());
        let dside_any = ev_out.iter().any(|c| !c.dz.dside.is_empty());
        let ibase = ws.div_euclid(1000);
        // the same for dense record l (chunk << 32 | position)
        let dexact = |l: u64, k: u32| -> i64 {
            let sd = &ev_out[(l >> 32) as usize].dz.dside;
            match sd.binary_search_by_key(&(l as u32), |x| x.0) {
                Ok(i) => sd[i].1,
                Err(_) => (k as i64 - IOFF + ibase) * 1000,
            }
        };
        // exact ingest of line g with key k (only needed on key ties when some key is inexact)
        let exact = |g: u32, k: u32| -> i64 {
            let c = mbase.partition_point(|&b| b <= g) - 1;
            let loc = g - mbase[c];
            let sd = &ev_out[c].side;
            match sd.binary_search_by_key(&loc, |x| x.0) {
                Ok(i) => sd[i].1,
                Err(_) => (k as i64 - IOFF + ibase) * 1000,
            }
        };
        // the bytes of line g's event_id when pack6 rejected it (its record sits in `fb`, loc order)
        let fbb = |g: u32| -> B {
            let c = mbase.partition_point(|&b| b <= g) - 1;
            let loc = g - mbase[c];
            let f = &ev_out[c].fb.v;
            unpack(f[f.binary_search_by_key(&loc, |x| x.2).unwrap()].0)
        };
        // does line g (ingest key k) beat line og (key ok) for the same event_id: smaller ingest, then exact ingest
        let better = |g: u32, k: u32, og: u32, ok: u32| k < ok || (k == ok && side_any && exact(g, k) < exact(og, ok));
        let line = |g: u32| -> E {
            let c = mbase.partition_point(|&b| b <= g) - 1;
            ev_out[c].m[(g - mbase[c]) as usize]
        };
        rayon::join(
            || {
                // dense buckets: one slot per index; the slot keeps the winning copy (smallest
                // ingest), then the slots are read in index order
                const G: usize = 4;
                let span = 1usize << cfg.sh;
                let mask = span as u64 - 1;
                (0..ndb.div_ceil(G))
                    .into_par_iter()
                    .map(|g| {
                        let mut tm = now();
                        let mut t: Vec<u64> = vec![u64::MAX; span];
                        let mut lb: Vec<E> = Vec::new();
                        let mut lg: Vec<u64> = Vec::new();
                        // em outlives the task (sessionize reads its runs), so size it once from the
                        // record counts instead of growing it by doubling
                        let ub: usize = (g * G..((g + 1) * G).min(ndb))
                            .map(|b| {
                                ev_out
                                    .iter()
                                    .map(|c| {
                                        let (sa, _, oa, _) = c.dz.seg(b);
                                        sa.len() + oa.len()
                                    })
                                    .sum::<usize>()
                            })
                            .sum();
                        let mut em: Vec<E::S> = Vec::with_capacity(ub);
                        let mut ds: Vec<RunD> = Vec::new();
                        // open run: person key, start in em, ts sorted so far, last ts
                        let (mut rk, mut rs, mut rok, mut lts) = (0u64, 0usize, true, 0u64);
                        let mut pc = PkCache { k: [0; 4], p: [0; 4], n: 0 };
                        for b in g * G..((g + 1) * G).min(ndb) {
                            lb.clear();
                            lg.clear();
                            let bn: usize = ev_out
                                .iter()
                                .map(|c| {
                                    let (sa, _, oa, _) = c.dz.seg(b);
                                    sa.len() + oa.len()
                                })
                                .sum();
                            lb.reserve(bn);
                            if dside_any {
                                lg.reserve(bn);
                            }
                            let lbp = lb.as_mut_ptr();
                            let lgp = lg.as_mut_ptr();
                            let mut qn = 0usize;
                            for (ci, c) in ev_out.iter().enumerate() {
                                // each chunk keeps bucket b in its own far-off stretch: fetch the count a few
                                // chunks ahead and the records of the next chunk but one
                                if let Some(n) = ev_out.get(ci + 4) {
                                    if b < n.dz.dn.len() {
                                        prefetch(n.dz.dn.as_ptr().wrapping_add(b));
                                    }
                                }
                                if let Some(n) = ev_out.get(ci + 2) {
                                    if b < n.dz.dn.len() {
                                        let p = (n.dz.r.as_ptr() as *const DR<E>).wrapping_add(b * n.dz.cap) as *const u8;
                                        let len = n.dz.dn[b] as usize * std::mem::size_of::<DR<E>>();
                                        for o in (0..len.min(1024)).step_by(64) {
                                            prefetch(p.wrapping_add(o));
                                        }
                                    }
                                }
                                let (sa, sp, oa, op) = c.dz.seg(b);
                                for (part, p0) in [(sa, sp), (oa, op)] {
                                    for (j, x) in part.iter().enumerate() {
                                        if let Some(y) = part.get(j + 16) {
                                            prefetch(t.as_ptr().wrapping_add((y.i as u64 & mask) as usize) as *const u8);
                                        }
                                        let q = qn as u32;
                                        // SAFETY: qn < bn <= capacity of lb (and lg when dside_any): bn is the sum of
                                        // all seg lengths of b, and exactly one record is written per seg record
                                        unsafe {
                                            lbp.add(qn).write(x.e);
                                            if dside_any {
                                                lgp.add(qn).write(((ci as u64) << 32) | (p0 + j) as u64);
                                            }
                                        }
                                        qn += 1;
                                        let pk = ((x.ing as u64) << 32) | q as u64;
                                        // SAFETY: the slot index is masked to span
                                        let o = unsafe { t.get_unchecked_mut((x.i as u64 & mask) as usize) };
                                        if *o == u64::MAX {
                                            *o = pk;
                                            continue;
                                        }
                                        let ok = (*o >> 32) as u32;
                                        if x.ing < ok
                                            || (x.ing == ok
                                                && dside_any
                                                // SAFETY: q and *o as u32 are < qn and lg[..qn] is written
                                                && dexact(unsafe { *lgp.add(q as usize) }, x.ing)
                                                    < dexact(unsafe { *lgp.add(*o as u32 as usize) }, ok))
                                        {
                                            *o = pk;
                                        }
                                    }
                                }
                            }
                            // SAFETY: qn records were written to lb (and lg when dside_any), qn <= capacity
                            unsafe {
                                lb.set_len(qn);
                                if dside_any {
                                    lg.set_len(qn);
                                }
                            }
                            pt(2, &mut tm);
                            if !lb.is_empty() {
                                const PD: usize = 16;
                                em.reserve(lb.len());
                                let emp = em.as_mut_ptr();
                                let mut en = em.len();
                                let tl = t.len();
                                for j in 0..tl {
                                    if j + PD < tl {
                                        let n = t[j + PD];
                                        if n != u64::MAX {
                                            prefetch(lb.as_ptr().wrapping_add(n as u32 as usize) as *const u8);
                                        }
                                    }
                                    let o = &mut t[j];
                                    if *o != u64::MAX {
                                        let mut e = lb[*o as u32 as usize];
                                        *o = u64::MAX;
                                        let k = e.key();
                                        if k != 0 {
                                            let pk = pk_of(&mut pc, k, tab, mf);
                                            e.set_key(pk);
                                            let ts = e.rts();
                                            if pk != rk || en == rs {
                                                if en > rs {
                                                    ds.push(RunD::new(rk, rs, en - rs, rok));
                                                }
                                                rk = pk;
                                                rs = en;
                                                rok = true;
                                            } else {
                                                rok &= ts >= lts;
                                            }
                                            lts = ts;
                                            // SAFETY: en < em.len() at bucket start + lb.len() <= capacity (reserved
                                            // above), and at most one write per lb record
                                            unsafe { emp.add(en).write(e.slim()) };
                                            en += 1;
                                        }
                                    }
                                }
                                // SAFETY: en - old len records were written, en <= capacity
                                unsafe { em.set_len(en) };
                            }
                            pt(4, &mut tm);
                        }
                        if em.len() > rs {
                            ds.push(RunD::new(rk, rs, em.len() - rs, rok));
                        }
                        let (dv, dof, eo) = rd_scatter(&ds);
                        pt(5, &mut tm);
                        (em, dv, dof, eo)
                    })
                    .collect::<Vec<_>>()
            },
            || {
                // every other event_id: hash buckets by packed id, then exact bytes for the rest
                let hb: Vec<Vec<E>> = (0..EB / DDG)
                    .into_par_iter()
                    .map(|g| {
                        let mut tm = now();
                        let mut em: Vec<E> = Vec::new();
                        let totals: [usize; DDG] = std::array::from_fn(|i| {
                            ev_out
                                .iter()
                                .map(|c| (c.doff[g * DDG + i + 1] - c.doff[g * DDG + i]) as usize)
                                .sum()
                        });
                        let xany = ev_out.iter().any(|c| c.dxoff[g * DDG] != c.dxoff[(g + 1) * DDG]);
                        if totals.iter().all(|&x| x == 0) && !xany {
                            return em;
                        }
                        let maxcap = (totals.iter().max().unwrap() * 2).max(16).next_power_of_two();
                        let mut tbuf: Vec<[u64; 2]> = vec![[0, 0]; maxcap];
                        let mut xbuf: Vec<[u64; 4]> = Vec::new();
                        for (i, &total) in totals.iter().enumerate() {
                            // Hex event ids: one winner per full slot (the order of `em` is free)
                            let xn = dx_table(&ev_out, &mbase, g * DDG + i, &mut xbuf, &better, |_| {});
                            for o in &xbuf[..xn] {
                                if o[3] != 0 {
                                    em.push(line(o[2] as u32));
                                }
                            }
                            if total == 0 {
                                continue;
                            }
                            let b = g * DDG + i;
                            let cap = (total * 2).max(16).next_power_of_two();
                            let mask = cap - 1;
                            let t = &mut tbuf[..cap];
                            t.fill([0, 0]);
                            for (ci, c) in ev_out.iter().enumerate() {
                                let mb = mbase[ci];
                                for r in &c.d[c.doff[b] as usize..c.doff[b + 1] as usize] {
                                    let gi = mb + r.loc;
                                    let pk = ((r.ing as u64) << 32) | gi as u64;
                                    let mut p = (mix(r.eid) >> 8) as usize & mask;
                                    loop {
                                        // SAFETY: p is masked to cap
                                        let o = unsafe { t.get_unchecked_mut(p) };
                                        if o[0] == 0 {
                                            *o = [r.eid, pk];
                                            break;
                                        }
                                        if o[0] == r.eid {
                                            if r.eid & HBIT != 0 && fbb(gi) != fbb(o[1] as u32) {
                                                p = (p + 1) & mask;
                                                continue;
                                            }
                                            let ok = (o[1] >> 32) as u32;
                                            let og = o[1] as u32;
                                            if r.ing < ok || (r.ing == ok && side_any && exact(gi, r.ing) < exact(og, ok)) {
                                                o[1] = pk;
                                            }
                                            break;
                                        }
                                        p = (p + 1) & mask;
                                    }
                                }
                            }
                            for o in t.iter() {
                                if o[0] != 0 {
                                    em.push(line(o[1] as u32));
                                }
                            }
                        }
                        pt(2, &mut tm);
                        em
                    })
                    .collect();
                let mut tm = now();
                let mut em: Vec<E> = hb.into_iter().flatten().collect();
                pt(3, &mut tm);
                let mut pc = PkCache { k: [0; 4], p: [0; 4], n: 0 };
                em.retain_mut(|e| {
                    let k = e.key();
                    if k == 0 {
                        return false;
                    }
                    e.set_key(pk_of(&mut pc, k, tab, mf));
                    true
                });
                let r = pb_scatter(&em);
                pt(5, &mut tm);
                r
            },
        )
    };
    drop(ev_out);
    tick("dedup done");

    let offs: Vec<&[u32]> = dparts.iter().map(|x| &x.3[..]).chain(std::iter::once(&rest.1[..])).collect();
    let cost = pb_cost(&offs, click_out);
    lpt_run(&cost, || SsW::new(ncamp, d0, nd), |p, w: &mut SsW<E>| {
        w.gather_rd(p, &dparts, &rest, click_out);
        let mut uds = Vec::new();
        process_runs(&w.evs_s, &mut w.tmp_s, &w.runs, &mut w.cnt, &mut w.cl, &mut w.ctmp, &mut w.win, ncamp as u32, &mut uds, &mut w.cm, ws);
        w.out.push(uds);
    })
    .into_iter()
    .map(|w| (w.out, w.cm))
    .collect()
}

/// A stretch of one person's lines in a bucket: digit, person key, start, length, ts sorted.
#[derive(Clone, Copy)]
struct Run {
    d: u32,
    k: u64,
    s: u32,
    l: u32,
    ok: bool,
}

/// process_bucket for lines that arrive mostly as one ts-sorted run per person: the runs are
/// ordered by (digit, key) (gather_rd); a person given as one sorted run is sessionized in place,
/// any other person (several runs, or a run out of ts order) is gathered and sorted by ts first.
#[allow(clippy::too_many_arguments)]
fn process_runs<E: Tp>(
    evs: &[E],
    tmp: &mut Vec<E>,
    runs: &[Run],
    cnt: &mut Vec<u32>,
    clicks: &mut Vec<ClickRec>,
    ctmp: &mut Vec<ClickRec>,
    win: &mut Win,
    organic: u32,
    uds: &mut Vec<UD>,
    cm: &mut CampAcc,
    base: i64,
) {
    let mut tm = now();
    sort_clicks(clicks, ctmp, cnt);
    pt(11, &mut tm);
    let mut ci = 0;
    let mut day = DayAcc { key: 0, day: 0, sessions: 0, events: 0, rev: 0, cats: vec![] };
    let mut a = 0;
    while a < runs.len() {
        let pid = runs[a].k;
        let pd = runs[a].d as usize;
        let mut b = a + 1;
        while b < runs.len() && runs[b].k == pid {
            b += 1;
        }
        let pes: &[E] = if b == a + 1 && runs[a].ok {
            &evs[runs[a].s as usize..(runs[a].s + runs[a].l) as usize]
        } else {
            tmp.clear();
            for r in &runs[a..b] {
                tmp.extend_from_slice(&evs[r.s as usize..(r.s + r.l) as usize]);
            }
            tmp.sort_by_key(|e| e.rts()); // stable driftsort merges the sorted runs
            &tmp[..]
        };
        while ci < clicks.len() && (dg(clicks[ci].key), clicks[ci].key) < (pd, pid) {
            ci += 1;
        }
        let mut cj = ci;
        while cj < clicks.len() && clicks[cj].key == pid {
            cj += 1;
        }
        person_sess(pes, pid, &clicks[ci..cj], win, organic, &mut day, uds, cm, base);
        ci = cj;
        a = b;
    }
    pt(8, &mut tm);
}

/// Sessions of one person (lines sorted by ts, clicks sorted by ts); the loop body of
/// process_bucket.
#[allow(clippy::too_many_arguments)]
#[inline]
fn person_sess<E: Tp>(
    pes: &[E],
    pid: u64,
    cs: &[ClickRec],
    win: &mut Win,
    organic: u32,
    day: &mut DayAcc,
    uds: &mut Vec<UD>,
    cm: &mut CampAcc,
    base: i64,
) {
    let (mut lo, mut hi) = (0usize, 0usize);
    day.key = pid;
    day.day = 0;
    let mut prev = 0i64;
    let mut first = true;
    let mut s_camp = 0u32;
    let mut s_day = 0i64;
    let mut s_rev = 0u64;
    let mut s_purch = false;
    let mut in_sess = false;
    for e in pes {
        let et = e.rts() as i64 + base;
        if first || et - prev > GAP {
            if in_sess {
                cm.sess(s_camp, s_day, s_purch, s_rev);
            }
            first = false;
            in_sess = true;
            let s = et;
            let d = s.div_euclid(DAY);
            s_day = d;
            s_rev = 0;
            s_purch = false;
            while hi < cs.len() && cs[hi].ts <= s {
                win.add(cs[hi].camp);
                hi += 1;
            }
            while lo < hi && cs[lo].ts < s - WEEK {
                win.remove(cs[lo].camp);
                lo += 1;
            }
            s_camp = if hi > lo {
                // the clicks at the latest ts; ties are not in click_id order
                let t = cs[hi - 1].ts;
                let c0 = cs[hi - 1].camp;
                let mut g = hi - 1;
                let mut mixed = false;
                while g > lo && cs[g - 1].ts == t {
                    g -= 1;
                    mixed |= cs[g].camp != c0;
                }
                if mixed {
                    min_cid_camp(&cs[g..hi])
                } else {
                    c0
                }
            } else {
                organic
            };
            cm.touch(&win.active, d);
            if day.day != d {
                flush_day(day, uds);
                day.day = d;
            }
            day.sessions += 1;
        }
        prev = et;
        day.events += 1;
        if let Some((p, ecat)) = e.pc() {
            day.rev += p;
            s_rev += p;
            s_purch = true;
            match day.cats.iter_mut().find(|x| x.0 == ecat) {
                Some(x) => x.1 += p,
                None => day.cats.push((ecat, p)),
            }
        }
    }
    if in_sess {
        cm.sess(s_camp, s_day, s_purch, s_rev);
    }
    flush_day(day, uds);
    for &c in &win.active {
        win.cnt[c as usize] = 0;
    }
    win.active.clear();
}

// ---------- generator event ids -> dense indices ----------
// The published generator writes an event_id as 'e' + base36(scramble(index)), a 40-bit
// scramble with a per-dataset key. The key is recovered at run time from the first IDs of the
// input; a record whose event_id decodes to an index below a size bound goes to a dense bucket
// keyed by that index. Exactness never depends on the key: canonical IDs and their 40-bit
// values are a bijection, and so is the scramble, so equal indices mean equal event_ids. Every
// other event_id keeps the hash path.

const GM: u64 = (1 << 40) - 1;
const GC1: u64 = 0x9E37_79B9_7F4B;
const GC2: u64 = 0xD6E8_FEB8_6659;

const fn inv_odd(a: u64) -> u64 {
    let mut x = a;
    let mut i = 0;
    while i < 6 {
        x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
        i += 1;
    }
    x
}
const GC1I: u64 = inv_odd(GC1) & GM;
const GC2I: u64 = inv_odd(GC2) & GM;

/// 40-bit value of a canonical generator ID packed by `pack6`: prefix code `pre`, then a base-36
/// body of 1-8 chars [0-9a-z] with no leading '0'. None for anything else.
#[inline(always)]
fn gen_val(k: u64, pre: u64) -> Option<u64> {
    if (k >> 54) != pre || (k & 63) != 0 {
        return None;
    }
    let mut x = (k >> 6) & ((1 << 48) - 1);
    if x == 0 {
        return None;
    }
    // eight 6-bit codes -> eight bytes, the first char in the top byte
    x = ((x & 0xFFFF_FF00_0000) << 8) | (x & 0xFF_FFFF);
    x = ((x & 0x00FF_F000_00FF_F000) << 4) | (x & 0x0000_0FFF_0000_0FFF);
    x = ((x & 0x0FC0_0FC0_0FC0_0FC0) << 2) | (x & 0x003F_003F_003F_003F);
    let tz = x.trailing_zeros() / 8;
    let y = x >> (8 * tz);
    let lm = u64::MAX >> (8 * tz);
    let hi = h8(0x80);
    let ge = |c: u8| y.wrapping_add(h8(0x80 - c)) & hi;
    let low = ge(38);
    let ok = (ge(1) & !ge(11)) | low;
    if (ok & lm) != (hi & lm) || (y >> (56 - 8 * tz)) == 1 {
        return None;
    }
    let v = y - (h8(1) & lm) - ((low >> 7) & lm) * 27;
    let t = ((v >> 8) & 0x00FF_00FF_00FF_00FF) * 36 + (v & 0x00FF_00FF_00FF_00FF);
    let t = ((t >> 16) & 0x0000_FFFF_0000_FFFF) * 1296 + (t & 0x0000_FFFF_0000_FFFF);
    let val = (t >> 32) * 1_679_616 + (t & 0xFFFF_FFFF);
    if val > GM {
        None
    } else {
        Some(val)
    }
}

/// `gen_val(pack6(id), CODE[pre])` straight from the ID's bytes, without packing: the ID is
/// `pre` then a base-36 body of 1-8 chars [0-9a-z] with no leading '0', worth at most GM.
/// SAFETY: the 9 bytes from `p` are readable (bytes past `n` are ignored).
#[inline(always)]
unsafe fn gen_val_txt(p: *const u8, n: usize, pre: u8) -> Option<u64> {
    let l = n.wrapping_sub(2);
    if l >= 8 || *p != pre {
        return None;
    }
    // the body's l + 1 bytes right-aligned, its first char in the top byte
    let sh = 8 * (7 - l) as u32;
    let y = (p.add(1) as *const u64).read_unaligned().swap_bytes() >> sh;
    let lm = u64::MAX >> sh;
    let hi = h8(0x80);
    if (y & hi) != 0 {
        return None;
    }
    let ge = |c: u8| y.wrapping_add(h8(0x80 - c)) & hi;
    let low = ge(b'a');
    let ok = (ge(b'0') & !ge(b'9' + 1)) | (low & !ge(b'z' + 1));
    if (ok & lm) != (hi & lm) || (y >> (8 * l)) == b'0' as u64 {
        return None;
    }
    let v = y - (h8(b'0') & lm) - ((low >> 7) & lm) * (b'a' - 10 - b'0') as u64;
    let t = ((v >> 8) & 0x00FF_00FF_00FF_00FF) * 36 + (v & 0x00FF_00FF_00FF_00FF);
    let t = ((t >> 16) & 0x0000_FFFF_0000_FFFF) * 1296 + (t & 0x0000_FFFF_0000_FFFF);
    let val = (t >> 32) * 1_679_616 + (t & 0xFFFF_FFFF);
    if val > GM {
        None
    } else {
        Some(val)
    }
}

/// The scramble without its key: (index * GC1) ^ key.
#[inline(always)]
fn gen_unmix(val: u64) -> u64 {
    let x = val.wrapping_mul(GC2I) & GM;
    x ^ (x >> 17) ^ (x >> 34)
}

#[inline(always)]
fn gen_idx(val: u64, key: u64) -> u64 {
    ((gen_unmix(val) ^ key).wrapping_mul(GC1I)) & GM
}

#[derive(Clone, Copy, Debug)]
struct IdxCfg {
    key: u64,
    lim: u64,
    sh: u32,
    ndb: usize,
    /// lines per byte in the sample
    lpb: f64,
}

static IDXC: std::sync::OnceLock<Option<IdxCfg>> = std::sync::OnceLock::new();

fn idx_cfg() -> Option<IdxCfg> {
    IDXC.get().copied().flatten()
}

/// True when at least 1/32 of the first (up to 512) lines of `c` have an event_id or user_id made of
/// a prefix letter and 32 hex digits (quoted or not). The index scramble never gives such IDs, and
/// data that has them has no key to find, so the brute-force search is skipped (the key decides
/// only the speed, never the output).
fn hex_ids_seen(c: &[u8]) -> bool {
    let is_hex = |f: &[u8]| {
        let f = f.strip_prefix(b"\"").and_then(|f| f.strip_suffix(b"\"")).unwrap_or(f);
        f.len() == 33 && f[1..].iter().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
    };
    let (mut pos, mut lines, mut hex) = (0usize, 0usize, 0usize);
    while lines < 512 {
        let Some(e) = memchr(b'\n', &c[pos..]) else { break };
        let line = &c[pos..pos + e];
        lines += 1;
        pos += e + 1;
        let mut it = line.splitn(3, |&b| b == b',');
        if it.next().is_some_and(|f| is_hex(f)) || it.next().is_some_and(|f| is_hex(f)) {
            hex += 1;
        }
    }
    hex >= 4 && hex * 32 >= lines
}

/// Recovers the event_id key from the first canonical IDs of the first chunk: an index guess for
/// one sample fixes the key, which must put most other samples below the size bound too. None
/// (every record keeps the hash path) when SHOP_NOIDX is set or no key fits.
fn idx_find(ev_files: &[B], ev_chunks: &[B]) -> Option<IdxCfg> {
    if std::env::var_os("SHOP_NOIDX").is_some() {
        return None;
    }
    let c = *ev_chunks.first()?;
    if hex_ids_seen(c) {
        return None;
    }
    let total: usize = ev_files.iter().map(|f| f.len()).sum();
    let mut xs: Vec<u64> = vec![];
    let (mut pos, mut lines) = (0usize, 0usize);
    while xs.len() < 64 && lines < 4096 {
        let Some(e) = memchr(b'\n', &c[pos..]) else { break };
        let line = &c[pos..pos + e];
        lines += 1;
        pos += e + 1;
        let f = &line[..memchr(b',', line).unwrap_or(line.len())];
        if let Some(v) = pack6(f).and_then(|k| gen_val(k, 42)) {
            xs.push(gen_unmix(v));
        }
    }
    if xs.len() < 2 {
        return None;
    }
    let lim = (total as f64 * lines as f64 / pos as f64 * 1.3) as u64 + 64;
    if lim > 1 << 31 {
        return None;
    }
    let need = (xs.len() * 3).div_ceil(4);
    let fits = |k: u64| xs.iter().filter(|&&x| ((x ^ k).wrapping_mul(GC1I) & GM) < lim).count() >= need;
    // One pass tries the pairs p = 0..np at once; the key is the one of the smallest p that has a
    // fit, at its smallest i (as separate passes p = 0, 1, 2 would give). Runs of SUB indices are
    // first screened by a branch-free loop that vectorizes; most runs have no candidate at all.
    let np = 3.min(xs.len() - 1);
    let ds: [u64; 3] = std::array::from_fn(|p| xs[p.min(np - 1)] ^ xs[p.min(np - 1) + 1]);
    let hit = |d: u64, t: u64| ((d ^ t).wrapping_mul(GC1I) & GM) < lim;
    let later = [AtomicU64::new(u64::MAX), AtomicU64::new(u64::MAX)];
    const SUB: u64 = 512;
    let blk = 1u64 << 16;
    let k0 = (0..lim.div_ceil(blk)).into_par_iter().find_map_first(|b| {
        let (mut s, hi) = (b * blk, ((b + 1) * blk).min(lim));
        while s < hi {
            let e = (s + SUB).min(hi);
            let mut any = false;
            for i in s..e {
                let t = i.wrapping_mul(GC1) & GM;
                any |= hit(ds[0], t) | hit(ds[1], t) | hit(ds[2], t);
            }
            if any {
                for i in s..e {
                    let t = i.wrapping_mul(GC1) & GM;
                    if hit(ds[0], t) && fits(xs[0] ^ t) {
                        return Some(xs[0] ^ t);
                    }
                    for p in 1..np {
                        if hit(ds[p], t) && fits(xs[p] ^ t) {
                            later[p - 1].fetch_min(i, Relaxed);
                        }
                    }
                }
            }
            s = e;
        }
        None
    });
    let key = k0.or_else(|| {
        (1..np).find_map(|p| {
            let i = later[p - 1].load(Relaxed);
            (i != u64::MAX).then(|| xs[p] ^ (i.wrapping_mul(GC1) & GM))
        })
    })?;
    let sh = (64 - (lim / 2048).max(1).leading_zeros()).clamp(10, 15);
    Some(IdxCfg { key, lim, sh, ndb: (lim >> sh) as usize + 1, lpb: lines as f64 / pos as f64 })
}

/// Recovers the user_id key from the first lines of the first chunk whose event_id and user_id are
/// both canonical: the sample with the smallest event index has a small generator person (event
/// indices are given person by person), so a guess of its person and alias fixes the key, which
/// must then give most samples a small alias. None when SHOP_NOGP is set or no key fits; the key
/// only decides the speed (GpTab is exact for any key).
fn ukey_find(ev_chunks: &[B], ekey: u64) -> Option<u64> {
    if std::env::var_os("SHOP_NOGP").is_some() {
        return None;
    }
    let c = *ev_chunks.first()?;
    let mut xs: Vec<(u64, u64)> = vec![];
    let (mut pos, mut lines) = (0usize, 0usize);
    while xs.len() < 256 && lines < 4096 {
        let Some(e) = memchr(b'\n', &c[pos..]) else { break };
        let line = &c[pos..pos + e];
        lines += 1;
        pos += e + 1;
        let mut it = line.splitn(3, |&b| b == b',');
        let (Some(f0), Some(f1)) = (it.next(), it.next()) else { continue };
        if let (Some(ve), Some(vu)) = (pack6(f0).and_then(|k| gen_val(k, 42)), pack6(f1).and_then(|k| gen_val(k, UPRE))) {
            xs.push((gen_idx(ve, ekey), gen_unmix(vu)));
        }
    }
    if xs.len() < 8 {
        return None;
    }
    xs.sort_unstable();
    // a plausible index: alias below 64 and person below 2^28 (1 in 64 for a wrong key)
    let ok = |x: u64, k: u64| {
        let i = (x ^ k).wrapping_mul(GC1I) & GM;
        (i & 511) < 64 && (i >> 9) < (1 << 28)
    };
    let need = (xs.len() * 3).div_ceil(4);
    let fits = |k: u64| xs.iter().filter(|&&(_, x)| ok(x, k)).count() >= need;
    let (x1, x2) = (xs[xs.len() / 2].1, xs[xs.len() - 1].1);
    (0..3).find_map(|p| {
        let (e0, x0) = xs[p];
        // persons hold about 30 events each, so the first sampled event's person is far below e0 / 4
        let top = (e0 / 4 + 4096).min(1 << 28);
        let blk = 1u64 << 12;
        (0..top.div_ceil(blk)).into_par_iter().find_map_first(|b| {
            (b * blk..((b + 1) * blk).min(top)).find_map(|gp| {
                (0..GPA).find_map(|a| {
                    let k = x0 ^ ((gp * 512 + a).wrapping_mul(GC1) & GM);
                    if ok(x1, k) && ok(x2, k) && fits(k) {
                        Some(k)
                    } else {
                        None
                    }
                })
            })
        })
    })
}

/// Parses one chunk: dense records go straight to their buckets, the rest are grouped by event_id
/// hash bucket.
fn ev_chunk_parse<E: Ev>(c: B, cx: &EvCtx) -> EvChunk<E> {
    let mut dz = Dz::new(idx_cfg(), c.len());
    let cap = if dz.on { c.len() / 4096 } else { c.len() / 64 };
    let mut m: Vec<E> = Vec::with_capacity(cap);
    let mut d: Vec<DRec> = Vec::with_capacity(cap);
    let (mut fb, mut fbn, mut side) = (Fb::default(), vec![], vec![]);
    let mut tm = now();
    static OLD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OLD.get_or_init(|| std::env::var_os("SHOP_EVOLD").is_some()) {
        parse_events(c, cx, &mut m, &mut d, &mut fb, &mut fbn, &mut side);
    } else {
        if dz.on {
            parse_events_sol::<E, true>(c, cx, &mut m, &mut d, &mut fb, &mut fbn, &mut side, &mut dz);
        } else {
            parse_events_sol::<E, false>(c, cx, &mut m, &mut d, &mut fb, &mut fbn, &mut side, &mut dz);
        }
    }
    pt(0, &mut tm);
    let (d, doff) = bucketize(&d, EB, |r| (mix(r.eid) >> (64 - EBB)) as usize);
    let (dx, dxoff) = bucketize(&fb.x, EB, |r| (dx_hash(r) >> (64 - EBB)) as usize);
    fb.x = Vec::new();
    dz.finish();
    pt(1, &mut tm);
    EvChunk { m, d, doff, dz, fb, dx, dxoff, fbn, side }
}

fn ev_parse<E: Ev>(ev_chunks: &[B], cx: &EvCtx) -> Vec<EvChunk<E>> {
    ev_chunks.par_iter().map(|&c| ev_chunk_parse(c, cx)).collect()
}

/// measurement aid: order-sensitive checksum of one parsed chunk, chained on h
fn hash_chunk<E: Ev>(mut h: u64, r: &EvChunk<E>) -> u64 {
    let mut add = |x: u64| h = mix(h ^ x).wrapping_add(x);
    add(r.m.len() as u64);
    for e in &r.m {
        add(e.key() ^ (e.rts() << 1));
        if let Some((p, cat)) = e.pc() {
            add(p ^ ((cat as u64) << 32));
        }
    }
    for x in &r.d {
        add(x.eid ^ ((x.ing as u64) << 1) ^ ((x.loc as u64) << 40));
    }
    for &x in &r.doff {
        add(x as u64);
    }
    for x in &r.fb.v {
        add(x.0 ^ (x.1 as u64) ^ ((x.2 as u64) << 40));
    }
    for x in &r.dx {
        add(x.hi ^ x.lo.rotate_left(17) ^ ((x.ing as u64) << 1) ^ ((x.lp as u64) << 32));
    }
    for &x in &r.dxoff {
        add(x as u64);
    }
    for x in &r.fbn {
        add(x.0 ^ x.1);
    }
    for x in &r.side {
        add(x.0 as u64 ^ x.1 as u64);
    }
    for b in 0..r.dz.dn.len() {
        let (sa, sp, oa, op) = r.dz.seg(b);
        for (part, p0) in [(sa, sp), (oa, op)] {
            add(part.len() as u64 ^ ((p0 as u64) << 32));
            for x in part {
                add(x.i as u64 ^ ((x.ing as u64) << 32));
                add(x.e.key() ^ (x.e.rts() << 1));
                if let Some((p, cat)) = x.e.pc() {
                    add(p ^ ((cat as u64) << 32));
                }
            }
        }
    }
    for x in &r.dz.dside {
        add(x.0 as u64 ^ x.1 as u64);
    }
    h
}

/// measurement aid: cycles this thread has run (Windows; 0 elsewhere), unaffected by preemption
fn thread_cycles() -> u64 {
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentThread() -> isize;
            fn QueryThreadCycleTime(h: isize, c: *mut u64) -> i32;
        }
        let mut c = 0u64;
        // SAFETY: plain Win32 calls on the current thread's pseudo handle
        unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut c) };
        c
    }
    #[cfg(not(windows))]
    0
}

/// measurement aid: event parsing alone, serial (SHOP_PARSE=reps); prints the best time and a
/// checksum of everything the parse produces
fn parse_aid<E: Ev>(ev_chunks: &[B], cx: &EvCtx) {
    let reps = std::env::var("SHOP_PARSE").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(3).max(1);
    if std::env::var_os("SHOP_PARSE_SCAN").is_some() {
        // the line scanner alone
        let mut bc = u64::MAX;
        for rep in 0..reps {
            let t = std::time::Instant::now();
            let c0 = thread_cycles();
            let mut n = 0u64;
            for &c in ev_chunks {
                scan7(c, |f| n += (f[0].len() + f[3].len() + f[6].len()) as u64);
            }
            bc = bc.min(thread_cycles() - c0);
            eprintln!("scan rep{} {:.3}s ({})", rep, t.elapsed().as_secs_f64(), n);
        }
        eprintln!("parse best 0s sum 0 gcyc {:.4}", bc as f64 / 1e9);
        return;
    }
    let mut best = f64::MAX;
    let mut sum = 0u64;
    let mut bc = u64::MAX;
    for rep in 0..reps {
        let t = std::time::Instant::now();
        let c0 = thread_cycles();
        let all: Vec<EvChunk<E>> = ev_chunks.iter().map(|&c| ev_chunk_parse(c, cx)).collect();
        bc = bc.min(thread_cycles() - c0);
        let el = t.elapsed().as_secs_f64();
        let h = all.iter().fold(0u64, |h, r| hash_chunk(h, r));
        best = best.min(el);
        sum = h;
        eprintln!("parse rep{} {:.3}s", rep, el);
    }
    eprintln!("parse best {:.3}s sum {:016x} gcyc {:.4}", best, sum, bc as f64 / 1e9);
}

fn fbn_merge<E: Ev>(v: &[EvChunk<E>], names_map: &mut AHashMap<u64, u64>) {
    for c in v {
        for &(k, r) in &c.fbn {
            names_map.entry(k).or_insert(r);
        }
    }
}

enum Evs {
    L(Vec<EvChunk<EvL>>),
    W(Vec<EvChunk<EvW>>),
}

fn main() {
    hpa::init();
    let ep = || std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    let ep0 = ep();
    let data = std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into());
    let outd = std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into());
    let ncpu: usize = std::env::var("NCPU").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(0);
    let mut tb = rayon::ThreadPoolBuilder::new().stack_size(8 << 20);
    if ncpu > 0 {
        tb = tb.num_threads(ncpu);
    }
    tb.build_global().unwrap();
    let data = Path::new(&data);
    let outd = Path::new(&outd);
    let timing = std::env::var("SHOP_TIMING").is_ok();
    TIMING.store(timing, Relaxed);
    let t0 = std::time::Instant::now();
    let tick = |s: &str| {
        if timing {
            eprintln!("[{:.2}s] {}", t0.elapsed().as_secs_f64(), s);
        }
        hpa::diag(s);
    };

    let mf = std::fs::read_to_string(data.join("manifest.json")).unwrap_or_default();
    let ws = manifest_ts(&mf, "window_start").unwrap_or(0);
    let we = manifest_ts(&mf, "window_end").unwrap_or(0);

    // the event files, chunks and id keys need nothing else: find them on their own thread while
    // products load (joined before the pool runs the two branches below; campaigns and links load
    // in the links branch while the other threads parse events)
    let ev_dir = data.join("events");
    let ev_setup = std::thread::spawn(move || {
        let ev_files = csv_files(&ev_dir);
        let ev_chunks = all_chunks(&ev_files, ev_chunk(&ev_files));
        let _ = IDXC.set(idx_find(&ev_files, &ev_chunks));
        let _ = UKEYC.set(idx_cfg().and_then(|c| ukey_find(&ev_chunks, c.key)));
        ev_chunks
    });

    // products
    let mut prod_raw: Vec<(B, B)> = vec![];
    if let Some(f) = leak_file(&data.join("products.csv")) {
        for c in chunks(f, CHUNK) {
            scan::<2>(c, |&[p, c]| prod_raw.push((p, c)));
        }
    }
    let mut cat_names: Vec<B> = prod_raw.iter().map(|x| x.1).collect();
    cat_names.push(b"unknown");
    cat_names.sort();
    cat_names.dedup();
    let unknown_cat = cat_names.binary_search(&(b"unknown" as B)).unwrap() as u32;
    // 16-byte event records need a window of at most 2^32 ms and at most 256 categories
    let lean = std::env::var_os("SHOP_WIDE").is_none() && cat_names.len() <= 256 && we.saturating_sub(ws) <= 1i64 << 32;
    // first occurrence wins in a map insert loop? original used insert (last wins); keep last wins
    let mut prod_items: Vec<(B, u32)> = vec![];
    {
        let mut seen: AHashMap<B, usize> = AHashMap::new();
        for &(p, c) in &prod_raw {
            let cat = cat_names.binary_search(&c).unwrap() as u32;
            match seen.get(p) {
                Some(&i) => prod_items[i].1 = cat,
                None => {
                    seen.insert(p, prod_items.len());
                    prod_items.push((p, cat));
                }
            }
        }
    }
    let prod = IdLookup::new(b"p", &prod_items);

    tick("prod loaded");
    let rs = hasher();
    let ev_chunks = ev_setup.join().unwrap();
    match idx_cfg() {
        Some(c) => tick(&format!("idx key found=1 lim={} sh={} ndb={}", c.lim, c.sh, c.ndb)),
        None => tick("idx key found=0"),
    }
    tick(&format!("user key found={}", UKEYC.get().copied().flatten().is_some() as u8));
    // campaigns, links load and the serial union-find overlap the parallel event parse; events do not
    // need the id table
    let ((mut tab, root, bk, nl, nroot, mut names_map, camp, ncamp, camp_names, channels), ev_out) = rayon::join(
        || {
            // campaigns
            let mut camp_items: Vec<(B, u32)> = vec![];
            let mut camp_names: Vec<B> = vec![];
            let mut channels: Vec<B> = vec![];
            if let Some(f) = leak_file(&data.join("campaigns.csv")) {
                let mut seen: AHashMap<B, u32> = AHashMap::new();
                for c in chunks(f, CHUNK) {
                    scan::<2>(c, |&[id, ch]| {
                        if !seen.contains_key(id) {
                            seen.insert(id, camp_names.len() as u32);
                            camp_items.push((id, camp_names.len() as u32));
                            camp_names.push(id);
                            channels.push(ch);
                        }
                    });
                }
            }
            let camp = IdLookup::new(b"c_", &camp_items);
            let ncamp = camp_names.len();
            camp_names.push(b"organic");
            channels.push(b"organic");
            tick("camp loaded");
            // links load and parse stay on this thread, with no rayon call: if a links job were
            // stolen, this thread would steal event chunks while it waits and could return only after
            // the event parse, which leaves the serial insert/union/best below without overlap
            let link_chunks = all_chunks(&csv_files_in(&data.join("links"), false), CHUNK);
            let link_pairs: Vec<Vec<[u64; 4]>> = link_chunks
                .iter()
                .map(|&c| {
                    let mut v = vec![];
                    scan::<2>(c, |&[a, b]| {
                        if !a.is_empty() && !b.is_empty() {
                            v.push([ukey(&rs, a), pack(a), ukey(&rs, b), pack(b)]);
                        }
                    });
                    v
                })
                .collect();
            tick("links parsed");
            let occ: usize = link_pairs.iter().map(|v| v.len()).sum::<usize>() * 2;
            let mut names_map: AHashMap<u64, u64> = AHashMap::new();
            let (mut t, root, bk, nl, nroot) = {
                // unique-id estimate by linear counting over the parsed keys (bitmap of 1-4 MB)
                let est = {
                    let m = (occ * 2).next_power_of_two().clamp(1 << 16, 1 << 25);
                    let mut bits = vec![0u64; m / 64];
                    let sh = 64 - m.trailing_zeros();
                    for v in &link_pairs {
                        for e in v {
                            for k in [e[0], e[2]] {
                                let h = (mix(k) >> sh) as usize;
                                bits[h >> 6] |= 1u64 << (h & 63);
                            }
                        }
                    }
                    let set: usize = bits.iter().map(|w| w.count_ones() as usize).sum();
                    let z = (m - set).max(1) as f64;
                    let e = -(m as f64) * (z / m as f64).ln();
                    (e as usize).min(occ)
                };
                // slots for the estimate plus a margin, load at most about 0.7
                let mut slots = ((est + est / 8) as f64 * 1.45) as usize;
                slots = slots.max(16).next_power_of_two();
                let mut t;
                let mut idx_key: Vec<u64>;
                let mut idx_ref: Vec<u64>;
                let mut parent: Vec<u32>;
                let mut edges: Vec<(u32, u32)>;
                'retry: loop {
                t = IdTab::with_slots(slots);
                idx_key = vec![];
                idx_ref = vec![];
                parent = vec![];
                let limit = slots / 8 * 7;
                edges = Vec::with_capacity(occ / 2);
                for v in &link_pairs {
                    for (n, e) in v.iter().enumerate() {
                        if n + PFD / 2 < v.len() {
                            t.pf(v[n + PFD / 2][0]);
                            t.pf(v[n + PFD / 2][2]);
                        }
                        let mut g = |k: u64, r: u64| -> u32 {
                            let (id, new) = t.get_or_insert(k, idx_key.len() as u64);
                            if new {
                                idx_key.push(k);
                                idx_ref.push(r);
                                parent.push(id as u32);
                            }
                            id as u32
                        };
                        let a = g(e[0], e[1]);
                        let b = g(e[2], e[3]);
                        edges.push((a, b));
                        if idx_key.len() > limit {
                            slots *= 2;
                            continue 'retry;
                        }
                    }
                }
                break;
                }
                fn find(p: &mut [u32], mut x: u32) -> u32 {
                    while p[x as usize] != x {
                        let g = p[p[x as usize] as usize];
                        p[x as usize] = g;
                        x = g;
                    }
                    x
                }
                tick(&format!("links inserted {} est {} slots {}", idx_key.len(), est, slots));
                // union by index: the larger root goes under the smaller one, so parent[g] <= g always
                // holds (path halving keeps it too)
                for (a, b) in edges {
                    let ra = find(&mut parent, a);
                    let rb = find(&mut parent, b);
                    if ra < rb {
                        parent[rb as usize] = ra;
                    } else if rb < ra {
                        parent[ra as usize] = rb;
                    }
                }
                tick("links union");
                let nl = idx_key.len();
                // one forward pass sets every entry to its root: parent[g] <= g and lower entries are final
                let mut root = parent;
                for g in 0..nl {
                    let p = root[g] as usize;
                    root[g] = root[p];
                }
                // smallest id of each component as [key, index] at its root; the root is the component's
                // lowest index, so it comes before every other member
                let mut bk: Vec<[u64; 2]> = vec![[0u64; 2]; nl];
                for g in 0..nl {
                    if g + RPD < nl {
                        prefetch(bk.as_ptr().wrapping_add(root[g + RPD] as usize));
                    }
                    let r = root[g] as usize;
                    let kg = idx_key[g];
                    if r == g {
                        bk[g] = [kg, g as u64];
                    } else {
                        let [kb, b] = bk[r];
                        let less = if kg & HBIT == 0 && kb & HBIT == 0 { kg < kb } else { unpack(idx_ref[g]) < unpack(idx_ref[b as usize]) };
                        if less {
                            bk[r] = [kg, g as u64];
                        }
                    }
                }
                tick("links best");
                let mut nroot = 0usize;
                for r in 0..nl {
                    if root[r] as usize == r {
                        nroot += 1;
                        let [pk, b] = bk[r];
                        if pk & HBIT != 0 {
                            names_map.insert(pk, idx_ref[b as usize]);
                        }
                    }
                }
                (t, root, bk, nl, nroot)
            };
            drop(link_pairs);
            tick("links done");
            (t, root, bk, nl, nroot, names_map, camp, ncamp, camp_names, channels)
        },
        || {
            let cx = EvCtx { prod: &prod, unknown_cat, ws, we, ibase: ws.div_euclid(1000), rs: &rs };
            if std::env::var_os("SHOP_PARSE").is_some() {
                if lean {
                    parse_aid::<EvL>(&ev_chunks, &cx);
                } else {
                    parse_aid::<EvW>(&ev_chunks, &cx);
                }
                std::process::exit(0);
            }
            if std::env::var_os("SHOP_SOLBENCH").is_some() {
                if lean {
                    sol_bench::<EvL>(&ev_chunks, &cx);
                } else {
                    sol_bench::<EvW>(&ev_chunks, &cx);
                }
                std::process::exit(0);
            }
            if std::env::var_os("SHOP_PARTS").is_some() {
                // measurement aid: each per-field step alone over the real fields
                let mut fs: [Vec<B>; 7] = Default::default();
                for &c in &ev_chunks {
                    scan::<7>(c, |f| {
                        for k in 0..7 {
                            fs[k].push(f[k]);
                        }
                    });
                }
                let tm = |name: &str, f: &mut dyn FnMut() -> u64| {
                    let mut best = f64::MAX;
                    let mut r = 0;
                    for _ in 0..3 {
                        let t = std::time::Instant::now();
                        r = f();
                        best = best.min(t.elapsed().as_secs_f64());
                    }
                    eprintln!("part {name} {:.3}s ({})", best, r & 1);
                };
                tm("ts", &mut || {
                    let mut c = TsCache::new();
                    fs[2].iter().map(|b| parse_ts_c(b, &mut c).unwrap_or(0) as u64).fold(0, u64::wrapping_add)
                });
                tm("ingest", &mut || {
                    let mut c = TsCache::new();
                    fs[3].iter().map(|b| parse_ts_c(b, &mut c).unwrap_or(0) as u64).fold(0, u64::wrapping_add)
                });
                tm("pack-eid", &mut || fs[0].iter().map(|b| pack6(b).unwrap_or(0)).fold(0, u64::wrapping_add));
                tm("pack-uid", &mut || fs[1].iter().map(|b| pack6(b).unwrap_or(0)).fold(0, u64::wrapping_add));
                tm("prod", &mut || fs[5].iter().map(|b| prod.get(b).unwrap_or(0) as u64).fold(0, u64::wrapping_add));
                tm("et", &mut || {
                    fs[4].iter()
                        .map(|b| match *b {
                            b"view" | b"cart" => 1u64,
                            b"purchase" => 2,
                            _ => 0,
                        })
                        .fold(0, u64::wrapping_add)
                });
                tm("price", &mut || fs[6].iter().map(|b| parse_price(b).unwrap_or(0) as u64).fold(0, u64::wrapping_add));
            }
            if std::env::var("SHOP_PROBE").is_ok() {
                let t = std::time::Instant::now();
                let mut acc = 0u64;
                for c in &ev_chunks {
                    let mut i = 0;
                    while i < c.len() {
                        acc = acc.wrapping_add(c[i] as u64);
                        i += 4096;
                    }
                }
                eprintln!("probe touch {:.3}s ({})", t.elapsed().as_secs_f64(), acc & 1);
                let t = std::time::Instant::now();
                let mut n = 0u64;
                for &c in &ev_chunks {
                    scan::<7>(c, |f| n += f[0].len() as u64 + f[6].len() as u64);
                }
                eprintln!("probe scan-only {:.3}s ({})", t.elapsed().as_secs_f64(), n);
                let t = std::time::Instant::now();
                let mut n = 0u64;
                for &c in &ev_chunks {
                    let (mut tci, mut tct) = (TsCache::new(), TsCache::new());
                    scan::<7>(c, |f| {
                        n += parse_ts_c(f[2], &mut tct).unwrap_or(0) as u64 ^ parse_ts_c(f[3], &mut tci).unwrap_or(0) as u64;
                        n += pack6(f[0]).unwrap_or(0) ^ pack6(f[1]).unwrap_or(0);
                    });
                }
                eprintln!("probe scan+ts+pack {:.3}s ({})", t.elapsed().as_secs_f64(), n & 1);
                let t = std::time::Instant::now();
                let mut n = 0u64;
                for &c in &ev_chunks {
                    let mut p = 0;
                    while let Some(i) = memchr(10u8, &c[p..]) {
                        n += 1;
                        p += i + 1;
                    }
                }
                eprintln!("probe memchr-lines {:.3}s ({})", t.elapsed().as_secs_f64(), n);
            }
            let ev_out = if lean { Evs::L(ev_parse(&ev_chunks, &cx)) } else { Evs::W(ev_parse(&ev_chunks, &cx)) };
            tick("events parsed");
            ev_out
        },
    );

    // the insert table becomes the id table in place (a slot's index turns into its person key), with the whole pool
    {
        let (root, bk) = (&root[..], &bk[..]);
        tab.s.par_chunks_mut(1 << 14).for_each(|c| {
            let n = c.len();
            for i in 0..n {
                if i + 2 * PFD < n {
                    prefetch(root.as_ptr().wrapping_add(c[i + 2 * PFD][1] as usize));
                }
                if i + PFD < n {
                    let e = c[i + PFD];
                    if e[0] != 0 {
                        prefetch(bk.as_ptr().wrapping_add(root[e[1] as usize] as usize));
                    }
                }
                let e = &mut c[i];
                if e[0] != 0 {
                    e[1] = bk[root[e[1] as usize] as usize][0];
                }
            }
        });
    }
    tab.mov = nl - nroot;
    drop((root, bk));
    tick(&format!("links table movers {}", tab.mov));

    // clicks
    let click_chunks = all_chunks(&csv_files(&data.join("clicks")), CHUNK);
    let mut tm = now();
    let mf = MoverF::build(&tab);
    pt(4, &mut tm);
    // the names of hashed user keys (one map insert per such line, serial) merge on one thread
    // while the others parse the clicks; the map is read only by the writers
    let (click_out, ()) = rayon::join(
        || -> Vec<(Vec<ClickRec>, Vec<u32>)> {
            click_chunks
            .par_iter()
            .map(|&c| {
                let mut tmp: Vec<ClickRec> = Vec::with_capacity(c.len() / 40);
                let mut tm = now();
                parse_clicks(c, &camp, &rs, ws, we, &mut tmp);
                pt(9, &mut tm);
                // keys that miss the mover filter map to themselves without a table lookup. The
                // filter pass lists the hits without branching on them, the table pass resolves only
                // those, and both count the person buckets of the final keys on the way.
                let n = tmp.len();
                let mut hit = vec![0u32; n + 1];
                let mut cnt = vec![0u32; PB + 1];
                let mut m = 0usize;
                for i in 0..n {
                    if i + RPD < n {
                        mf.pf(tmp[i + RPD].key);
                    }
                    let k = tmp[i].key;
                    let h = mf.hit(k);
                    hit[m] = i as u32;
                    m += h as usize;
                    cnt[pbk(k) + 1] += (!h) as u32;
                }
                for j in 0..m {
                    if j + RPD < m {
                        tab.pf(tmp[hit[j + RPD] as usize].key);
                    }
                    let i = hit[j] as usize;
                    let k = tab.get(tmp[i].key);
                    tmp[i].key = k;
                    cnt[pbk(k) + 1] += 1;
                }
                drop(hit);
                let r = scatter_counted(&tmp, cnt, |x| pbk(x.key));
                pt(10, &mut tm);
                r
            })
            .collect()
        },
        || match &ev_out {
            Evs::L(v) => fbn_merge(v, &mut names_map),
            Evs::W(v) => fbn_merge(v, &mut names_map),
        },
    );
    tick("clicks parsed");

    // person buckets
    let d0 = ws.div_euclid(DAY);
    let nd = if we > ws { ((we - 1).div_euclid(DAY) - d0 + 1) as usize } else { 0 };
    let results = match ev_out {
        Evs::L(v) => ev_after(v, &tab, &mf, &click_out, ws, we, ncamp, d0, nd, &tick),
        Evs::W(v) => ev_after(v, &tab, &mf, &click_out, ws, we, ncamp, d0, nd, &tick),
    };
    tick("sessionized");

    // outputs
    let names = Names { map: names_map };
    let ud_dir = outd.join("user_daily");
    let cd_dir = outd.join("campaign_daily");
    std::fs::create_dir_all(&ud_dir).unwrap();
    std::fs::create_dir_all(&cd_dir).unwrap();
    let mut cm = CampAcc::new(ncamp + 1, d0, nd);
    let mut ud_groups: Vec<&Vec<UD>> = vec![];
    for (u, m) in &results {
        ud_groups.extend(u.iter());
        cm.merge(m);
    }
    let rows = cm.rows();
    let nf = 16usize;
    let mut files: Vec<Vec<&Vec<UD>>> = (0..nf).map(|_| vec![]).collect();
    let mut load = vec![0usize; nf];
    ud_groups.sort_by_key(|g| std::cmp::Reverse(g.len()));
    for g in ud_groups {
        let k = (0..nf).min_by_key(|&k| load[k]).unwrap();
        load[k] += g.len();
        files[k].push(g);
    }
    let any_rows = files.iter().any(|g| g.iter().any(|x| !x.is_empty()));
    // one revenue width for every file of the table, so all part files share a schema
    let rev32 = std::env::var_os("SHOP_REV64").is_none()
        && files.par_iter().all(|g| g.iter().all(|x| x.iter().all(|r| r.rev <= i32::MAX as u64)));
    rayon::join(
        || {
            files.par_iter().enumerate().filter(|(i, g)| g.iter().any(|x| !x.is_empty()) || (*i == 0 && !any_rows)).for_each(|(i, g)| {
                write_user_daily(&ud_dir.join(format!("part-{i}.parquet")), g, &cat_names, &names, rev32);
            })
        },
        || write_campaign_daily(&cd_dir.join("part-0.parquet"), &rows, &camp_names, &channels),
    );
    tick("written");
    if timing {
        let v: Vec<String> = (0..PROF.len()).map(|i| format!("{}={:.2}", PROF_NAMES[i], PROF[i].load(Relaxed) as f64 / 1e9)).collect();
        eprintln!("prof(cpu s): {}", v.join(" "));
    }
    if timing {
        eprintln!("epoch main={:.3} end={:.3}", ep0, ep());
    }
    if std::env::var("SHOP_DROPIN").map_or(true, |v| v != "off") {
        let t = ep();
        drop_inputs();
        if timing {
            eprintln!("dropin end {:.3}s", ep() - t);
        }
    }
    // all output files are written and closed; skip freeing the large tables
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
        fn chance(&mut self, pct: u64) -> bool {
            self.below(100) < pct
        }
    }

    /// One id class check on `b` placed at the start of a 64-byte buffer padded with `pad`.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
    fn idk_check(b: &[u8], pad: u8, other: &[u8]) {
        let r = idk_ref(b);
        let mut buf = [pad; 128];
        buf[..b.len()].copy_from_slice(b);
        let mut buf2 = [pad; 128];
        buf2[..other.len()].copy_from_slice(other);
        let norm = |x: (u64, u128, u8)| if x.2 == 2 { (0, 0, 2) } else { x };
        let x = unsafe { idk_x(buf.as_ptr(), b.len()) };
        assert_eq!(norm(x), norm(r), "idk_x {:?}", String::from_utf8_lossy(b));
        let (xa, xb) = unsafe { idk_x2(buf.as_ptr(), b.len(), buf2.as_ptr(), other.len()) };
        assert_eq!(norm(xa), norm(r));
        assert_eq!(norm(xb), norm(idk_ref(other)));
        let (xa, xb) = unsafe { idk_x2(buf2.as_ptr(), other.len(), buf.as_ptr(), b.len()) };
        assert_eq!(norm(xb), norm(r));
        assert_eq!(norm(xa), norm(idk_ref(other)));
        match r.2 {
            0 => assert_eq!(Some(r.0), pack6(b)),
            1 => {
                assert_eq!(hex_full(b), Some((r.0, r.1)));
                assert_eq!(r.0 >> 62, 3);
                assert_eq!((r.0 >> 56) & 63, CODE[b[0] as usize] as u64);
            }
            _ => assert_eq!(hex_full(b), None),
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
    fn idk_classes_match_ref() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let alpha: Vec<u8> = (0u8..=255).filter(|&c| CODE[c as usize] != 0).collect();
        let hexd = b"0123456789abcdef";
        let mut cases: Vec<Vec<u8>> = vec![
            b"u".to_vec(),
            b"u0123456789abcdef0123456789abcdef".to_vec(),
            b"uffffffffffffffffffffffffffffffff".to_vec(),
            b"u00000000000000000000000000000000".to_vec(),
            b"_ffffffffffffffffffffffffffffffff".to_vec(),
            b"zffffffffffffffffffffffffffffffff".to_vec(),
            b"u0123456789ABCDEF0123456789abcdef".to_vec(),
            b"-0123456789abcdef0123456789abcdef".to_vec(),
            b"u0123456789abcdef0123456789abcdeg".to_vec(),
            b"u0123456789abcdef0123456789abcde/".to_vec(),
            b"u0123456789abcdef0123456789abcde:".to_vec(),
            b"u0123456789abcdef0123456789abcde`".to_vec(),
            b"u0123456789abcdef0123456789abcde\xe1".to_vec(),
            b"u0123456789abcdef0123456789abcde".to_vec(),
            b"u0123456789abcdef0123456789abcdef0".to_vec(),
            b"0123456789abcdef0123456789abcdef0".to_vec(),
        ];
        for n in 1..=12 {
            for _ in 0..50 {
                cases.push((0..n).map(|_| alpha[rng.below(alpha.len() as u64) as usize]).collect());
            }
        }
        for _ in 0..3000 {
            let mut v: Vec<u8> = vec![alpha[rng.below(alpha.len() as u64) as usize]];
            v.extend((0..32).map(|_| hexd[rng.below(16) as usize]));
            if rng.chance(30) {
                let i = rng.below(33) as usize;
                v[i] = rng.below(256) as u8;
            }
            if rng.chance(5) {
                let n = 31 + rng.below(5) as usize;
                v.resize(n, b'a');
            }
            cases.push(v);
        }
        for c in &cases {
            for pad in [0u8, b'a', b',', 0xff] {
                let other = &cases[rng.below(cases.len() as u64) as usize];
                idk_check(c, pad, other);
            }
        }
        // key order equals byte order within Hex ids of one prefix length
        let mut hx: Vec<&Vec<u8>> = cases.iter().filter(|c| idk_ref(c).2 == 1).collect();
        assert!(hx.len() > 1500);
        hx.sort();
        for w in hx.windows(2) {
            let (a, b) = (idk_ref(w[0]), idk_ref(w[1]));
            assert!((a.0, a.1) <= (b.0, b.1));
        }
    }

    fn field(r: &mut Rng, out: &mut Vec<u8>, maxlen: u64) {
        const AL: &[u8] = b"0123456789abZT:-+.";
        let n = if r.chance(10) { 0 } else { 1 + r.below(maxlen) };
        let quoted = r.chance(25);
        if quoted {
            out.push(b'"');
        }
        for _ in 0..n {
            out.push(match r.below(100) {
                0..=1 => b'"',
                2 => b'\r',
                _ => AL[r.below(AL.len() as u64) as usize],
            });
        }
        if quoted {
            out.push(b'"');
        }
    }

    fn line(r: &mut Rng, out: &mut Vec<u8>) {
        let start = out.len();
        match r.below(10) {
            0..=3 => {
                // event-shaped: eid, uid, ts, ingest_ts, event_type, product, price
                let lens = [12u64, 9, 25, 20, 8, 10, 7];
                for (k, &ml) in lens.iter().enumerate() {
                    if k > 0 {
                        out.push(b',');
                    }
                    field(r, out, ml);
                }
            }
            4..=7 => {
                let nf = if r.chance(70) { 7 } else { 1 + r.below(10) };
                let ml = [3u64, 20, 60, 130][r.below(4) as usize];
                for k in 0..nf {
                    if k > 0 {
                        out.push(b',');
                    }
                    field(r, out, ml);
                }
            }
            _ => {
                const AL: &[u8] = b"a0,,\"\r";
                for _ in 0..r.below(160) {
                    out.push(AL[r.below(AL.len() as u64) as usize]);
                }
            }
        }
        for _ in 0..r.below(3) {
            if r.chance(30) && out.len() > start {
                let at = start + r.below((out.len() - start) as u64) as usize;
                match r.below(4) {
                    0 => out.insert(at, b'"'),
                    1 => out.insert(at, b','),
                    2 => out.insert(at, b'\r'),
                    _ => {
                        out.remove(at);
                    }
                }
            }
        }
        if r.chance(10) {
            out.push(b'\r');
        }
    }

    fn lines_of(chunk: B, f: impl Fn(B, &mut dyn FnMut(&[B; 7]))) -> Vec<[(usize, usize); 7]> {
        let mut v = vec![];
        let b0 = chunk.as_ptr() as usize;
        f(chunk, &mut |l: &[B; 7]| v.push(l.map(|x| (x.as_ptr() as usize - b0, x.len()))));
        v
    }

    /// scan7 (vector windows, mask quote rule) yields exactly the lines and fields of scan::<7>
    #[test]
    fn scan7_matches_scan() {
        assert!(cfg!(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), "build with -C target-cpu=x86-64-v3");
        let buf: &'static mut [u8] = Box::leak(vec![0u8; 1 << 16].into_boxed_slice());
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        let mut text = Vec::new();
        let (mut cases, mut total, mut fast_lines) = (0u64, 0u64, 0u64);
        for _ in 0..400_000 {
            text.clear();
            for _ in 0..r.below(40) {
                line(&mut r, &mut text);
                text.push(b'\n');
            }
            if r.chance(50) && !text.is_empty() {
                text.pop();
            }
            if text.len() + 64 > buf.len() {
                continue;
            }
            let off = r.below(64) as usize;
            buf[off..off + text.len()].copy_from_slice(&text);
            // SAFETY: buf is leaked (lives forever) and not written while `chunk` is in use
            let chunk: B = unsafe { std::slice::from_raw_parts(buf.as_ptr().add(off), text.len()) };
            let want = lines_of(chunk, |c, f| scan::<7>(c, f));
            let got = lines_of(chunk, |c, f| scan7(c, f));
            assert_eq!(got, want, "case {cases}: {:?}", String::from_utf8_lossy(chunk));
            cases += 1;
            total += want.len() as u64;
            fast_lines += SCAN7_FAST.with(|c| c.replace(0));
        }
        eprintln!("scan7 test: {cases} chunks, {total} accepted lines, {fast_lines} via the vector path");
        assert!(total > 1_000_000 && fast_lines > total / 4);
    }

    /// one click-shaped line (four fields) or a random one, with the mutations of `line`
    fn line4t(r: &mut Rng, out: &mut Vec<u8>) {
        let start = out.len();
        match r.below(10) {
            0..=5 => {
                let lens = [10u64, 10, 7, 27];
                for (k, &ml) in lens.iter().enumerate() {
                    if k > 0 {
                        out.push(b',');
                    }
                    field(r, out, ml);
                }
            }
            6..=8 => {
                let nf = if r.chance(60) { 4 } else { 1 + r.below(6) };
                let ml = [3u64, 20, 40, 90][r.below(4) as usize];
                for k in 0..nf {
                    if k > 0 {
                        out.push(b',');
                    }
                    field(r, out, ml);
                }
            }
            _ => {
                const AL: &[u8] = b"a0,,\"\r";
                for _ in 0..r.below(90) {
                    out.push(AL[r.below(AL.len() as u64) as usize]);
                }
            }
        }
        for _ in 0..r.below(3) {
            if r.chance(30) && out.len() > start {
                let at = start + r.below((out.len() - start) as u64) as usize;
                match r.below(4) {
                    0 => out.insert(at, b'"'),
                    1 => out.insert(at, b','),
                    2 => out.insert(at, b'\r'),
                    _ => {
                        out.remove(at);
                    }
                }
            }
        }
        if r.chance(10) {
            out.push(b'\r');
        }
    }

    fn lines_of4(chunk: B, f: impl Fn(B, &mut dyn FnMut(&[B; 4]))) -> Vec<[(usize, usize); 4]> {
        let mut v = vec![];
        let b0 = chunk.as_ptr() as usize;
        f(chunk, &mut |l: &[B; 4]| v.push(l.map(|x| (x.as_ptr() as usize - b0, x.len()))));
        v
    }

    /// scan4 (one vector window, mask quote rule) yields exactly the lines and fields of scan::<4>
    #[test]
    fn scan4_matches_scan() {
        assert!(cfg!(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), "build with -C target-cpu=x86-64-v3");
        let buf: &'static mut [u8] = Box::leak(vec![0u8; 1 << 16].into_boxed_slice());
        let mut r = Rng(0x2545_f491_4f6c_dd1d);
        let mut text = Vec::new();
        let (mut cases, mut total, mut fast_lines) = (0u64, 0u64, 0u64);
        for _ in 0..400_000 {
            text.clear();
            for _ in 0..r.below(40) {
                line4t(&mut r, &mut text);
                text.push(b'\n');
            }
            if r.chance(50) && !text.is_empty() {
                text.pop();
            }
            if text.len() + 64 > buf.len() {
                continue;
            }
            let off = r.below(64) as usize;
            buf[off..off + text.len()].copy_from_slice(&text);
            // SAFETY: buf is leaked (lives forever) and not written while `chunk` is in use
            let chunk: B = unsafe { std::slice::from_raw_parts(buf.as_ptr().add(off), text.len()) };
            let want = lines_of4(chunk, |c, f| scan::<4>(c, f));
            let got = lines_of4(chunk, |c, f| scan4(c, f));
            assert_eq!(got, want, "case {cases}: {:?}", String::from_utf8_lossy(chunk));
            cases += 1;
            total += want.len() as u64;
            fast_lines += SCAN7_FAST.with(|c| c.replace(0));
        }
        eprintln!("scan4 test: {cases} chunks, {total} accepted lines, {fast_lines} via the vector path");
        assert!(total > 1_000_000 && fast_lines > total / 4);
    }

    /// one click line: ids, campaign and time in the shapes `parse_clicks` special-cases, with
    /// quotes, empty and long fields, bad times, CR, extra commas and random edits
    fn click_text(r: &mut Rng, out: &mut Vec<u8>) {
        use std::io::Write;
        const ID: &[u8] = b"0123456789ABCXYZabcxyz_-.:\"";
        let start = out.len();
        let q = |r: &mut Rng, out: &mut Vec<u8>, s: &[u8]| {
            if r.chance(15) {
                out.push(b'"');
                out.extend_from_slice(s);
                out.push(b'"');
            } else {
                out.extend_from_slice(s);
            }
        };
        let id = |r: &mut Rng| -> Vec<u8> {
            let n = if r.chance(10) { r.below(40) } else { 8 + r.below(3) };
            (0..n).map(|_| if r.chance(98) { ID[r.below(23) as usize] } else { ID[r.below(ID.len() as u64) as usize] }).collect()
        };
        let ts = |r: &mut Rng| -> Vec<u8> {
            let mut s = Vec::new();
            let (y, mo) = match r.below(20) {
                0 => (2024, 2),
                1 => (2023, 2),
                2 => (1999, 12),
                3 => (2023, 13),
                _ => (2023, 12),
            };
            let d = if r.chance(4) { [0u64, 29, 30, 31, 32][r.below(5) as usize] } else { 1 + r.below(4) };
            let hm = if r.chance(2) { 30 } else { 24 };
            let mm = if r.chance(1) { 70 } else { 60 };
            let sm = if r.chance(2) { 70 } else { 60 };
            let (h, mi, sec) = (r.below(hm), r.below(mm), r.below(sm));
            match r.below(13) {
                0..=5 => write!(s, "{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{sec:02}Z").unwrap(),
                6..=7 => write!(s, "{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{sec:02}{}{:02}:{:02}", if r.chance(50) { '+' } else { '-' }, r.below(26), r.below(62)).unwrap(),
                8..=9 => {
                    let base = days_from_civil(2023, 12, 1) * DAY;
                    let ms = if r.chance(50) { r.below(5 * 86_400) * 1000 } else { r.below(5 * DAY as u64) };
                    write!(s, "{}", base as u64 + ms).unwrap()
                }
                10 => write!(s, "{}", r.next() % 10u64.pow(r.below(20) as u32)).unwrap(),
                11 => write!(s, "{}{:010}", r.next(), r.below(1_000_000_000)).unwrap(),
                _ => write!(s, "{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{sec:02}.{}", r.below(1000)).unwrap(),
            }
            if r.chance(3) && !s.is_empty() {
                let at = r.below(s.len() as u64) as usize;
                s[at] = b"9x:-T Z+"[r.below(8) as usize];
            }
            s
        };
        let cid = id(r);
        q(r, out, &cid);
        out.push(b',');
        let uid = id(r);
        q(r, out, &uid);
        out.push(b',');
        let cmp: Vec<u8> = match r.below(10) {
            0..=6 => format!("c_{}", 1 + r.below(300)).into_bytes(),
            7 => format!("c_0{}", r.below(50)).into_bytes(),
            8 => [&b"camp_x"[..], b"c_", b"", b"c_abc", b"C_12", b"c_1234567890"][r.below(6) as usize].to_vec(),
            _ => format!("c_{}", r.below(1 << 25)).into_bytes(),
        };
        q(r, out, &cmp);
        out.push(b',');
        let t = ts(r);
        q(r, out, &t);
        if r.chance(3) {
            out.push(b',');
        }
        if r.chance(5) {
            out.push(b'\r');
        }
        if r.chance(10) && out.len() > start {
            let at = start + r.below((out.len() - start) as u64) as usize;
            match r.below(4) {
                0 => out.insert(at, b'"'),
                1 => out.insert(at, b','),
                2 => out.insert(at, b'\r'),
                _ => {
                    out.remove(at);
                }
            }
        }
    }

    /// parse_clicks (one-window kernel, fixed-offset times) gives exactly the records of
    /// parse_clicks_old (scan4 + click_line + ts_fast), in the same order
    #[test]
    fn parse_clicks_matches_old() {
        assert!(cfg!(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")), "build with -C target-cpu=x86-64-v3");
        let mut names: Vec<B> = (1..=250u32).map(|i| &*Box::leak(format!("c_{i}").into_bytes().into_boxed_slice())).collect();
        names.extend([&b"c_007"[..], b"camp_x", b"c_abc"]);
        let items: Vec<(B, u32)> = names.iter().enumerate().map(|(i, &n)| (n, i as u32)).collect();
        let camp = IdLookup::new(b"c_", &items);
        let rs = hasher();
        let buf: &'static mut [u8] = Box::leak(vec![0u8; 1 << 16].into_boxed_slice());
        let mut r = Rng(0x3c6e_f372_fe94_f82b);
        let mut text = Vec::new();
        let (mut cases, mut lines, mut total, mut fast_lines) = (0u64, 0u64, 0u64, 0u64);
        let key = |v: &[ClickRec]| v.iter().map(|c| (c.key, c.ts, c.cid, c.camp)).collect::<Vec<_>>();
        for _ in 0..200_000 {
            text.clear();
            if r.chance(20) {
                text.extend_from_slice(b"click_id,user_id,campaign_id,ts\n");
            }
            for _ in 0..r.below(40) {
                match r.below(20) {
                    0..=16 => click_text(&mut r, &mut text),
                    17..=18 => line4t(&mut r, &mut text),
                    _ => {}
                }
                text.push(b'\n');
                lines += 1;
            }
            if r.chance(50) && !text.is_empty() {
                text.pop();
            }
            if text.len() + 64 > buf.len() {
                continue;
            }
            let off = r.below(64) as usize;
            buf[off..off + text.len()].copy_from_slice(&text);
            // SAFETY: buf is leaked (lives forever) and not written while `chunk` is in use
            let chunk: B = unsafe { std::slice::from_raw_parts(buf.as_ptr().add(off), text.len()) };
            let w0 = days_from_civil(2023, 12, 1) * DAY + DAY / 2;
            let (ws, we) = match r.below(3) {
                0 => (w0, w0 + 2 * DAY),
                1 => (-(1i64 << 62), 1i64 << 62),
                _ => (w0 - 3 * DAY, w0 + 3 * DAY),
            };
            let mut want = vec![];
            parse_clicks_old(chunk, &camp, &rs, ws, we, &mut want);
            SCAN7_FAST.with(|c| c.set(0));
            let mut got = vec![];
            parse_clicks(chunk, &camp, &rs, ws, we, &mut got);
            fast_lines += SCAN7_FAST.with(|c| c.replace(0));
            assert_eq!(key(&got), key(&want), "case {cases} ws {ws} we {we}: {:?}", String::from_utf8_lossy(chunk));
            cases += 1;
            total += want.len() as u64;
        }
        eprintln!("parse_clicks test: {cases} chunks, {lines} lines, {total} records, {fast_lines} lines via the kernel");
        assert!(total > 300_000 && fast_lines > lines / 2);
    }

    /// IdLookup::get_fast2 returns exactly what get returns, for ids near and far from canonical
    #[test]
    fn get_fast2_matches_get() {
        let mut r = Rng(0x6a09_e667_f3bc_c908);
        let mut items: Vec<(B, u32)> = vec![];
        for i in 0..3000u32 {
            let s = match r.below(4) {
                0 => format!("c_{}", r.below(5000)),
                1 => format!("c_{}", r.below(1 << 26)),
                2 => format!("c_0{}", r.below(100)),
                _ => format!("x{}", r.below(1000)),
            };
            items.push((leak_str(s), i));
        }
        let lk = IdLookup::new(b"c_", &items);
        let buf: &'static mut [u8] = Box::leak(vec![b'#'; 64].into_boxed_slice());
        const AL: &[u8] = b"0123456789c_x/:";
        for _ in 0..2_000_000 {
            let n = r.below(14) as usize;
            let mut id: Vec<u8> = if r.chance(70) { b"c_".to_vec() } else { vec![] };
            while id.len() < n {
                id.push(if r.chance(85) { b'0' + r.below(10) as u8 } else { AL[r.below(AL.len() as u64) as usize] });
            }
            id.truncate(n);
            let off = 8 + r.below(30) as usize;
            buf[off..off + n].copy_from_slice(&id);
            let lo = buf.as_ptr() as usize + r.below(off as u64 + 1) as usize;
            // SAFETY: buf is leaked and not written while b is in use
            let b: B = unsafe { std::slice::from_raw_parts(buf.as_ptr().add(off), n) };
            assert_eq!(lk.get_fast2(b, lo), lk.get(b), "{:?}", String::from_utf8_lossy(b));
        }
    }

    /// old and new parse of one chunk; the record count, or the first difference
    fn sol_cmp<E: Ev>(c: B, cx: &EvCtx) -> Result<(usize, usize), String> {
        let (mut a, mut b) = (EvBufs::<E>::new(), EvBufs::<E>::new());
        parse_events(c, cx, &mut a.m, &mut a.d, &mut a.fb, &mut a.fbn, &mut a.side);
        parse_events_sol::<E, false>(c, cx, &mut b.m, &mut b.d, &mut b.fb, &mut b.fbn, &mut b.side, &mut Dz::off());
        match a.diff(&b) {
            Some(e) => Err(e),
            None => Ok((a.m.len(), a.m.iter().filter(|e| e.pc().is_some()).count())),
        }
    }

    fn leak_str(s: String) -> B {
        Box::leak(s.into_bytes().into_boxed_slice())
    }

    /// one event line of random, mostly well-formed fields (ids with a few bad bytes, the three
    /// time shapes around early December 2023 with some invalid dates, all event types)
    fn ev_text(r: &mut Rng, out: &mut Vec<u8>) {
        use std::io::Write;
        const ID: &[u8] = b"0123456789ABCXYZabcxyz_-.:\"";
        let q = |r: &mut Rng, out: &mut Vec<u8>, s: &[u8]| {
            if r.chance(20) {
                out.push(b'"');
                out.extend_from_slice(s);
                out.push(b'"');
            } else {
                out.extend_from_slice(s);
            }
        };
        let id = |r: &mut Rng| -> Vec<u8> {
            let n = if r.chance(10) { r.below(14) } else if r.chance(15) { 30 + r.below(6) } else { 8 + r.below(2) };
            (0..n).map(|_| if r.chance(98) { ID[r.below(23) as usize] } else { ID[r.below(ID.len() as u64) as usize] }).collect()
        };
        let ts = |r: &mut Rng, iso_only: bool| -> Vec<u8> {
            let mut s = Vec::new();
            let d = if r.chance(2) { [0u64, 31, 32][r.below(3) as usize] } else { 1 + r.below(4) };
            let hm = if r.chance(1) { 30 } else { 24 };
            let sm = if r.chance(1) { 70 } else { 60 };
            let (h, mi, sec) = (r.below(hm), r.below(60), r.below(sm));
            match if iso_only { 0 } else { r.below(10) } {
                0..=5 => write!(s, "2023-12-{d:02}T{h:02}:{mi:02}:{sec:02}Z").unwrap(),
                6 => write!(s, "2023-12-{d:02}T{h:02}:{mi:02}:{sec:02}{}{:02}:{:02}", if r.chance(50) { '+' } else { '-' }, r.below(25), r.below(60)).unwrap(),
                7..=8 => {
                    let base = days_from_civil(2023, 12, 1) * DAY;
                    let ms = if r.chance(50) { r.below(5 * 86_400) * 1000 } else { r.below(5 * DAY as u64) };
                    write!(s, "{}", base as u64 + ms).unwrap()
                }
                _ => write!(s, "2023-12-{d:02} {h:02}:{mi:02}:{sec:02}.{}", r.below(1000)).unwrap(),
            }
            if r.chance(2) && !s.is_empty() {
                let at = r.below(s.len() as u64) as usize;
                s[at] = b"9x:-T "[r.below(6) as usize];
            }
            s
        };
        let eid = id(r);
        q(r, out, &eid);
        out.push(b',');
        let uid = id(r);
        q(r, out, &uid);
        out.push(b',');
        let t = ts(r, false);
        q(r, out, &t);
        out.push(b',');
        let iso = r.chance(85);
        let ing = ts(r, iso);
        q(r, out, &ing);
        out.push(b',');
        let et: &[u8] = [&b"view"[..], b"cart", b"purchase", b"purchase", b"purchasx", b"", b"View"][r.below(7) as usize];
        q(r, out, et);
        out.push(b',');
        let pr = match r.below(20) {
            0 => vec![],
            1 => b"x5".to_vec(),
            _ => format!("p{}", r.below(14)).into_bytes(),
        };
        q(r, out, &pr);
        out.push(b',');
        let price: &[u8] = [&b"12.34"[..], b"5", b"0", b"", b"abc", b"-1", b"999999.99"][r.below(7) as usize];
        q(r, out, price);
        if r.chance(3) {
            out.push(b',');
        }
        if r.chance(5) {
            out.push(b'\r');
        }
    }

    /// parse_events_sol gives the same records, dedup entries and side lists as parse_events:
    /// on every chunk of the fixtures and of each directory in SOL_DATA (';'-separated), and on
    /// random event text
    #[test]
    fn sol_matches_old() {
        let mut dirs: Vec<std::path::PathBuf> = vec![];
        if let Ok(rd) = std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures")) {
            for e in rd.flatten() {
                dirs.push(e.path().join("input"));
            }
        }
        if let Ok(v) = std::env::var("SOL_DATA") {
            dirs.extend(v.split(';').filter(|s| !s.is_empty()).map(std::path::PathBuf::from));
        }
        let rs = hasher();
        let (mut n_chunks, mut n_recs) = (0usize, 0usize);
        for dir in dirs.iter().filter(|d| d.join("events").is_dir()) {
            let mf = std::fs::read_to_string(dir.join("manifest.json")).unwrap_or_default();
            let ws = manifest_ts(&mf, "window_start").unwrap_or(0);
            let we = manifest_ts(&mf, "window_end").unwrap_or(0);
            let mut prod_raw: Vec<(B, B)> = vec![];
            if let Some(f) = leak_file(&dir.join("products.csv")) {
                for c in chunks(f, CHUNK) {
                    scan::<2>(c, |&[p, c]| prod_raw.push((p, c)));
                }
            }
            let mut cat_names: Vec<B> = prod_raw.iter().map(|x| x.1).collect();
            cat_names.push(b"unknown");
            cat_names.sort();
            cat_names.dedup();
            let unknown_cat = cat_names.binary_search(&(b"unknown" as B)).unwrap() as u32;
            let lean = cat_names.len() <= 256 && we.saturating_sub(ws) <= 1i64 << 32;
            let mut prod_items: Vec<(B, u32)> = vec![];
            let mut seen: AHashMap<B, usize> = AHashMap::new();
            for &(p, c) in &prod_raw {
                let cat = cat_names.binary_search(&c).unwrap() as u32;
                match seen.get(p) {
                    Some(&i) => prod_items[i].1 = cat,
                    None => {
                        seen.insert(p, prod_items.len());
                        prod_items.push((p, cat));
                    }
                }
            }
            let prod = IdLookup::new(b"p", &prod_items);
            let cx = EvCtx { prod: &prod, unknown_cat, ws, we, ibase: ws.div_euclid(1000), rs: &rs };
            let ev_files = csv_files(&dir.join("events"));
            let ev_chunks = all_chunks(&ev_files, ev_chunk(&ev_files));
            for (i, &c) in ev_chunks.iter().enumerate() {
                let (n, _) = sol_cmp::<EvW>(c, &cx).unwrap_or_else(|e| panic!("{} chunk {i} EvW: {e}", dir.display()));
                if lean {
                    sol_cmp::<EvL>(c, &cx).unwrap_or_else(|e| panic!("{} chunk {i} EvL: {e}", dir.display()));
                }
                n_chunks += 1;
                n_recs += n;
            }
            eprintln!("sol test: {} ({} chunks so far, {n_recs} records)", dir.display(), n_chunks);
        }
        assert!(n_chunks > 0);

        // random event text around a window of 2023-12-01 12:00 .. 2023-12-04 00:00
        let items: Vec<(B, u32)> = (0..10).map(|i| (leak_str(format!("p{i}")), (i % 3) as u32)).collect();
        let prod = IdLookup::new(b"p", &items);
        let ws = days_from_civil(2023, 12, 1) * DAY + DAY / 2;
        let cx = EvCtx { prod: &prod, unknown_cat: 3, ws, we: ws + 2 * DAY + DAY / 2, ibase: ws.div_euclid(1000), rs: &rs };
        let buf: &'static mut [u8] = Box::leak(vec![0u8; 1 << 17].into_boxed_slice());
        let mut r = Rng(0x2545_f491_4f6c_dd1d);
        let mut text = Vec::new();
        let (mut recs, mut purch) = (0usize, 0usize);
        for case in 0..20_000 {
            text.clear();
            for _ in 0..r.below(300) {
                if r.chance(95) {
                    ev_text(&mut r, &mut text);
                } else {
                    line(&mut r, &mut text);
                }
                text.push(b'\n');
            }
            if r.chance(50) && !text.is_empty() {
                text.pop();
            }
            if text.len() + 64 > buf.len() {
                continue;
            }
            let off = r.below(64) as usize;
            buf[off..off + text.len()].copy_from_slice(&text);
            // SAFETY: buf is leaked (lives forever) and not written while `chunk` is in use
            let chunk: B = unsafe { std::slice::from_raw_parts(buf.as_ptr().add(off), text.len()) };
            let (n, p) = sol_cmp::<EvW>(chunk, &cx).unwrap_or_else(|e| panic!("random case {case} EvW: {e}\n{}", String::from_utf8_lossy(chunk)));
            sol_cmp::<EvL>(chunk, &cx).unwrap_or_else(|e| panic!("random case {case} EvL: {e}\n{}", String::from_utf8_lossy(chunk)));
            recs += n;
            purch += p;
        }
        eprintln!("sol test random: {recs} records, {purch} priced");
        assert!(recs > 1_000_000 && purch > 10_000);
    }
    #[test]
    fn scatter_counted_matches_bucketize() {
        let mut r = Rng(0x5eed_0bad_cafe_f00d);
        for case in 0..200 {
            let nb = 1usize << (r.below(11) as u32);
            let n = r.below(if case % 4 == 0 { 50_000 } else { 300 }) as usize;
            let v: Vec<(u64, u32)> = (0..n).map(|i| (r.next(), i as u32)).collect();
            let bk = |x: &(u64, u32)| (mix(x.0) >> 20) as usize & (nb - 1);
            let mut cnt = vec![0u32; nb + 1];
            for x in &v {
                cnt[bk(x) + 1] += 1;
            }
            let (a, ao) = bucketize(&v, nb, bk);
            let (b, bo) = scatter_counted(&v, cnt, bk);
            assert_eq!(ao, bo);
            assert!(a.iter().zip(&b).all(|(x, y)| x == y), "nb={nb} n={n}");
        }
    }
}

#[cfg(all(test, target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
mod pack_simd_tests {
    use super::*;

    fn chk(a: &[u8], b: &[u8]) {
        let mut ba = [b','; 32];
        ba[..a.len()].copy_from_slice(a);
        let mut bb = [b','; 32];
        bb[..b.len()].copy_from_slice(b);
        let (ka, oka, kb, okb) = unsafe { pack6x2(ba.as_ptr(), a.len(), bb.as_ptr(), b.len()) };
        for (x, k, ok) in [(a, ka, oka), (b, kb, okb)] {
            match pack6(x) {
                Some(e) => assert!(ok && k == e, "pack6x2 differs on {:?}", x),
                None => assert!(!ok, "pack6x2 accepts {:?}", x),
            }
        }
    }

    fn chk_txt(a: &[u8], pad: u8) {
        let mut buf = [pad; 32];
        buf[..a.len()].copy_from_slice(a);
        let want = pack6(a).and_then(|k| gen_val(k, 42));
        let got = unsafe { gen_val_txt(buf.as_ptr(), a.len(), b'e') };
        assert_eq!(got, want, "gen_val_txt differs on {:?}", String::from_utf8_lossy(a));
    }

    #[test]
    fn gen_val_txt_matches_gen_val() {
        let body: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let mut nok = 0usize;
        for n in 0..=12usize {
            for pos in 0..n {
                for v in 0..=255u8 {
                    let mut a: Vec<u8> = (0..n).map(|i| if i == 0 { b'e' } else { body[(i * 7 + n) % 36] }).collect();
                    a[pos] = v;
                    for pad in [b',', b'0', b'z', 0xFF, 0] {
                        chk_txt(&a, pad);
                    }
                    nok += pack6(&a).and_then(|k| gen_val(k, 42)).is_some() as usize;
                }
            }
        }
        assert!(nok > 1000);
        // edges: GM and GM + 1 in base 36, leading zeros, every length
        let b36 = |mut x: u64| {
            let mut d = vec![];
            while x > 0 {
                d.push(body[(x % 36) as usize]);
                x /= 36;
            }
            d.reverse();
            let mut s = vec![b'e'];
            s.extend(d);
            s
        };
        for x in [0, 1, 35, 36, GM - 1, GM, GM + 1, GM + 36, 36u64.pow(8) - 1] {
            for pad in [b',', b'z'] {
                chk_txt(&b36(x), pad);
            }
        }
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let alpha: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCZ_e,\"";
        for _ in 0..3_000_000 {
            let r = rnd();
            let n = (r % 12) as usize;
            let mut a: Vec<u8> = (0..n).map(|i| {
                let q = rnd();
                if i == 0 && q % 8 != 0 { b'e' } else if q % 4 != 0 { body[(q >> 8) as usize % 36] } else { alpha[(q >> 8) as usize % alpha.len()] }
            }).collect();
            if n > 1 && r & 0x100 != 0 {
                a[1] = body[1 + (r >> 9) as usize % 35];
            }
            chk_txt(&a, (r >> 32) as u8);
        }
    }

    /// the per-person mover table answers what the filter and table answer, for any user_id key:
    /// movers and non-movers, aliases from 16, persons with mixed movers, persons beyond the
    /// table, kiosk and hashed IDs
    #[test]
    fn gp_table_matches_table() {
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0
            }
            fn below(&mut self, n: u64) -> u64 {
                self.next() % n
            }
        }
        let mut r = Rng(0x2545_F491_4F6C_DD1D);
        for _ in 0..6 {
            let key = r.next() & GM;
            let enc = |idx: u64| -> u64 {
                let m = (idx.wrapping_mul(GC1) ^ key) & GM;
                let mut v = ((m ^ (m >> 17)).wrapping_mul(GC2)) & GM;
                let mut b = vec![];
                loop {
                    b.push(b"0123456789abcdefghijklmnopqrstuvwxyz"[(v % 36) as usize]);
                    v /= 36;
                    if v == 0 {
                        break;
                    }
                }
                b.push(b'u');
                b.reverse();
                pack6(&b).unwrap()
            };
            let mut tab = IdTab::with_cap(1 << 16);
            let (mut keys, mut mov) = (vec![], 0usize);
            let gps: Vec<u64> = (0..400).chain((0..40).map(|i| (1 << 20) + i)).collect();
            for &gp in &gps {
                let rep = if r.below(5) == 0 { r.next() | HBIT } else { enc(gp * 512 + r.below(4)) };
                let mixed = r.below(6) == 0;
                for a in (0..24).chain([100 + r.below(400)]) {
                    let k = enc(gp * 512 + a);
                    assert_eq!(gen_val(k, UPRE).map(|v| gen_idx(v, key)), Some(gp * 512 + a));
                    keys.push(k);
                    let p = match r.below(4) {
                        0 | 1 if mixed && r.below(2) == 0 => enc((gp + 1) * 512),
                        0 | 1 => rep,
                        2 => k,
                        _ => continue,
                    };
                    let (_, new) = tab.get_or_insert(k, p);
                    mov += (new && p != k) as usize;
                }
            }
            for i in 0..200u64 {
                let k = if i % 2 == 0 { enc((1 << 39) + i) } else { r.next() | HBIT };
                keys.push(k);
                if i % 3 == 0 {
                    tab.get_or_insert(k, enc(i * 512));
                    mov += 1;
                }
            }
            tab.mov = mov;
            let (with, without) = (MoverF::build_with(&tab, Some(key)), MoverF::build_with(&tab, None));
            assert!(with.g.is_some() && without.g.is_none());
            let (mut ga, mut gm) = (0usize, 0usize);
            for &k in &keys {
                let want = tab.get(k);
                assert_eq!(with.person(k, &tab), want);
                assert_eq!(without.person(k, &tab), want);
                if let Some(p) = with.g.as_ref().unwrap().get(k) {
                    ga += 1;
                    gm += (p != k) as usize;
                }
            }
            // the per-person table itself answers most keys, movers among them
            assert!(ga * 3 > keys.len() && gm * 4 > ga, "{ga} {gm} {}", keys.len());
        }
    }

    #[test]
    fn fuse_eu_matches() {
        let body: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let alpha: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";
        let mut dz = Dz::<EvL>::off();
        dz.on = true;
        dz.key = 0x0123_4567_89AB & GM;
        dz.lim = GM / 2;
        let edge: [u8; 16] = [0x00, b'/', b'0', b'9', b':', b'@', b'A', b'Z', b'_', b'`', b'a', b'z', b'{', 0x7f, 0x80, 0xff];
        let one = |a: &[u8], u: &[u8], pad: u8| {
            let mut be = [pad; 32];
            be[..a.len()].copy_from_slice(a);
            let mut bu = [pad; 32];
            bu[..u.len()].copy_from_slice(u);
            let (oke, val, ku, oku) = unsafe { fuse_eu(be.as_ptr(), a.len(), bu.as_ptr(), u.len()) };
            let want = unsafe { gen_val_txt(be.as_ptr(), a.len(), b'e') };
            assert_eq!(oke.then_some(val), want, "eid differs on {:?}", String::from_utf8_lossy(a));
            if want.is_some() {
                let wi = unsafe { dz.idx_txt(be.as_ptr(), a.len()) };
                let gi = dz.idx_val(val);
                assert_eq!(gi, wi);
            }
            let (_, _, wk, wok) = unsafe { pack6x2(be.as_ptr(), a.len(), bu.as_ptr(), u.len()) };
            assert_eq!(oku, wok, "uid ok differs on {:?}", u);
            if wok {
                assert_eq!(ku, wk);
            }
        };
        for n in 0..=16usize {
            for pos in 0..n {
                for &v in edge.iter().chain(body.iter()) {
                    for lead in [b'e', b'x'] {
                        let mut a: Vec<u8> = (0..n).map(|i| if i == 0 { lead } else { body[(i * 7 + n) % 36] }).collect();
                        a[pos] = v;
                        let u: Vec<u8> = (0..n).map(|i| alpha[(i * 5 + n) % 63]).collect();
                        for pad in [b',', 0u8, 0xFF] {
                            one(&a, &u, pad);
                            one(&u, &a, pad);
                        }
                    }
                }
            }
            // leading zero body and plain all-zero / all-z
            for fill in [b'0', b'z', b'1'] {
                let mut a = vec![fill; n];
                if n > 0 {
                    a[0] = b'e';
                }
                one(&a, b"u_1", b',');
                if n > 1 {
                    a[1] = b'0';
                    one(&a, b"u_1", b',');
                }
            }
        }
        let b36 = |mut x: u64| {
            let mut d = vec![];
            while x > 0 {
                d.push(body[(x % 36) as usize]);
                x /= 36;
            }
            d.reverse();
            let mut s = vec![b'e'];
            s.extend(d);
            s
        };
        for x in [0, 1, 35, 36, GM - 1, GM, GM + 1, GM + 36, 36u64.pow(8) - 1, 36u64.pow(7)] {
            one(&b36(x), b"u_9", b',');
        }
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..1_500_000 {
            let n = (rnd() % 14) as usize;
            let a: Vec<u8> = (0..n)
                .map(|i| match rnd() % 40 {
                    0 => rnd() as u8,
                    1 if i > 0 => alpha[(rnd() % 63) as usize],
                    _ if i == 0 && rnd() % 16 != 0 => b'e',
                    _ => body[(rnd() % 36) as usize],
                })
                .collect();
            let m = (rnd() % 14) as usize;
            let u: Vec<u8> = (0..m).map(|_| if rnd() % 30 == 0 { rnd() as u8 } else { alpha[(rnd() % 63) as usize] }).collect();
            one(&a, &u, (rnd() & 0xFF) as u8);
        }
    }

    #[test]
    fn pack6x1_matches_pack6() {
        let alpha: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";
        let one = |a: &[u8]| {
            let mut ba = [b','; 32];
            ba[..a.len()].copy_from_slice(a);
            let (k, ok) = unsafe { pack6x1(ba.as_ptr(), a.len()) };
            match pack6(a) {
                Some(e) => assert!(ok && k == e, "pack6x1 differs on {:?}", a),
                None => assert!(!ok, "pack6x1 accepts {:?}", a),
            }
        };
        one(b"");
        for n in 1..=16usize {
            for pos in 0..n {
                for v in 0..=255u8 {
                    let mut a: Vec<u8> = (0..n).map(|i| alpha[(i * 7 + n) % alpha.len()]).collect();
                    a[pos] = v;
                    one(&a);
                }
            }
        }
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..200_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let n = 1 + (s % 10) as usize;
            let a: Vec<u8> = (0..n).map(|i| alpha[((s >> (6 * i)) % 63) as usize]).collect();
            one(&a);
        }
    }

    #[test]
    fn pack6x2_matches_pack6() {
        let alpha: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";
        for n in 1..=12usize {
            for pos in 0..n {
                for v in 0..=255u8 {
                    let mut a: Vec<u8> = (0..n).map(|i| alpha[(i * 7 + n) % alpha.len()]).collect();
                    a[pos] = v;
                    chk(&a, b"u_1");
                    chk(b"e_9", &a);
                }
            }
        }
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..200_000 {
            let n = 1 + (rnd() % 10) as usize;
            let a: Vec<u8> = (0..n).map(|_| alpha[(rnd() % 63) as usize]).collect();
            let m = 1 + (rnd() % 10) as usize;
            let b: Vec<u8> = (0..m).map(|_| alpha[(rnd() % 63) as usize]).collect();
            chk(&a, &b);
        }
        assert_eq!(pack6(b"u_123"), Some(1055258694234669056));
    }

    #[test]
    fn f1_fixed_matches_ts_iso() {
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0
            }
            fn below(&mut self, n: u64) -> u64 {
                self.next() % n
            }
            fn chance(&mut self, pct: u64) -> bool {
                self.below(100) < pct
            }
        }
        let mut r = Rng(0x5EED_F1F1_0BAD_CAFE);
        let alpha = b"0123456789:-TZtz +\"
,";
        let mut dc = DayCache::new();
        let mut buf = [0u8; 32];
        let mut ok = 0u32;
        for i in 0..400_000u32 {
            let y = 1990 + r.below(60);
            let mo = 1 + r.below(13);
            let d = 1 + r.below(31);
            let (h, mi, s) = (r.below(25), r.below(61), r.below(61));
            let t = format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z");
            buf[..20].copy_from_slice(t.as_bytes());
            let k = if i % 4 == 0 { 0 } else { 1 + r.below(3) };
            for _ in 0..k {
                let at = r.below(20) as usize;
                buf[at] = if r.chance(70) { alpha[r.below(alpha.len() as u64) as usize] } else { r.next() as u8 };
            }
            let b = &buf[..20];
            let want = if b[19] == b'Z' { ts_iso(b, &mut TsCache::with_bits(1)) } else { None };
            let p = buf.as_ptr();
            let got = if b[19] == b'Z' {
                // SAFETY: buf has 32 bytes
                unsafe {
                    match (day_get(ldp(p), ldp(p.add(3)), p, &mut dc), hms(p.add(11))) {
                        (Some(d), Some(h)) => Some((d + h) * 1000),
                        _ => None,
                    }
                }
            } else {
                None
            };
            assert_eq!(got, want, "{:?}", String::from_utf8_lossy(b));
            ok += want.is_some() as u32;
        }
        assert!(ok > 50_000, "{ok}");
    }

    #[test]
    fn ts25_fixed_matches_parse_ts_c() {
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0
            }
            fn below(&mut self, n: u64) -> u64 {
                self.next() % n
            }
            fn chance(&mut self, pct: u64) -> bool {
                self.below(100) < pct
            }
        }
        let mut r = Rng(0x25AD_0FF5_E7C0_FFEE);
        let alpha = b"0123456789:-+TZ ,\"";
        let (mut dc, mut dk) = (DayCache::new(), DayCache::new());
        let mut buf = [0u8; 32];
        let (mut ok, mut fast) = (0u32, 0u32);
        for i in 0..400_000u32 {
            let y = 1990 + r.below(60);
            let mo = 1 + r.below(13);
            let d = 1 + r.below(31);
            let (h, mi, s) = (r.below(25), r.below(61), r.below(61));
            let (oh, om) = (r.below(26), r.below(61));
            let sg = if r.chance(50) { '+' } else { '-' };
            let t = format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}{sg}{oh:02}:{om:02}");
            buf[..25].copy_from_slice(t.as_bytes());
            // the date key handed in: this value's own date half the time when valid
            let p = buf.as_ptr();
            // SAFETY: buf has 32 bytes
            let (ra, rb, rd) = unsafe {
                match day_get(ldp(p), ldp(p.add(3)), p, &mut dk) {
                    Some(d) if r.chance(50) => (ldp(p), ldp(p.add(3)), d),
                    _ => (D0_A, D0_B, D0_S),
                }
            };
            let k = if i % 4 == 0 { 0 } else { 1 + r.below(3) };
            for _ in 0..k {
                let at = r.below(25) as usize;
                buf[at] = if r.chance(70) { alpha[r.below(alpha.len() as u64) as usize] } else { r.next() as u8 };
            }
            let b = &buf[..25];
            let want = parse_ts_c(b, &mut TsCache::with_bits(1));
            let p = buf.as_ptr();
            // SAFETY: buf has 32 bytes
            if let Some(g) = unsafe { ts25(p, ra, rb, rd, &mut dc) } {
                assert_eq!(Some(g), want, "{:?}", String::from_utf8_lossy(b));
                fast += 1;
            }
            ok += want.is_some() as u32;
        }
        assert!(ok > 50_000 && fast > 50_000, "{ok} {fast}");
    }

    #[test]
    fn hms2_matches_hms() {
        let mut x = 0x4A5D_2EED_0DD5_1234u64;
        let mut nx = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let alpha = b"0123456789:/;,+-Z 9";
        let mut buf = [0u8; 16];
        let mut ok = 0u32;
        for i in 0..1_000_000u32 {
            for h in 0..2 {
                let s = (nx() % 86_400 + if nx() % 8 == 0 { 86_400 + nx() % 20_000 } else { 0 }) as u32;
                let t = format!("{:02}:{:02}:{:02}", (s / 3600) % 100, s / 60 % 60, s % 60);
                buf[8 * h..8 * h + 8].copy_from_slice(t.as_bytes());
            }
            let k = if i % 3 == 0 { 0 } else { 1 + nx() % 3 };
            for _ in 0..k {
                let at = (nx() % 16) as usize;
                buf[at] = if nx() % 100 < 70 { alpha[(nx() % alpha.len() as u64) as usize] } else { nx() as u8 };
            }
            // SAFETY: both 8-byte reads lie inside buf
            let (sa, oa, sb, ob) = unsafe { hms2(buf.as_ptr(), buf.as_ptr().add(8)) };
            let (wa, wb) = unsafe { (hms(buf.as_ptr()), hms(buf.as_ptr().add(8))) };
            assert_eq!(if oa { Some(sa) } else { None }, wa, "{:?}", &buf[..8]);
            assert_eq!(if ob { Some(sb) } else { None }, wb, "{:?}", &buf[8..]);
            ok += oa as u32 + ob as u32;
        }
        assert!(ok > 300_000, "{ok}");
    }

    #[test]
    fn ts13_matches_ts_num() {
        let mut x = 0x1313_5EED_0DD5_EEDEu64;
        let mut nx = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let alpha = b"0123456789/:;,\"+- ";
        let mut buf = [0u8; 32];
        let mut ok = 0u32;
        for i in 0..400_000u32 {
            for j in 0..32 {
                buf[j] = if j < 3 || j >= 16 { nx() as u8 } else { b'0' + (nx() % 10) as u8 };
            }
            let k = if i % 3 == 0 { 0 } else { 1 + nx() % 2 };
            for _ in 0..k {
                let at = 3 + (nx() % 13) as usize;
                buf[at] = if nx() % 100 < 70 { alpha[(nx() % alpha.len() as u64) as usize] } else { nx() as u8 };
            }
            let want = ts_num(&buf[3..16]);
            // SAFETY: buf has 3 bytes before and 3 after the 13
            let got = unsafe { ts13(buf.as_ptr().add(3)) };
            assert_eq!(got, want, "{:?}", &buf[3..16]);
            ok += want.is_some() as u32;
        }
        assert!(ok > 100_000, "{ok}");
    }

    fn bz(x: u64, n: u32) -> u64 {
        if n >= 64 { x } else { x & ((1u64 << n) - 1) }
    }

    /// The earlier quote rule (open/close pairs by a subtraction), kept as the reference
    fn quote_bad_old(q: u64, c: u64, end: u32) -> u64 {
        let q1 = bz(q, end);
        let x = q1 & ((c << 1) | 1);
        let y = q1 & (c >> 1);
        let r = y.wrapping_sub(x);
        (q1 & !(x | y)) | (x & y) | (r & (bz(c, end + 1) | y))
    }

    /// Byte-level rule: every field before `end` is free of quotes or is "..." with no other quote
    fn quote_ok_ref(q: u64, c: u64, end: u32) -> bool {
        let mut s = 0u32;
        for i in 0..=end {
            if (c >> i) & 1 == 1 {
                let n = i - s;
                let fq = bz(q >> s, n);
                let ok = fq == 0 || (n >= 2 && fq == (1 | (1u64 << (n - 1))));
                if !ok {
                    return false;
                }
                s = i + 1;
            }
        }
        true
    }

    fn quote_case(q: u64, c: u64, end: u32) {
        let c = c | (1u64 << end);
        let q = q & !c;
        let (x, bad) = quote_bad(bz(q, end), c);
        let old = quote_bad_old(q, c, end);
        assert_eq!(x, bz(q, end) & ((c << 1) | 1));
        assert_eq!(bad == 0, old == 0, "q={q:#x} c={c:#x} end={end}");
        assert_eq!(bad == 0, quote_ok_ref(q, c, end), "q={q:#x} c={c:#x} end={end}");
    }

    /// The quote check equals the earlier rule and the byte-level rule: every class string
    /// (other, ',', '"') of up to 12 bytes at the start, one byte in, and against bit 63
    #[test]
    fn quote_bad_exhaustive() {
        for n in 0..=12u32 {
            let total = 3u64.pow(n);
            for code in 0..total {
                let (mut q, mut c, mut k) = (0u64, 0u64, code);
                for i in 0..n {
                    match k % 3 {
                        1 => c |= 1 << i,
                        2 => q |= 1 << i,
                        _ => {}
                    }
                    k /= 3;
                }
                for o in [0u32, 1, 63 - n] {
                    let above = !bz(!0, o + n + 1);
                    for (qa, ca) in [(0, 0), (above, 0), (0, above), (above & 0x5555_5555_5555_5555, above & 0xaaaa_aaaa_aaaa_aaaa)] {
                        quote_case((q << o) | qa, (c << o) | ca, o + n);
                    }
                }
                if n <= 59 {
                    // '"' 'a' '"' ',' prefix (a well-formed quoted field), then the string
                    quote_case((q << 4) | 0b101, (c << 4) | 0b1000, 4 + n);
                    // '"' 'a' ',' prefix (an unclosed quote), then the string
                    quote_case((q << 3) | 0b1, (c << 3) | 0b100, 3 + n);
                }
            }
        }
    }

    /// 3 * 10^7 random windows: random masks of several densities, and lines built from
    /// well-formed and broken fields with the end at a random separator
    #[test]
    fn quote_bad_random() {
        struct Xs(u64);
        impl Xs {
            fn next(&mut self) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0
            }
            fn below(&mut self, n: u64) -> u64 {
                self.next() % n
            }
            fn chance(&mut self, pct: u64) -> bool {
                self.below(100) < pct
            }
        }
        let mut r = Xs(0x9e37_79b9_7f4a_7c15);
        for i in 0..10_000_000u64 {
            let (a, b) = ((i % 4) as u32, ((i / 4) % 5) as u32);
            let mut c = r.next();
            for _ in 0..a {
                c &= r.next();
            }
            let mut q = r.next();
            for _ in 0..b {
                q &= r.next();
            }
            quote_case(q, c, (r.next() % 64) as u32);
        }
        let mut acc = 0u64;
        for _ in 0..20_000_000u64 {
            let (mut q, mut c, mut pos) = (0u64, 0u64, 0u32);
            let mut seps: Vec<u32> = Vec::new();
            while pos < 64 {
                let n = r.below(9) as u32;
                let kind = r.below(10);
                for j in 0..n.min(64 - pos) {
                    let quote = match kind {
                        0..=4 => false,
                        5 | 6 => j == 0 || j + 1 == n,
                        7 => j == 0,
                        8 => j + 1 == n,
                        _ => r.chance(33),
                    };
                    if quote {
                        q |= 1 << (pos + j);
                    }
                }
                pos += n;
                if pos < 64 {
                    c |= 1 << pos;
                    seps.push(pos);
                    pos += 1;
                }
            }
            if seps.is_empty() {
                continue;
            }
            let end = seps[r.below(seps.len() as u64) as usize];
            let junk = r.next() & !bz(!0, end + 1);
            let (q, c) = if r.chance(50) { (q | (junk & 0x5555_5555_5555_5555), c) } else { (q, c | (junk & r.next())) };
            let (_, bad) = quote_bad(bz(q & !c, end), c);
            acc += (bad == 0) as u64;
            quote_case(q, c, end);
        }
        assert!(acc > 1_000_000, "{acc}");
    }
}

#[cfg(test)]
mod ck_sort_tests {
    use super::*;

    fn rnd(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    fn mk(key: u64, ts: i64, id: B, camp: u32) -> ClickRec {
        ClickRec { key, ts, cid: pack(id), camp }
    }

    #[test]
    fn sort_clicks_matches_reference() {
        let mut s = 0x1234_5678_9abc_def1u64;
        let ids: &'static [Vec<u8>] = Vec::leak((0..4000)
            .map(|_| {
                let l = 1 + (rnd(&mut s) % 9) as usize;
                (0..l).map(|_| b"k0a\x00z"[(rnd(&mut s) % 5) as usize]).collect()
            })
            .collect::<Vec<Vec<u8>>>());
        // (clicks, distinct keys, ts span, ts step, one far outlier)
        let cases: [(usize, u64, i64, i64, bool); 8] = [
            (300, 3, 2_000_000_000, 1000, false),
            (5000, 1, 5_000_000, 1000, false),
            (20000, 500, 2_000_000_000, 1, false),
            (10000, 1, 1000, 1, true),
            (3000, 2000, 30_000, 1000, false),
            (700, 2, 60_000, 1, true),
            (256, 1, 10, 1, false),
            (40000, 9000, 1_800_000_000, 1000, false),
        ];
        for (ci, &(n, nk, span, step, far)) in cases.iter().enumerate() {
            let keys: Vec<u64> = (0..nk).map(|_| rnd(&mut s)).collect();
            let base = if ci % 2 == 0 { 1_830_000_000_000i64 } else { -5_000_000_000 };
            let mut v: Vec<ClickRec> = (0..n)
                .map(|j| {
                    let k = keys[(rnd(&mut s) % nk) as usize];
                    let t = base + (rnd(&mut s) % (span as u64 / step as u64 + 1)) as i64 * step;
                    let t = if far && j == n / 2 { base + (1i64 << 40) } else { t };
                    mk(k, t, &ids[(rnd(&mut s) % ids.len() as u64) as usize], j as u32)
                })
                .collect();
            if ci == 3 {
                // one clicker, one quantum, reversed: the insertion sort must give up
                v.sort_by(|a, b| b.ts.cmp(&a.ts));
            }
            // the contract: a permutation of the input in (dg, key, ts) order; equal (key, ts) in
            // any order, so both sides are compared after a full sort that includes click_id and camp
            let full = |v: &mut Vec<ClickRec>| {
                v.sort_by(|a, b| (dg(a.key), a.key, a.ts).cmp(&(dg(b.key), b.key, b.ts)).then_with(|| unpack(a.cid).cmp(unpack(b.cid))).then(a.camp.cmp(&b.camp)))
            };
            let mut want = v.clone();
            full(&mut want);
            let f = |v: &[ClickRec]| v.iter().map(|c| (c.key, c.ts, unpack(c.cid).to_vec(), c.camp)).collect::<Vec<_>>();
            let ord = |v: &[ClickRec]| v.windows(2).all(|w| (dg(w[0].key), w[0].key, w[0].ts) <= (dg(w[1].key), w[1].key, w[1].ts));
            let (mut tmp, mut cnt) = (vec![], vec![]);
            let mut a = v.clone();
            sort_clicks(&mut a, &mut tmp, &mut cnt);
            assert!(ord(&a), "sort_clicks order case {ci}");
            full(&mut a);
            assert!(f(&a) == f(&want), "sort_clicks case {ci}");
            let mut b = v.clone();
            assert!(sort_clicks_rdx(&mut b, &mut tmp, &mut cnt), "rdx fits case {ci}");
            assert!(ord(&b), "sort_clicks_rdx order case {ci}");
            full(&mut b);
            assert!(f(&b) == f(&want), "sort_clicks_rdx case {ci}");
        }
    }

    #[test]
    fn last_touch_tie_takes_smallest_click_id() {
        let mut s = 0x0bad_cafe_1234_5678u64;
        let ids: &'static [Vec<u8>] = Vec::leak((0..64)
            .map(|_| {
                let l = 1 + (rnd(&mut s) % 4) as usize;
                (0..l).map(|_| b"a0\x00\xffz"[(rnd(&mut s) % 5) as usize]).collect()
            })
            .collect::<Vec<Vec<u8>>>());
        for _ in 0..20000 {
            let n = 2 + (rnd(&mut s) % 6) as usize;
            // copies of one click share their campaign
            let g: Vec<ClickRec> = (0..n)
                .map(|_| {
                    let i = (rnd(&mut s) % ids.len() as u64) as usize;
                    mk(7, 0, &ids[i], (i % 3) as u32)
                })
                .collect();
            let want = g.iter().min_by(|a, b| unpack(a.cid).cmp(unpack(b.cid))).unwrap().camp;
            assert_eq!(min_cid_camp(&g), want);
        }
    }

    #[test]
    fn slim_runs_match_full() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _case in 0..200 {
            let ncamp = 3usize;
            let nd = 12usize;
            let mut keys: Vec<u64> = (0..1 + rnd() % 30).map(|_| 1 + rnd() % 5000).collect();
            keys.sort_by_key(|&k| (dg(k), k));
            keys.dedup();
            let mut full: Vec<EvL> = vec![];
            let mut runs: Vec<Run> = vec![];
            let mut clicks: Vec<ClickRec> = vec![];
            for &k in &keys {
                let nr = 1 + rnd() % 3;
                let mut ok = true;
                let mut last = 0u64;
                let s0 = full.len();
                for r in 0..nr {
                    let n = 1 + rnd() % 12;
                    let mut ts = rnd() % (8 * DAY as u64);
                    let rs = full.len();
                    let mut rok = true;
                    for i in 0..n {
                        let step = rnd() % 5;
                        ts = if rnd() % 7 == 0 { ts.saturating_sub(rnd() % 3000) } else { ts + step * 1_000_000 };
                        if i > 0 && ts < last {
                            rok = false;
                        }
                        if i == 0 && r > 0 && ts < last {
                            ok = false;
                        }
                        last = ts;
                        let price = if rnd() % 3 == 0 { 1 + (rnd() % 1000) as u32 } else { 0 };
                        full.push(EvL::mk(k, ts, price, (rnd() % 3) as u32));
                    }
                    ok &= rok;
                    let _ = rs;
                }
                runs.push(Run { d: dg(k) as u32, k, s: s0 as u32, l: (full.len() - s0) as u32, ok });
                for c in 0..rnd() % 6 {
                    clicks.push(ClickRec { key: k, ts: (rnd() % (8 * DAY as u64)) as i64, cid: c + rnd() % 100, camp: (rnd() % 3) as u32 });
                }
            }
            // split each person's lines into the pieces gather_rd would join: one Run per person here,
            // ok = false forces the tmp sort path, ok = true the in-place path only if truly sorted
            for r in runs.iter_mut() {
                if r.ok {
                    let v = &full[r.s as usize..(r.s + r.l) as usize];
                    r.ok = v.windows(2).all(|w| w[0].ts <= w[1].ts);
                }
            }
            let slim: Vec<EvS> = full.iter().map(|e| e.slim()).collect();
            let run = |a: &dyn Fn(&mut Vec<UD>, &mut CampAcc, &mut Vec<ClickRec>)| {
                let mut uds: Vec<UD> = vec![];
                let mut cm = CampAcc::new(ncamp + 1, 0, nd);
                let mut cl = clicks.clone();
                a(&mut uds, &mut cm, &mut cl);
                (uds, cm, cl)
            };
            let (u1, c1, _) = run(&|u, c, cl| {
                let (mut tmp, mut cnt, mut ct) = (vec![], vec![], vec![]);
                let mut win = Win { cnt: vec![0; ncamp + 1], pos: vec![0; ncamp + 1], active: vec![] };
                process_runs(&full, &mut tmp, &runs, &mut cnt, cl, &mut ct, &mut win, ncamp as u32, u, c, 0);
            });
            let (u2, c2, _) = run(&|u, c, cl| {
                let (mut tmp, mut cnt, mut ct) = (vec![], vec![], vec![]);
                let mut win = Win { cnt: vec![0; ncamp + 1], pos: vec![0; ncamp + 1], active: vec![] };
                process_runs(&slim, &mut tmp, &runs, &mut cnt, cl, &mut ct, &mut win, ncamp as u32, u, c, 0);
            });
            assert_eq!(u1.len(), u2.len());
            for (a, b) in u1.iter().zip(&u2) {
                assert_eq!((a.key, a.day, a.sessions, a.events, a.rev, a.top), (b.key, b.day, b.sessions, b.events, b.rev, b.top));
            }
            assert_eq!(c1.tch, c2.tch);
            for (a, b) in c1.dense.iter().zip(&c2.dense) {
                assert_eq!((a.s, a.p, a.r), (b.s, b.p, b.r));
            }
        }
    }
}

#[cfg(test)]
mod ev_sort_tests {
    use super::*;

    fn rnd(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    #[test]
    fn sort_evs_matches_reference() {
        let mut s = 0x0bad_cafe_1234_5679u64;
        // (events, distinct keys, ts span, reversed input)
        let cases: [(usize, u64, u64, bool); 9] = [
            (300, 3, 2_000_000_000, false),
            (5000, 1, 5_000_000, true),
            (20000, 500, 4_000_000_000, false),
            (10000, 1, 1000, false),
            (60000, 200_000, 86_400_000, false),
            (40000, 30, 10, true),
            (256, 1, 0, false),
            (100_000, 20_000, 3_600_000, false),
            (9000, 8000, 1 << 31, true),
        ];
        for &(n, nk, span, rev) in cases.iter() {
            let mut keys: Vec<u64> = (0..nk).map(|_| rnd(&mut s) | 1).collect();
            if nk > 100 {
                // many keys on a few digits: multi-key ranges, some above 16 keys
                let d0: Vec<u64> = keys.iter().copied().filter(|&k| dg(k) < 4).collect();
                keys.extend(d0.iter().cycle().take(nk as usize / 4));
            }
            let mut v: Vec<EvL> = (0..n)
                .map(|j| EvL::mk(keys[(rnd(&mut s) % keys.len() as u64) as usize], rnd(&mut s) % (span + 1), j as u32 & 0xffffff, 0))
                .collect();
            if rev {
                v.sort_by(|a, b| b.rts().cmp(&a.rts()));
            }
            let full = |v: &mut Vec<EvL>| v.sort_by_key(|e| (dg(e.key()), e.key(), e.rts(), e.meta));
            let mut want = v.clone();
            full(&mut want);
            let (mut tmp, mut cnt) = (vec![], vec![]);
            let mut a = v.clone();
            sort_evs(&mut a, &mut tmp, &mut cnt, span as i64);
            assert!(a.windows(2).all(|w| (dg(w[0].key()), w[0].key(), w[0].rts()) <= (dg(w[1].key()), w[1].key(), w[1].rts())), "order n={n} nk={nk}");
            full(&mut a);
            assert!(a.iter().zip(&want).all(|(x, y)| x.key() == y.key() && x.rts() == y.rts() && x.meta == y.meta), "perm n={n} nk={nk}");
        }
    }
}

#[cfg(test)]
mod idk_tests {
    use super::*;

    #[test]
    fn idk_ref_classes() {
        // Short ids: pack6
        for id in ["e1", "E0ab_9", "x123456789"] {
            assert_eq!(idk_ref(id.as_bytes()), (pack6(id.as_bytes()).unwrap(), 0, 0));
        }
        // Hex ids: prefix + 32 lowercase hex digits
        let h = "e0123456789abcdef00112233445566ff";
        let full = 0x0123456789abcdef00112233445566ffu128;
        let key = HBIT | 1 << 62 | (CODE[b'e' as usize] as u64) << 56 | (full >> 72) as u64;
        assert_eq!(idk_ref(h.as_bytes()), (key, full, 1));
        assert_eq!(idk_ref(b"Zffffffffffffffffffffffffffffffff"), (HBIT | 1 << 62 | (CODE[b'Z' as usize] as u64) << 56 | (u128::MAX >> 72) as u64, u128::MAX, 1));
        // numeric key order equals byte order for Hex ids
        let (ka, _, _) = idk_ref(b"a00000000000000000000000000000000");
        let (kb, _, _) = idk_ref(b"a00000000000001000000000000000000");
        assert!(ka < kb);
        // Other: upper-case hex, wrong length, bad prefix, non-hex
        for id in ["e0123456789ABCDEF00112233445566ff", "e0123456789abcdef00112233445566f", "-0123456789abcdef00112233445566ff", "e0123456789abcdeg00112233445566ff", "e0123456789abcdef00112233445566ff0", "abcdefghijk"] {
            assert_eq!(idk_ref(id.as_bytes()).2, 2, "{id}");
        }
        // DX round trip: hi/lo/code give back the id bytes
        let (_, f, _) = idk_ref(h.as_bytes());
        let x = DX { hi: (f >> 64) as u64, lo: f as u64, ing: 7, lp: 5 | (CODE[b'e' as usize] as u32) << 26 };
        let back = format!("{}{:016x}{:016x}", DECODE[(x.lp >> 26) as usize] as char, x.hi, x.lo);
        assert_eq!(back, h);
        assert_eq!(x.lp & DX_LOC, 5);
    }
}
