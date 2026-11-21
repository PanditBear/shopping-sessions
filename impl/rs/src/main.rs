use ahash::{AHashMap, RandomState};
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

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
static TIMING: AtomicBool = AtomicBool::new(false);
static PROF: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];
const PROF_NAMES: [&str; 13] =
    ["ev.scan", "ev.bucket", "dd.table", "dd.collect", "dd.resolve", "dd.bucket", "ss.gather", "ss.sort", "ss.proc", "ck.scan", "ck.res", "ss.csort", "wr.ud"];
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
    let mut v = vec![];
    if let Ok(rd) = std::fs::read_dir(dir) {
        let mut paths: Vec<_> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "csv").unwrap_or(false) && p.is_file())
            .collect();
        paths.sort();
        let r: Vec<Option<B>> = paths.par_iter().map(|p| leak_file(p)).collect();
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
    fn get(&self, k: u64) -> u64 {
        let mut i = self.slot(k);
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
    w: Vec<AtomicU64>,
    mask: usize,
}
impl MoverF {
    fn build(tab: &IdTab) -> Self {
        let nw = (tab.s.len() / 16).max(1);
        let w: Vec<AtomicU64> = (0..nw).map(|_| AtomicU64::new(0)).collect();
        tab.s.par_chunks(1 << 15).for_each(|c| {
            for e in c {
                if e[0] != 0 && e[0] != e[1] {
                    let h = mix(e[0]);
                    w[h as usize & (nw - 1)].fetch_or(Self::bits(h), Relaxed);
                }
            }
        });
        MoverF { w, mask: nw - 1 }
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
        unsafe { self.w.get_unchecked(h as usize & self.mask) }.load(Relaxed) & b == b
    }
}

const PFD: usize = 12;
/// prefetch distance of the user -> person passes (mover filter, then table), sized for DRAM latency
const RPD: usize = 32;
/// id table slices of the partitioned user -> person pass (by the top bits of the key hash, the
/// bits that pick the table slot): 1024 slices of the 128 MB table at 60M are 128 KB each
const RSB: u32 = 10;
const RNS: usize = 1 << RSB;

// ---------- records ----------

/// Dedup record of one event line: packed event_id, ingest key (`ing32`) and the line's index in
/// its chunk's record list.
#[derive(Clone, Copy)]
struct DRec {
    eid: u64,
    ing: u32,
    loc: u32,
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
trait Ev: Copy + Send + Sync {
    fn nil() -> Self;
    fn mk(key: u64, rts: u64, price: u32, cat: u32) -> Self;
    fn key(&self) -> u64;
    fn set_key(&mut self, k: u64);
    fn rts(&self) -> u64;
    /// (price, category) of a purchase
    fn pc(&self) -> Option<(u64, u32)>;
}

/// 16-byte record: needs a window of at most 2^32 ms and at most 256 categories.
#[derive(Clone, Copy)]
struct EvL {
    key: u64,
    ts: u32,
    meta: u32,
}

impl Ev for EvL {
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

/// 24-byte record for any window and category count.
#[derive(Clone, Copy)]
struct EvW {
    key: u64,
    ts: u64,
    meta: u64,
}

impl Ev for EvW {
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

#[derive(Clone, Copy)]
struct ClickRec {
    key: u64,
    ts: i64,
    cid: u64,
    camp: u32,
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

/// Sort by (dg(key), key, ts); equal (key, ts) pairs may end up in any order. `span` bounds
/// every rts. A counting pass on the digit, then per range: a range shorter than RADIX_MIN with
/// few keys is ordered as packed u64s (`pack_sort`), a long one-key range with `radix_rts`.
fn sort_evs<E: Ev>(evs: &mut [E], tmp: &mut Vec<E>, cnt: &mut Vec<u32>, span: i64) {
    let n = evs.len();
    if n < 256 {
        evs.sort_unstable_by_key(|e| (dg(e.key()), k128(e)));
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
    match (a.key, a.ts).cmp(&(b.key, b.ts)) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => unpack(a.cid) < unpack(b.cid),
    }
}

/// Sort clicks by (dg(key), key, ts, click_id bytes): a counting pass on the digit, then a small
/// sort per range.
fn sort_clicks(cl: &mut Vec<ClickRec>, tmp: &mut Vec<ClickRec>, cnt: &mut Vec<u32>) {
    let n = cl.len();
    if n < 256 {
        cl.sort_unstable_by(|a, b| (dg(a.key), a.key, a.ts).cmp(&(dg(b.key), b.key, b.ts)).then_with(|| unpack(a.cid).cmp(unpack(b.cid))));
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
                o.sort_unstable_by(|a, b| (a.key, a.ts).cmp(&(b.key, b.ts)).then_with(|| unpack(a.cid).cmp(unpack(b.cid))));
            }
        }
        st = en;
    }
}

/// A long digit range is nearly always one heavy clicker. When every click in `src` has the same key,
/// sort (ts, index) pairs packed into u64s, gather into `out`, then order each run of equal ts by
/// click_id bytes. Returns false (and leaves `out` alone) for a range with several keys.
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
    let mut a = 0;
    while a < m {
        let mut b = a + 1;
        while b < m && out[b].ts == out[a].ts {
            b += 1;
        }
        if b - a > 1 {
            out[a..b].sort_unstable_by(|x, y| unpack(x.cid).cmp(unpack(y.cid)));
        }
        a = b;
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn process_bucket<E: Ev>(
    evs: &mut [E],
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
    sort_evs(evs, tmp, cnt, span);
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
                    let t = cs[hi - 1].ts;
                    let mut g = hi - 1;
                    while g > lo && cs[g - 1].ts == t {
                        g -= 1;
                    }
                    cs[g].camp
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
// one row group per at most 2^20 rows, zstd pages (SHOP_CODEC=none stores them uncompressed).

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
}

fn cm(ty: i32, name: &'static str) -> Cm {
    Cm { ty, name, encs: vec![0, 3], nv: 0, unc: 0, comp: 0, dpo: 0, dico: None }
}

struct Pqf {
    out: Vec<u8>,
    z: Option<zstd::bulk::Compressor<'static>>,
    zb: Vec<u8>,
    codec: i32,
}
impl Pqf {
    fn new(cap: usize) -> Self {
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
            Some(z) => {
                self.zb.clear();
                self.zb.reserve(zstd::zstd_safe::compress_bound(body.len()));
                z.compress_to_buffer(body, &mut self.zb).unwrap();
                &self.zb
            }
            None => body,
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
fn ud_row_group(f: &mut Pqf, rows: &[&[UD]], n: usize, cats: &[B], names: &Names) -> Vec<Cm> {
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
        cm(2, "revenue_cents"),
        cm(6, "top_category"),
    ];
    // user_id: dictionary of the persons in row order, indices as one RLE run per person
    {
        let w = bwid(persons.saturating_sub(1) as u64);
        let vb = ((w + 7) / 8) as usize;
        let mut dict: Vec<u8> = Vec::with_capacity(persons * 14);
        let mut idx: Vec<u8> = Vec::with_capacity(1 + persons * (3 + vb));
        idx.push(w as u8);
        let mut tmp: Vec<u8> = Vec::with_capacity(16);
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
                        tmp.clear();
                        decode6(r.key, &mut tmp);
                        dict.extend_from_slice(&(tmp.len() as u32).to_le_bytes());
                        dict.extend_from_slice(&tmp);
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
    int_col(f, &mut cols[1], rows, n, |r| r.day as i64, dlo, dhi);
    int_col(f, &mut cols[2], rows, n, |r| r.sessions as i64, slo as i64, shi as i64);
    int_col(f, &mut cols[3], rows, n, |r| r.events as i64, elo as i64, ehi as i64);
    // revenue: PLAIN (wide range, mostly zeros)
    {
        let mut b: Vec<u8> = Vec::with_capacity(n * 8);
        for g in rows {
            for r in g.iter() {
                b.extend_from_slice(&(r.rev as i64).to_le_bytes());
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
    cols
}

fn write_user_daily(path: &Path, groups: &[&Vec<UD>], cats: &[B], names: &Names) {
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
                rgs.push((ud_row_group(&mut f, &cur, n, cats, names), n));
                cur.clear();
                n = 0;
            }
        }
    }
    if n > 0 {
        rgs.push((ud_row_group(&mut f, &cur, n, cats, names), n));
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
        (2, 0, "revenue_cents", -1, 0),
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
            t.i32(4, f.codec);
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
    std::fs::write(path, &f.out).unwrap();
    pt(12, &mut tm);
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
fn parse_events<E: Ev>(c: B, cx: &EvCtx, m: &mut Vec<E>, d: &mut Vec<DRec>, fb: &mut Vec<(u64, i64, u32)>, fbn: &mut Vec<(u64, u64)>, side: &mut Vec<(u32, i64)>) {
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
                None => fb.push((pack(eid), ingest, loc)),
            }
        }
    });
}

/// The body of `parse_events` for one line, as a plain function (the slow lines of
/// `parse_events_sol` go through it).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn ev_line<E: Ev>(f: &[B; 7], cx: &EvCtx, c0: usize, tci: &mut TsCache, tct: &mut TsCache, m: &mut Vec<E>, d: &mut Vec<DRec>, fb: &mut Vec<(u64, i64, u32)>, fbn: &mut Vec<(u64, u64)>, side: &mut Vec<(u32, i64)>) {
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
    let loc = m.len() as u32;
    m.push(r);
    let (ik, exact) = ing32(ingest, cx.ibase);
    if !exact {
        side.push((loc, ingest));
    }
    match pack6(eid) {
        Some(k) => d.push(DRec { eid: k, ing: ik, loc }),
        None => fb.push((pack(eid), ingest, loc)),
    }
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

/// `ts_iso` of the 20 bytes "YYYY-MM-DDTHH:MM:SSZ" at `p`, in whole seconds (the caller checks
/// the length and byte 19).
#[inline(always)]
unsafe fn iso_secs(p: *const u8, c: &mut TsCache) -> Option<i64> {
    let b = std::slice::from_raw_parts(p, 20);
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
    debug_assert!(hi < c.ents.len());
    // SAFETY: the shift leaves 64 - sh bits and ents has 1 << (64 - sh) entries
    let ent = c.ents.get_unchecked_mut(hi);
    if ent.p8 == p8 && ent.tag2 == tag2 {
        days = ent.days as i64;
    } else {
        days = ts_days(b)?;
        if days == days as i32 as i64 {
            *ent = TsEnt { p8, tag2, days: days as i32 };
        }
    }
    Some(days * 86_400 + (h * 60 + mi) * 60 + s)
}

/// Same records, dedup entries and side lists, in the same order, as `parse_events`, except
/// that the category of a record with no price is 0 (no product lookup for views and carts;
/// `Ev::pc` never reads it). One fused pass: the `scan7` line classifier with the fields kept
/// as pointer and length, both ids packed in one vector, ISO times straight to seconds.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(never)]
fn parse_events_sol<E: Ev>(c: B, cx: &EvCtx, m: &mut Vec<E>, d: &mut Vec<DRec>, fb: &mut Vec<(u64, i64, u32)>, fbn: &mut Vec<(u64, u64)>, side: &mut Vec<(u32, i64)>) {
    use core::arch::x86_64::_bzhi_u64;
    let mut tci = TsCache::new();
    let mut tct = TsCache::new();
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
    'ln: while pos + 128 <= len {
        // SAFETY: pos + 128 <= len, and the second window starts at most 64 bytes in
        let p = unsafe { base.add(pos) };
        let (ca, qa, na) = masks64(p);
        let e0 = ca.trailing_zeros();
        let c1 = blsr(ca);
        let e1 = c1.trailing_zeros();
        let c2 = blsr(c1);
        let e2 = c2.trailing_zeros();
        'slow: {
            if !((c2 != 0) & (bzhi(na, e2) == 0)) {
                break 'slow;
            }
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
            if !((nb != 0) & (e5 < l) & (bzhi(d3, l) == 0) & ((bad_a | bad_b) == 0)) {
                break 'slow;
            }
            pos += (e2 + l + 2) as usize;
            let fld = |p: *const u8, x: u64, s: u32, e: u32| {
                let q = ((x >> s) & 1) as u32;
                // SAFETY: s <= e <= 63 inside a window; a quoted field has e - s >= 2
                (unsafe { p.add((s + q) as usize) }, (e - s - 2 * q) as usize)
            };
            let (pe, ne) = fld(p, xa, 0, e0);
            if ne == 0 {
                continue 'ln;
            }
            let (pu, nu) = fld(p, xa, e0 + 1, e1);
            let (pt, nt) = fld(p, xa, e1 + 1, e2);
            let (pi, ni) = fld(pb, xb, 0, e3);
            let (pet, net) = fld(pb, xb, e3 + 1, e4);
            let (pp, np) = fld(pb, xb, e4 + 1, e5);
            let (ppr, npr) = fld(pb, xb, e5 + 1, l);
            // SAFETY: both ids start in the first window, so 16 bytes from each stay inside
            // the 128 readable bytes at p
            let (ke, oke, ku, oku) = unsafe { pack6x2(pe, ne, pu, nu) };
            // SAFETY (iso_secs): a 20-byte field lies inside its window
            let iso_i = if ni == 20 && unsafe { *pi.add(19) } == b'Z' { Some(unsafe { iso_secs(pi, &mut tci) }) } else { None };
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
                    match unsafe { iso_secs(pt, &mut tct) } {
                        Some(s) => s * 1000,
                        None => break 'v,
                    }
                } else if nt == 13 {
                    match ts_num(sl(pt, 13)) {
                        Some(t) => t,
                        None => break 'v,
                    }
                } else {
                    match parse_ts_c(sl(pt, nt), &mut tct) {
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
            let loc = m.len() as u32;
            m.push(r);
            if !iex {
                side.push((loc, ingest));
            }
            if oke {
                d.push(DRec { eid: ke, ing: ik, loc });
            } else {
                fb.push((pack(sl(pe, ne)), ingest, loc));
            }
            continue 'ln;
        }
        sbuf.clear();
        pos = scan7_line(c, pos, &mut sbuf);
        for f in &sbuf {
            ev_line(f, cx, c0, &mut tci, &mut tct, m, d, fb, fbn, side);
        }
    }
    if pos < len {
        scan::<7>(&c[pos..], |f| ev_line(f, cx, c0, &mut tci, &mut tct, m, d, fb, fbn, side));
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
fn parse_events_sol<E: Ev>(c: B, cx: &EvCtx, m: &mut Vec<E>, d: &mut Vec<DRec>, fb: &mut Vec<(u64, i64, u32)>, fbn: &mut Vec<(u64, u64)>, side: &mut Vec<(u32, i64)>) {
    let mut tci = TsCache::new();
    let mut tct = TsCache::new();
    let c0 = c.as_ptr() as usize;
    scan::<7>(c, |f| ev_line(f, cx, c0, &mut tci, &mut tct, m, d, fb, fbn, side));
}

/// Output buffers of one event-parse call.
struct EvBufs<E> {
    m: Vec<E>,
    d: Vec<DRec>,
    fb: Vec<(u64, i64, u32)>,
    fbn: Vec<(u64, u64)>,
    side: Vec<(u32, i64)>,
}
impl<E: Ev> EvBufs<E> {
    fn new() -> Self {
        EvBufs { m: Vec::with_capacity(CHUNK / 32), d: Vec::with_capacity(CHUNK / 32), fb: vec![], fbn: vec![], side: vec![] }
    }
    fn clear(&mut self) {
        self.m.clear();
        self.d.clear();
        self.fb.clear();
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
            return Some(format!("fb differs ({} vs {})", self.fb.len(), o.fb.len()));
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
                parse_events_sol(c, cx, &mut b.m, &mut b.d, &mut b.fb, &mut b.fbn, &mut b.side);
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
    fb: Vec<(u64, i64, u32)>,
    fbn: Vec<(u64, u64)>,
    side: Vec<(u32, i64)>,
}

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct DLine([u64; 8]);

// (slots whose event_id equals k, free slots) of a dedup table line, as 4-bit masks
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn line4(l: &DLine, k: u64) -> (u32, u32) {
    use core::arch::x86_64::*;
    unsafe {
        let v = _mm256_load_si256(l.0.as_ptr() as *const __m256i);
        let eq = _mm256_cmpeq_epi64(v, _mm256_set1_epi64x(k as i64));
        let z = _mm256_cmpeq_epi64(v, _mm256_setzero_si256());
        (_mm256_movemask_pd(_mm256_castsi256_pd(eq)) as u32, _mm256_movemask_pd(_mm256_castsi256_pd(z)) as u32)
    }
}
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
fn line4(l: &DLine, k: u64) -> (u32, u32) {
    let mut eq = 0;
    let mut z = 0;
    for j in 0..4 {
        eq |= ((l.0[j] == k) as u32) << j;
        z |= ((l.0[j] == 0) as u32) << j;
    }
    (eq, z)
}

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
        rayon::join(
            || {
                (0..EB / DDG).into_par_iter().for_each(|g| {
                    let mut tm = now();
                    let totals: [usize; DDG] =
                        std::array::from_fn(|i| ev_out.iter().map(|c| (c.doff[g * DDG + i + 1] - c.doff[g * DDG + i]) as usize).sum());
                    let maxl = (totals.iter().max().unwrap() / 2).max(4).next_power_of_two();
                    // 64-byte lines of 4 slots: event_ids, then (ingest key << 32 | global index of the
                    // current winner). A line fills in slot order, so a key is absent once a line has a
                    // free slot; one line holds the probe almost always (load ~45%).
                    let mut tbuf: Vec<DLine> = vec![DLine([0; 8]); maxl];
                    for (i, &total) in totals.iter().enumerate() {
                        let b = g * DDG + i;
                        let nl = (total / 2).max(4).next_power_of_two();
                        let lmask = nl - 1;
                        let t = &mut tbuf[..nl];
                        if i > 0 {
                            t.fill(DLine([0; 8]));
                        }
                        for (ci, c) in ev_out.iter().enumerate() {
                            let mb = mbase[ci];
                            // the next chunk's segment starts at an unpredictable address: fetch its head early
                            if let Some(nx) = ev_out.get(ci + 1) {
                                let s = nx.doff[b] as usize;
                                prefetch(nx.d.as_ptr().wrapping_add(s));
                                prefetch(nx.d.as_ptr().wrapping_add(s + 4));
                            }
                            for r in &c.d[c.doff[b] as usize..c.doff[b + 1] as usize] {
                                let gi = mb + r.loc;
                                let pk = ((r.ing as u64) << 32) | gi as u64;
                                let mut p = (mix(r.eid) >> 8) as usize & lmask;
                                loop {
                                    let l = unsafe { t.get_unchecked_mut(p) };
                                    let (meq, mz) = line4(l, r.eid);
                                    if meq != 0 {
                                        let j = meq.trailing_zeros() as usize + 4;
                                        let ok = (l.0[j] >> 32) as u32;
                                        let og = l.0[j] as u32;
                                        let win = r.ing < ok || (r.ing == ok && side_any && exact(gi, r.ing) < exact(og, ok));
                                        if win {
                                            l.0[j] = pk;
                                            kill(og);
                                        } else {
                                            kill(gi);
                                        }
                                        break;
                                    }
                                    if mz != 0 {
                                        let j = mz.trailing_zeros() as usize;
                                        l.0[j] = r.eid;
                                        l.0[j + 4] = pk;
                                        break;
                                    }
                                    p = (p + 1) & lmask;
                                }
                            }
                        }
                    }
                    pt(2, &mut tm);
                })
            },
            || {
                // events whose event_id has no packed key: exact byte comparison
                let mut tm = now();
                let mut m: AHashMap<B, (i64, u32)> = AHashMap::new();
                for (ci, c) in ev_out.iter().enumerate() {
                    for &(e, ing, loc) in &c.fb {
                        let gi = mbase[ci] + loc;
                        match m.get_mut(unpack(e)) {
                            Some(o) => {
                                if ing < o.0 {
                                    kill(o.1);
                                    *o = (ing, gi);
                                } else {
                                    kill(gi);
                                }
                            }
                            None => {
                                m.insert(unpack(e), (ing, gi));
                            }
                        }
                    }
                }
                pt(3, &mut tm);
            },
        );
    }
    // kept lines: keys that miss the mover filter map to themselves and are grouped by person
    // bucket inside their chunk (a cache-sized scatter). Keys that hit it (candidate movers) are
    // grouped by id table slice (the top bits of their table slot) instead; each slice is then
    // resolved on its own, so the slice of the table it probes stays cache resident, and the
    // resolved lines are grouped by person bucket into extra parts. The sessionizer gathers a
    // bucket from every part; the order inside a bucket does not matter.
    let sl = |k: u64| (mix(k) >> (64 - RSB)) as usize;
    let pass: Vec<((Vec<E>, Vec<u32>), (Vec<E>, Vec<u32>))> = ev_out
        .into_par_iter()
        .enumerate()
        .map(|(ci, c)| {
            let mut tm = now();
            let mb = mbase[ci] as usize;
            let m = c.m;
            let n = m.len();
            // 0 dropped, 1 own key, 2 candidate mover
            let mut cls = vec![0u8; n];
            let mut cnt = vec![0u32; PB + 1];
            let mut ccnt = vec![0u32; RNS + 1];
            for i in 0..n {
                if i + 2 * RPD < n {
                    mf.pf(m[i + 2 * RPD].key());
                }
                let g = mb + i;
                let k = m[i].key();
                if k != 0 && dead[g >> 6].load(Relaxed) & (1u64 << (g & 63)) == 0 {
                    if mf.hit(k) {
                        cls[i] = 2;
                        ccnt[sl(k) + 1] += 1;
                    } else {
                        cls[i] = 1;
                        cnt[pbk(k) + 1] += 1;
                    }
                }
            }
            pt(4, &mut tm);
            for p in 0..PB {
                cnt[p + 1] += cnt[p];
            }
            for p in 0..RNS {
                ccnt[p + 1] += ccnt[p];
            }
            let tot = cnt[PB] as usize;
            let ctot = ccnt[RNS] as usize;
            let mut out: Vec<E> = Vec::with_capacity(tot);
            let mut cand: Vec<E> = Vec::with_capacity(ctot);
            {
                let mut w = cnt.clone();
                let mut cw = ccnt.clone();
                let sp = out.spare_capacity_mut();
                let cp = cand.spare_capacity_mut();
                for (e, &t) in m.iter().zip(cls.iter()) {
                    if t == 1 {
                        let p = pbk(e.key());
                        sp[w[p] as usize].write(*e);
                        w[p] += 1;
                    } else if t == 2 {
                        let p = sl(e.key());
                        cp[cw[p] as usize].write(*e);
                        cw[p] += 1;
                    }
                }
            }
            unsafe {
                out.set_len(tot);
                cand.set_len(ctot);
            }
            pt(5, &mut tm);
            ((out, cnt), (cand, ccnt))
        })
        .collect();
    drop(dead);
    let (mut parts, cands): (Vec<(Vec<E>, Vec<u32>)>, Vec<(Vec<E>, Vec<u32>)>) = pass.into_iter().unzip();
    let ng = (rayon::current_num_threads() * 8).clamp(1, RNS);
    let per = RNS.div_ceil(ng);
    let mparts: Vec<(Vec<E>, Vec<u32>)> = (0..RNS.div_ceil(per))
        .into_par_iter()
        .map(|gi| {
            let mut tm = now();
            let (s0, s1) = (gi * per, ((gi + 1) * per).min(RNS));
            let tot: usize = cands.iter().map(|(_, o)| (o[s1] - o[s0]) as usize).sum();
            let mut gl: Vec<E> = Vec::with_capacity(tot);
            let mut cnt = vec![0u32; PB + 1];
            for s in s0..s1 {
                let a = gl.len();
                for (v, o) in cands.iter() {
                    gl.extend_from_slice(&v[o[s] as usize..o[s + 1] as usize]);
                }
                let b = gl.len();
                for i in a..b {
                    if i + PFD < b {
                        tab.pf(gl[i + PFD].key());
                    }
                    let pk = tab.get(gl[i].key());
                    gl[i].set_key(pk);
                    cnt[pbk(pk) + 1] += 1;
                }
            }
            pt(4, &mut tm);
            for p in 0..PB {
                cnt[p + 1] += cnt[p];
            }
            let mut out: Vec<E> = Vec::with_capacity(tot);
            {
                let mut w = cnt.clone();
                let sp = out.spare_capacity_mut();
                for e in &gl {
                    let p = pbk(e.key());
                    sp[w[p] as usize].write(*e);
                    w[p] += 1;
                }
            }
            unsafe { out.set_len(tot) };
            pt(5, &mut tm);
            (out, cnt)
        })
        .collect();
    drop(cands);
    parts.extend(mparts);
    tick("dedup done");

    let ngroups = (rayon::current_num_threads() * 4).clamp(1, PB);
    let per = PB.div_ceil(ngroups);
    (0..ngroups)
        .into_par_iter()
        .map(|g| {
            let mut cm = CampAcc::new(ncamp + 1, d0, nd);
            let mut win = Win { cnt: vec![0; ncamp + 1], pos: vec![0; ncamp + 1], active: vec![] };
            let mut out = vec![];
            let mut evs: Vec<E> = vec![];
            let mut tmp: Vec<E> = vec![];
            let mut cnt: Vec<u32> = vec![];
            let mut cl: Vec<ClickRec> = vec![];
            let mut ctmp: Vec<ClickRec> = vec![];
            for p in g * per..((g + 1) * per).min(PB) {
                let mut tm = now();
                cl.clear();
                for (v, o) in click_out.iter() {
                    cl.extend_from_slice(&v[o[p] as usize..o[p + 1] as usize]);
                }
                evs.clear();
                for (v, o) in parts.iter() {
                    evs.extend_from_slice(&v[o[p] as usize..o[p + 1] as usize]);
                }
                pt(6, &mut tm);
                let mut uds = Vec::new();
                process_bucket(&mut evs, &mut tmp, &mut cnt, &mut cl, &mut ctmp, &mut win, ncamp as u32, &mut uds, &mut cm, ws, we.wrapping_sub(ws));
                out.push(uds);
            }
            (out, cm)
        })
        .collect()
}

/// Parses one chunk and groups its dedup records by event_id bucket.
fn ev_chunk_parse<E: Ev>(c: B, cx: &EvCtx) -> EvChunk<E> {
    let mut m: Vec<E> = Vec::with_capacity(c.len() / 64);
    let mut d: Vec<DRec> = Vec::with_capacity(c.len() / 64);
    let (mut fb, mut fbn, mut side) = (vec![], vec![], vec![]);
    let mut tm = now();
    static OLD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OLD.get_or_init(|| std::env::var_os("SHOP_EVOLD").is_some()) {
        parse_events(c, cx, &mut m, &mut d, &mut fb, &mut fbn, &mut side);
    } else {
        parse_events_sol(c, cx, &mut m, &mut d, &mut fb, &mut fbn, &mut side);
    }
    pt(0, &mut tm);
    let (d, doff) = bucketize(&d, EB, |r| (mix(r.eid) >> (64 - EBB)) as usize);
    pt(1, &mut tm);
    EvChunk { m, d, doff, fb, fbn, side }
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
    for x in &r.fb {
        add(x.0 ^ (x.1 as u64) ^ ((x.2 as u64) << 40));
    }
    for x in &r.fbn {
        add(x.0 ^ x.1);
    }
    for x in &r.side {
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

    tick("prod/camp loaded");
    let rs = hasher();

    let link_chunks = all_chunks(&csv_files(&data.join("links")), CHUNK);
    let link_pairs: Vec<Vec<[u64; 4]>> = link_chunks
        .par_iter()
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
    // links (serial union-find) overlaps the parallel event parse; events do not need the id table
    let ((tab, mut names_map), ev_out) = rayon::join(
        || {
            let occ: usize = link_pairs.iter().map(|v| v.len()).sum::<usize>() * 2;
            let mut names_map: AHashMap<u64, u64> = AHashMap::new();
            let tab = {
                let mut t = IdTab::with_cap(occ);
                let mut idx_key: Vec<u64> = vec![];
                let mut idx_ref: Vec<u64> = vec![];
                let mut parent: Vec<u32> = vec![];
                fn find(p: &mut [u32], mut x: u32) -> u32 {
                    while p[x as usize] != x {
                        let g = p[p[x as usize] as usize];
                        p[x as usize] = g;
                        x = g;
                    }
                    x
                }
                let mut edges: Vec<(u32, u32)> = Vec::with_capacity(occ / 2);
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
                    }
                }
                tick(&format!("links inserted {}", idx_key.len()));
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
                // the insert table becomes the id table in place: a slot's index turns into its person key
                {
                    let (root, bk) = (&root[..], &bk[..]);
                    t.s.par_chunks_mut(1 << 14).for_each(|c| {
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
                t.mov = nl - nroot;
                tick(&format!("links table movers {} pairs {}", t.mov, occ / 2));
                t
            };
            drop(link_pairs);
            tick("links done");
            (tab, names_map)
        },
        || {
            let ev_files = csv_files(&data.join("events"));
            let ev_chunks = all_chunks(&ev_files, ev_chunk(&ev_files));
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

    // clicks
    let click_chunks = all_chunks(&csv_files(&data.join("clicks")), CHUNK);
    let mut tm = now();
    let mf = MoverF::build(&tab);
    pt(4, &mut tm);
    let click_out: Vec<(Vec<ClickRec>, Vec<u32>)> = click_chunks
        .par_iter()
        .map(|&c| {
            let mut tmp: Vec<ClickRec> = Vec::with_capacity(c.len() / 40);
            let mut tc = TsCache::new();
            let mut tm = now();
            scan::<4>(c, |&[cid, uid, cmp, ts]| {
                if cid.is_empty() || uid.is_empty() {
                    return;
                }
                let Some(ci) = camp.get(cmp) else { return };
                let Some(t) = parse_ts_c(ts, &mut tc) else { return };
                if t < ws - WEEK || t >= we {
                    return;
                }
                tmp.push(ClickRec { key: ukey(&rs, uid), ts: t, cid: pack(cid), camp: ci });
            });
            pt(9, &mut tm);
            // keys that miss the mover filter map to themselves without a table lookup
            let n = tmp.len();
            for i in 0..n {
                if i + 2 * RPD < n {
                    mf.pf(tmp[i + 2 * RPD].key);
                }
                if i + RPD < n {
                    let k = tmp[i + RPD].key;
                    if mf.hit(k) {
                        tab.pf(k);
                    }
                }
                let k = tmp[i].key;
                if mf.hit(k) {
                    tmp[i].key = tab.get(k);
                }
            }
            let r = bucketize(&tmp, PB, |x| pbk(x.key));
            pt(10, &mut tm);
            r
        })
        .collect();
    tick("clicks parsed");
    match &ev_out {
        Evs::L(v) => fbn_merge(v, &mut names_map),
        Evs::W(v) => fbn_merge(v, &mut names_map),
    }

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
    rayon::join(
        || {
            files.par_iter().enumerate().filter(|(i, g)| g.iter().any(|x| !x.is_empty()) || (*i == 0 && !any_rows)).for_each(|(i, g)| {
                write_user_daily(&ud_dir.join(format!("part-{i}.parquet")), g, &cat_names, &names);
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

    /// old and new parse of one chunk; the record count, or the first difference
    fn sol_cmp<E: Ev>(c: B, cx: &EvCtx) -> Result<(usize, usize), String> {
        let (mut a, mut b) = (EvBufs::<E>::new(), EvBufs::<E>::new());
        parse_events(c, cx, &mut a.m, &mut a.d, &mut a.fb, &mut a.fbn, &mut a.side);
        parse_events_sol(c, cx, &mut b.m, &mut b.d, &mut b.fb, &mut b.fbn, &mut b.side);
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
            let n = if r.chance(10) { r.below(14) } else { 8 + r.below(2) };
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
}
