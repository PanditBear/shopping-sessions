use ahash::{AHashMap, RandomState};
use memchr::memchr;
use rayon::prelude::*;
use std::fs::File;
use std::mem::MaybeUninit;
use std::path::Path;
use std::sync::Arc;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

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
static PROF: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
const PROF_NAMES: [&str; 12] =
    ["ev.scan", "ev.bucket", "dd.table", "dd.collect", "dd.resolve", "dd.bucket", "ss.gather", "ss.sort", "ss.proc", "ck.scan", "ck.res", "ss.csort"];
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
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
fn scan7(chunk: B, mut f: impl FnMut(&[B; 7])) {
    use core::arch::x86_64::_bzhi_u64;
    // SAFETY (bzhi): BMI2 is enabled for this build
    let bzhi = |x: u64, n: u32| unsafe { _bzhi_u64(x, n) };
    let blsr = |x: u64| x & x.wrapping_sub(1);
    let len = chunk.len();
    let base = chunk.as_ptr();
    let mut pos = 0usize;
    while pos + 128 <= len {
        // SAFETY: pos + 128 <= len, and the second window starts at most 64 bytes in
        let p = unsafe { base.add(pos) };
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
                f(&out);
                #[cfg(test)]
                SCAN7_FAST.with(|c| c.set(c.get() + 1));
                pos += (e2 + l + 2) as usize;
                continue;
            }
        }
        pos = scan7_line(chunk, pos, &mut f);
    }
    if pos < len {
        scan::<7>(&chunk[pos..], f);
    }
}

#[cfg(test)]
thread_local! {
    static SCAN7_FAST: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The slow path of `scan7`: the one line at `pos` through `scan`; returns the next line start.
#[cold]
#[inline(never)]
fn scan7_line<F: FnMut(&[B; 7])>(chunk: B, pos: usize, f: &mut F) -> usize {
    let end = match memchr(b'\n', &chunk[pos..]) {
        Some(i) => pos + i + 1,
        None => chunk.len(),
    };
    scan::<7>(&chunk[pos..end], |l| f(l));
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
}
impl IdTab {
    fn with_cap(n: usize) -> Self {
        let cap = (n + n / 2).max(16).next_power_of_two();
        IdTab { s: vec![[0u64; 2]; cap], shift: 64 - cap.trailing_zeros(), mask: cap - 1 }
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

const PFD: usize = 12;

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

/// (campaign, day) accumulator: dense when the window is small, a map otherwise.
struct CampAcc {
    dense: Vec<[u64; 4]>,
    d0: i64,
    nd: usize,
    map: CampMap,
}
impl CampAcc {
    fn new(ncamp1: usize, d0: i64, nd: usize) -> Self {
        let dense = if nd > 0 && nd * ncamp1 <= (1 << 22) { vec![[0u64; 4]; nd * ncamp1] } else { vec![] };
        CampAcc { dense, d0, nd, map: CampMap::new() }
    }
    #[inline(always)]
    fn at(&mut self, c: u32, d: i64) -> &mut [u64; 4] {
        let di = d - self.d0;
        if !self.dense.is_empty() && di >= 0 && (di as usize) < self.nd {
            &mut self.dense[c as usize * self.nd + di as usize]
        } else {
            self.map.entry(((c as u64) << 32) | (d as u32 as u64)).or_insert([0; 4])
        }
    }
    fn merge(&mut self, o: &CampAcc) {
        for (a, b) in self.dense.iter_mut().zip(o.dense.iter()) {
            for i in 0..4 {
                a[i] += b[i];
            }
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
        if !self.dense.is_empty() {
            for (i, v) in self.dense.iter().enumerate() {
                if v[0] != 0 || v[3] != 0 {
                    rows.push(((i / self.nd) as u32, (self.d0 + (i % self.nd) as i64) as i32, *v));
                }
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
/// every rts. A counting pass on the digit, then per
/// range: when the range holds one key, its (ts, index) pairs are sorted as u64s and gathered back.
fn sort_evs<E: Ev>(evs: &mut [E], tmp: &mut Vec<E>, cnt: &mut Vec<u32>, span: i64) {
    let n = evs.len();
    if n < 256 {
        evs.sort_unstable_by_key(|e| (dg(e.key()), k128(e)));
        return;
    }
    hk_scatter(evs, tmp, cnt, |e| e.key());
    let fast = span >= 0 && span < (1i64 << 55);
    let mut ks = [0u64; 256];
    let mut st = 0usize;
    for d in 0..1usize << NBK {
        let en = cnt[d] as usize;
        let m = en - st;
        let r = &tmp[st..en];
        if m == 1 {
            evs[st] = r[0];
        } else if m > 1 {
            let k0 = r[0].key();
            let mut same = fast && m <= 256;
            if same {
                for (i, e) in r.iter().enumerate() {
                    ks[i] = (e.rts() << 8) | i as u64;
                    same &= e.key() == k0;
                }
            }
            if same {
                let ks = &mut ks[..m];
                if m <= 32 {
                    isort_by(ks, |a, b| a < b);
                } else {
                    ks.sort_unstable();
                }
                for (j, &k) in ks.iter().enumerate() {
                    evs[st + j] = r[(k & 0xFF) as usize];
                }
            } else {
                evs[st..en].copy_from_slice(r);
                sort_small(&mut evs[st..en]);
            }
        }
        st = en;
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
    let mut st = 0usize;
    for d in 0..1usize << NBK {
        let en = cnt[d] as usize;
        let m = en - st;
        if m == 1 {
            cl[st] = tmp[st];
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
                    let en = cm.at(s_camp, s_day);
                    en[0] += 1;
                    en[1] += s_purch as u64;
                    en[2] += s_rev;
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
                for &c in &win.active {
                    cm.at(c, d)[3] += 1;
                }
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
            let en = cm.at(s_camp, s_day);
            en[0] += 1;
            en[1] += s_purch as u64;
            en[2] += s_rev;
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
use parquet::schema::types::ColumnPath;

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
    // Dictionary encoding on the low-cardinality columns makes the output about 2.5x smaller for no
    // measurable write time, and fewer bytes means less writeback when the container exits. user_id stays
    // plain: its dictionary (or a delta encoding) costs more write time than the smaller output saves.
    // A dictionary that outgrows the default page limit falls back to plain on its own.
    Arc::new(
        WriterProperties::builder()
            .set_compression(codec())
            .set_dictionary_enabled(true)
            .set_column_dictionary_enabled(ColumnPath::from("user_id"), false)
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
impl Names {
    fn get(&self, key: u64) -> ByteArray {
        if key & HBIT == 0 {
            let mut v = Vec::with_capacity(10);
            decode6(key, &mut v);
            ByteArray::from(v)
        } else {
            ba(unpack(*self.map.get(&key).expect("name for hashed key")))
        }
    }
}

fn write_user_daily(path: &Path, groups: &[&Vec<UD>], cats: &[B], names: &Names) {
    let schema = Arc::new(
        parse_message_type(
            "message user_daily { required binary user_id (STRING); required int32 day (DATE); required int64 sessions; required int64 events; required int64 revenue_cents; optional binary top_category (STRING); }",
        )
        .unwrap(),
    );
    let f = File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(f, schema, props()).unwrap();
    const RG: usize = 1 << 20;
    let mut buf: Vec<&UD> = Vec::with_capacity(RG);
    let mut nrg = 0usize;
    let flush = |buf: &mut Vec<&UD>, w: &mut SerializedFileWriter<File>, force: bool, nrg: &mut usize| {
        if buf.is_empty() && !force {
            return;
        }
        *nrg += 1;
        let mut rg = w.next_row_group().unwrap();
        // rows of one person are adjacent: share the name's bytes instead of decoding it again
        let mut users: Vec<ByteArray> = Vec::with_capacity(buf.len());
        let mut last: Option<u64> = None;
        for r in buf.iter() {
            if last == Some(r.key) {
                let v = users[users.len() - 1].clone();
                users.push(v);
            } else {
                users.push(names.get(r.key));
                last = Some(r.key);
            }
        }
        let mut c = rg.next_column().unwrap().unwrap();
        c.typed::<ByteArrayType>().write_batch(&users, None, None).unwrap();
        c.close().unwrap();
        let v: Vec<i32> = buf.iter().map(|r| r.day).collect();
        let mut c = rg.next_column().unwrap().unwrap();
        c.typed::<Int32Type>().write_batch(&v, None, None).unwrap();
        c.close().unwrap();
        let v: Vec<i64> = buf.iter().map(|r| r.sessions as i64).collect();
        let mut c = rg.next_column().unwrap().unwrap();
        c.typed::<Int64Type>().write_batch(&v, None, None).unwrap();
        c.close().unwrap();
        let v: Vec<i64> = buf.iter().map(|r| r.events as i64).collect();
        let mut c = rg.next_column().unwrap().unwrap();
        c.typed::<Int64Type>().write_batch(&v, None, None).unwrap();
        c.close().unwrap();
        let v: Vec<i64> = buf.iter().map(|r| r.rev as i64).collect();
        let mut c = rg.next_column().unwrap().unwrap();
        c.typed::<Int64Type>().write_batch(&v, None, None).unwrap();
        c.close().unwrap();
        let mut defs: Vec<i16> = Vec::with_capacity(buf.len());
        let mut tv: Vec<ByteArray> = vec![];
        for r in buf.iter() {
            if r.top == u32::MAX {
                defs.push(0);
            } else {
                defs.push(1);
                tv.push(ba(cats[r.top as usize]));
            }
        }
        let mut c = rg.next_column().unwrap().unwrap();
        c.typed::<ByteArrayType>().write_batch(&tv, Some(&defs), None).unwrap();
        c.close().unwrap();
        rg.close().unwrap();
        buf.clear();
    };
    for g in groups {
        for r in g.iter() {
            buf.push(r);
            if buf.len() >= RG {
                flush(&mut buf, &mut w, false, &mut nrg);
            }
        }
    }
    flush(&mut buf, &mut w, nrg == 0, &mut nrg);
    w.close().unwrap();
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

struct EvChunk<E> {
    m: Vec<E>,
    d: Vec<DRec>,
    doff: Vec<u32>,
    fb: Vec<(u64, i64, u32)>,
    fbn: Vec<(u64, u64)>,
    side: Vec<(u32, i64)>,
}

#[allow(clippy::too_many_arguments)]
fn ev_after<E: Ev>(
    mut ev_out: Vec<EvChunk<E>>,
    tab: &IdTab,
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
                    let maxcap = (totals.iter().max().unwrap() * 2).max(16).next_power_of_two();
                    // slot: (event_id, ingest key << 32 | global index of the current winner)
                    let mut tbuf: Vec<[u64; 2]> = vec![[0, 0]; maxcap];
                    for (i, &total) in totals.iter().enumerate() {
                        let b = g * DDG + i;
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
    // kept lines: user key -> person key (0 marks a dropped line), counted per (chunk, bucket)
    let mut cnts: Vec<Vec<u32>> = ev_out
        .par_iter_mut()
        .enumerate()
        .map(|(ci, c)| {
            let mut tm = now();
            let mut cnt = vec![0u32; PB];
            let mb = mbase[ci] as usize;
            let m = &mut c.m;
            let n = m.len();
            for i in 0..n {
                if i + PFD < n {
                    tab.pf(m[i + PFD].key());
                }
                let g = mb + i;
                let k = m[i].key();
                if k != 0 && dead[g >> 6].load(Relaxed) & (1u64 << (g & 63)) == 0 {
                    let pk = tab.get(k);
                    m[i].set_key(pk);
                    cnt[pbk(pk)] += 1;
                } else {
                    m[i].set_key(0);
                }
            }
            pt(4, &mut tm);
            cnt
        })
        .collect();
    drop(dead);
    // one scatter of every kept line into a person-bucket-major array
    let mut bstart = vec![0usize; PB + 1];
    for p in 0..PB {
        let mut a = bstart[p];
        for cnt in cnts.iter_mut() {
            let x = cnt[p] as usize;
            cnt[p] = a as u32;
            a += x;
        }
        bstart[p + 1] = a;
    }
    let tot = bstart[PB];
    let mut evg: Vec<E> = Vec::with_capacity(tot);
    {
        let gp = SPtr(evg.as_mut_ptr());
        ev_out.par_iter().zip(cnts.into_par_iter()).for_each(|(c, mut pos)| {
            let mut tm = now();
            for e in &c.m {
                if e.key() != 0 {
                    let p = pbk(e.key());
                    unsafe { gp.at(pos[p] as usize).write(*e) };
                    pos[p] += 1;
                }
            }
            pt(5, &mut tm);
        });
    }
    unsafe { evg.set_len(tot) };
    drop(ev_out);
    tick("dedup done");

    let ngroups = (rayon::current_num_threads() * 4).clamp(1, PB);
    let per = PB.div_ceil(ngroups);
    let mut gsl: Vec<(usize, &mut [E])> = Vec::with_capacity(ngroups);
    {
        let mut rest: &mut [E] = &mut evg;
        for g in 0..ngroups {
            let (a, b) = (bstart[(g * per).min(PB)], bstart[((g + 1) * per).min(PB)]);
            let (x, y) = rest.split_at_mut(b - a);
            gsl.push((g, x));
            rest = y;
        }
    }
    gsl
        .into_par_iter()
        .map(|(g, sl)| {
            let mut cm = CampAcc::new(ncamp + 1, d0, nd);
            let mut win = Win { cnt: vec![0; ncamp + 1], pos: vec![0; ncamp + 1], active: vec![] };
            let mut out = vec![];
            let mut tmp: Vec<E> = vec![];
            let mut cnt: Vec<u32> = vec![];
            let mut cl: Vec<ClickRec> = vec![];
            let mut ctmp: Vec<ClickRec> = vec![];
            let g0 = bstart[(g * per).min(PB)];
            for p in g * per..((g + 1) * per).min(PB) {
                let mut tm = now();
                cl.clear();
                for (v, o) in click_out.iter() {
                    cl.extend_from_slice(&v[o[p] as usize..o[p + 1] as usize]);
                }
                pt(6, &mut tm);
                let evs = &mut sl[bstart[p] - g0..bstart[p + 1] - g0];
                let mut uds = Vec::new();
                process_bucket(evs, &mut tmp, &mut cnt, &mut cl, &mut ctmp, &mut win, ncamp as u32, &mut uds, &mut cm, ws, we.wrapping_sub(ws));
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
    parse_events(c, cx, &mut m, &mut d, &mut fb, &mut fbn, &mut side);
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
                for (a, b) in edges {
                    let ra = find(&mut parent, a);
                    let rb = find(&mut parent, b);
                    if ra != rb {
                        parent[ra as usize] = rb;
                    }
                }
                let nl = idx_key.len();
                let mut best: Vec<u32> = (0..nl as u32).collect();
                for g in 0..nl {
                    let r = find(&mut parent, g as u32) as usize;
                    let (kg, kb) = (idx_key[g], idx_key[best[r] as usize]);
                    let less = if kg & HBIT == 0 && kb & HBIT == 0 { kg < kb } else { unpack(idx_ref[g]) < unpack(idx_ref[best[r] as usize]) };
                    if less {
                        best[r] = g as u32;
                    }
                }
                let mut ft = IdTab::with_cap(nl);
                for g in 0..nl {
                    let b = best[find(&mut parent, g as u32) as usize] as usize;
                    let pk = idx_key[b];
                    ft.get_or_insert(idx_key[g], pk);
                    if pk & HBIT != 0 && b == g {
                        names_map.insert(pk, idx_ref[b]);
                    }
                }
                ft
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
            let n = tmp.len();
            for i in 0..n {
                if i + PFD < n {
                    tab.pf(tmp[i + PFD].key);
                }
                tmp[i].key = tab.get(tmp[i].key);
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
        Evs::L(v) => ev_after(v, &tab, &click_out, ws, we, ncamp, d0, nd, &tick),
        Evs::W(v) => ev_after(v, &tab, &click_out, ws, we, ncamp, d0, nd, &tick),
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
        let v: Vec<String> = (0..12).map(|i| format!("{}={:.2}", PROF_NAMES[i], PROF[i].load(Relaxed) as f64 / 1e9)).collect();
        eprintln!("prof(cpu s): {}", v.join(" "));
    }
    if timing {
        eprintln!("epoch main={:.3} end={:.3}", ep0, ep());
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
}
