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
        let (k0, ok0) = code8(ld8(b, 0));
        let (k1, ok1) = code8(ld8(b, n - 8));
        if !(ok0 & ok1) {
            return None;
        }
        let t = 6 * (n - 8) as u32;
        let tail = k1 & ((1u64 << t) - 1);
        return Some((k0 << 12) | (tail << (12 - t)));
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
#[inline]
fn parse_ts_c(b: &[u8], c: &mut TsCache) -> Option<i64> {
    let n = b.len();
    if n == 0 {
        return None;
    }
    if (n == 20 && b[19] == b'Z') || (n == 25 && (b[19] == b'+' || b[19] == b'-')) {
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
            days = days_from_civil(y, mo, d);
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
        return Some(days * DAY + ((h * 60 + mi) * 60 + s) * 1000 - off);
    }
    // F3: digits only
    if (9..=16).contains(&n) {
        let sh = 8 * (16 - n) as u32;
        let hw = (ld8(b, 0) << sh) | (h8(0x30) & ((1u64 << sh) - 1));
        let lw = ld8(b, n - 8);
        if !(dig8(hw) & dig8(lw)) {
            return None;
        }
        return Some((parse8(hw) * 100_000_000 + parse8(lw)) as i64);
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

fn chunks(data: B) -> Vec<B> {
    let start = match memchr(b'\n', data) {
        Some(i) => i + 1,
        None => return vec![],
    };
    let mut v = vec![];
    let mut a = start;
    while a < data.len() {
        let mut b = (a + CHUNK).min(data.len());
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

fn all_chunks(files: &[B]) -> Vec<B> {
    files.iter().flat_map(|f| chunks(f)).collect()
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

const META_PURCH: u64 = 1 << 56;
const META_OK: u64 = 1 << 57;

#[derive(Clone, Copy)]
struct EvRec {
    eid: u64,
    ingest: i64,
    uid: u64,
    ts: i64,
    meta: u64,
}

#[derive(Clone, Copy)]
struct EvOut {
    key: u64,
    ts: i64,
    meta: u64,
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
fn k128(e: &EvOut) -> u128 {
    ((e.key as u128) << 64) | (e.ts as u64 ^ (1u64 << 63)) as u128
}

fn sort_small(v: &mut [EvOut]) {
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

/// Sort by (key, ts). One counting-sort pass on the key bits below the common prefix of the
/// packed keys (hashed keys, which sort after all packed keys, go to one extra last range),
/// then a small sort per range. Equal (key, ts) pairs may end up in any order.
fn sort_evs(evs: &mut Vec<EvOut>, tmp: &mut Vec<EvOut>, cnt: &mut Vec<u32>) {
    let n = evs.len();
    if n < 512 {
        evs.sort_unstable_by_key(k128);
        return;
    }
    let mut k0 = None;
    let mut kx = 0u64;
    for e in evs.iter() {
        if e.key & HBIT == 0 {
            let b = *k0.get_or_insert(e.key);
            kx |= e.key ^ b;
        }
    }
    let bits = 64 - kx.leading_zeros();
    let w = (usize::BITS - n.leading_zeros()).min(13).min(bits);
    let sh = bits - w;
    let mask = (1u64 << w) - 1;
    let nb = 1usize << w;
    let dig = |e: &EvOut| -> usize {
        if e.key & HBIT != 0 {
            nb
        } else {
            ((e.key >> sh) & mask) as usize
        }
    };
    cnt.clear();
    cnt.resize(nb + 2, 0);
    for e in evs.iter() {
        cnt[dig(e) + 1] += 1;
    }
    for i in 0..=nb {
        cnt[i + 1] += cnt[i];
    }
    tmp.clear();
    tmp.reserve(n);
    {
        let sp = tmp.spare_capacity_mut();
        for e in evs.iter() {
            let d = dig(e);
            sp[cnt[d] as usize].write(*e);
            cnt[d] += 1;
        }
    }
    unsafe { tmp.set_len(n) };
    // cnt[d] is now the end of range d
    let mut st = 0usize;
    for d in 0..=nb {
        let en = cnt[d] as usize;
        if en - st > 1 {
            sort_small(&mut tmp[st..en]);
        }
        st = en;
    }
    std::mem::swap(evs, tmp);
}

fn process_bucket(
    evs: &mut Vec<EvOut>,
    tmp: &mut Vec<EvOut>,
    cnt: &mut Vec<u32>,
    clicks: &mut Vec<ClickRec>,
    win: &mut Win,
    organic: u32,
    uds: &mut Vec<UD>,
    cm: &mut CampAcc,
) {
    let mut tm = now();
    sort_evs(evs, tmp, cnt);
    pt(7, &mut tm);
    clicks.sort_unstable_by(|a, b| (a.key, a.ts).cmp(&(b.key, b.ts)).then_with(|| unpack(a.cid).cmp(unpack(b.cid))));
    pt(11, &mut tm);
    let mut ci = 0;
    let mut i = 0;
    let mut day = DayAcc { key: 0, day: 0, sessions: 0, events: 0, rev: 0, cats: vec![] };
    while i < evs.len() {
        let pid = evs[i].key;
        let mut j = i;
        while j < evs.len() && evs[j].key == pid {
            j += 1;
        }
        while ci < clicks.len() && clicks[ci].key < pid {
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
            if first || e.ts - prev > GAP {
                if in_sess {
                    let en = cm.at(s_camp, s_day);
                    en[0] += 1;
                    en[1] += s_purch as u64;
                    en[2] += s_rev;
                }
                first = false;
                in_sess = true;
                let s = e.ts;
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
            prev = e.ts;
            day.events += 1;
            if e.meta & META_PURCH != 0 {
                let p = e.meta & 0xFF_FFFF;
                let ecat = ((e.meta >> 24) & 0xFFFF_FFFF) as u32;
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

fn props() -> Arc<WriterProperties> {
    Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .set_dictionary_enabled(false)
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
        let users: Vec<ByteArray> = buf.iter().map(|r| names.get(r.key)).collect();
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
        for c in chunks(f) {
            scan::<2>(c, |&[p, c]| prod_raw.push((p, c)));
        }
    }
    let mut cat_names: Vec<B> = prod_raw.iter().map(|x| x.1).collect();
    cat_names.push(b"unknown");
    cat_names.sort();
    cat_names.dedup();
    let unknown_cat = cat_names.binary_search(&(b"unknown" as B)).unwrap() as u32;
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
        for c in chunks(f) {
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

    let link_chunks = all_chunks(&csv_files(&data.join("links")));
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
    type EvChunk = (Vec<EvRec>, Vec<u32>, Vec<EvRec>, Vec<(u64, u64)>);
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
            let ev_chunks = all_chunks(&csv_files(&data.join("events")));
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
            let ev_out: Vec<EvChunk> = ev_chunks
                .par_iter()
                .map(|&c| {
                    let mut flat: Vec<EvRec> = Vec::with_capacity(c.len() / 64);
                    let mut fb: Vec<EvRec> = vec![];
                    let mut fbn: Vec<(u64, u64)> = vec![];
                    let mut tci = TsCache::new();
                    let mut tct = TsCache::new();
                    let mut tm = now();
                    scan::<7>(c, |&[eid, uid, ts, ing, et, pr, price]| {
                        if eid.is_empty() {
                            return;
                        }
                        let ingest = parse_ts_c(ing, &mut tci).unwrap_or(i64::MAX);
                        let mut r = EvRec { eid: 0, ingest, uid: 0, ts: 0, meta: 0 };
                        'v: {
                            if uid.is_empty() || pr.is_empty() {
                                break 'v;
                            }
                            let Some(t) = parse_ts_c(ts, &mut tct) else { break 'v };
                            if t < ws || t >= we {
                                break 'v;
                            }
                            let purch = match et {
                                b"view" | b"cart" => false,
                                b"purchase" => true,
                                _ => break 'v,
                            };
                            let mut pv = 0u64;
                            if purch {
                                match parse_price(price) {
                                    Some(p) => pv = p as u64,
                                    None => break 'v,
                                }
                            }
                            if ingest.wrapping_sub(t) > DAY {
                                break 'v;
                            }
                            r.ts = t;
                            let cat = prod.get(pr).unwrap_or(unknown_cat) as u64;
                            r.meta = pv | (cat << 24) | META_OK | if purch { META_PURCH } else { 0 };
                            r.uid = match pack6(uid) {
                                Some(k) => k,
                                None => {
                                    let k = rs.hash_one(uid) | HBIT;
                                    fbn.push((k, pack(uid)));
                                    k
                                }
                            };
                        }
                        match pack6(eid) {
                            Some(k) => {
                                r.eid = k;
                                flat.push(r);
                            }
                            None => {
                                r.eid = pack(eid);
                                fb.push(r);
                            }
                        }
                    });
                    pt(0, &mut tm);
                    let (v, off) = bucketize(&flat, EB, |r| (mix(r.eid) >> (64 - EBB)) as usize);
                    pt(1, &mut tm);
                    (v, off, fb, fbn)
                })
                .collect();
            tick("events parsed");
            ev_out
        },
    );

    // clicks
    let click_chunks = all_chunks(&csv_files(&data.join("clicks")));
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
    for (_, _, _, fbn) in &ev_out {
        for &(k, r) in fbn {
            names_map.entry(k).or_insert(r);
        }
    }

    let resolve_out = |mut sv: Vec<EvOut>| -> (Vec<EvOut>, Vec<u32>) {
        let mut tm = now();
        let n = sv.len();
        for i in 0..n {
            if i + PFD < n {
                tab.pf(sv[i + PFD].key);
            }
            sv[i].key = tab.get(sv[i].key);
        }
        pt(4, &mut tm);
        let r = bucketize(&sv, PB, |x| pbk(x.key));
        pt(5, &mut tm);
        r
    };

    // dedup per event_id bucket -> person buckets. Each task takes DDG consecutive event_id
    // buckets, one small (L2-sized) table at a time, and resolves and buckets them together.
    let mut dd: Vec<(Vec<EvOut>, Vec<u32>)> = (0..EB / DDG)
        .into_par_iter()
        .map(|g| {
            let mut tm = now();
            let totals: [usize; DDG] =
                std::array::from_fn(|i| ev_out.iter().map(|c| (c.1[g * DDG + i + 1] - c.1[g * DDG + i]) as usize).sum());
            let maxcap = (totals.iter().max().unwrap() * 2).max(16).next_power_of_two();
            let mut tbuf: Vec<(u64, *const EvRec)> = vec![(0, std::ptr::null()); maxcap];
            let mut sv: Vec<EvOut> = Vec::with_capacity(totals.iter().sum());
            for (i, &total) in totals.iter().enumerate() {
                let b = g * DDG + i;
                let cap = (total * 2).max(16).next_power_of_two();
                let mask = cap - 1;
                let t = &mut tbuf[..cap];
                if i > 0 {
                    t.fill((0, std::ptr::null()));
                }
                for c in &ev_out {
                    for r in &c.0[c.1[b] as usize..c.1[b + 1] as usize] {
                        let mut p = (mix(r.eid) >> 8) as usize & mask;
                        loop {
                            let o = &mut t[p];
                            if o.0 == 0 {
                                *o = (r.eid, r as *const EvRec);
                                break;
                            }
                            if o.0 == r.eid {
                                if r.ingest < unsafe { (*o.1).ingest } {
                                    o.1 = r as *const EvRec;
                                }
                                break;
                            }
                            p = (p + 1) & mask;
                        }
                    }
                }
                pt(2, &mut tm);
                // Second pass in record order (sequential reads, no pointer chasing): a record is
                // kept when it is valid and is the one its event_id's slot points to.
                for c in &ev_out {
                    for r in &c.0[c.1[b] as usize..c.1[b + 1] as usize] {
                        if r.meta & META_OK == 0 {
                            continue;
                        }
                        let mut p = (mix(r.eid) >> 8) as usize & mask;
                        loop {
                            let o = unsafe { t.get_unchecked(p) };
                            if o.0 == r.eid {
                                if std::ptr::eq(o.1, r) {
                                    sv.push(EvOut { key: r.uid, ts: r.ts, meta: r.meta });
                                }
                                break;
                            }
                            p = (p + 1) & mask;
                        }
                    }
                }
                pt(3, &mut tm);
            }
            drop(tbuf);
            resolve_out(sv)
        })
        .collect();
    {
        // events whose event_id has no packed key: exact byte comparison
        let mut m: AHashMap<B, &EvRec> = AHashMap::new();
        let mut order: Vec<B> = vec![];
        for c in &ev_out {
            for r in &c.2 {
                let k = unpack(r.eid);
                match m.get_mut(k) {
                    Some(o) => {
                        if r.ingest < o.ingest {
                            *o = r;
                        }
                    }
                    None => {
                        m.insert(k, r);
                        order.push(k);
                    }
                }
            }
        }
        let sv: Vec<EvOut> =
            order.iter().map(|k| m[k]).filter(|r| r.meta & META_OK != 0).map(|r| EvOut { key: r.uid, ts: r.ts, meta: r.meta }).collect();
        dd.push(resolve_out(sv));
    }
    drop(ev_out);
    tick("dedup done");

    // person buckets
    let d0 = ws.div_euclid(DAY);
    let nd = if we > ws { ((we - 1).div_euclid(DAY) - d0 + 1) as usize } else { 0 };
    let ngroups = (rayon::current_num_threads() * 4).clamp(1, PB);
    let per = PB.div_ceil(ngroups);
    let results: Vec<(Vec<Vec<UD>>, CampAcc)> = (0..ngroups)
        .into_par_iter()
        .map(|g| {
            let mut cm = CampAcc::new(ncamp + 1, d0, nd);
            let mut win = Win { cnt: vec![0; ncamp + 1], pos: vec![0; ncamp + 1], active: vec![] };
            let mut out = vec![];
            let mut evs: Vec<EvOut> = vec![];
            let mut tmp: Vec<EvOut> = vec![];
            let mut cnt: Vec<u32> = vec![];
            let mut cl: Vec<ClickRec> = vec![];
            for p in g * per..((g + 1) * per).min(PB) {
                let mut tm = now();
                evs.clear();
                cl.clear();
                for (v, o) in &dd {
                    evs.extend_from_slice(&v[o[p] as usize..o[p + 1] as usize]);
                }
                for (v, o) in &click_out {
                    cl.extend_from_slice(&v[o[p] as usize..o[p + 1] as usize]);
                }
                pt(6, &mut tm);
                let mut uds = Vec::new();
                process_bucket(&mut evs, &mut tmp, &mut cnt, &mut cl, &mut win, ncamp as u32, &mut uds, &mut cm);
                out.push(uds);
            }
            (out, cm)
        })
        .collect();
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
