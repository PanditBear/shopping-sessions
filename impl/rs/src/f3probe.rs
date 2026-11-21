#![allow(dead_code, unused)]
// F3 probe: dense event index facts + microbenchmark of (dz scatter+finish, dd.table) vs (global idx table writes, walk).
macro_rules! out { ($($t:tt)*) => { eprintln!("probe f3 {}", format!($($t)*)) } }
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::*};
use std::time::Instant;

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
fn gen_unmix(val: u64) -> u64 {
    let x = val.wrapping_mul(GC2I) & GM;
    x ^ (x >> 17) ^ (x >> 34)
}
fn gen_idx(val: u64, key: u64) -> u64 {
    ((gen_unmix(val) ^ key).wrapping_mul(GC1I)) & GM
}
fn unq(f: &[u8]) -> &[u8] {
    if f.len() >= 2 && f[0] == b'"' && f[f.len() - 1] == b'"' { &f[1..f.len() - 1] } else { f }
}
fn val36(f: &[u8], pre: u8) -> Option<u64> {
    let f = unq(f);
    if f.len() < 2 || f.len() > 9 || f[0] != pre || f[1] == b'0' {
        return None;
    }
    let mut v = 0u64;
    for &c in &f[1..] {
        let d = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'z' => c - b'a' + 10,
            _ => return None,
        };
        v = v * 36 + d as u64;
    }
    if v > GM { None } else { Some(v) }
}
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}
fn f1secs(f: &[u8]) -> Option<i64> {
    let f = unq(f);
    if f.len() != 20 {
        return None;
    }
    let n = |a: usize, b: usize| -> i64 { f[a..b].iter().fold(0i64, |s, &c| s * 10 + (c.wrapping_sub(b'0')) as i64) };
    Some(days_from_civil(n(0, 4), n(5, 7), n(8, 10)) * 86400 + n(11, 13) * 3600 + n(14, 16) * 60 + n(17, 19))
}

#[derive(Clone, Copy, Default)]
struct EvL {
    key: u64,
    ts: u32,
    meta: u32,
}
#[derive(Clone, Copy)]
struct DR {
    i: u32,
    ing: u32,
    e: EvL,
}
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Line([u64; 8]);
const WK: usize = 8;

struct Cfg {
    key: u64,
    lim: u64,
    sh: u32,
    ndb: usize,
    lpb: f64,
}

struct Chunk {
    len: usize,
    lines: usize,
    idx: Vec<u32>,
    ing: Vec<u32>,
}

fn chunks(data: &[u8], sz: usize) -> Vec<(usize, usize)> {
    let start = match data.iter().position(|&b| b == b'\n') {
        Some(i) => i + 1,
        None => return vec![],
    };
    let mut v = vec![];
    let mut a = start;
    while a < data.len() {
        let mut b = (a + sz).min(data.len());
        if b < data.len() {
            b = match data[b - 1..].iter().position(|&c| c == b'\n') {
                Some(i) => b - 1 + i + 1,
                None => data.len(),
            };
        }
        v.push((a, b));
        a = b;
    }
    v
}

fn idx_find(total: usize, c: &[u8]) -> Option<Cfg> {
    let mut xs = vec![];
    let (mut pos, mut lines) = (0usize, 0usize);
    while xs.len() < 64 && lines < 4096 {
        let Some(e) = c[pos..].iter().position(|&b| b == b'\n') else { break };
        let line = &c[pos..pos + e];
        lines += 1;
        pos += e + 1;
        let f = &line[..line.iter().position(|&b| b == b',').unwrap_or(line.len())];
        // the shipped code uses pack6 (no quotes); a quoted id does not pack there
        if f.first() == Some(&b'"') {
            continue;
        }
        if let Some(v) = val36(f, b'e') {
            xs.push(gen_unmix(v));
        }
    }
    let lim = (total as f64 * lines as f64 / pos as f64 * 1.3) as u64 + 64;
    let need = (xs.len() * 3).div_ceil(4);
    let fits = |k: u64| xs.iter().filter(|&&x| ((x ^ k).wrapping_mul(GC1I) & GM) < lim).count() >= need;
    let key = (0..3.min(xs.len() - 1)).find_map(|p| {
        let (x0, d) = (xs[p], xs[p] ^ xs[p + 1]);
        (0..lim).find_map(|i| {
            let t = i.wrapping_mul(GC1) & GM;
            if ((d ^ t).wrapping_mul(GC1I) & GM) < lim && fits(x0 ^ t) { Some(x0 ^ t) } else { None }
        })
    })?;
    let sh = (64 - (lim / 2048).max(1).leading_zeros()).clamp(10, 15);
    Some(Cfg { key, lim, sh, ndb: (lim >> sh) as usize + 1, lpb: lines as f64 / pos as f64 })
}

fn load(dir: &str) -> (Cfg, Vec<Chunk>, usize) {
    let mut paths: Vec<_> = std::fs::read_dir(format!("{dir}/events")).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map(|x| x == "csv").unwrap_or(false)).collect();
    paths.sort();
    let total: usize = paths.iter().map(|p| std::fs::metadata(p).unwrap().len() as usize).sum();
    let sz = (total / 300).next_multiple_of(1 << 20).clamp(4 << 20, 32 << 20);
    let mut cfg: Option<Cfg> = None;
    let mut out = vec![];
    let mut base_s: Option<i64> = None;
    let mut nlines = 0usize;
    for p in &paths {
        let data = std::fs::read(p).unwrap();
        for (a, b) in chunks(&data, sz) {
            let c = &data[a..b];
            if cfg.is_none() {
                cfg = Some(idx_find(total, c).expect("no key"));
            }
            let cf = cfg.as_ref().unwrap();
            let mut ch = Chunk { len: c.len(), lines: 0, idx: vec![], ing: vec![] };
            for line in c.split(|&b| b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                ch.lines += 1;
                let mut it = line.split(|&b| b == b',');
                let f0 = it.next().unwrap();
                let _ = it.next();
                let _ = it.next();
                let Some(f3) = it.next() else { continue };
                let Some(s) = f1secs(f3) else { continue };
                let bs = *base_s.get_or_insert(s - 40 * 86400);
                let Some(v) = val36(f0, b'e') else { continue };
                let i = gen_idx(v, cf.key);
                if i >= cf.lim {
                    continue;
                }
                ch.idx.push(i as u32);
                ch.ing.push((s - bs + (1 << 31)) as u32);
            }
            nlines += ch.lines;
            out.push(ch);
        }
    }
    (cfg.unwrap(), out, nlines)
}

fn stats(cfg: &Cfg, ch: &[Chunk], nlines: usize) {
    let lim = cfg.lim as usize;
    let mut first: Vec<u16> = vec![u16::MAX; lim];
    let mut cnt: Vec<u8> = vec![0; lim];
    let (mut dense, mut same_chunk, mut cross) = (0usize, 0usize, 0usize);
    let mut maxi = 0usize;
    for (ci, c) in ch.iter().enumerate() {
        for &i in &c.idx {
            let i = i as usize;
            dense += 1;
            maxi = maxi.max(i);
            if first[i] == u16::MAX {
                first[i] = ci as u16;
            } else if first[i] == ci as u16 {
                same_chunk += 1;
            } else {
                cross += 1;
            }
            cnt[i] = cnt[i].saturating_add(1);
        }
    }
    let distinct = cnt.iter().filter(|&&x| x > 0).count();
    let under = cnt[..=maxi].iter().filter(|&&x| x == 0).count();
    let mut hist = [0usize; 6];
    for &x in &cnt {
        hist[(x as usize).min(5)] += 1;
    }
    let span = 1usize << cfg.sh;
    let tr = 4 * span;
    let mut fills = vec![];
    let mut r = 0;
    while (r + 1) * tr <= maxi {
        fills.push(cnt[r * tr..(r + 1) * tr].iter().filter(|&&x| x > 0).count() as f64 / tr as f64);
        r += 1;
    }
    fills.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out!(
        "lines {nlines} chunks {} lim {} sh {} ndb {} dense_lines {dense} ({:.4}) distinct_idx {distinct} max_idx {maxi} holes_below_max {under} ({:.5}) extra_copies {} same_chunk {same_chunk} cross_chunk {cross} copies_hist {:?}",
        ch.len(),
        cfg.lim,
        cfg.sh,
        cfg.ndb,
        dense as f64 / nlines as f64,
        under as f64 / (maxi + 1) as f64,
        dense - distinct,
        &hist[1..]
    );
    if !fills.is_empty() {
        out!("task-range fill (4*span idx) min {:.4} p1 {:.4} median {:.4} max {:.4} (ranges {})", fills[0], fills[fills.len() / 100], fills[fills.len() / 2], fills[fills.len() - 1], fills.len());
    }
}

// ---------- memory ----------
#[cfg(windows)]
extern "system" {
    fn VirtualAlloc(a: *mut u8, n: usize, t: u32, p: u32) -> *mut u8;
    fn VirtualFree(a: *mut u8, n: usize, t: u32) -> i32;
}
#[cfg(unix)]
extern "C" {
    fn mmap(a: *mut u8, n: usize, prot: i32, fl: i32, fd: i32, off: i64) -> *mut u8;
    fn munmap(a: *mut u8, n: usize) -> i32;
    fn madvise(a: *mut u8, n: usize, adv: i32) -> i32;
}
struct Mem {
    p: *mut u8,
    n: usize,
}
unsafe impl Send for Mem {}
unsafe impl Sync for Mem {}
impl Mem {
    #[cfg(windows)]
    fn new(n: usize) -> Mem {
        let p = unsafe { VirtualAlloc(std::ptr::null_mut(), n.max(4096), 0x3000, 4) };
        assert!(!p.is_null(), "VirtualAlloc {n}");
        Mem { p, n }
    }
    #[cfg(unix)]
    fn new(n: usize) -> Mem {
        // 2 MiB aligned + MADV_HUGEPAGE, like the shipped allocator setup
        let n2 = n.max(4096).next_multiple_of(1 << 21) + (1 << 21);
        let p = unsafe { mmap(std::ptr::null_mut(), n2, 3, 0x22, -1, 0) };
        assert!(p as isize != -1, "mmap {n}");
        let a = (p as usize).next_multiple_of(1 << 21) as *mut u8;
        unsafe { madvise(a, n2 - (a as usize - p as usize), 14) };
        let _ = p;
        Mem { p: a, n: n2 | ((a as usize - p as usize) << 48) }
    }
}
impl Drop for Mem {
    #[cfg(windows)]
    fn drop(&mut self) {
        unsafe { VirtualFree(self.p, 0, 0x8000) };
    }
    #[cfg(unix)]
    fn drop(&mut self) {
        let off = self.n >> 48;
        let n2 = self.n & ((1 << 48) - 1);
        unsafe { munmap(self.p.sub(off), n2) };
    }
}

#[inline(always)]
unsafe fn stream_lines(s: *const Line, d: *mut Line, n: usize) {
    use std::arch::x86_64::*;
    for k in 0..n {
        let a = _mm256_load_si256(s.add(k) as *const __m256i);
        let b = _mm256_load_si256((s.add(k) as *const __m256i).add(1));
        _mm256_stream_si256(d.add(k) as *mut __m256i, a);
        _mm256_stream_si256((d.add(k) as *mut __m256i).add(1), b);
    }
}
#[inline(always)]
fn prefetch<T>(p: *const T) {
    unsafe { std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0) }
}
#[inline(always)]
fn prefetchw<T>(p: *const T) {
    unsafe { std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_ET0) }
}

// ---------- today's dz ----------
struct Dz {
    sh: u32,
    ndb: usize,
    cap: usize,
    r: Mem,
    dn: Vec<u32>,
    ov: Vec<DR>,
    ovb: Vec<u32>,
    ovoff: Vec<u32>,
}
impl Dz {
    fn cap_of(cfg: &Cfg, len: usize) -> usize {
        let lines = len as f64 * cfg.lpb;
        let mean = lines * (1u64 << cfg.sh) as f64 / ((cfg.lim - 64) as f64 / 1.3).max(1.0) * 1.04;
        ((mean + 1.5 * mean.sqrt() + 1.0) as usize).min(lines as usize + 16).next_multiple_of(WK)
    }
    fn new(cfg: &Cfg, len: usize, r: Mem) -> Dz {
        let cap = Self::cap_of(cfg, len);
        Dz { sh: cfg.sh, ndb: cfg.ndb, cap, r, dn: vec![0; cfg.ndb], ov: vec![], ovb: vec![], ovoff: vec![] }
    }
    #[inline(always)]
    fn put(&mut self, wc: &mut [MaybeUninit<Line>], i: u64, ing: u32, e: EvL) {
        let b = (i >> self.sh) as usize;
        let x = DR { i: i as u32, ing, e };
        let n = unsafe { *self.dn.get_unchecked(b) } as usize;
        if n < self.cap {
            unsafe {
                let w = (wc.as_mut_ptr() as *mut DR).add(b * WK);
                w.add(n % WK).write(x);
                *self.dn.get_unchecked_mut(b) = n as u32 + 1;
                if (n + 1) % WK == 0 {
                    let d = (self.r.p as *mut DR).add(b * self.cap + n + 1 - WK);
                    stream_lines(w as *const Line, d as *mut Line, WK * 24 / 64);
                }
            }
        } else {
            self.ov.push(x);
            self.ovb.push(b as u32);
        }
    }
    fn finish(&mut self, wc: &[MaybeUninit<Line>]) {
        for b in 0..self.ndb {
            let n = self.dn[b] as usize;
            let t = n % WK;
            unsafe {
                let w = (wc.as_ptr() as *const DR).add(b * WK);
                let d = (self.r.p as *mut DR).add(b * self.cap + n - t);
                std::ptr::copy_nonoverlapping(w, d, t);
            }
        }
        unsafe { std::arch::x86_64::_mm_sfence() };
        if !self.ov.is_empty() {
            let mut off = vec![0u32; self.ndb + 1];
            for &b in &self.ovb {
                off[b as usize + 1] += 1;
            }
            for i in 0..self.ndb {
                off[i + 1] += off[i];
            }
            let mut w = off.clone();
            let mut ov = self.ov.clone();
            for (j, &b) in self.ovb.iter().enumerate() {
                ov[w[b as usize] as usize] = self.ov[j];
                w[b as usize] += 1;
            }
            self.ov = ov;
            self.ovoff = off;
        }
    }
    fn seg(&self, b: usize) -> (&[DR], &[DR]) {
        let s = unsafe { std::slice::from_raw_parts((self.r.p as *const DR).add(b * self.cap), self.dn[b] as usize) };
        let o = if self.ovoff.is_empty() { &self.ov[..0] } else { &self.ov[self.ovoff[b] as usize..self.ovoff[b + 1] as usize] };
        (s, o)
    }
}

#[inline(always)]
fn payload(idx: u32, ing: u32) -> EvL {
    let key = if idx % 97 == 0 { 0 } else { (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1 };
    EvL { key, ts: ing, meta: idx }
}

#[inline(always)]
fn work(n: u32, seed: u64) -> u64 {
    let mut x = seed;
    for _ in 0..n {
        x = (x ^ (x >> 29)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    }
    x
}

/// runs f(item) for items 0..n on nt threads (dynamic), returns summed busy seconds
fn par<F: Fn(usize, usize) + Sync>(nt: usize, n: usize, f: F) -> f64 {
    let next = AtomicUsize::new(0);
    let busy = AtomicU64::new(0);
    std::thread::scope(|s| {
        for t in 0..nt {
            let (next, busy, f) = (&next, &busy, &f);
            s.spawn(move || {
                let mut acc = 0f64;
                loop {
                    let i = next.fetch_add(1, Relaxed);
                    if i >= n {
                        break;
                    }
                    let t0 = Instant::now();
                    f(t, i);
                    acc += t0.elapsed().as_secs_f64();
                }
                busy.fetch_add((acc * 1e9) as u64, Relaxed);
            });
        }
    });
    busy.load(Relaxed) as f64 / 1e9
}

#[repr(C, align(32))]
struct Slot {
    w: AtomicU64,
    key: u64,
    tm: u64,
    _pad: u64,
}

#[inline(always)]
unsafe fn f3_insert(s: *mut Slot, ing: u32, e: EvL) {
    let want = (ing as u64 + 1) << 32;
    let w = &(*s).w;
    loop {
        let cur = w.load(Relaxed);
        if cur & 1 != 0 {
            std::hint::spin_loop();
            continue;
        }
        if cur != 0 && (cur >> 32) <= (want >> 32) {
            return;
        }
        if w.compare_exchange_weak(cur, want | 1, Acquire, Relaxed).is_ok() {
            std::ptr::addr_of_mut!((*s).key).write(e.key);
            std::ptr::addr_of_mut!((*s).tm).write(e.ts as u64 | ((e.meta as u64) << 32));
            w.store(want, Release);
            return;
        }
    }
}

#[inline(always)]
unsafe fn f3_ins<const SS: usize, const AT: bool>(s: *mut u8, ing: u32, e: EvL) {
    let want = (ing as u64 + 1) << 32;
    let w = &*(s as *const AtomicU64);
    if !AT {
        let cur = w.load(Relaxed);
        if cur == 0 || (cur >> 32) > (want >> 32) {
            w.store(want, Relaxed);
            (s.add(8) as *mut u64).write(e.key);
            if SS == 32 {
                (s.add(16) as *mut u64).write(e.ts as u64 | ((e.meta as u64) << 32));
            }
        }
        return;
    }
    loop {
        let cur = w.load(Relaxed);
        if cur & 1 != 0 {
            std::hint::spin_loop();
            continue;
        }
        if cur != 0 && (cur >> 32) <= (want >> 32) {
            return;
        }
        if w.compare_exchange_weak(cur, want | 1, Acquire, Relaxed).is_ok() {
            (s.add(8) as *mut u64).write(e.key);
            if SS == 32 {
                (s.add(16) as *mut u64).write(e.ts as u64 | ((e.meta as u64) << 32));
            }
            w.store(want, Release);
            return;
        }
    }
}
fn f3_put<const SS: usize, const AT: bool>(base: *mut u8, c: &Chunk, pd: usize, wk: u32) -> u64 {
    let n = c.idx.len();
    let mut x = 0u64;
    for j in 0..n {
        if j + pd < n {
            prefetchw(base.wrapping_add(c.idx[j + pd] as usize * SS));
        }
        if wk > 0 {
            x ^= work(wk, c.idx[j] as u64);
        }
        unsafe { f3_ins::<SS, AT>(base.add(c.idx[j] as usize * SS), c.ing[j], payload(c.idx[j], c.ing[j])) };
    }
    x
}
fn f3_walk<const SS: usize>(base: *const u8, lo: usize, hi: usize) -> Vec<EvL> {
    let mut em = vec![];
    for i in lo..hi {
        unsafe {
            let s = base.add(i * SS);
            let w = *(s as *const u64);
            let key = *(s.add(8) as *const u64);
            if w != 0 && key != 0 {
                if SS == 32 {
                    let tm = *(s.add(16) as *const u64);
                    em.push(EvL { key, ts: tm as u32, meta: (tm >> 32) as u32 });
                } else {
                    em.push(EvL { key, ts: ((w >> 32) - 1) as u32, meta: i as u32 });
                }
            }
        }
    }
    em
}

struct SPtr(*mut u8);
unsafe impl Send for SPtr {}
unsafe impl Sync for SPtr {}
impl SPtr {
    fn p(&self) -> *mut u8 {
        self.0
    }
}

pub fn run() {
    let dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into());
    let nt: usize = std::env::var("NCPU").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(4);
    let t0 = Instant::now();
    let (cfg, ch, nlines) = load(&dir);
    out!("loaded in {:.1}s nt {nt}", t0.elapsed().as_secs_f64());
    stats(&cfg, &ch, nlines);
    for (ss, at, reps) in [(32usize, true, 2usize), (16, true, 2), (32, false, 1), (16, false, 1)] {
        bench(&cfg, &ch, nt, reps, 0, 16, ss, at);
    }
    out!("done in {:.1}s", t0.elapsed().as_secs_f64());
}

fn bench(cfg: &Cfg, ch: &[Chunk], nt: usize, reps: usize, wk: u32, pd: usize, ss: usize, at: bool) {
    let ndense: usize = ch.iter().map(|c| c.idx.len()).sum();
    let nd = ndense as f64;
    let maxi = ch.iter().flat_map(|c| c.idx.iter()).copied().max().unwrap() as usize;
    let span = 1usize << cfg.sh;
    const G: usize = 4;
    let ntask = cfg.ndb.div_ceil(G);
    // pure work baseline
    if wk > 0 {
        let sink = AtomicU64::new(0);
        let tw = par(nt, ch.len(), |_, ci| {
            let c = &ch[ci];
            let mut x = 0u64;
            for j in 0..c.idx.len() {
                x ^= work(wk, c.idx[j] as u64);
            }
            sink.fetch_xor(x, Relaxed);
        });
        out!("work-only {:.2} ns/ev", tw * 1e9 / nd);
    }
    for rep in 0..reps {
        // ---------- today ----------
        let sink = AtomicU64::new(0);
        let put_pass = |pool: &Vec<std::sync::Mutex<Option<Mem>>>| -> (Vec<Dz>, f64) {
        let dzs: Vec<std::sync::Mutex<Option<Dz>>> = (0..ch.len()).map(|_| std::sync::Mutex::new(None)).collect();
        let tput = par(nt, ch.len(), |_, ci| {
            let c = &ch[ci];
            let cap = Dz::cap_of(&cfg, c.len);
            let mem = pool[ci].lock().unwrap().take().unwrap_or_else(|| Mem::new(cfg.ndb * cap * 24));
            let mut dz = Dz::new(&cfg, c.len, mem);
            let wl = (cfg.ndb * WK * 24).div_ceil(64);
            thread_local! { static WC: std::cell::RefCell<Vec<MaybeUninit<Line>>> = const { std::cell::RefCell::new(Vec::new()) }; }
            WC.with(|wcc| {
                let mut wc = wcc.borrow_mut();
                if wc.len() < wl {
                    *wc = Vec::with_capacity(wl);
                    unsafe { wc.set_len(wl) };
                }
                let mut x = 0u64;
                for j in 0..c.idx.len() {
                    if wk > 0 {
                        x ^= work(wk, c.idx[j] as u64);
                    }
                    dz.put(&mut wc, c.idx[j] as u64, c.ing[j], payload(c.idx[j], c.ing[j]));
                }
                dz.finish(&wc);
                sink.fetch_xor(x, Relaxed);
            });
            *dzs[ci].lock().unwrap() = Some(dz);
        });
        (dzs.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect(), tput)
        };
        let pool0: Vec<std::sync::Mutex<Option<Mem>>> = (0..ch.len()).map(|_| std::sync::Mutex::new(None)).collect();
        let (dz, tput) = put_pass(&pool0);
        let touched: usize = dz.iter().map(|d| d.ndb * d.cap * 24).sum();
        let tgat = AtomicU64::new(0);
        let twalk = AtomicU64::new(0);
        let ems: Vec<std::sync::Mutex<Vec<EvL>>> = (0..ntask).map(|_| std::sync::Mutex::new(vec![])).collect();
        let ttab = par(nt, ntask, |_, g| {
            let mask = span as u64 - 1;
            let mut t0 = Instant::now();
            let mut t: Vec<u64> = vec![u64::MAX; span];
            let mut lb: Vec<EvL> = vec![];
            let mut em: Vec<EvL> = vec![];
            let (mut ga, mut wa) = (0f64, 0f64);
            for b in g * G..((g + 1) * G).min(cfg.ndb) {
                lb.clear();
                for ci in 0..dz.len() {
                    if let Some(n) = dz.get(ci + 4) {
                        prefetch(n.dn.as_ptr().wrapping_add(b));
                    }
                    if let Some(n) = dz.get(ci + 2) {
                        let p = (n.r.p as *const DR).wrapping_add(b * n.cap) as *const u8;
                        let len = n.dn[b] as usize * 24;
                        for o in (0..len.min(1024)).step_by(64) {
                            prefetch(p.wrapping_add(o));
                        }
                    }
                    let (sa, oa) = dz[ci].seg(b);
                    for part in [sa, oa] {
                        for x in part {
                            let q = lb.len() as u32;
                            lb.push(x.e);
                            let pk = ((x.ing as u64) << 32) | q as u64;
                            let o = unsafe { t.get_unchecked_mut((x.i as u64 & mask) as usize) };
                            if *o == u64::MAX {
                                *o = pk;
                                continue;
                            }
                            if x.ing < (*o >> 32) as u32 {
                                *o = pk;
                            }
                        }
                    }
                }
                let t1 = Instant::now();
                ga += (t1 - t0).as_secs_f64();
                if !lb.is_empty() {
                    for o in t.iter_mut() {
                        if *o != u64::MAX {
                            let e = lb[*o as u32 as usize];
                            *o = u64::MAX;
                            if e.key != 0 {
                                em.push(e);
                            }
                        }
                    }
                }
                t0 = Instant::now();
                wa += (t0 - t1).as_secs_f64();
            }
            tgat.fetch_add((ga * 1e9) as u64, Relaxed);
            twalk.fetch_add((wa * 1e9) as u64, Relaxed);
            *ems[g].lock().unwrap() = em;
        });
        let (mut n1, mut h1) = (0usize, 0u64);
        for m in &ems {
            let v = m.lock().unwrap();
            n1 += v.len();
            for e in v.iter() {
                h1 = h1.wrapping_add(e.key.wrapping_mul(e.ts as u64 | 1));
            }
        }
        drop(ems);
        // warm today: same dz buffers again (already touched)
        let pool1: Vec<std::sync::Mutex<Option<Mem>>> = dz.into_iter().map(|d| std::sync::Mutex::new(Some(d.r))).collect();
        let (dzw, tputw) = put_pass(&pool1);
        let tfree = Instant::now();
        drop(dzw);
        drop(pool1);
        let tfree = tfree.elapsed().as_secs_f64();
        // ---------- F3 ----------
        let tabn = (maxi + 1) * ss;
        let f3put = |tp: &SPtr| -> f64 {
            par(nt, ch.len(), |_, ci| {
                let x = match (ss, at) {
                    (16, true) => f3_put::<16, true>(tp.p(), &ch[ci], pd, wk),
                    (16, false) => f3_put::<16, false>(tp.p(), &ch[ci], pd, wk),
                    (_, true) => f3_put::<32, true>(tp.p(), &ch[ci], pd, wk),
                    (_, false) => f3_put::<32, false>(tp.p(), &ch[ci], pd, wk),
                };
                sink.fetch_xor(x, Relaxed);
            })
        };
        let tab = Mem::new(tabn);
        let tp = SPtr(tab.p);
        let tf3 = f3put(&tp);
        let tr = G * span;
        let nr = (maxi + 1).div_ceil(tr);
        let ems2: Vec<std::sync::Mutex<Vec<EvL>>> = (0..nr).map(|_| std::sync::Mutex::new(vec![])).collect();
        let twalk3 = par(nt, nr, |_, g| {
            let (lo, hi) = (g * tr, ((g + 1) * tr).min(maxi + 1));
            let em = if ss == 16 { f3_walk::<16>(tp.p(), lo, hi) } else { f3_walk::<32>(tp.p(), lo, hi) };
            *ems2[g].lock().unwrap() = em;
        });
        let (mut n2, mut h2) = (0usize, 0u64);
        for m in &ems2 {
            let v = m.lock().unwrap();
            n2 += v.len();
            for e in v.iter() {
                h2 = h2.wrapping_add(e.key.wrapping_mul(e.ts as u64 | 1));
            }
        }
        drop(ems2);
        // warm F3: clear (untimed), insert again into the same touched memory
        let tz = Instant::now();
        unsafe { std::ptr::write_bytes(tab.p, 0, tabn) };
        let tz = tz.elapsed().as_secs_f64();
        let tf3w = f3put(&tp);
        drop(tab);
        // parallel first touch of a fresh table (fault cost only)
        let tab2 = Mem::new(tabn);
        let tp2 = SPtr(tab2.p);
        let pg = 1 << 21;
        let npg = tabn.div_ceil(pg);
        let tfault = par(nt, npg, |_, k| {
            let p = tp2.p();
            let mut o = k * pg;
            while o < ((k + 1) * pg).min(tabn) {
                unsafe { p.add(o).write_volatile(0) };
                o += 4096;
            }
        });
        drop(tab2);
        out!(
            "rep{rep} nt{nt} wk{wk} pd{pd} ss{ss} at{at} dense {ndense} | today: put+finish cold {:.2} warm {:.2} gather(dd.table) {:.2} walk {:.2} ns/ev, dz bytes {:.2} GB, em {n1} | F3 cold put {:.2} warm put {:.2} walk {:.2} ns/ev, table {:.2} GB, fault-only(par) {:.2} ns/ev, em {n2} {} | free dz {:.2}s zero {:.2}s",
            tput * 1e9 / nd,
            tputw * 1e9 / nd,
            tgat.load(Relaxed) as f64 / nd,
            twalk.load(Relaxed) as f64 / nd,
            touched as f64 / 1e9,
            tf3 * 1e9 / nd,
            tf3w * 1e9 / nd,
            twalk3 * 1e9 / nd,
            tabn as f64 / 1e9,
            tfault * 1e9 / nd,
            if n1 == n2 && h1 == h2 { "MATCH" } else { "MISMATCH" },
            tfree,
            tz
        );
        let _ = sink.load(Relaxed);
    }
}
