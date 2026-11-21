//! Floor kernel for the events parse: per line the same results as `parse_events_sol` (K1), in
//! the same output structures. Two passes per window of a chunk: pass 1 turns each 64-byte block
//! into comma|newline and quote bitmaps and flattens them into a u32 list of comma/newline
//! positions plus a list of newline positions (bit 31: the line holds a quote); pass 2 takes the
//! lines in order, reads the 7 field ends from the list and decodes the fields. A line the fast
//! path does not take (quote, wrong comma count, longer than 127 bytes, or in the last 160 bytes
//! of the chunk) goes to the slow list, or with FB to K1's own per-line path.
use super::*;
use std::time::Instant;

/// bytes per pass-1 round
const W: usize = 4 << 10;
/// comma/newline entries kept from a partial line across rounds; past this the line is slow
/// anyway (it is longer than 127 bytes) and its entries are dropped
const CARRY: usize = 512;

#[derive(Default, Clone, Copy, Debug)]
pub struct Stats {
    pub fast: u64,
    pub slow_q: u64,
    pub slow_f: u64,
    pub slow_l: u64,
    pub tail: u64,
}
impl Stats {
    pub fn slow(&self) -> u64 {
        self.slow_q + self.slow_f + self.slow_l + self.tail
    }
    pub fn lines(&self) -> u64 {
        self.fast + self.slow()
    }
    pub fn add(&mut self, o: &Stats) {
        self.fast += o.fast;
        self.slow_q += o.slow_q;
        self.slow_f += o.slow_f;
        self.slow_l += o.slow_l;
        self.tail += o.tail;
    }
}

pub struct Scratch {
    s: Vec<u32>,
    nl: Vec<u32>,
    /// one bit per entry of `s`: the byte after the separator is a quote (the next field opens)
    of: Vec<u64>,
    /// one bit per entry of `s`: the byte before the separator is a quote (its field closes)
    cf: Vec<u64>,
    /// (start, end) of the lines refused by the fast path (end = newline position; the tail
    /// is one entry with end = chunk length)
    pub slow: Vec<(u32, u32)>,
    pub st: Stats,
}
impl Scratch {
    pub fn new() -> Self {
        Scratch { s: vec![0; W + CARRY + 128], nl: vec![0; W + 128], of: vec![0; (W + CARRY) / 64 + 8], cf: vec![0; (W + CARRY) / 64 + 8], slow: Vec::with_capacity(4096), st: Stats::default() }
    }
}

struct Lx {
    dci: DayCache,
    dct: DayCache,
    tci: TsCache,
    tct: TsCache,
    ws: i64,
    we: i64,
    ibase: i64,
    imax_k: u32,
    imax_x: bool,
}

/// FLOOR_SKIP: parts of the line body left out, for the cost split only (1 event ts, 2 ingest
/// ts, 4 dz, 8 user key, 16 event type/price/product, 32 record pushes, 64 dz put (idx kept),
/// 128 id packing)
fn skip_mask() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("FLOOR_SKIP").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// lines in the chunk tail (from `pos`), counting a last line without newline
fn tail_lines(t: &[u8]) -> u64 {
    let n = memchr::memchr_iter(b'\n', t).count() as u64;
    n + (t.last().map(|&b| b != b'\n').unwrap_or(false)) as u64
}

/// The K1 record of one fast line. `f` holds the six comma positions; `end` is the newline.
/// SAFETY: the line lies in the chunk at `base`, starts at least 160 bytes before its end and
/// is shorter than 128 bytes, so every read below (16 bytes from each id, the ts fields) stays
/// inside the chunk.
#[inline(always)]
unsafe fn line_body<E: Ev, const DISC: bool>(
    base: *const u8,
    st: usize,
    f: &[u32; 6],
    q: u32,
    end: usize,
    cx: &EvCtx,
    c0: usize,
    x: &mut Lx,
    m: &mut Vec<E>,
    d: &mut Vec<DRec>,
    fb: &mut Vec<(u64, i64, u32)>,
    fbn: &mut Vec<(u64, u64)>,
    side: &mut Vec<(u32, i64)>,
    dz: &mut Dz<E>,
    sk: u32,
) {
    let sl = |p: *const u8, n: usize| -> B { std::slice::from_raw_parts(p, n) };
    // field i spans [a, b); when quoted (bit i of q) the quotes are dropped
    let fs = |a: usize, b: usize, i: u32| {
        let qi = ((q >> i) & 1) as usize;
        (base.add(a + qi), b - a - 2 * qi)
    };
    let [f0, f1, f2, f3, f4, f5] = f.map(|v| v as usize);
    let (pe, ne) = fs(st, f0, 0);
    if ne == 0 {
        return;
    }
    let (pu, nu) = fs(f0 + 1, f1, 1);
    let (pt, nt) = fs(f1 + 1, f2, 2);
    let (pi, ni) = fs(f2 + 1, f3, 3);
    let (pet, net) = fs(f3 + 1, f4, 4);
    let (pp, np) = fs(f4 + 1, f5, 5);
    let (ppr, npr) = fs(f5 + 1, end, 6);
    let (ws, we, ibase) = (x.ws, x.we, x.ibase);
    let (ke, oke, ku, oku) = if sk & 128 != 0 { (pe as u64 ^ ne as u64, true, pu as u64 ^ nu as u64, true) } else { pack6x2(pe, ne, pu, nu) };
    let (mut ra, mut rb, mut rd) = (D0_A, D0_B, D0_S);
    let iso_i = if sk & 2 != 0 {
        Some(Some(x.ws / 1000 + 1))
    } else if ni == 20 && *pi.add(19) == b'Z' {
        let (ia, ib) = (ldp(pi), ldp(pi.add(3)));
        let (dd, hs) = (day_get(ia, ib, pi, &mut x.dci), hms(pi.add(11)));
        Some(match dd {
            Some(dv) => {
                (ra, rb, rd) = (ia, ib, dv);
                hs.map(|h| dv + h)
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
        Some(None) => (i64::MAX, x.imax_k, x.imax_x),
        None => {
            let ingest = ts_fast(sl(pi, ni), &mut x.tci).unwrap_or(i64::MAX);
            let (k, e) = ing32(ingest, ibase);
            (ingest, k, e)
        }
    };
    let mut r = E::nil();
    'v: {
        if (nu == 0) | (np == 0) {
            break 'v;
        }
        let t = if sk & 1 != 0 {
            ingest.clamp(ws, we - 1)
        } else if nt == 20 && *pt.add(19) == b'Z' {
            let (ta, tb) = (ldp(pt), ldp(pt.add(3)));
            let dv = if (ta == ra) & (tb == rb) {
                rd
            } else {
                match day_get(ta, tb, pt, &mut x.dct) {
                    Some(dv) => dv,
                    None => break 'v,
                }
            };
            match hms(pt.add(11)) {
                Some(h) => (dv + h) * 1000,
                None => break 'v,
            }
        } else if nt == 13 {
            match ts13(pt) {
                Some(t) => t,
                None => break 'v,
            }
        } else {
            let fz = if nt == 25 { ts25(pt, ra, rb, rd, &mut x.dct) } else { None };
            match fz.or_else(|| parse_ts_c(sl(pt, nt), &mut x.tct)) {
                Some(t) => t,
                None => break 'v,
            }
        };
        if (t < ws) | (t >= we) | (ingest.wrapping_sub(t) > DAY) {
            break 'v;
        }
        let mut pv = 0u32;
        let mut cat = 0u32;
        match if sk & 16 != 0 { 1 } else { et_code(sl(pet, net)) } {
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
        let key = if sk & 8 != 0 { nu as u64 ^ pu as u64 } else if oku { ku } else { ev_key_slow(sl(pu, nu), cx.rs, fbn) };
        r = E::mk(key, (t - ws) as u64, pv, cat);
    }
    if let Some(i) = if sk & 4 != 0 { None } else { dz.idx_txt(pe, ne) } {
        if DISC {
            dz.sk = dz.sk.wrapping_add(i ^ ((ik as u64) << 32) ^ fold(&r) ^ ingest as u64 ^ iex as u64);
            return;
        } else if sk & 64 != 0 {
            x.imax_k ^= i as u32;
        } else {
            dz.put(i, ik, r, iex, ingest);
            return;
        }
    }
    if sk & 32 != 0 {
        x.imax_k ^= ke as u32 ^ ik ^ r.key() as u32;
        return;
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
}

/// Floor parse of one chunk. FB: refused lines go through K1's per-line path in line order (the
/// checker's mode, output equal to K1 as a whole); otherwise they are only listed in `scr.slow`.
pub fn parse<E: Ev, const FB: bool>(
    c: B,
    cx: &EvCtx,
    m: &mut Vec<E>,
    d: &mut Vec<DRec>,
    fb: &mut Vec<(u64, i64, u32)>,
    fbn: &mut Vec<(u64, u64)>,
    side: &mut Vec<(u32, i64)>,
    dz: &mut Dz<E>,
    scr: &mut Scratch,
) {
    parse_x::<E, FB, true, false>(c, cx, m, d, fb, fbn, side, dz, scr)
}

/// the probe's floor-discard kernel: as `parse`, with the dense records folded into `dz.sk`
#[allow(clippy::too_many_arguments)]
pub fn parse_disc<E: Ev, const FB: bool>(
    c: B,
    cx: &EvCtx,
    m: &mut Vec<E>,
    d: &mut Vec<DRec>,
    fb: &mut Vec<(u64, i64, u32)>,
    fbn: &mut Vec<(u64, u64)>,
    side: &mut Vec<(u32, i64)>,
    dz: &mut Dz<E>,
    scr: &mut Scratch,
) {
    parse_x::<E, FB, true, true>(c, cx, m, d, fb, fbn, side, dz, scr)
}

/// BODY = false: structure only (no field decode), for the cost split
#[inline(always)]
pub fn parse_x<E: Ev, const FB: bool, const BODY: bool, const DISC: bool>(
    c: B,
    cx: &EvCtx,
    m: &mut Vec<E>,
    d: &mut Vec<DRec>,
    fb: &mut Vec<(u64, i64, u32)>,
    fbn: &mut Vec<(u64, u64)>,
    side: &mut Vec<(u32, i64)>,
    dz: &mut Dz<E>,
    scr: &mut Scratch,
) {
    let c0 = c.as_ptr() as usize;
    let len = c.len();
    let base = c.as_ptr();
    let (imax_k, imax_x) = ing32(i64::MAX, cx.ibase);
    let mut x = Lx { dci: DayCache::new(), dct: DayCache::new(), tci: TsCache::new(), tct: TsCache::new(), ws: cx.ws, we: cx.we, ibase: cx.ibase, imax_k, imax_x };
    let mut sbuf: Vec<[B; 7]> = Vec::new();
    let sk = skip_mask();
    scr.slow.clear();
    let mut stt = Stats::default();
    // lines whose newline lies below lim start at least 160 bytes before the chunk end
    let lim = len.saturating_sub(160) & !63;
    let sp = scr.s.as_mut_ptr();
    let np = scr.nl.as_mut_ptr();
    let mut sn = 0usize; // entries in s
    let mut pend = 0u64; // a misplaced quote since the last newline, from earlier blocks
    let mut scar = 1u64; // the byte before the block is a separator (chunk start counts as one)
    let mut qcar = 0u64; // the byte before the block is a quote
    let ofp = scr.of.as_mut_ptr();
    let cfp = scr.cf.as_mut_ptr();
    let mut pos = 0usize; // start of the next line
    let mut w0 = 0usize;
    while w0 < lim {
        let w1 = (w0 + W).min(lim);
        let mut nn = 0usize;
        let mut b = w0;
        // pass 1
        while b < w1 {
            // SAFETY: b + 64 <= lim <= len; s has room for W + CARRY entries plus 64 of slack,
            // nl for W plus 64
            unsafe {
                let (ca, qa, na) = masks64(base.add(b));
                let b32 = b as u32;
                let sm = ca | na;
                let cnt = sm.count_ones() as usize;
                // quotes: a quote is well placed iff exactly one of its neighbours is a
                // separator (it opens or closes its field); every other quote is misplaced
                let nx = *base.add(b + 64);
                let sepl = (sm << 1) | scar;
                let sepr = (sm >> 1) | ((((nx == b',') | (nx == b'\n')) as u64) << 63);
                let badq = qa & !(sepl ^ sepr);
                let openq = sm & ((qa >> 1) | (((nx == b'"') as u64) << 63));
                let closeq = sm & ((qa << 1) | qcar);
                scar = sm >> 63;
                qcar = qa >> 63;
                {
                    use core::arch::x86_64::{_bzhi_u64, _pext_u64};
                    let (ob, cb) = (_pext_u64(openq, sm), _pext_u64(closeq, sm));
                    let (w, sh) = (sn >> 6, (sn & 63) as u32);
                    let lo = _bzhi_u64(!0, sh);
                    *ofp.add(w) = (*ofp.add(w) & lo) | (ob << sh);
                    *ofp.add(w + 1) = (ob >> 1) >> (63 - sh);
                    *cfp.add(w) = (*cfp.add(w) & lo) | (cb << sh);
                    *cfp.add(w + 1) = (cb >> 1) >> (63 - sh);
                }
                let o = sp.add(sn);
                let mut y = sm;
                for i in 0..8 {
                    *o.add(i) = b32.wrapping_add(y.trailing_zeros());
                    y &= y.wrapping_sub(1);
                }
                if cnt > 8 {
                    let mut i = 8;
                    while y != 0 {
                        *o.add(i) = b32.wrapping_add(y.trailing_zeros());
                        y &= y.wrapping_sub(1);
                        i += 1;
                    }
                }
                sn += cnt;
                // newline bit n stays set in r2 iff no quote lies between the previous newline
                // and n (the borrow of each quote run stops at the next newline)
                let (r1, b1) = na.overflowing_sub(badq);
                let (r2, b2) = r1.overflowing_sub(pend);
                pend = (b1 | b2) as u64;
                let qt = !r2;
                let ncnt = na.count_ones() as usize;
                let o = np.add(nn);
                let t0 = na.trailing_zeros();
                let y1 = na & na.wrapping_sub(1);
                let t1 = y1.trailing_zeros();
                *o = b32.wrapping_add(t0) | ((((qt >> (t0 & 63)) & 1) as u32) << 31);
                *o.add(1) = b32.wrapping_add(t1) | ((((qt >> (t1 & 63)) & 1) as u32) << 31);
                if ncnt > 2 {
                    let mut y = y1 & y1.wrapping_sub(1);
                    let mut i = 2;
                    while y != 0 {
                        let t = y.trailing_zeros();
                        *o.add(i) = b32.wrapping_add(t) | ((((qt >> t) & 1) as u32) << 31);
                        y &= y.wrapping_sub(1);
                        i += 1;
                    }
                }
                nn += ncnt;
            }
            b += 64;
        }
        // pass 2
        let mut k = 0usize;
        for j in 0..nn {
            // SAFETY: j < nn; k + 6 < sn + 64 (k is at most the index of this line's newline
            // entry, and s has 64 entries of slack)
            unsafe {
                let e = *np.add(j);
                let end = (e & 0x7FFF_FFFF) as usize;
                let st = pos;
                pos = end + 1;
                let fp = sp.add(k);
                let f6 = *fp.add(6) as usize;
                // quote flags of the 7 separators from entry k: field i opens with a quote
                // (field 0: the line's first byte), field i closes with one
                let rd = |p: *const u64| (p as *const u8).add(k >> 3).cast::<u64>().read_unaligned() >> (k & 7);
                let op = ((rd(ofp) & 0x3F) << 1) | (*base.add(st) == b'"') as u64;
                let cl = rd(cfp) & 0x7F;
                let ok = (f6 == end) & (e >> 31 == 0) & (end - st < 128) & (op == cl);
                if ok {
                    let f = &*(fp as *const [u32; 6]);
                    k += 7;
                    stt.fast += 1;
                    if !BODY {
                        stt.slow_l = stt.slow_l.wrapping_add(f[0] as u64 ^ f[5] as u64 ^ cl);
                        continue;
                    }
                    line_body::<E, DISC>(base, st, f, cl as u32, end, cx, c0, &mut x, m, d, fb, fbn, side, dz, sk);
                } else {
                    while *sp.add(k) as usize != end {
                        k += 1;
                    }
                    k += 1;
                    if e >> 31 != 0 || (f6 == end && op != cl) {
                        stt.slow_q += 1;
                    } else if f6 != end {
                        stt.slow_f += 1;
                    } else {
                        stt.slow_l += 1;
                    }
                    if FB {
                        sbuf.clear();
                        let e2 = scan7_line(c, st, &mut sbuf);
                        debug_assert_eq!(e2, end + 1);
                        for fl in &sbuf {
                            ev_line(fl, cx, c0, &mut x.tci, &mut x.tct, m, d, fb, fbn, side, dz);
                        }
                    } else {
                        scr.slow.push((st as u32, end as u32));
                    }
                }
            }
        }
        // carry the partial line's entries
        let carry = sn - k;
        if carry > CARRY {
            sn = 0;
        } else {
            scr.s.copy_within(k..sn, 0);
            // shift the flag bits k.. down to 0.. (in place: word i reads words >= i)
            for v in [&mut scr.of, &mut scr.cf] {
                let p = v.as_mut_ptr();
                // SAFETY: k + carry = sn, inside the streams with 8 words of slack
                unsafe {
                    for i in 0..(carry + 63) / 64 {
                        let bi = k + 64 * i;
                        let (w, sh) = (bi >> 6, (bi & 63) as u32);
                        let lo = *p.add(w) >> sh;
                        let hi = (*p.add(w + 1) << 1) << (63 - sh);
                        *p.add(i) = lo | hi;
                    }
                }
            }
            sn = carry;
        }
        w0 = w1;
    }
    if pos < len {
        stt.tail += tail_lines(&c[pos..]);
        if FB {
            scan::<7>(&c[pos..], |fl| ev_line(fl, cx, c0, &mut x.tci, &mut x.tct, m, d, fb, fbn, side, dz));
        } else {
            scr.slow.push((pos as u32, len as u32));
        }
    }
    scr.st = stt;
}

// ---------------------------------------------------------------- harness

#[cfg(windows)]
fn pin(cpu: usize) {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadAffinityMask(h: isize, m: usize) -> usize;
    }
    // SAFETY: plain Win32 call on the current thread's pseudo handle
    let r = unsafe { SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu) };
    if r == 0 {
        eprintln!("pin {cpu} failed");
    }
}
#[cfg(target_os = "linux")]
fn pin(cpu: usize) {
    extern "C" {
        fn sched_setaffinity(pid: i32, sz: usize, mask: *const u64) -> i32;
    }
    let mut m = [0u64; 16];
    m[(cpu / 64) % 16] |= 1 << (cpu % 64);
    // SAFETY: plain libc call on the calling thread with a 128-byte mask
    if unsafe { sched_setaffinity(0, 128, m.as_ptr()) } != 0 {
        eprintln!("pin {cpu} failed");
    }
}
#[cfg(not(any(windows, target_os = "linux")))]
fn pin(_cpu: usize) {}

fn mkdz<E: Ev>(on: bool, len: usize) -> Dz<E> {
    if on {
        Dz::new(idx_cfg(), len)
    } else {
        Dz::off()
    }
}

/// mismatching entries between two equal-length-or-not vectors (length difference counts too)
fn vdiff<T, F: Fn(&T, &T) -> bool>(a: &[T], b: &[T], eq: F) -> u64 {
    let n = a.len().min(b.len());
    let mut bad = (a.len().max(b.len()) - n) as u64;
    for i in 0..n {
        if !eq(&a[i], &b[i]) {
            bad += 1;
        }
    }
    bad
}

fn eeq<E: Ev>(a: &E, b: &E) -> bool {
    a.key() == b.key() && a.rts() == b.rts() && a.pc() == b.pc()
}

fn bufs_diff<E: Ev>(a: &EvBufs<E>, b: &EvBufs<E>) -> [u64; 5] {
    [
        vdiff(&a.m, &b.m, eeq),
        vdiff(&a.d, &b.d, |x, y| (x.eid, x.ing, x.loc) == (y.eid, y.ing, y.loc)),
        vdiff(&a.fb, &b.fb, |x, y| x == y),
        vdiff(&a.fbn, &b.fbn, |x, y| x == y),
        vdiff(&a.side, &b.side, |x, y| x == y),
    ]
}

/// staged dedup records after `finish`: slots per bucket, overflow, positions of inexact keys
fn dz_diff<E: Ev>(a: &Dz<E>, b: &Dz<E>) -> u64 {
    let dreq = |x: &DR<E>, y: &DR<E>| x.i == y.i && x.ing == y.ing && eeq(&x.e, &y.e);
    let mut bad = vdiff(&a.dn, &b.dn, |x, y| x == y);
    bad += (a.cap != b.cap) as u64 + (a.ndb != b.ndb) as u64;
    for bk in 0..a.dn.len().min(b.dn.len()) {
        let (sa, pa, oa, qa) = a.seg(bk);
        let (sb, pb, ob, qb) = b.seg(bk);
        bad += vdiff(sa, sb, dreq) + vdiff(oa, ob, dreq) + (pa != pb) as u64 + (qa != qb) as u64;
    }
    bad += vdiff(&a.ov, &b.ov, dreq);
    bad += vdiff(&a.dside, &b.dside, |x, y| x == y);
    bad
}

fn check<E: Ev>(chunks: &[B], cx: &EvCtx) {
    let t = Instant::now();
    // per chunk: [dz off: 5 buffer diffs, dz on: 5 buffer diffs + dz diff], stats, lines counted
    let res: Vec<([u64; 11], Stats, u64)> = chunks
        .par_iter()
        .map(|&c| {
            let mut out = [0u64; 11];
            let mut st = Stats::default();
            let mut scr = Scratch::new();
            for (pass, on) in [false, true].into_iter().enumerate() {
                let (mut a, mut b) = (EvBufs::<E>::new(), EvBufs::<E>::new());
                let mut da = mkdz::<E>(on, c.len());
                parse_events_sol(c, cx, &mut a.m, &mut a.d, &mut a.fb, &mut a.fbn, &mut a.side, &mut da);
                da.finish();
                let mut db = mkdz::<E>(on, c.len());
                parse::<E, true>(c, cx, &mut b.m, &mut b.d, &mut b.fb, &mut b.fbn, &mut b.side, &mut db, &mut scr);
                db.finish();
                let bd = bufs_diff(&a, &b);
                out[pass * 5..pass * 5 + 5].copy_from_slice(&bd);
                if on {
                    out[10] = dz_diff(&da, &db);
                }
                st = scr.st;
            }
            (out, st, tail_lines(c))
        })
        .collect();
    let mut tot = [0u64; 11];
    let mut st = Stats::default();
    let mut lines = 0u64;
    for (o, s, l) in &res {
        for i in 0..11 {
            tot[i] += o[i];
        }
        st.add(s);
        lines += l;
    }
    let mism: u64 = tot.iter().sum();
    println!("check chunks {} lines(counted) {} lines(floor) {} fast {} slow {} (quote {} fields {} long {} tail {}) slow% {:.4}", chunks.len(), lines, st.lines(), st.fast, st.slow(), st.slow_q, st.slow_f, st.slow_l, st.tail, 100.0 * st.slow() as f64 / st.lines().max(1) as f64);
    println!("check dzoff [m d fb fbn side] {:?} dzon {:?} dz {}", &tot[0..5], &tot[5..10], tot[10]);
    println!("check MISMATCHES {} ({:.1}s)", mism, t.elapsed().as_secs_f64());
}

/// one serial pass over all chunks: seconds inside the parse calls
fn pass_k1<E: Ev>(chunks: &[B], cx: &EvCtx, on: bool, bufs: &mut EvBufs<E>) -> f64 {
    let mut el = 0f64;
    for &c in chunks {
        bufs.clear();
        let mut dz = mkdz::<E>(on, c.len());
        let t = Instant::now();
        parse_events_sol(c, cx, &mut bufs.m, &mut bufs.d, &mut bufs.fb, &mut bufs.fbn, &mut bufs.side, &mut dz);
        el += t.elapsed().as_secs_f64();
        std::hint::black_box(&bufs.m);
        dz.finish();
    }
    el
}

fn pass_fl<E: Ev, const FB: bool>(chunks: &[B], cx: &EvCtx, on: bool, bufs: &mut EvBufs<E>, scr: &mut Scratch) -> f64 {
    let mut el = 0f64;
    for &c in chunks {
        bufs.clear();
        let mut dz = mkdz::<E>(on, c.len());
        let t = Instant::now();
        parse::<E, FB>(c, cx, &mut bufs.m, &mut bufs.d, &mut bufs.fb, &mut bufs.fbn, &mut bufs.side, &mut dz, scr);
        el += t.elapsed().as_secs_f64();
        std::hint::black_box(&bufs.m);
        dz.finish();
    }
    el
}

fn bench<E: Ev>(chunks: &[B], cx: &EvCtx, lines: u64) {
    let reps: usize = std::env::var("FLOOR_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let cpu: usize = std::env::var("FLOOR_CPU").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let which = std::env::var("FLOOR_WHICH").unwrap_or_else(|_| "k1off,floff,k1on,flon,flonfb".into());
    let thr: usize = std::env::var("FLOOR_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    pin(cpu);
    let mut bufs = EvBufs::<E>::new();
    let mut scr = Scratch::new();
    let names: Vec<&str> = which.split(',').collect();
    let mut best = vec![f64::MAX; names.len()];
    for rep in 0..reps {
        for (i, &nm) in names.iter().enumerate() {
            let s = match nm {
                "k1off" => pass_k1(chunks, cx, false, &mut bufs),
                "k1on" => pass_k1(chunks, cx, true, &mut bufs),
                "floff" => pass_fl::<E, false>(chunks, cx, false, &mut bufs, &mut scr),
                "flon" => pass_fl::<E, false>(chunks, cx, true, &mut bufs, &mut scr),
                "floffb" => pass_fl::<E, true>(chunks, cx, false, &mut bufs, &mut scr),
                "flstruct" => {
                    let mut el = 0f64;
                    for &c in chunks {
                        bufs.clear();
                        let mut dz = Dz::<E>::off();
                        let t = Instant::now();
                        parse_x::<E, false, false, false>(c, cx, &mut bufs.m, &mut bufs.d, &mut bufs.fb, &mut bufs.fbn, &mut bufs.side, &mut dz, &mut scr);
                        el += t.elapsed().as_secs_f64();
                        std::hint::black_box(&scr.st);
                    }
                    el
                }
                "flonfb" => pass_fl::<E, true>(chunks, cx, true, &mut bufs, &mut scr),
                _ => continue,
            };
            let ns = s * 1e9 / lines as f64;
            best[i] = best[i].min(ns);
            eprintln!("st rep{rep} {nm} {ns:.2} ns/line");
        }
    }
    for (i, nm) in names.iter().enumerate() {
        println!("single cpu{cpu} best-of-{reps} {nm} {:.2} ns/line", best[i]);
    }
    if thr == 0 {
        return;
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(thr).start_handler(|i| pin(i)).stack_size(8 << 20).build().unwrap();
    for &nm in &names {
        let (fl, on, fbk) = match nm {
            "k1off" => (false, false, false),
            "k1on" => (false, true, false),
            "floff" => (true, false, false),
            "flon" => (true, true, false),
            "floffb" => (true, false, true),
            "flonfb" => (true, true, true),
            _ => continue,
        };
        let mut bt = f64::MAX;
        for _ in 0..reps {
            let t = pool.install(|| {
                let t = Instant::now();
                chunks.par_iter().for_each_init(
                    || (EvBufs::<E>::new(), Scratch::new()),
                    |(b, s), &c| {
                        b.clear();
                        let mut dz = mkdz::<E>(on, c.len());
                        match (fl, fbk) {
                            (false, _) => parse_events_sol(c, cx, &mut b.m, &mut b.d, &mut b.fb, &mut b.fbn, &mut b.side, &mut dz),
                            (true, false) => parse::<E, false>(c, cx, &mut b.m, &mut b.d, &mut b.fb, &mut b.fbn, &mut b.side, &mut dz, s),
                            (true, true) => parse::<E, true>(c, cx, &mut b.m, &mut b.d, &mut b.fb, &mut b.fbn, &mut b.side, &mut dz, s),
                        }
                        std::hint::black_box(&b.m);
                        dz.finish();
                    },
                );
                t.elapsed().as_secs_f64()
            });
            bt = bt.min(t);
        }
        println!("mt {thr} threads cpus 0-{} best-of-{reps} {nm} {:.3} s {:.1} Mlines/s {:.2} ns/line/thread", thr - 1, bt, lines as f64 / bt / 1e6, bt * 1e9 * thr as f64 / lines as f64);
    }
}


// ---------------------------------------------------------------- input and probe CLI

/// the parse input: event chunks and the parse context, all leaked ('static)
struct Inp {
    chunks: Vec<B>,
    cx: EvCtx<'static>,
    lines: u64,
    files: usize,
    bytes: usize,
    io_s: f64,
}

/// `data`: the directory with manifest.json and products.csv (either may be missing); `ev`: the
/// event files (None: the sorted DATA_DIR/events/*.csv). Empty files are skipped, header-only
/// files kept. Every input page is faulted in before returning, so no timed pass reads files.
fn load(data: &Path, ev: Option<&[std::path::PathBuf]>) -> Inp {
    let mf = std::fs::read_to_string(data.join("manifest.json")).unwrap_or_default();
    let ws = manifest_ts(&mf, "window_start").unwrap_or(0);
    let we = manifest_ts(&mf, "window_end").unwrap_or(0);
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
    let lean = cat_names.len() <= 256 && we.saturating_sub(ws) <= 1i64 << 32;
    assert!(lean, "this harness runs lean records only (at most 256 categories, window under 2^32 ms)");
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
    let prod: &'static IdLookup = Box::leak(Box::new(IdLookup::new(b"p", &prod_items)));
    let rs: &'static RandomState = Box::leak(Box::new(hasher()));
    let t_io = Instant::now();
    let ev_files: Vec<B> = match ev {
        None => csv_files(&data.join("events")),
        Some(ps) => ps.iter().filter_map(|p| leak_file(p)).collect(),
    };
    let mut acc = 0u8;
    let mut bytes = 0usize;
    for f in &ev_files {
        let mut i = 0;
        while i < f.len() {
            acc ^= f[i];
            i += 4096;
        }
        bytes += f.len();
    }
    std::hint::black_box(acc);
    let io_s = t_io.elapsed().as_secs_f64();
    let ev_chunks = all_chunks(&ev_files, ev_chunk(&ev_files));
    let _ = IDXC.set(idx_find(&ev_files, &ev_chunks));
    let cx = EvCtx { prod, unknown_cat, ws, we, ibase: ws.div_euclid(1000), rs };
    let lines: u64 = ev_chunks.par_iter().map(|c| tail_lines(c)).sum();
    Inp { chunks: ev_chunks, cx, lines, files: ev_files.len(), bytes, io_s }
}

/// One chunk the way `ev_chunk_parse` does it (dense records to their dz buckets, the rest
/// bucketed by event_id hash, then `dz.finish`), minus its timing ticks. kind: 0 K1, 1 K1 with
/// the dz put replaced by the discard sink, 2 floor, 3 floor with the discard sink. The floor
/// kinds send the lines their fast path refuses through K1's per-line path, in line order.
fn chunk_x<E: Ev>(c: B, cx: &EvCtx, kind: u8, scr: &mut Scratch) -> EvChunk<E> {
    let mut dz = Dz::new(idx_cfg(), c.len());
    let cap = if dz.on { c.len() / 4096 } else { c.len() / 64 };
    let mut m: Vec<E> = Vec::with_capacity(cap);
    let mut d: Vec<DRec> = Vec::with_capacity(cap);
    let (mut fb, mut fbn, mut side) = (vec![], vec![], vec![]);
    match kind {
        0 => parse_events_sol(c, cx, &mut m, &mut d, &mut fb, &mut fbn, &mut side, &mut dz),
        1 => parse_events_disc(c, cx, &mut m, &mut d, &mut fb, &mut fbn, &mut side, &mut dz),
        2 => parse::<E, true>(c, cx, &mut m, &mut d, &mut fb, &mut fbn, &mut side, &mut dz, scr),
        _ => parse_disc::<E, true>(c, cx, &mut m, &mut d, &mut fb, &mut fbn, &mut side, &mut dz, scr),
    }
    let (d, doff) = bucketize(&d, EB, |r| (mix(r.eid) >> (64 - EBB)) as usize);
    dz.finish();
    EvChunk { m, d, doff, dz, fb, fbn, side }
}

/// one timed pass of a kernel over all chunks on the current pool: (seconds, checksum). The
/// outputs are hashed and freed after the clock stops.
fn run_kernel<E: Ev>(kind: u8, inp: &Inp) -> (f64, u64) {
    let t = Instant::now();
    let out: Vec<EvChunk<E>> = inp.chunks.par_iter().map_init(Scratch::new, |scr, &c| chunk_x::<E>(c, &inp.cx, kind, scr)).collect();
    let s = t.elapsed().as_secs_f64();
    let mut h = 0x9e37_79b9_7f4a_7c15u64;
    for r in &out {
        h = hash_chunk(h, r);
        h = mix(h ^ r.dz.sk);
    }
    drop(out);
    (s, h)
}

const USAGE: &str = "usage: floor [EVENTS_DIR | DATA_DIR | FILE.csv ...] [--data DATA_DIR] [--threads N] [--kernel k1,k1-discard,floor,floor-discard,check] [--reps R] [--pin FIRST_CPU] [-v]
  no path: $DATA_DIR/events (DATA_DIR defaults to /data). manifest.json and products.csv come
  from --data, else the DATA_DIR given, else the parent of the events directory.
  One stderr line per kernel: median ns per line over R reps (wall time of the parallel parse
  of all files together / all lines), ns per line-thread (x N), min, first rep, and an output
  checksum. k1 and floor must print the same checksum; the -discard checksums fold only what
  each kernel discards and are kernel-specific.";

fn cli(args: Vec<String>) {
    let mut paths: Vec<std::path::PathBuf> = vec![];
    let mut data_arg: Option<std::path::PathBuf> = None;
    let mut threads = 8usize;
    let mut kernels: Vec<String> = vec![];
    let mut reps = 3usize;
    let mut pin0: Option<usize> = None;
    let mut verbose = false;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        let mut val = |name: &str| it.next().unwrap_or_else(|| panic!("{name} needs a value\n{USAGE}"));
        match a.as_str() {
            "--threads" => threads = val("--threads").parse().expect("--threads N"),
            "--reps" => reps = val("--reps").parse().expect("--reps R"),
            "--kernel" => kernels.extend(val("--kernel").split(',').filter(|s| !s.is_empty()).map(String::from)),
            "--data" => data_arg = Some(val("--data").into()),
            "--pin" => pin0 = Some(val("--pin").parse().expect("--pin FIRST_CPU")),
            "-v" => verbose = true,
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return;
            }
            s if s.starts_with('-') => panic!("unknown flag {s}\n{USAGE}"),
            _ => paths.push(a.into()),
        }
    }
    let threads = threads.max(1);
    let reps = reps.max(1);
    if kernels.is_empty() {
        kernels = ["k1", "k1-discard", "floor", "floor-discard"].iter().map(|s| s.to_string()).collect();
    }
    let mut b = rayon::ThreadPoolBuilder::new().num_threads(threads).stack_size(8 << 20);
    if let Some(p0) = pin0 {
        b = b.start_handler(move |i| pin(p0 + i));
    }
    b.build_global().unwrap();
    // where the data directory and the event files are
    let (data, ev): (std::path::PathBuf, Option<Vec<std::path::PathBuf>>) = if paths.is_empty() {
        (std::path::PathBuf::from(std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into())), None)
    } else if paths.len() == 1 && paths[0].is_dir() {
        let p = paths[0].clone();
        if p.join("events").is_dir() {
            (p, None)
        } else {
            let parent = p.parent().map(|x| x.to_path_buf()).unwrap_or_else(|| ".".into());
            let mut fs: Vec<std::path::PathBuf> = std::fs::read_dir(&p)
                .expect("events dir")
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|q| q.extension().map(|x| x == "csv").unwrap_or(false) && q.is_file())
                .collect();
            fs.sort();
            (parent, Some(fs))
        }
    } else {
        let parent = paths[0].parent().and_then(|x| x.parent()).map(|x| x.to_path_buf()).unwrap_or_else(|| ".".into());
        (parent, Some(paths.clone()))
    };
    let data = data_arg.unwrap_or(data);
    let inp = load(&data, ev.as_deref());
    eprintln!(
        "input files={} bytes={} lines={} chunks={} page_in_s={:.3} threads={} pin={:?} idx={:?} data={}",
        inp.files,
        inp.bytes,
        inp.lines,
        inp.chunks.len(),
        inp.io_s,
        threads,
        pin0,
        idx_cfg().map(|c| (c.lim, c.sh, c.ndb)),
        data.display()
    );
    let mut kinds: Vec<(String, u8)> = vec![];
    for k in &kernels {
        match k.as_str() {
            "k1" => kinds.push((k.clone(), 0)),
            "k1-discard" => kinds.push((k.clone(), 1)),
            "floor" => kinds.push((k.clone(), 2)),
            "floor-discard" => kinds.push((k.clone(), 3)),
            "check" => check::<EvL>(&inp.chunks, &inp.cx),
            _ => eprintln!("unknown kernel {k}"),
        }
    }
    let mut times: Vec<Vec<f64>> = vec![vec![]; kinds.len()];
    let mut sums: Vec<u64> = vec![0; kinds.len()];
    let mut sum_ok = vec![true; kinds.len()];
    let lines = inp.lines.max(1) as f64;
    // reps interleaved across kernels, so drift (heat, other load) hits every kernel alike
    for rep in 0..reps {
        for (i, (name, kind)) in kinds.iter().enumerate() {
            let (s, h) = run_kernel::<EvL>(*kind, &inp);
            if rep == 0 {
                sums[i] = h;
            } else if sums[i] != h {
                sum_ok[i] = false;
            }
            times[i].push(s);
            if verbose {
                eprintln!("rep {rep} kernel={name} {:.3} ns/line {:.4} s checksum {h:016x}", s * 1e9 / lines, s);
            }
        }
    }
    for (i, (name, _)) in kinds.iter().enumerate() {
        let first = times[i][0];
        let mut v = times[i].clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = v[v.len() / 2];
        let ns = |s: f64| s * 1e9 / lines;
        eprintln!(
            "probe kernel={name} median_ns_line={:.3} ns_line_thread={:.2} min_ns_line={:.3} first_ns_line={:.3} reps={reps} threads={threads} files={} lines={} checksum={:016x}{}",
            ns(med),
            ns(med) * threads as f64,
            ns(v[0]),
            ns(first),
            inp.files,
            inp.lines,
            sums[i],
            if sum_ok[i] { "" } else { " CHECKSUM-VARIES" }
        );
    }
}

/// the older env-driven harness (FLOOR_MODE=check|bench, DATA_DIR, NCPU, FLOOR_*)
fn legacy() {
    let data = std::env::var("DATA_DIR").unwrap_or_else(|_| "C:/bxdata/shop/gen/t20m".into());
    let ncpu: usize = std::env::var("NCPU").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(8);
    rayon::ThreadPoolBuilder::new().num_threads(ncpu).stack_size(8 << 20).build_global().unwrap();
    let inp = load(Path::new(&data), None);
    println!("input {} files {:.1} MB {} chunks {} lines; map+page-in {:.3} s; idx cfg {:?}", inp.files, inp.bytes as f64 / 1e6, inp.chunks.len(), inp.lines, inp.io_s, idx_cfg());
    let mode = std::env::var("FLOOR_MODE").unwrap_or_else(|_| "check".into());
    for md in mode.split(',') {
        match md {
            "check" => check::<EvL>(&inp.chunks, &inp.cx),
            "bench" => bench::<EvL>(&inp.chunks, &inp.cx, inp.lines),
            _ => eprintln!("unknown mode {md}"),
        }
    }
}

pub fn main() {
    hpa::init();
    if std::env::var_os("FLOOR_MODE").is_some() {
        legacy();
    } else {
        cli(std::env::args().skip(1).collect());
    }
}
