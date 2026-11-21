use ahash::{AHashMap, RandomState};
use memchr::{memchr, memchr_iter};
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

type B = &'static [u8];

const DAY: i64 = 86_400_000;
const WEEK: i64 = 7 * DAY;
const GAP: i64 = 1_800_000;
const CHUNK: usize = 4 << 20;
const EB: usize = 512; // event dedup buckets
const PB: usize = 1024; // person buckets

fn hasher() -> RandomState {
    RandomState::with_seeds(0x243f6a8885a308d3, 0x13198a2e03707344, 0xa4093822299f31d0, 0x082efa98ec4e6c89)
}

// ---------- parsing helpers ----------

#[inline]
fn unq(f: &'static [u8]) -> Option<B> {
    if f.first() == Some(&b'"') {
        if f.len() < 2 || *f.last().unwrap() != b'"' {
            return None;
        }
        let inner = &f[1..f.len() - 1];
        if memchr(b'"', inner).is_some() {
            return None;
        }
        Some(inner)
    } else if memchr(b'"', f).is_some() {
        None
    } else {
        Some(f)
    }
}

#[inline]
fn split_fields<const N: usize>(line: &'static [u8]) -> Option<[B; N]> {
    let mut out: [B; N] = [&[]; N];
    let mut n = 0;
    let mut start = 0;
    for i in memchr_iter(b',', line) {
        if n >= N - 1 {
            return None;
        }
        out[n] = unq(&line[start..i])?;
        n += 1;
        start = i + 1;
    }
    if n != N - 1 {
        return None;
    }
    out[n] = unq(&line[start..])?;
    Some(out)
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

struct TsCache {
    pre: [u8; 10],
    days: i64,
    ok: bool,
}
fn parse_ts(b: &[u8]) -> Option<i64> {
    parse_ts_c(b, &mut TsCache { pre: [0; 10], days: 0, ok: false })
}
fn parse_ts_c(b: &[u8], c: &mut TsCache) -> Option<i64> {
    let n = b.len();
    if n == 0 {
        return None;
    }
    if (n == 20 && b[19] == b'Z') || (n == 25 && (b[19] == b'+' || b[19] == b'-')) {
        if b[10] != b'T' || b[13] != b':' || b[16] != b':' {
            return None;
        }
        let h = d2(&b[11..13])?;
        let mi = d2(&b[14..16])?;
        let s = d2(&b[17..19])?;
        if h > 23 || mi > 59 || s > 59 {
            return None;
        }
        let days;
        if c.ok && c.pre[..] == b[0..10] {
            days = c.days;
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
            c.pre.copy_from_slice(&b[0..10]);
            c.days = days;
            c.ok = true;
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

fn parse_price(b: &[u8]) -> Option<u32> {
    if b.is_empty() || !b.iter().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let mut v: u64 = 0;
    for &c in b {
        v = v * 10 + (c - b'0') as u64;
        if v > 10_000_000 {
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

fn lines(chunk: B, mut f: impl FnMut(B)) {
    let mut pos = 0;
    while pos < chunk.len() {
        match memchr(b'\n', &chunk[pos..]) {
            Some(i) => {
                f(&chunk[pos..pos + i]);
                pos += i + 1;
            }
            None => {
                f(&chunk[pos..]);
                break;
            }
        }
    }
}

fn leak_file(p: &Path) -> Option<B> {
    let f = File::open(p).ok()?;
    let len = f.metadata().ok()?.len();
    if len == 0 {
        return None;
    }
    let m = unsafe { memmap2::Mmap::map(&f).ok()? };
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

// ---------- identity table ----------

#[inline]
fn pack(b: B) -> u64 {
    (b.as_ptr() as u64) | ((b.len() as u64) << 48)
}
#[inline]
fn unpack(r: u64) -> B {
    unsafe { std::slice::from_raw_parts((r & ((1u64 << 48) - 1)) as *const u8, (r >> 48) as usize) }
}
#[inline]
fn hid(rs: &RandomState, b: B) -> u64 {
    let h = rs.hash_one(b);
    if h == 0 {
        1
    } else {
        h
    }
}

// open addressing, 3 u64 per slot: hash, person key, person name ref
struct LinkTab {
    s: Vec<u64>,
    mask: usize,
}
impl LinkTab {
    #[inline]
    fn find(&self, h: u64) -> Option<(u64, u64)> {
        let mut i = h as usize & self.mask;
        loop {
            let sh = self.s[i * 3];
            if sh == h {
                return Some((self.s[i * 3 + 1], self.s[i * 3 + 2]));
            }
            if sh == 0 {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }
}

// ---------- records ----------

const META_PURCH: u64 = 1 << 56;
const META_OK: u64 = 1 << 57;

#[derive(Clone, Copy)]
struct EvRec {
    h: u64,
    eid: u64,
    ingest: i64,
    key: u64,
    name: u64,
    ts: i64,
    meta: u64,
}

#[derive(Clone, Copy)]
struct EvOut {
    key: u64,
    name: u64,
    ts: i64,
    meta: u64,
}

#[derive(Clone, Copy)]
struct ClickRec {
    key: u64,
    camp: u32,
    ts: i64,
    cid: B,
}

struct UD {
    name: u64,
    day: i32,
    sessions: u32,
    events: u32,
    rev: u64,
    top: u32,
}


// ---------- per-bucket processing ----------

struct DayAcc {
    name: u64,
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
    out.push(UD { name: d.name, day: d.day as i32, sessions: d.sessions, events: d.events, rev: d.rev, top });
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

fn process_bucket(mut evs: Vec<EvOut>, mut clicks: Vec<ClickRec>, ncamp: usize, uds: &mut Vec<UD>, cm: &mut CampMap) {
    let organic = ncamp as u32;
    evs.sort_unstable_by_key(|e| (e.key, e.ts));
    clicks.sort_unstable_by(|a, b| (a.key, a.ts).cmp(&(b.key, b.ts)).then(a.cid.cmp(b.cid)));
    let mut win = Win { cnt: vec![0; ncamp + 1], pos: vec![0; ncamp + 1], active: vec![] };
    let mut ci = 0;
    let mut i = 0;
    while i < evs.len() {
        let pid = evs[i].key;
        let pname = evs[i].name;
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
        let mut day = DayAcc { name: pname, day: 0, sessions: 0, events: 0, rev: 0, cats: vec![] };
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
                    let en = cm.entry(((s_camp as u64) << 32) | (s_day as u32 as u64)).or_insert([0; 4]);
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
                    cm.entry(((c as u64) << 32) | (d as u32 as u64)).or_insert([0; 4])[3] += 1;
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
            let en = cm.entry(((s_camp as u64) << 32) | (s_day as u32 as u64)).or_insert([0; 4]);
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

fn write_user_daily(path: &Path, groups: &[&Vec<UD>], cats: &[B]) {
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
        let users: Vec<ByteArray> = buf.iter().map(|r| ba(unpack(r.name))).collect();
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
            lines(c, |l| {
                if let Some([p, c]) = split_fields::<2>(l) {
                    prod_raw.push((p, c));
                }
            });
        }
    }
    let mut cat_names: Vec<B> = prod_raw.iter().map(|x| x.1).collect();
    cat_names.push(b"unknown");
    cat_names.sort();
    cat_names.dedup();
    let unknown_cat = cat_names.binary_search(&(b"unknown" as B)).unwrap() as u32;
    let mut prod: AHashMap<B, u32> = AHashMap::new();
    for &(p, c) in &prod_raw {
        prod.insert(p, cat_names.binary_search(&c).unwrap() as u32);
    }

    // campaigns
    let mut camp: AHashMap<B, u32> = AHashMap::new();
    let mut camp_names: Vec<B> = vec![];
    let mut channels: Vec<B> = vec![];
    if let Some(f) = leak_file(&data.join("campaigns.csv")) {
        for c in chunks(f) {
            lines(c, |l| {
                if let Some([id, ch]) = split_fields::<2>(l) {
                    if !camp.contains_key(id) {
                        camp.insert(id, camp_names.len() as u32);
                        camp_names.push(id);
                        channels.push(ch);
                    }
                }
            });
        }
    }
    let ncamp = camp_names.len();
    camp_names.push(b"organic");
    channels.push(b"organic");

    let rs = hasher();

    // links
    let link_chunks = all_chunks(&csv_files(&data.join("links")));
    let link_pairs: Vec<Vec<(u64, u64, u64, u64)>> = link_chunks
        .par_iter()
        .map(|&c| {
            let mut v = vec![];
            lines(c, |l| {
                if let Some([a, b]) = split_fields::<2>(l) {
                    if !a.is_empty() && !b.is_empty() {
                        v.push((hid(&rs, a), pack(a), hid(&rs, b), pack(b)));
                    }
                }
            });
            v
        })
        .collect();
    let occ: usize = link_pairs.iter().map(|v| v.len()).sum::<usize>() * 2;
    let cap = (occ.max(8) * 2).next_power_of_two();
    let mut tab = LinkTab { s: vec![0u64; cap * 3], mask: cap - 1 };
    let mut idx_h: Vec<u64> = vec![];
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
    {
        let mut get = |h: u64, r: u64, tab: &mut LinkTab| -> u32 {
            let mut i = h as usize & tab.mask;
            loop {
                let sh = tab.s[i * 3];
                if sh == h {
                    return tab.s[i * 3 + 1] as u32;
                }
                if sh == 0 {
                    let id = idx_h.len() as u32;
                    tab.s[i * 3] = h;
                    tab.s[i * 3 + 1] = id as u64;
                    idx_h.push(h);
                    idx_ref.push(r);
                    parent.push(id);
                    return id;
                }
                i = (i + 1) & tab.mask;
            }
        };
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(occ / 2);
        for v in &link_pairs {
            for &(ha, ra, hb, rb) in v {
                let a = get(ha, ra, &mut tab);
                let b = get(hb, rb, &mut tab);
                edges.push((a, b));
            }
        }
        for (a, b) in edges {
            let ra = find(&mut parent, a);
            let rb = find(&mut parent, b);
            if ra != rb {
                parent[ra as usize] = rb;
            }
        }
    }
    let nl = idx_h.len();
    let mut best: Vec<u32> = (0..nl as u32).collect();
    for g in 0..nl {
        let r = find(&mut parent, g as u32) as usize;
        if unpack(idx_ref[g]) < unpack(idx_ref[best[r] as usize]) {
            best[r] = g as u32;
        }
    }
    for i in 0..cap {
        if tab.s[i * 3] != 0 {
            let g = tab.s[i * 3 + 1] as u32;
            let b = best[find(&mut parent, g) as usize] as usize;
            tab.s[i * 3 + 1] = idx_h[b];
            tab.s[i * 3 + 2] = idx_ref[b];
        }
    }
    drop(link_pairs);
    let resolve = |uid: B| -> (u64, u64) {
        let h = hid(&rs, uid);
        tab.find(h).unwrap_or((h, pack(uid)))
    };
    tick("links done");

    // clicks
    let click_chunks = all_chunks(&csv_files(&data.join("clicks")));
    let click_out: Vec<Vec<Vec<ClickRec>>> = click_chunks
        .par_iter()
        .map(|&c| {
            let mut out: Vec<Vec<ClickRec>> = (0..PB).map(|_| Vec::new()).collect();
            let mut tc = TsCache { pre: [0; 10], days: 0, ok: false };
            lines(c, |l| {
                let Some([cid, uid, cmp, ts]) = split_fields::<4>(l) else { return };
                if cid.is_empty() || uid.is_empty() {
                    return;
                }
                let Some(&ci) = camp.get(cmp) else { return };
                let Some(t) = parse_ts_c(ts, &mut tc) else { return };
                if t < ws - WEEK || t >= we {
                    return;
                }
                let (key, _) = resolve(uid);
                out[key as usize % PB].push(ClickRec { key, camp: ci, ts: t, cid });
            });
            out
        })
        .collect();
    tick("clicks parsed");

    // events
    let ev_chunks = all_chunks(&csv_files(&data.join("events")));
    let ev_out: Vec<Vec<Vec<EvRec>>> = ev_chunks
        .par_iter()
        .map(|&c| {
            let mut out: Vec<Vec<EvRec>> = (0..EB).map(|_| Vec::new()).collect();
            let mut tc = TsCache { pre: [0; 10], days: 0, ok: false };
            lines(c, |l| {
                let Some([eid, uid, ts, ing, et, pr, price]) = split_fields::<7>(l) else { return };
                if eid.is_empty() {
                    return;
                }
                let ingest = parse_ts_c(ing, &mut tc).unwrap_or(i64::MAX);
                let h = hid(&rs, eid);
                let mut r = EvRec { h, eid: pack(eid), ingest, key: 0, name: 0, ts: 0, meta: 0 };
                'v: {
                    if uid.is_empty() || pr.is_empty() {
                        break 'v;
                    }
                    let Some(t) = parse_ts_c(ts, &mut tc) else { break 'v };
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
                    let cat = prod.get(pr).copied().unwrap_or(unknown_cat) as u64;
                    r.meta = pv | (cat << 24) | META_OK | if purch { META_PURCH } else { 0 };
                    let (k, n) = resolve(uid);
                    r.key = k;
                    r.name = n;
                }
                out[(h >> 32) as usize % EB].push(r);
            });
            out
        })
        .collect();
    tick("events parsed");

    // dedup per bucket -> person buckets
    let dd: Vec<Vec<Vec<EvOut>>> = (0..EB)
        .into_par_iter()
        .map(|b| {
            let total: usize = ev_out.iter().map(|c| c[b].len()).sum();
            let mut v: Vec<EvRec> = Vec::with_capacity(total);
            for c in &ev_out {
                v.extend_from_slice(&c[b]);
            }
            let cap = (total * 2).max(4).next_power_of_two();
            let mask = cap - 1;
            let mut t: Vec<u32> = vec![u32::MAX; cap];
            for (i, r) in v.iter().enumerate() {
                let mut p = (r.h >> 7) as usize & mask;
                loop {
                    let o = t[p];
                    if o == u32::MAX {
                        t[p] = i as u32;
                        break;
                    }
                    let k = &v[o as usize];
                    if k.h == r.h && unpack(k.eid) == unpack(r.eid) {
                        if r.ingest < k.ingest {
                            t[p] = i as u32;
                        }
                        break;
                    }
                    p = (p + 1) & mask;
                }
            }
            let mut out: Vec<Vec<EvOut>> = (0..PB).map(|_| Vec::new()).collect();
            for &o in &t {
                if o != u32::MAX {
                    let r = &v[o as usize];
                    if r.meta & META_OK != 0 {
                        out[r.key as usize % PB].push(EvOut { key: r.key, name: r.name, ts: r.ts, meta: r.meta });
                    }
                }
            }
            out
        })
        .collect();
    drop(ev_out);
    tick("dedup done");

    // person buckets
    let results: Vec<(Vec<Vec<UD>>, CampMap)> = (0..PB)
        .into_par_iter()
        .fold(
            || (Vec::new(), CampMap::new()),
            |mut acc: (Vec<Vec<UD>>, CampMap), p| {
                let evs: Vec<EvOut> = dd.iter().flat_map(|d| d[p].iter().copied()).collect();
                let cl: Vec<ClickRec> = click_out.iter().flat_map(|c| c[p].iter().copied()).collect();
                let mut uds = Vec::new();
                process_bucket(evs, cl, ncamp, &mut uds, &mut acc.1);
                acc.0.push(uds);
                acc
            },
        )
        .collect();
    tick("sessionized");

    // outputs
    let ud_dir = outd.join("user_daily");
    let cd_dir = outd.join("campaign_daily");
    std::fs::create_dir_all(&ud_dir).unwrap();
    std::fs::create_dir_all(&cd_dir).unwrap();
    let mut cm = CampMap::new();
    let mut ud_groups: Vec<&Vec<UD>> = vec![];
    for (u, m) in &results {
        ud_groups.extend(u.iter());
        for (&k, v) in m {
            let e = cm.entry(k).or_insert([0; 4]);
            for i in 0..4 {
                e[i] += v[i];
            }
        }
    }
    let rows: Vec<(u32, i32, [u64; 4])> = cm.iter().map(|(&k, &v)| ((k >> 32) as u32, k as u32 as i32, v)).collect();
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
                write_user_daily(&ud_dir.join(format!("part-{i}.parquet")), g, &cat_names);
            })
        },
        || write_campaign_daily(&cd_dir.join("part-0.parquet"), &rows, &camp_names, &channels),
    );
    tick("written");
}
