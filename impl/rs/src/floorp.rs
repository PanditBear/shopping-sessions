// Floor probe: FLOOR=<mode> runs one measurement instead of the pipeline and exits.
// Modes: exit0 | pool | <read>_<work> | wr<MB>
//   read: mmap mmapseq mmapwill popread populate pieces read<KB>
//   work: sum (add all bytes as u64 lanes) nl (count newlines) page (one byte per 4 KB)
// FLOOR_T=threads (default NCPU), FLOOR_PIN=1 pins pool thread i to cpu i, FLOOR_PIECE=MB (default 4).
#![allow(dead_code)]
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

fn ep() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

#[cfg(target_os = "linux")]
mod sys {
    use std::ffi::c_void;
    extern "C" {
        pub fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
        pub fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        pub fn pread(fd: i32, buf: *mut c_void, n: usize, off: i64) -> isize;
        pub fn sched_setaffinity(pid: i32, size: usize, mask: *const u8) -> i32;
        pub fn getrusage(who: i32, usage: *mut [i64; 18]) -> i32;
    }
    pub const PROT_READ: i32 = 1;
    pub const MAP_PRIVATE: i32 = 2;
    pub const MAP_POPULATE: i32 = 0x8000;
    pub const MADV_SEQUENTIAL: i32 = 2;
    pub const MADV_WILLNEED: i32 = 3;
    pub const MADV_POPULATE_READ: i32 = 22;
    pub fn rusage() -> [i64; 18] {
        let mut u = [0i64; 18];
        unsafe { getrusage(0, &mut u) };
        u
    }
    pub fn pin(cpu: usize) {
        let mut m = [0u8; 128];
        m[cpu / 8] |= 1 << (cpu % 8);
        unsafe { sched_setaffinity(0, 128, m.as_ptr()) };
    }
}
#[cfg(not(target_os = "linux"))]
mod sys {
    pub fn rusage() -> [i64; 18] {
        [0; 18]
    }
    pub fn pin(_: usize) {}
}

#[inline(never)]
fn w_sum(s: &[u8]) -> u64 {
    #[cfg(target_feature = "avx2")]
    unsafe {
        use std::arch::x86_64::*;
        let mut a = [_mm256_setzero_si256(); 4];
        let mut i = 0;
        let p = s.as_ptr();
        while i + 128 <= s.len() {
            for k in 0..4 {
                a[k] = _mm256_add_epi64(a[k], _mm256_loadu_si256(p.add(i + 32 * k) as *const __m256i));
            }
            i += 128;
        }
        let mut r = 0u64;
        for k in 0..4 {
            let mut t = [0u64; 4];
            _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, a[k]);
            r = r.wrapping_add(t[0] ^ t[1] ^ t[2] ^ t[3]);
        }
        for &b in &s[i..] {
            r = r.wrapping_add(b as u64);
        }
        r
    }
    #[cfg(not(target_feature = "avx2"))]
    s.iter().fold(0u64, |a, &b| a.wrapping_add(b as u64))
}

#[inline(never)]
fn w_nl(s: &[u8]) -> u64 {
    #[cfg(target_feature = "avx2")]
    unsafe {
        use std::arch::x86_64::*;
        let nl = _mm256_set1_epi8(b'\n' as i8);
        let mut n = 0u64;
        let mut i = 0;
        let p = s.as_ptr();
        while i + 64 <= s.len() {
            let a = _mm256_cmpeq_epi8(_mm256_loadu_si256(p.add(i) as *const __m256i), nl);
            let b = _mm256_cmpeq_epi8(_mm256_loadu_si256(p.add(i + 32) as *const __m256i), nl);
            let m = (_mm256_movemask_epi8(a) as u32 as u64) | ((_mm256_movemask_epi8(b) as u32 as u64) << 32);
            n += m.count_ones() as u64;
            i += 64;
        }
        n + s[i..].iter().filter(|&&b| b == b'\n').count() as u64
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        s.iter().filter(|&&b| b == b'\n').count() as u64
    }
}

#[inline(never)]
fn w_page(s: &[u8]) -> u64 {
    let mut a = 0u64;
    let mut i = 0;
    while i < s.len() {
        a = a.wrapping_add(unsafe { std::ptr::read_volatile(s.as_ptr().add(i)) } as u64);
        i += 4096;
    }
    a
}

fn csvs(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map_or(false, |x| x == "csv")).collect())
        .unwrap_or_default();
    v.sort();
    v
}

fn report_end(ep0: f64, t0: Instant, extra: &str) -> ! {
    let u = sys::rusage();
    eprintln!("[{:.3}s] written", t0.elapsed().as_secs_f64());
    eprintln!(
        "floorp {extra} utime={:.3} stime={:.3} maxrss_mb={} minflt={} majflt={} nvcsw={} nivcsw={}",
        u[0] as f64 + u[1] as f64 * 1e-6,
        u[2] as f64 + u[3] as f64 * 1e-6,
        u[4] >> 10,
        u[8],
        u[9],
        u[14],
        u[15]
    );
    eprintln!("epoch main={:.3} end={:.3}", ep0, ep());
    std::process::exit(0);
}

pub fn run(mode: &str, ep0: f64) -> ! {
    let t0 = Instant::now();
    if mode == "exit0" {
        report_end(ep0, t0, "mode=exit0");
    }
    let ncpu: usize = std::env::var("FLOOR_T")
        .or_else(|_| std::env::var("NCPU"))
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let pin = std::env::var_os("FLOOR_PIN").is_some();
    rayon::ThreadPoolBuilder::new().stack_size(8 << 20).num_threads(ncpu).build_global().unwrap();
    rayon::broadcast(|c| {
        if pin {
            sys::pin(c.index());
        }
    });
    let t_pool = t0.elapsed().as_secs_f64();
    if mode == "pool" {
        report_end(ep0, t0, &format!("mode=pool threads={ncpu} pool_ms={:.2}", t_pool * 1e3));
    }
    if let Some(mb) = mode.strip_prefix("wr") {
        let mb: usize = mb.parse().unwrap_or(12);
        let outd = std::path::PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into()));
        let ud = outd.join("user_daily");
        let cd = outd.join("campaign_daily");
        std::fs::create_dir_all(&ud).unwrap();
        std::fs::create_dir_all(&cd).unwrap();
        let per = (mb << 20) / 16;
        let tw = Instant::now();
        rayon::join(
            || {
                use rayon::prelude::*;
                (0..16usize).into_par_iter().for_each(|i| {
                    let mut v = vec![0u8; per];
                    let mut x = 0x9e3779b97f4a7c15u64 ^ i as u64;
                    for c in v.chunks_mut(8) {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        let b = x.to_le_bytes();
                        c.copy_from_slice(&b[..c.len()]);
                    }
                    std::fs::write(ud.join(format!("part-{i}.parquet")), &v).unwrap();
                })
            },
            || std::fs::write(cd.join("part-0.parquet"), vec![1u8; 200 << 10]).unwrap(),
        );
        report_end(ep0, t0, &format!("mode={mode} write_ms={:.2} pool_ms={:.2}", tw.elapsed().as_secs_f64() * 1e3, t_pool * 1e3));
    }
    let (rd, work) = mode.split_once('_').unwrap_or((mode, "sum"));
    let data = std::path::PathBuf::from(std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let mut paths = vec![];
    for d in ["events", "clicks", "links"] {
        paths.extend(csvs(&data.join(d)));
    }
    paths.push(data.join("products.csv"));
    paths.push(data.join("campaigns.csv"));
    let piece: usize = std::env::var("FLOOR_PIECE").ok().and_then(|s| s.parse().ok()).unwrap_or(4usize) << 20;
    let tl = Instant::now();
    let files: Vec<std::fs::File> = paths.iter().filter_map(|p| std::fs::File::open(p).ok()).collect();
    let lens: Vec<usize> = files.iter().map(|f| f.metadata().map(|m| m.len() as usize).unwrap_or(0)).collect();
    // whole-file maps made by the main thread (like the pipeline), for the map modes
    let maps: Vec<Option<memmap2::Mmap>> = files
        .iter()
        .zip(&lens)
        .map(|(f, &n)| {
            if n == 0 || !(rd.starts_with("mmap") || rd == "popread") {
                return None;
            }
            let m = unsafe { memmap2::Mmap::map(f).ok()? };
            #[cfg(target_os = "linux")]
            {
                if rd == "mmapseq" {
                    let _ = m.advise(memmap2::Advice::Sequential);
                }
                if rd == "mmapwill" {
                    let _ = m.advise(memmap2::Advice::WillNeed);
                }
            }
            Some(m)
        })
        .collect();
    let mut pieces: Vec<(usize, usize, usize)> = vec![];
    for (i, &n) in lens.iter().enumerate() {
        let mut o = 0;
        while o < n {
            let k = piece.min(n - o);
            pieces.push((i, o, k));
            o += k;
        }
    }
    let t_open = tl.elapsed().as_secs_f64();
    let bytes: usize = lens.iter().sum();
    let next = AtomicUsize::new(0);
    let acc = AtomicU64::new(0);
    let busy = AtomicU64::new(0);
    let tmax = AtomicU64::new(0);
    let tmin = AtomicU64::new(u64::MAX);
    let wf: fn(&[u8]) -> u64 = match work {
        "nl" => w_nl,
        "page" => w_page,
        _ => w_sum,
    };
    let bufkb: usize = rd.strip_prefix("read").and_then(|s| s.parse().ok()).unwrap_or(0);
    let tr = Instant::now();
    rayon::broadcast(|_| {
        let ts = Instant::now();
        let mut a = 0u64;
        let mut buf: Vec<u8> = if bufkb > 0 { vec![0u8; bufkb << 10] } else { vec![] };
        loop {
            let j = next.fetch_add(1, Relaxed);
            if j >= pieces.len() {
                break;
            }
            let (fi, o, k) = pieces[j];
            if bufkb > 0 {
                #[cfg(target_os = "linux")]
                {
                    use std::os::fd::AsRawFd;
                    let fd = files[fi].as_raw_fd();
                    let mut p = o;
                    while p < o + k {
                        let want = buf.len().min(o + k - p);
                        let got = unsafe { sys::pread(fd, buf.as_mut_ptr() as *mut _, want, p as i64) };
                        if got <= 0 {
                            break;
                        }
                        a = a.wrapping_add(wf(&buf[..got as usize]));
                        p += got as usize;
                    }
                }
                #[cfg(not(target_os = "linux"))]
                {
                    use std::os::windows::fs::FileExt;
                    let mut p = o;
                    while p < o + k {
                        let want = buf.len().min(o + k - p);
                        let got = files[fi].seek_read(&mut buf[..want], p as u64).unwrap_or(0);
                        if got == 0 {
                            break;
                        }
                        a = a.wrapping_add(wf(&buf[..got]));
                        p += got;
                    }
                }
                continue;
            }
            #[cfg(target_os = "linux")]
            if rd == "populate" || rd == "pieces" {
                use std::os::fd::AsRawFd;
                let fl = sys::MAP_PRIVATE | if rd == "populate" { sys::MAP_POPULATE } else { 0 };
                let p = unsafe { sys::mmap(std::ptr::null_mut(), k, sys::PROT_READ, fl, files[fi].as_raw_fd(), o as i64) };
                let s = unsafe { std::slice::from_raw_parts(p as *const u8, k) };
                a = a.wrapping_add(wf(s));
                continue;
            }
            let m = maps[fi].as_ref().unwrap();
            let s = &m[o..o + k];
            #[cfg(target_os = "linux")]
            if rd == "popread" {
                unsafe { sys::madvise(s.as_ptr() as *mut _, k, sys::MADV_POPULATE_READ) };
            }
            a = a.wrapping_add(wf(s));
        }
        let ns = ts.elapsed().as_nanos() as u64;
        busy.fetch_add(ns, Relaxed);
        tmax.fetch_max(ns, Relaxed);
        tmin.fetch_min(ns, Relaxed);
        acc.fetch_add(a, Relaxed);
    });
    let t_read = tr.elapsed().as_secs_f64();
    report_end(
        ep0,
        t0,
        &format!(
            "mode={mode} threads={ncpu} pin={} files={} pieces={} bytes={bytes} acc={} pool_ms={:.2} open_ms={:.2} read_ms={:.2} ns_per_byte={:.4} gbps={:.2} busy_s={:.3} thread_ms_min={:.2} thread_ms_max={:.2}",
            pin as u8,
            files.len(),
            pieces.len(),
            acc.load(Relaxed),
            t_pool * 1e3,
            t_open * 1e3,
            t_read * 1e3,
            t_read * 1e9 / bytes as f64,
            bytes as f64 / t_read / 1e9,
            busy.load(Relaxed) as f64 / 1e9,
            tmin.load(Relaxed) as f64 / 1e6,
            tmax.load(Relaxed) as f64 / 1e6
        ),
    );
}
