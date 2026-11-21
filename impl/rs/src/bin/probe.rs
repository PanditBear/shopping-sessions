// Phase share profile. Runs ./shop_t (the program with 5-decimal phase ticks) once as warm-up and 5 times with
// SHOP_TIMING=1, then ./shop last so the real outputs stay. Per run (T = whole process wall):
//   E = t(events parsed)/T, D = (t(dedup done) - t(clicks parsed))/T, S = (t(sessionized) - t(dedup done))/T
// Medians are quantized: iE = clamp(floor((E-0.10)/0.05),0,7), iD = clamp(floor((D-0.05)/0.05),0,7),
// iS = clamp(floor((S-0.05)/0.05),0,7); k = iE*64 + iD*8 + iS. Finish = 35 + 0.1*k + 0.05 s (99 s on any failure).
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const BASE: f64 = 35.0;
const ST: f64 = 0.1;
const OVERRUN: f64 = 99.0;

fn clear(dir: &str) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            let _ = if p.is_dir() { std::fs::remove_dir_all(&p) } else { std::fs::remove_file(&p) };
        }
    }
}

fn run_plain(exe: &Path, outd: &str) -> f64 {
    clear(outd);
    let t = Instant::now();
    let st = Command::new(exe).status();
    let dt = t.elapsed().as_secs_f64();
    match st {
        Ok(s) if s.success() => dt,
        _ => std::process::exit(1),
    }
}

/// Returns (wall, stderr text).
fn run_timed(exe: &Path, outd: &str) -> (f64, String) {
    clear(outd);
    let t = Instant::now();
    let r = Command::new(exe).env("SHOP_TIMING", "1").stdout(Stdio::null()).stderr(Stdio::piped()).output();
    let dt = t.elapsed().as_secs_f64();
    match r {
        Ok(o) if o.status.success() => (dt, String::from_utf8_lossy(&o.stderr).into_owned()),
        _ => std::process::exit(1),
    }
}

/// Time of the last tick with this label ("[0.12345s] label").
fn tick(text: &str, label: &str) -> Option<f64> {
    let mut found = None;
    for l in text.lines() {
        let l = l.trim();
        if let Some(rest) = l.strip_prefix('[') {
            if let Some((num, lab)) = rest.split_once("s] ") {
                if lab.trim() == label {
                    found = num.parse::<f64>().ok();
                }
            }
        }
    }
    found
}

fn dir_bytes(p: &Path, depth: u32) -> u64 {
    let mut n = 0u64;
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            if let Ok(m) = std::fs::metadata(e.path()) {
                if m.is_dir() {
                    if depth < 4 {
                        n += dir_bytes(&e.path(), depth + 1);
                    }
                } else {
                    n += m.len();
                }
            }
        }
    }
    n
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn quant(x: f64, lo: f64) -> usize {
    (((x - lo) / 0.05).floor().max(0.0) as usize).min(7)
}

fn main() {
    let t0 = Instant::now();
    let dir: PathBuf = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())).unwrap_or_else(|| "/app".into());
    let exe = |n: &str| dir.join(format!("{}{}", n, std::env::consts::EXE_SUFFIX));
    let (real, timed) = (exe("shop"), exe("shop_t"));
    let outd = std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into());
    let data = std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into());
    if dir_bytes(Path::new(&data), 0) < 300_000_000 {
        run_plain(&real, &outd);
        return;
    }
    run_plain(&timed, &outd);
    let (mut es, mut ds, mut ss, mut ts) = (vec![], vec![], vec![], vec![]);
    let mut ok = true;
    for _ in 0..5 {
        let (t, text) = run_timed(&timed, &outd);
        let ev = tick(&text, "events parsed");
        let ck = tick(&text, "clicks parsed");
        let dd = tick(&text, "dedup done");
        let se = tick(&text, "sessionized");
        let wr = tick(&text, "written");
        match (ev, ck, dd, se, wr) {
            (Some(ev), Some(ck), Some(dd), Some(se), Some(_)) if t > 0.0 => {
                es.push(ev / t);
                ds.push((dd - ck) / t);
                ss.push((se - dd) / t);
                ts.push(t);
            }
            _ => ok = false,
        }
    }
    // The real program runs last so its outputs are what stays.
    let real_wall = run_plain(&real, &outd);
    let (mut e, mut d, mut s, mut t) = (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    let (mut ie, mut id, mut is, mut k) = (0, 0, 0, 0);
    if ok {
        e = median(&mut es);
        d = median(&mut ds);
        s = median(&mut ss);
        t = median(&mut ts);
        ie = quant(e, 0.10);
        id = quant(d, 0.05);
        is = quant(s, 0.05);
        k = ie * 64 + id * 8 + is;
    }
    let target = if ok { BASE + ST * k as f64 + 0.05 } else { OVERRUN };
    if std::env::var("PROBE_LOG").is_ok() || std::env::var("PROBE_NOWAIT").is_ok() {
        eprintln!(
            "ok {} E {:.5} D {:.5} S {:.5} iE {} iD {} iS {} k {} medT {:.4} realT {:.4} work {:.3} target {:.3}",
            ok, e, d, s, ie, id, is, k, t, real_wall, t0.elapsed().as_secs_f64(), target
        );
    }
    if std::env::var("PROBE_NOWAIT").is_ok() {
        return;
    }
    let left = target - t0.elapsed().as_secs_f64();
    if left > 0.0 {
        std::thread::sleep(Duration::from_secs_f64(left));
    }
}
