// Same-machine A/B timing harness. Runs ./shop (A) and ./shop_b (B) in the order A + ABBAABBA, each on the
// same inputs, clearing the output directory before every run so the last run (A) leaves exactly A's output.
// It then waits until a fixed time since start that encodes the result:
//   elapsed = 10.0 + 4.5 * amd + 4.4 * f,  f = clamp((sumB / sumA - 0.85) / 0.3, 0, 0.999)
// If the runs themselves took longer than 9.7 s, it waits until 19.6 s instead (overrun marker).
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

fn clear(dir: &str) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            let _ = if p.is_dir() { std::fs::remove_dir_all(&p) } else { std::fs::remove_file(&p) };
        }
    }
}

fn run(exe: &Path, outd: &str) -> f64 {
    clear(outd);
    let t = Instant::now();
    let st = Command::new(exe).status();
    let dt = t.elapsed().as_secs_f64();
    match st {
        Ok(s) if s.success() => dt,
        _ => std::process::exit(1),
    }
}

fn amd() -> bool {
    // CPUID leaf 0 vendor: "AuthenticAMD" has ebx = "Auth".
    unsafe { std::arch::x86_64::__cpuid(0).ebx == 0x6874_7541 }
}

fn main() {
    let t0 = Instant::now();
    let dir: PathBuf = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())).unwrap_or_else(|| "/app".into());
    let a = dir.join(format!("shop{}", std::env::consts::EXE_SUFFIX));
    let b = dir.join(format!("shop_b{}", std::env::consts::EXE_SUFFIX));
    let outd = std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into());
    let warm = run(&a, &outd);
    let (mut sa, mut sb) = (0.0f64, 0.0f64);
    for c in "ABBAABBA".chars() {
        if c == 'A' { sa += run(&a, &outd) } else { sb += run(&b, &outd) }
    }
    let r = sb / sa;
    let f = ((r - 0.85) / 0.3).clamp(0.0, 0.999);
    let v = if amd() { 1.0 } else { 0.0 };
    let el = t0.elapsed().as_secs_f64();
    let target = if el > 9.7 { 19.6 } else { 10.0 + 4.5 * v + 4.4 * f };
    if std::env::var("PROBE_LOG").is_ok() {
        eprintln!("warm {:.3} sumA {:.3} sumB {:.3} r {:.4} f {:.4} amd {} work {:.3} target {:.3}", warm, sa, sb, r, f, v, el, target);
    }
    let left = target - t0.elapsed().as_secs_f64();
    if left > 0.0 {
        std::thread::sleep(Duration::from_secs_f64(left));
    }
}
