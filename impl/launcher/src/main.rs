// Picks the build that matches the CPU and replaces this process with it
// (same argv, same environment). Any doubt selects the baseline build.
use std::os::unix::process::CommandExt;
use std::process::Command;

const WIDE: &str = "/app/shop-wide";
const BASE: &str = "/app/shop-base";

#[cfg(target_arch = "x86_64")]
fn wide_ok() -> bool {
    // std's detection reads CPUID and checks XGETBV (XMM/YMM/opmask/ZMM state enabled by the OS).
    std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vl")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512cd")
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("bmi1")
        && std::is_x86_feature_detected!("bmi2")
        && std::is_x86_feature_detected!("fma")
        && std::is_x86_feature_detected!("f16c")
        && std::is_x86_feature_detected!("lzcnt")
        && std::is_x86_feature_detected!("movbe")
        && std::is_x86_feature_detected!("popcnt")
}

#[cfg(not(target_arch = "x86_64"))]
fn wide_ok() -> bool {
    false
}

fn main() {
    let mut args = std::env::args_os();
    let argv0 = args.next();
    let pick = if std::env::var_os("SHOP_FORCE_BASE").is_none() && wide_ok() { WIDE } else { BASE };
    if std::env::var_os("SHOP_LAUNCHER_LOG").is_some() {
        eprintln!("launcher: {}", pick);
    }
    let mut c = Command::new(pick);
    c.args(args);
    if let Some(a) = argv0 {
        c.arg0(a);
    }
    let err = c.exec();
    // exec failed: try the other binary, then give up
    let other = if pick == WIDE { BASE } else { WIDE };
    let mut c2 = Command::new(other);
    c2.args(std::env::args_os().skip(1));
    let err2 = c2.exec();
    eprintln!("launcher: exec failed: {} / {}", err, err2);
    std::process::exit(127);
}
