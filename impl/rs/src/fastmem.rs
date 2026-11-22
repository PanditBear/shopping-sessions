// memcpy / memmove / memcmp / bcmp for the static musl build.
// musl's x86_64 memcpy starts every copy with `rep movsq` (slow below a few hundred
// bytes) and its memcmp is a byte loop; Rust code, mimalloc and zstd all call these.
// Defining the symbols here makes the static link use these instead of libc.a's.
// Short sizes use overlapping head/tail loads with every load issued before any store,
// so they are also correct for overlapping memmove. Loops move 64 or 128 bytes per
// step through 32-byte registers so the optimizer cannot turn them back into calls.

use core::arch::x86_64::*;

#[inline(always)]
unsafe fn ld(p: *const u8) -> __m256i {
    _mm256_loadu_si256(p as *const __m256i)
}
#[inline(always)]
unsafe fn st(p: *mut u8, v: __m256i) {
    _mm256_storeu_si256(p as *mut __m256i, v)
}

// n <= 256, any overlap.
#[inline(always)]
unsafe fn copy_small(d: *mut u8, s: *const u8, n: usize) {
    if n <= 16 {
        if n >= 8 {
            let a = (s as *const u64).read_unaligned();
            let b = (s.add(n - 8) as *const u64).read_unaligned();
            (d as *mut u64).write_unaligned(a);
            (d.add(n - 8) as *mut u64).write_unaligned(b);
        } else if n >= 4 {
            let a = (s as *const u32).read_unaligned();
            let b = (s.add(n - 4) as *const u32).read_unaligned();
            (d as *mut u32).write_unaligned(a);
            (d.add(n - 4) as *mut u32).write_unaligned(b);
        } else if n >= 2 {
            let a = (s as *const u16).read_unaligned();
            let b = (s.add(n - 2) as *const u16).read_unaligned();
            (d as *mut u16).write_unaligned(a);
            (d.add(n - 2) as *mut u16).write_unaligned(b);
        } else if n == 1 {
            *d = *s;
        }
    } else if n <= 32 {
        let a = _mm_loadu_si128(s as *const __m128i);
        let b = _mm_loadu_si128(s.add(n - 16) as *const __m128i);
        _mm_storeu_si128(d as *mut __m128i, a);
        _mm_storeu_si128(d.add(n - 16) as *mut __m128i, b);
    } else if n <= 64 {
        let a = ld(s);
        let b = ld(s.add(n - 32));
        st(d, a);
        st(d.add(n - 32), b);
    } else if n <= 128 {
        let a = ld(s);
        let b = ld(s.add(32));
        let c = ld(s.add(n - 64));
        let e = ld(s.add(n - 32));
        st(d, a);
        st(d.add(32), b);
        st(d.add(n - 64), c);
        st(d.add(n - 32), e);
    } else {
        let a0 = ld(s);
        let a1 = ld(s.add(32));
        let a2 = ld(s.add(64));
        let a3 = ld(s.add(96));
        let b0 = ld(s.add(n - 128));
        let b1 = ld(s.add(n - 96));
        let b2 = ld(s.add(n - 64));
        let b3 = ld(s.add(n - 32));
        st(d, a0);
        st(d.add(32), a1);
        st(d.add(64), a2);
        st(d.add(96), a3);
        st(d.add(n - 128), b0);
        st(d.add(n - 96), b1);
        st(d.add(n - 64), b2);
        st(d.add(n - 32), b3);
    }
}

// n > 256; correct when d does not overlap s, or d < s.
#[inline(always)]
unsafe fn copy_fwd_big(d: *mut u8, s: *const u8, n: usize) {
    if n >= 8192 {
        core::arch::asm!("rep movsb",
            inout("rcx") n => _, inout("rdi") d => _, inout("rsi") s => _,
            options(nostack, preserves_flags));
        return;
    }
    let t0 = ld(s.add(n - 128));
    let t1 = ld(s.add(n - 96));
    let t2 = ld(s.add(n - 64));
    let t3 = ld(s.add(n - 32));
    let mut i = 0usize;
    while i + 128 < n {
        let a = ld(s.add(i));
        let b = ld(s.add(i + 32));
        let c = ld(s.add(i + 64));
        let e = ld(s.add(i + 96));
        st(d.add(i), a);
        st(d.add(i + 32), b);
        st(d.add(i + 64), c);
        st(d.add(i + 96), e);
        i += 128;
    }
    st(d.add(n - 128), t0);
    st(d.add(n - 96), t1);
    st(d.add(n - 64), t2);
    st(d.add(n - 32), t3);
}

// n > 256; s < d < s + n.
#[inline(always)]
unsafe fn copy_bwd_big(d: *mut u8, s: *const u8, n: usize) {
    let h0 = ld(s);
    let h1 = ld(s.add(32));
    let mut e = n;
    while e > 64 {
        e -= 64;
        let a = ld(s.add(e));
        let b = ld(s.add(e + 32));
        st(d.add(e), a);
        st(d.add(e + 32), b);
    }
    st(d, h0);
    st(d.add(32), h1);
}

#[inline(always)]
pub unsafe fn fm_memcpy(d: *mut u8, s: *const u8, n: usize) -> *mut u8 {
    if n <= 256 {
        copy_small(d, s, n);
    } else {
        copy_fwd_big(d, s, n);
    }
    d
}

#[inline(always)]
pub unsafe fn fm_memmove(d: *mut u8, s: *const u8, n: usize) -> *mut u8 {
    if n <= 256 {
        copy_small(d, s, n);
    } else if (d as usize).wrapping_sub(s as usize) >= n {
        copy_fwd_big(d, s, n);
    } else {
        copy_bwd_big(d, s, n);
    }
    d
}

#[inline(always)]
unsafe fn diff_at(a: *const u8, b: *const u8, mask: u32) -> i32 {
    let k = mask.trailing_zeros() as usize;
    *a.add(k) as i32 - *b.add(k) as i32
}

#[inline(always)]
pub unsafe fn fm_memcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
    if n < 32 {
        if n >= 16 {
            let x = _mm_loadu_si128(a as *const __m128i);
            let y = _mm_loadu_si128(b as *const __m128i);
            let m = !(_mm_movemask_epi8(_mm_cmpeq_epi8(x, y)) as u32) & 0xffff;
            if m != 0 {
                return diff_at(a, b, m);
            }
            let x = _mm_loadu_si128(a.add(n - 16) as *const __m128i);
            let y = _mm_loadu_si128(b.add(n - 16) as *const __m128i);
            let m = !(_mm_movemask_epi8(_mm_cmpeq_epi8(x, y)) as u32) & 0xffff;
            if m != 0 {
                return diff_at(a.add(n - 16), b.add(n - 16), m);
            }
            return 0;
        }
        if n >= 8 {
            let x = (a as *const u64).read_unaligned().swap_bytes();
            let y = (b as *const u64).read_unaligned().swap_bytes();
            if x != y {
                return if x < y { -1 } else { 1 };
            }
            let x = (a.add(n - 8) as *const u64).read_unaligned().swap_bytes();
            let y = (b.add(n - 8) as *const u64).read_unaligned().swap_bytes();
            return if x == y { 0 } else if x < y { -1 } else { 1 };
        }
        if n >= 4 {
            let x = (a as *const u32).read_unaligned().swap_bytes();
            let y = (b as *const u32).read_unaligned().swap_bytes();
            if x != y {
                return if x < y { -1 } else { 1 };
            }
            let x = (a.add(n - 4) as *const u32).read_unaligned().swap_bytes();
            let y = (b.add(n - 4) as *const u32).read_unaligned().swap_bytes();
            return if x == y { 0 } else if x < y { -1 } else { 1 };
        }
        let mut i = 0;
        while i < n {
            let r = *a.add(i) as i32 - *b.add(i) as i32;
            if r != 0 {
                return r;
            }
            i += 1;
        }
        return 0;
    }
    let mut i = 0usize;
    while i + 32 <= n {
        let m = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(ld(a.add(i)), ld(b.add(i)))) as u32);
        if m != 0 {
            return diff_at(a.add(i), b.add(i), m);
        }
        i += 32;
    }
    if i < n {
        let j = n - 32;
        let m = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(ld(a.add(j)), ld(b.add(j)))) as u32);
        if m != 0 {
            return diff_at(a.add(j), b.add(j), m);
        }
    }
    0
}

#[cfg(all(target_os = "linux", target_env = "musl", target_feature = "avx2", not(test)))]
mod exported {
    #[no_mangle]
    pub unsafe extern "C" fn memcpy(d: *mut u8, s: *const u8, n: usize) -> *mut u8 {
        super::fm_memcpy(d, s, n)
    }
    #[no_mangle]
    pub unsafe extern "C" fn memmove(d: *mut u8, s: *const u8, n: usize) -> *mut u8 {
        super::fm_memmove(d, s, n)
    }
    #[no_mangle]
    pub unsafe extern "C" fn memcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
        super::fm_memcmp(a, b, n)
    }
    #[no_mangle]
    pub unsafe extern "C" fn bcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
        super::fm_memcmp(a, b, n)
    }
}
