//! Series colors `--c1`…`--c6` in OKLCH.
//!
//! Ported from oh-my-pi (`packages/utils/src/color.ts`, `seriesColors` in
//! `packages/tui/src/chat/svg-figure.ts`; MIT). OKLab matrices and the gamut
//! cusp fit are Björn Ottosson's reference values.

/// Hue families (OKLCH degrees): blue, orange, green, purple, pink, teal.
const SERIES_HUES: [f64; 6] = [255.0, 55.0, 145.0, 305.0, 0.0, 200.0];
/// Gold, for a sixth series when the accent's hue crowds out two families.
const RESERVE_HUE: f64 = 90.0;
/// Least hue distance between a family and the accent's hue for both to lead series.
const MIN_SERIES_HUE_GAP: f64 = 30.0;
/// Chroma below which the accent reads as gray and blue leads instead.
const MIN_ACCENT_CHROMA: f64 = 0.04;
/// Lightness band: deep enough on light surfaces, bright enough on dark ones.
const LIGHTNESS_LIGHT: (f64, f64) = (0.52, 0.66);
const LIGHTNESS_DARK: (f64, f64) = (0.68, 0.82);
/// Chroma ceiling: vivid, not neon.
const SERIES_CHROMA: f64 = 0.19;

/// Six series colors of even visual weight, led by the accent's hue (blue
/// when the accent is gray or unparseable), each next hue the family
/// farthest from those already taken.
pub fn series_colors(accent: &str, is_dark: bool) -> [String; 6] {
    let lead = match parse_hex(accent).map(rgb_to_oklch) {
        Some((_, chroma, hue)) if chroma >= MIN_ACCENT_CHROMA => hue,
        _ => SERIES_HUES[0],
    };
    let mut hues = vec![lead];
    let mut pool: Vec<f64> = SERIES_HUES
        .iter()
        .copied()
        .filter(|hue| hue_distance(*hue, lead) >= MIN_SERIES_HUE_GAP)
        .collect();
    while !pool.is_empty() {
        let (best, _) = pool
            .iter()
            .enumerate()
            .map(|(at, hue)| {
                let gap = hues
                    .iter()
                    .map(|taken| hue_distance(*hue, *taken))
                    .fold(f64::INFINITY, f64::min);
                (at, gap)
            })
            .fold((0, f64::NEG_INFINITY), |best, item| {
                if item.1 > best.1 { item } else { best }
            });
        hues.push(pool.remove(best));
    }
    if hues.len() < 6 {
        hues.push(RESERVE_HUE);
    }
    let (low, high) = if is_dark {
        LIGHTNESS_DARK
    } else {
        LIGHTNESS_LIGHT
    };
    std::array::from_fn(|at| {
        let hue = hues[at];
        let (cusp_l, cusp_c) = oklch_cusp(hue);
        oklch_to_hex(cusp_l.clamp(low, high), cusp_c.min(SERIES_CHROMA), hue)
    })
}

/// Whether `color` reads as gray (or is not a parseable hex color), so it
/// cannot serve as a colored accent.
pub(super) fn is_achromatic(color: &str) -> bool {
    !matches!(parse_hex(color).map(rgb_to_oklch), Some((_, chroma, _)) if chroma >= MIN_ACCENT_CHROMA)
}

/// Whether `color` is a `#rgb` / `#rrggbb` color the figure can use as is.
pub(super) fn parse_hex_color(color: &str) -> bool {
    parse_hex(color).is_some()
}

/// `#rgb` / `#rrggbb` (alpha suffixes ignored) to 0..255 channels.
fn parse_hex(value: &str) -> Option<(f64, f64, f64)> {
    let hex = value.trim().strip_prefix('#')?;
    let channel = |s: &str| u8::from_str_radix(s, 16).ok().map(f64::from);
    match hex.len() {
        3 | 4 => {
            let digit = |at: usize| channel(&hex[at..at + 1]).map(|v| v * 17.0);
            Some((digit(0)?, digit(1)?, digit(2)?))
        }
        6 | 8 => Some((
            channel(&hex[0..2])?,
            channel(&hex[2..4])?,
            channel(&hex[4..6])?,
        )),
        _ => None,
    }
}

fn linearize(channel: f64) -> f64 {
    let c = channel / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn delinearize(linear: f64) -> f64 {
    let c = if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    c * 255.0
}

fn linear_rgb_to_oklab(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let l = (0.412_221_470_8 * r + 0.536_332_536_3 * g + 0.051_445_992_9 * b).cbrt();
    let m = (0.211_903_498_2 * r + 0.680_699_545_1 * g + 0.107_396_956_6 * b).cbrt();
    let s = (0.088_302_461_9 * r + 0.281_718_837_6 * g + 0.629_978_700_5 * b).cbrt();
    (
        0.210_454_255_3 * l + 0.793_617_785 * m - 0.004_072_046_8 * s,
        1.977_998_495_1 * l - 2.428_592_205 * m + 0.450_593_709_9 * s,
        0.025_904_037_1 * l + 0.782_771_766_2 * m - 0.808_675_766 * s,
    )
}

fn oklab_to_linear_rgb(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let l_ = (l + 0.396_337_777_4 * a + 0.215_803_757_3 * b).powi(3);
    let m_ = (l - 0.105_561_345_8 * a - 0.063_854_172_8 * b).powi(3);
    let s_ = (l - 0.089_484_177_5 * a - 1.291_485_548 * b).powi(3);
    (
        4.076_741_662_1 * l_ - 3.307_711_591_3 * m_ + 0.230_969_929_2 * s_,
        -1.268_438_004_6 * l_ + 2.609_757_401_1 * m_ - 0.341_319_396_5 * s_,
        -0.004_196_086_3 * l_ - 0.703_418_614_7 * m_ + 1.707_614_701 * s_,
    )
}

/// (lightness, chroma, hue degrees).
fn rgb_to_oklch((r, g, b): (f64, f64, f64)) -> (f64, f64, f64) {
    let (l, a, b) = linear_rgb_to_oklab(linearize(r), linearize(g), linearize(b));
    let mut hue = b.atan2(a).to_degrees();
    if hue < 0.0 {
        hue += 360.0;
    }
    (l, a.hypot(b), hue)
}

fn hue_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).abs() % 360.0;
    d.min(360.0 - d)
}

/// Max OKLab saturation (C/L) inside sRGB for hue direction `(a, b)`:
/// polynomial fit plus one Halley step.
fn compute_max_saturation(a: f64, b: f64) -> f64 {
    let (k0, k1, k2, k3, k4, wl, wm, ws) = if -1.881_703_28 * a - 0.809_364_93 * b > 1.0 {
        (
            1.190_862_77,
            1.765_767_28,
            0.596_626_41,
            0.755_151_97,
            0.567_712_45,
            4.076_741_662_1,
            -3.307_711_591_3,
            0.230_969_929_2,
        )
    } else if 1.814_441_04 * a - 1.194_452_76 * b > 1.0 {
        (
            0.739_565_15,
            -0.459_544_04,
            0.082_854_27,
            0.125_410_7,
            0.145_032_04,
            -1.268_438_004_6,
            2.609_757_401_1,
            -0.341_319_396_5,
        )
    } else {
        (
            1.357_336_52,
            -0.009_157_99,
            -1.151_302_1,
            -0.505_596_06,
            0.006_921_67,
            -0.004_196_086_3,
            -0.703_418_614_7,
            1.707_614_701,
        )
    };
    let s = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;
    let kl = 0.396_337_777_4 * a + 0.215_803_757_3 * b;
    let km = -0.105_561_345_8 * a - 0.063_854_172_8 * b;
    let ks = -0.089_484_177_5 * a - 1.291_485_548 * b;
    let l_ = 1.0 + s * kl;
    let m_ = 1.0 + s * km;
    let s_ = 1.0 + s * ks;
    let f = wl * l_.powi(3) + wm * m_.powi(3) + ws * s_.powi(3);
    let f1 = wl * 3.0 * kl * l_ * l_ + wm * 3.0 * km * m_ * m_ + ws * 3.0 * ks * s_ * s_;
    let f2 = wl * 6.0 * kl * kl * l_ + wm * 6.0 * km * km * m_ + ws * 6.0 * ks * ks * s_;
    s - (f * f1) / (f1 * f1 - 0.5 * f * f2)
}

/// sRGB gamut cusp (lightness, chroma) for an OKLCH hue.
fn oklch_cusp(hue: f64) -> (f64, f64) {
    let radians = hue.to_radians();
    let (a, b) = (radians.cos(), radians.sin());
    let s_cusp = compute_max_saturation(a, b);
    let (r, g, b) = oklab_to_linear_rgb(1.0, s_cusp * a, s_cusp * b);
    let l_cusp = (1.0 / r.max(g).max(b)).cbrt();
    (l_cusp, l_cusp * s_cusp)
}

const GAMUT_EPSILON: f64 = 1e-4;

fn in_gamut((r, g, b): (f64, f64, f64)) -> bool {
    let fits = |c: f64| (-GAMUT_EPSILON..=1.0 + GAMUT_EPSILON).contains(&c);
    fits(r) && fits(g) && fits(b)
}

/// OKLCH to `#rrggbb`, gamut-mapped by chroma bisection.
fn oklch_to_hex(lightness: f64, chroma: f64, hue: f64) -> String {
    let l = lightness.clamp(0.0, 1.0);
    let radians = hue.to_radians();
    let at = |c: f64| oklab_to_linear_rgb(l, c * radians.cos(), c * radians.sin());
    let mut rgb = at(chroma);
    if !in_gamut(rgb) {
        let (mut lo, mut hi) = (0.0, chroma);
        for _ in 0..20 {
            let mid = (lo + hi) / 2.0;
            if in_gamut(at(mid)) {
                lo = mid
            } else {
                hi = mid
            }
        }
        rgb = at(lo);
    }
    let byte = |linear: f64| delinearize(linear).clamp(0.0, 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", byte(rgb.0), byte(rgb.1), byte(rgb.2))
}
