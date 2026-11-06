//! The dashboard: the run on one screen, redrawn in place.
//!
//! ```text
//!  puff-rs ▸ cartpole                          Metal · Apple M4 Max · mem 0.1/52G · ram 0.0G
//!  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━──────────────   64%   12.8M of 20.0M   eta 2s
//!
//!  score     476.604  ▁▁▂▂▃▅▆▇████████████████          best 480.614
//!
//!  run                       time                        loss
//!  params          12.7K     rollout   83ms  52%         total       82.672
//!  ...
//! ```

use std::io::{IsTerminal, Write};

use crate::pufferl::{Log, PuffeRL};

const WIDTH: usize = 80;
const MAX_STAT_ROWS: usize = 15;
const SPARK: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const SPARK_WIDTH: usize = 40;
const BAR_WIDTH: usize = 40;
/// Points of the score's history kept; the sparkline averages them down to its width
const HISTORY: usize = 4 * SPARK_WIDTH;

pub struct Dashboard {
    tty: bool,
    last_size: Option<(usize, usize)>,
    /// The sweep metric at each print that trained, oldest first
    history: Vec<f64>,
    best: Option<f64>,
    last_steps: f64,
}

impl Default for Dashboard {
    fn default() -> Self {
        Self::new()
    }
}

/// Rows and columns of the terminal, or what a terminal that does not say is taken to be.
fn terminal_size() -> (usize, usize) {
    #[cfg(unix)]
    unsafe {
        let mut size: libc::winsize = std::mem::zeroed();
        if libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut size) == 0 && size.ws_col > 0 && size.ws_row > 0 {
            return (size.ws_row as usize, size.ws_col as usize);
        }
    }
    (1000, WIDTH)
}

pub fn abbrev(mut value: f64) -> String {
    for suffix in ["", "K", "M", "B"] {
        if value < 1000.0 {
            return format!("{value:.1}{suffix}");
        }
        value /= 1000.0;
    }
    format!("{value:.1}T")
}

pub fn duration(seconds: f64) -> String {
    let ms = (seconds.max(0.0) * 1000.0 + 0.5) as u64;
    if ms < 1000 {
        return format!("{ms}ms");
    }
    if ms < 60_000 {
        return format!("{}s {:03}ms", ms / 1000, ms % 1000);
    }
    if ms < 3_600_000 {
        return format!("{}m {:02}s {:03}ms", ms / 60_000, (ms / 1000) % 60, ms % 1000);
    }
    let s = ms / 1000;
    format!("{}d {}h {}m {}s", s / 86_400, (s / 3600) % 24, (s / 60) % 60, s % 60)
}

/// A count: whole under ten thousand, abbreviated past it
fn count(value: f64) -> String {
    if value < 10_000.0 { format!("{}", value.round() as u64) } else { abbrev(value) }
}

/// A short duration, for the eta: no milliseconds past a second
fn eta(seconds: f64) -> String {
    let s = seconds.max(0.0).round() as u64;
    match s {
        0 if seconds > 0.0 => "<1s".into(),
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m {:02}s", s / 60, s % 60),
        _ => format!("{}h {:02}m", s / 3600, (s / 60) % 60),
    }
}

/// text cut or padded to width columns
fn fit(text: &str, width: usize, right: bool) -> String {
    let cut: String = text.chars().take(width).collect();
    if right { format!("{cut:>width$}") } else { format!("{cut:<width$}") }
}

/// values averaged down to at most width buckets, drawn from lowest to highest
fn sparkline(values: &[f64], width: usize) -> String {
    let n = values.len();
    if n == 0 {
        return String::new();
    }
    let buckets = n.min(width);
    let means: Vec<f64> = (0..buckets)
        .map(|i| {
            let part = &values[i * n / buckets..((i + 1) * n / buckets).max(i * n / buckets + 1)];
            part.iter().sum::<f64>() / part.len() as f64
        })
        .collect();
    let (lo, hi) = means.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    means
        .iter()
        .map(|&v| {
            let t = if hi > lo { (v - lo) / (hi - lo) } else { 0.5 };
            SPARK[((t * (SPARK.len() - 1) as f64).round() as usize).min(SPARK.len() - 1)]
        })
        .collect()
}

struct Paint {
    accent: &'static str,
    value: &'static str,
    dim: &'static str,
    bold: &'static str,
    reset: &'static str,
}

impl Dashboard {
    pub fn new() -> Self {
        Self { tty: std::io::stdout().is_terminal(), last_size: None, history: Vec::new(), best: None, last_steps: -1.0 }
    }

    fn paint(&self) -> Paint {
        if self.tty {
            Paint { accent: "\x1b[38;5;208m", value: "\x1b[97m", dim: "\x1b[90m", bold: "\x1b[1m", reset: "\x1b[0m" }
        } else {
            Paint { accent: "", value: "", dim: "", bold: "", reset: "" }
        }
    }

    /// Keep the metric for the sparkline, once for each print that trained
    fn record(&mut self, steps: f64, metric: Option<f64>) {
        let Some(value) = metric else { return };
        if steps <= self.last_steps {
            return;
        }
        self.last_steps = steps;
        self.history.push(value);
        if self.history.len() > HISTORY {
            // Halve the history by pairs, so the line keeps the whole run at less detail
            self.history = self.history.chunks(2).map(|pair| pair.iter().sum::<f64>() / pair.len() as f64).collect();
        }
        self.best = Some(self.best.map_or(value, |best| best.max(value)));
    }

    pub fn print(&mut self, pufferl: &PuffeRL, log: &Log, epoch: usize) {
        let p = self.paint();
        let (rows, cols) = if self.tty { terminal_size() } else { (1000, WIDTH) };
        let env_name = pufferl.ini.text("base", "env_name");
        let metric_name = pufferl.ini.text("sweep", "metric");
        let metric = log.get(&format!("env/{metric_name}"));
        let steps = log.or("agent_steps", pufferl.global_step as f64);
        let sps = log.or("SPS", 0.0);
        let target = (pufferl.total_epochs * pufferl.batch_size) as f64;
        let remaining = if sps > 0.0 && target > steps { (target - steps) / sps } else { 0.0 };
        let (rollout, train) = (log.or("perf/rollout", 0.0), log.or("perf/train", 0.0));
        let perf_total = rollout + train;
        self.record(steps, metric);

        let mut head = String::new();
        if self.tty {
            head.push_str("\x1b[?2026h");
            head.push_str(if Some((rows, cols)) != self.last_size { "\x1b[H\x1b[J" } else { "\x1b[H" });
            self.last_size = Some((rows, cols));
        }
        let mut stdout = std::io::stdout().lock();
        if self.tty && (cols < WIDTH || rows < 16) {
            let score = metric.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into());
            let compact = format!("puff-rs ▸ {env_name} · {} steps · {} sps · {metric_name} {score} · eta {}",
                abbrev(steps), abbrev(sps), eta(remaining));
            let compact: String = compact.chars().take(cols.saturating_sub(1)).collect();
            let _ = write!(stdout, "{head}{compact}\x1b[K\x1b[J\x1b[?2026l");
            let _ = stdout.flush();
            return;
        }

        let mut lines = Vec::new();

        // Title, and the machine on the right
        let info = &pufferl.gpu.info;
        let (used, total) = (log.or("util/vram_used_gb", 0.0), log.or("util/vram_total_gb", 0.0));
        let memory = if total > 0.0 { format!("{used:.1}/{total:.0}G") } else { format!("{used:.1}G") };
        let machine = format!("{} · {} · mem {memory} · ram {:.1}G", info.backend, info.name, log.or("util/cpu_mem_gb", 0.0));
        let title_width = 11 + env_name.chars().count();
        let machine = fit(&machine, WIDTH.saturating_sub(title_width + 3), true);
        lines.push(format!(" {}{}puff-rs{} {}▸{} {}{env_name}{}  {}{machine}{}",
            p.bold, p.accent, p.reset, p.dim, p.reset, p.value, p.reset, p.dim, p.reset));

        // Progress through total_timesteps
        let done = if target > 0.0 { (steps / target).clamp(0.0, 1.0) } else { 0.0 };
        let filled = (done * BAR_WIDTH as f64).round() as usize;
        let progress = format!("{:>4}%   {} of {}   eta {}", (100.0 * done) as u32, abbrev(steps), abbrev(target), eta(remaining));
        lines.push(format!(" {}{}{}{}{}  {}{progress}{}",
            p.accent, "━".repeat(filled), p.dim, "─".repeat(BAR_WIDTH - filled), p.reset, p.value, p.reset));
        lines.push(String::new());

        // The metric and its history
        let value = metric.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into());
        let best = self.best.map(|v| format!("best {v:.3}")).unwrap_or_default();
        lines.push(format!(" {}{}{} {}{}{}  {}{}{}  {}{}{}",
            p.accent, fit(&metric_name, 8, false), p.reset,
            p.bold, fit(&value, 10, true), p.reset,
            p.accent, fit(&sparkline(&self.history, SPARK_WIDTH), SPARK_WIDTH, false), p.reset,
            p.dim, fit(&best, 14, true), p.reset));
        lines.push(String::new());

        // Three columns: the run, where the time goes, the losses
        let kv = |label: &str, value: String, width: usize| {
            format!("{}{}{} {}{}{}", p.dim, fit(label, 9, false), p.reset, p.value, fit(&value, width, true), p.reset)
        };
        let run = [
            ("params", abbrev(pufferl.num_params() as f64)),
            ("agents", count(pufferl.ini.size("vec", "total_agents") as f64)),
            ("sps", abbrev(sps)),
            ("epoch", epoch.to_string()),
            ("uptime", eta(log.or("uptime", 0.0))),
        ];
        let timing = |sec: f64| {
            let percent = if perf_total > 0.0 { format!("{:.0}%", 100.0 * sec / perf_total) } else { String::new() };
            (duration(sec), percent)
        };
        let time = [
            ("rollout", timing(rollout), true),
            (" model", timing(log.or("perf/eval_model", 0.0)), false),
            (" env", timing(log.or("perf/eval_env", 0.0)), false),
            (" copy", timing(log.or("perf/eval_copy", 0.0)), false),
            ("train", timing(train), true),
            (" model", timing(log.or("perf/train_model", 0.0)), false),
            (" misc", timing(log.or("perf/train_misc", 0.0)), false),
        ];
        let mut loss = vec![
            ("total", "loss/total"), ("policy", "loss/policy"), ("value", "loss/value"), ("entropy", "loss/entropy"),
            ("old_kl", "loss/old_kl"), ("kl", "loss/kl"), ("clipfrac", "loss/clipfrac"),
        ];
        if pufferl.ini.num_opt("train", "target_entropy", 0.0) != 0.0 {
            loss.push(("ent_coef", "train/ent_coef"));
        }
        lines.push(format!(" {}{}run{:24}time{:24}loss{}", p.bold, p.accent, "", "", p.reset));
        for i in 0..run.len().max(time.len()).max(loss.len()) {
            let a = run.get(i).map(|(l, v)| kv(l, v.clone(), 14)).unwrap_or_else(|| " ".repeat(24));
            let b = time.get(i).map(|(l, (t, pct), main)| {
                let (lc, vc) = if *main { (p.value, p.value) } else { (p.dim, p.value) };
                format!("{lc}{}{} {vc}{}{} {}{}{}", fit(l, 9, false), p.reset, fit(t, 9, true), p.reset, p.dim, fit(pct, 4, true), p.reset)
            }).unwrap_or_else(|| " ".repeat(24));
            let c = loss.get(i).map(|(l, key)| kv(l, log.get(key).map(|v| format!("{v:.3}")).unwrap_or_default(), 12)).unwrap_or_default();
            lines.push(format!(" {a}   {b}   {c}"));
        }
        lines.push(String::new());

        // The env's own stats, two to a row
        let used_rows = lines.len() + 2;
        let stat_rows = if self.tty { rows.saturating_sub(used_rows).min(MAX_STAT_ROWS) } else { MAX_STAT_ROWS };
        let stats: Vec<(&str, f64)> = log.0.iter()
            .filter(|(key, _)| key.starts_with("env/") && key != "env/n")
            .map(|(key, value)| (&key[4..], *value)).take(2 * stat_rows).collect();
        if !stats.is_empty() {
            lines.push(format!(" {}{}env{}  {}{} episodes{}", p.bold, p.accent, p.reset, p.dim, count(log.or("env/n", 0.0)), p.reset));
        }
        for pair in stats.chunks(2) {
            let stat = |(key, value): &(&str, f64)| {
                format!("{}{}{} {}{}{}", p.dim, fit(key, 24, false), p.reset, p.value, fit(&format!("{value:.3}"), 12, true), p.reset)
            };
            let right = pair.get(1).map(stat).unwrap_or_default();
            lines.push(format!(" {}   {right}", stat(&pair[0])));
        }

        let eol = if self.tty { "\x1b[K\n" } else { "\n" };
        let mut out = head;
        for line in &lines {
            out.push_str(line);
            out.push_str(eol);
        }
        if self.tty {
            out.push_str("\x1b[J\x1b[?2026l");
        } else {
            out.push('\n');
        }
        let _ = stdout.write_all(out.as_bytes());
        let _ = stdout.flush();
    }
}
