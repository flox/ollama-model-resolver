//! Terminal/log sanitization helpers.
//!
//! Upstream-controlled strings from ollama.com, registry responses, local
//! Ollama-compatible endpoints, and pull streams must not be printed with raw
//! control characters. These helpers keep display code centralized and make the
//! intended behavior testable.

pub const ERROR_DETAIL_LIMIT_CHARS: usize = 4_096;
pub const RAW_STREAM_LINE_LIMIT_CHARS: usize = 1_024;

/// Sanitize text for terminal/log display while preserving newlines and tabs.
///
/// This is appropriate for multi-line error summaries assembled by this crate.
/// ASCII control characters other than `\n` and `\t` are replaced with a
/// visible replacement character so escape sequences cannot affect terminals.
pub fn terminal_text(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_control() && ch != '\n' && ch != '\t' {
                '\u{FFFD}'
            } else {
                ch
            }
        })
        .collect()
}

/// Sanitize text that must fit on one terminal/log line.
///
/// All ASCII controls are replaced, including `\n`, `\r`, and `\t`, so a hostile
/// upstream value cannot inject extra lines or move the cursor within tabular
/// output.
pub fn terminal_line(value: &str) -> String {
    value
        .chars()
        .map(|ch| if ch.is_ascii_control() { '\u{FFFD}' } else { ch })
        .collect()
}

/// Sanitize and cap a raw upstream payload before including it in an error.
pub fn capped_terminal_detail(value: &str, max_chars: usize) -> String {
    let sanitized = terminal_text(value);
    truncate_chars(&sanitized, max_chars)
}

/// Sanitize and cap a raw upstream payload that is expected to be one line.
pub fn capped_terminal_line(value: &str, max_chars: usize) -> String {
    let sanitized = terminal_line(value);
    truncate_chars(&sanitized, max_chars)
}

pub fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }

    if max_chars == 0 {
        return String::new();
    }

    let mut out: String = value.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_text_replaces_escape_but_keeps_newline_and_tab() {
        assert_eq!(terminal_text("ok\x1b[31m\nnext\tcol"), "ok�[31m\nnext\tcol");
    }

    #[test]
    fn terminal_line_replaces_all_ascii_controls() {
        assert_eq!(terminal_line("a\n\tb\r\x1b[2K"), "a��b��[2K");
    }

    #[test]
    fn capped_detail_sanitizes_and_truncates() {
        assert_eq!(capped_terminal_detail("abc\x1bdef", 5), "abc�…");
    }
}
