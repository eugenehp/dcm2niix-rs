//! GE Protocol Data Block `(0025,101B)` — gzip text with slice order / MB / group delay.
//!
//! Layout: 4-byte LE compressed length + gzip payload (C++ `geProtocolBlock`).

use flate2::read::GzDecoder;
use std::io::Read;

#[derive(Debug, Clone, Default)]
pub struct GeProtocolBlock {
    pub slice_order: i32,
    pub view_order: i32,
    pub mb_accel: i32,
    pub n_slices: i32,
    /// Seconds (from `DELACQNOAV`); may be forced to 0 / -1 for multiphase IOPT.
    pub group_delay_s: f64,
    pub iopt: String,
    pub seq: String,
}

/// Parse OB bytes from `(0025,101B)`. Returns `None` when signature / inflate fails.
pub fn parse_ge_protocol_block(ob: &[u8]) -> Option<GeProtocolBlock> {
    if ob.len() < 24 {
        return None;
    }
    let cmp_len = u32::from_le_bytes([ob[0], ob[1], ob[2], ob[3]]) as usize;
    let gzip = &ob[4..];
    if gzip.len() < 20 || cmp_len == 0 {
        return None;
    }
    let gzip = if cmp_len <= gzip.len() {
        &gzip[..cmp_len]
    } else {
        gzip
    };
    if gzip[0] != 31 || gzip[1] != 139 || gzip[2] != 8 {
        return None;
    }
    let mut dec = GzDecoder::new(gzip);
    let mut plain = Vec::new();
    if dec.read_to_end(&mut plain).is_err() || plain.is_empty() {
        return None;
    }
    if plain.first() == Some(&b'<') && plain.get(1) == Some(&b'?') {
        // XML protocol block not supported (same as C++).
        eprintln!(
            "New XML-based GE Protocol Block is not yet supported: please report issue on dcm2niix Github page"
        );
        return None;
    }
    Some(parse_plain(&plain))
}

fn parse_plain(plain: &[u8]) -> GeProtocolBlock {
    let text = String::from_utf8_lossy(plain);
    let mut out = GeProtocolBlock {
        slice_order: read_key_i(&text, "\nSLICEORDER").max(-1),
        view_order: read_key_i(&text, "VIEWORDER"),
        mb_accel: read_key_i(&text, "MBACCEL"),
        n_slices: read_key_i(&text, "NOSLC"),
        group_delay_s: read_key_f(&text, "DELACQNOAV"),
        iopt: read_key_str(&text, "IOPT"),
        seq: read_key_str(&text, "PSEQ"),
    };
    let delacq = read_key_str(&text, "DELACQ");
    if out.iopt.contains("MPh") {
        if delacq == "Minimum" {
            out.group_delay_s = 0.0;
        }
        if out.iopt.contains("MPhVar") {
            out.group_delay_s = -1.0;
        }
    }
    out
}

fn find_key<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let idx = text.find(key)?;
    Some(&text[idx + key.len()..])
}

fn read_key_i(text: &str, key: &str) -> i32 {
    let Some(rest) = find_key(text, key) else {
        return 0;
    };
    let rest = rest.trim_start_matches(|c: char| c == '=' || c.is_whitespace());
    let rest = rest.strip_prefix('"').unwrap_or(rest);
    let num: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-' || *c == '+')
        .collect();
    num.parse().unwrap_or(0)
}

fn read_key_f(text: &str, key: &str) -> f64 {
    let Some(rest) = find_key(text, key) else {
        return 0.0;
    };
    let rest = rest.trim_start_matches(|c: char| c == '=' || c.is_whitespace());
    let rest = rest.strip_prefix('"').unwrap_or(rest);
    let num: String = rest
        .chars()
        .take_while(|c| {
            c.is_ascii_digit() || *c == '-' || *c == '+' || *c == '.' || *c == 'e' || *c == 'E'
        })
        .collect();
    num.parse().unwrap_or(0.0)
}

fn read_key_str(text: &str, key: &str) -> String {
    let Some(rest) = find_key(text, key) else {
        return String::new();
    };
    let rest = rest.trim_start_matches(|c: char| c == '=' || c.is_whitespace());
    if let Some(stripped) = rest.strip_prefix('"') {
        return stripped.split('"').next().unwrap_or("").to_string();
    }
    rest.split(|c: char| c == '\n' || c == '\r')
        .next()
        .unwrap_or("")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sample_ge_protocol_from_qa() {
        let path = "/Users/Shared/dcm_qa_nih/In/20180918GE/mr_0006/axial_epi_fmri_interleaved_s_to_i-00084.dcm";
        let Ok(data) = std::fs::read(path) else {
            return;
        };
        let needle = [0x25u8, 0x00, 0x1b, 0x10];
        let Some(i) = data.windows(4).position(|w| w == needle) else {
            return;
        };
        let ln = u32::from_le_bytes(data[i + 8..i + 12].try_into().unwrap()) as usize;
        let ob = &data[i + 12..i + 12 + ln];
        let block = parse_ge_protocol_block(ob).expect("protocol block");
        assert!(block.n_slices > 0 || block.slice_order >= 0);
    }
}
