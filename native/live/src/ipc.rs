//! The Unix-socket protocol between the web process and the live engine (server/acw/stream/protocol.py, "IPC").
//! Every message: kind (1 byte) + u32 little-endian length + payload.

use std::io::{self, Read, Write};

pub const MSG_START: u8 = b'S';
pub const MSG_AUDIO: u8 = b'A';
pub const MSG_CONFIG: u8 = b'C';
pub const MSG_EVENT: u8 = b'T';
pub const MSG_END: u8 = b'E';
pub const MSG_REF: u8 = b'R';
pub const MSG_TAP: u8 = b'P';
pub const MSG_SILENCE: u8 = b'Z';

pub const MAX_IPC: usize = 4 * 1024 * 1024;

/// IPC audio: u32 seq + u8 kind + data; kind 0 = f32 samples, 3 = an Opus packet.
pub const IPC_F32: u8 = 0;
pub const CODEC_OPUS: u8 = 3;

pub fn read_msg(r: &mut impl Read) -> io::Result<(u8, Vec<u8>)> {
    let mut head = [0u8; 5];
    r.read_exact(&mut head)?;
    let n = u32::from_le_bytes([head[1], head[2], head[3], head[4]]) as usize;
    if n > MAX_IPC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "ipc message too large"));
    }
    let mut payload = vec![0u8; n];
    r.read_exact(&mut payload)?;
    Ok((head[0], payload))
}

/// One message in one write: the reader on the other side never sees half a header.
pub fn write_msg(w: &mut impl Write, kind: u8, payload: &[u8]) -> io::Result<()> {
    let mut buf = Vec::with_capacity(5 + payload.len());
    buf.push(kind);
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(payload);
    w.write_all(&buf)
}

pub enum Audio<'a> {
    F32(Vec<f32>),
    Opus(&'a [u8]),
}

pub fn unpack_audio(p: &[u8]) -> Option<(u32, Audio<'_>)> {
    if p.len() < 5 {
        return None;
    }
    let seq = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
    match p[4] {
        CODEC_OPUS => Some((seq, Audio::Opus(&p[5..]))),
        IPC_F32 => {
            let data = &p[5..];
            if data.len() % 4 != 0 {
                return None;
            }
            Some((seq, Audio::F32(data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())))
        }
        _ => None,
    }
}

pub fn pack_audio(seq: u32, x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + 4 * x.len());
    out.extend_from_slice(&seq.to_le_bytes());
    out.push(IPC_F32);
    for v in x {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn pack_opus(seq: u32, pkt: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + pkt.len());
    out.extend_from_slice(&seq.to_le_bytes());
    out.push(CODEC_OPUS);
    out.extend_from_slice(pkt);
    out
}

/// MSG_SILENCE: u32 seq + u16 samples.
pub fn unpack_silence(p: &[u8]) -> Option<(u32, usize)> {
    if p.len() < 6 {
        return None;
    }
    Some((u32::from_le_bytes([p[0], p[1], p[2], p[3]]), u16::from_le_bytes([p[4], p[5]]) as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut buf = Vec::new();
        write_msg(&mut buf, MSG_AUDIO, &pack_audio(7, &[0.5, -0.25])).unwrap();
        let (kind, payload) = read_msg(&mut &buf[..]).unwrap();
        assert_eq!(kind, MSG_AUDIO);
        match unpack_audio(&payload).unwrap() {
            (7, Audio::F32(x)) => assert_eq!(x, vec![0.5, -0.25]),
            _ => panic!(),
        }
        assert_eq!(unpack_silence(&[1, 0, 0, 0, 0xc0, 3]), Some((1, 960)));
    }
}
