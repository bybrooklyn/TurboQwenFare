//! Bind resolution: loopback-by-default with a reported port fallback, and
//! API-key provisioning for non-loopback exposure (spec Part IX sections 69,
//! 74).

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::error::{ConfigError, Result};

pub const DEFAULT_PORT: u16 = 11434;
pub const FALLBACK_PORT: u16 = 11435;

const PROBE_TIMEOUT: Duration = Duration::from_millis(300);

pub struct BoundServer {
    pub listener: TcpListener,
    pub addr: SocketAddr,
    /// `None` on loopback (no-auth, matching Ollama-style local convenience).
    /// `Some` on any non-loopback bind unless the caller explicitly opted
    /// into `--insecure`.
    pub api_key: Option<String>,
}

pub async fn resolve_and_bind(host: Option<&str>, insecure: bool) -> Result<BoundServer> {
    let ip: IpAddr = match host {
        None => IpAddr::V4(Ipv4Addr::LOCALHOST),
        Some(h) => h
            .parse()
            .map_err(|_| ConfigError::InvalidHost(h.to_string()))?,
    };

    let (listener, port) = bind_with_fallback(ip).await?;
    let addr = SocketAddr::new(ip, port);

    let api_key = if ip.is_loopback() {
        None
    } else if insecure {
        tracing::warn!(
            %addr,
            "binding to a non-loopback address with --insecure: anyone on the \
             network can reach model generation with no authentication"
        );
        None
    } else {
        let key = generate_api_key()?;
        tracing::warn!(
            %addr,
            "binding to a non-loopback address: API key required (spec Part IX section 74)"
        );
        println!("tqf: API key for non-loopback access: {key}");
        Some(key)
    };

    Ok(BoundServer {
        listener,
        addr,
        api_key,
    })
}

async fn bind_with_fallback(ip: IpAddr) -> Result<(TcpListener, u16)> {
    match TcpListener::bind(SocketAddr::new(ip, DEFAULT_PORT)).await {
        Ok(listener) => Ok((listener, DEFAULT_PORT)),
        Err(primary_err) => {
            if probe_health(SocketAddr::new(ip, DEFAULT_PORT)).await {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AddrInUse,
                    format!(
                        "tqf is already running on {ip}:{DEFAULT_PORT}; use --host/a different \
                         instance, or stop the existing server first"
                    ),
                )
                .into());
            }

            match TcpListener::bind(SocketAddr::new(ip, FALLBACK_PORT)).await {
                Ok(listener) => {
                    tracing::warn!(
                        default_port = DEFAULT_PORT,
                        fallback_port = FALLBACK_PORT,
                        %primary_err,
                        "default port unavailable (occupied by a non-tqf process); \
                         falling back"
                    );
                    Ok((listener, FALLBACK_PORT))
                }
                Err(fallback_err) => Err(fallback_err.into()),
            }
        }
    }
}

/// Best-effort check for "is an existing tqf instance already listening
/// here," so a busy default port produces "tqf is already running" instead
/// of a mysterious bind failure (spec Part IX section 69).
async fn probe_health(addr: SocketAddr) -> bool {
    let Ok(Ok(mut stream)) = tokio::time::timeout(PROBE_TIMEOUT, TcpStream::connect(addr)).await
    else {
        return false;
    };

    let request = format!("GET /health HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).await.is_err() {
        return false;
    }

    let mut buf = Vec::new();
    let Ok(Ok(_)) = tokio::time::timeout(PROBE_TIMEOUT, stream.read_to_end(&mut buf)).await else {
        return false;
    };

    let text = String::from_utf8_lossy(&buf);
    text.contains(r#""status":"ok""#)
}

fn generate_api_key() -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open("/dev/urandom")?;
    let mut buf = [0u8; 32];
    file.read_exact(&mut buf)?;
    Ok(buf.iter().map(|b| format!("{b:02x}")).collect())
}
