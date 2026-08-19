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

/// How firmly the caller wants a particular port.
///
/// The distinction is load-bearing. An explicit `--port` must be bound
/// exactly, because a client was told to use it and silently relocating
/// the server breaks precisely the caller who was most specific. A port
/// remembered from a previous run is only a preference: replaying it with
/// explicit-strictness would mean that using `--port` once permanently
/// disabled the 11434→11435 fallback for every later bare `tqf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortRequest {
    /// No preference: the default port, with fallback.
    Default,
    /// Remembered from a previous run: try it, fall back if it is busy.
    Preferred(u16),
    /// Named on this run's command line: bind it or fail.
    Explicit(u16),
}

pub async fn resolve_and_bind(
    host: Option<&str>,
    requested_port: PortRequest,
    insecure: bool,
) -> Result<BoundServer> {
    let ip: IpAddr = match host {
        None => IpAddr::V4(Ipv4Addr::LOCALHOST),
        Some(h) => h
            .parse()
            .map_err(|_| ConfigError::InvalidHost(h.to_string()))?,
    };

    let (listener, _port) = match requested_port {
        // A remembered preference still falls back, because nobody was
        // told to expect it on this run.
        PortRequest::Preferred(preferred) => {
            match TcpListener::bind(SocketAddr::new(ip, preferred)).await {
                Ok(listener) => (listener, preferred),
                Err(_) => {
                    tracing::warn!(
                        preferred,
                        "the remembered port is busy; falling back to the default"
                    );
                    bind_with_fallback(ip).await?
                }
            }
        }
        PortRequest::Explicit(requested) => {
            let listener = TcpListener::bind(SocketAddr::new(ip, requested))
                .await
                .map_err(|err| {
                    std::io::Error::new(
                        err.kind(),
                        format!(
                            "cannot bind the requested port {ip}:{requested}: {err}. Free that \
                             port, or omit --port to use the default with automatic fallback."
                        ),
                    )
                })?;
            (listener, requested)
        }
        PortRequest::Default => bind_with_fallback(ip).await?,
    };
    // Ask the socket, don't assume: with `--port 0` the caller wants an
    // OS-assigned ephemeral port, and reporting the *requested* number
    // would hand every downstream consumer (`--open`'s client config, the
    // GUI's base URL, the startup banner) an address nothing is listening
    // on.
    let addr = listener.local_addr()?;

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
                    // Printed, not only logged: this is the difference
                    // between "my Ollama client cannot see tqf" and a
                    // one-line explanation of why (spec §69 requires a
                    // *clearly reported* fallback). The occupant is very
                    // often a real Ollama, which owns this port by
                    // convention.
                    let occupant = match identify_occupant(SocketAddr::new(ip, DEFAULT_PORT)).await
                    {
                        Occupant::Ollama => "a real Ollama server",
                        Occupant::Unknown => "another process",
                        // `Tqf` is handled above and never reaches here.
                        Occupant::Tqf => "another tqf instance",
                    };
                    println!(
                        "tqf: port {DEFAULT_PORT} is already used by {occupant}; listening on \
                         {FALLBACK_PORT} instead.\n     Ollama-compatible clients pointed at the \
                         default http://{ip}:{DEFAULT_PORT} will NOT reach tqf — point them at \
                         http://{ip}:{FALLBACK_PORT}, or stop the other server and restart tqf."
                    );
                    Ok((listener, FALLBACK_PORT))
                }
                Err(fallback_err) => Err(fallback_err.into()),
            }
        }
    }
}

/// What is already listening on a port tqf wanted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occupant {
    /// Another tqf: `/health` answered with this crate's own payload.
    Tqf,
    /// A real Ollama: `GET /` answers with its fixed liveness string.
    /// Worth naming separately because 11434 is Ollama's port by
    /// convention, so this is the single most likely collision.
    Ollama,
    Unknown,
}

/// Best-effort identification of whoever holds a port, so the fallback
/// message can name the actual culprit instead of shrugging (spec Part IX
/// section 69: a "clearly reported fallback", not a mysterious one).
pub async fn identify_occupant(addr: SocketAddr) -> Occupant {
    if probe_health(addr).await {
        return Occupant::Tqf;
    }
    match http_get(addr, "/").await {
        Some(body) if body.contains(OLLAMA_LIVENESS_BODY) => Occupant::Ollama,
        _ => Occupant::Unknown,
    }
}

/// The exact string a real Ollama answers `GET /` with, and the one TQF's
/// own Ollama-compatible surface answers with too.
pub const OLLAMA_LIVENESS_BODY: &str = "Ollama is running";

/// Best-effort check for "is an existing tqf instance already listening
/// here," so a busy default port produces "tqf is already running" instead
/// of a mysterious bind failure (spec Part IX section 69).
async fn probe_health(addr: SocketAddr) -> bool {
    http_get(addr, "/health")
        .await
        .is_some_and(|body| body.contains(r#""status":"ok""#))
}

/// A deliberately tiny blocking-free HTTP/1.1 GET. `reqwest` is already a
/// dependency, but this path runs during bind — before any server exists
/// and, for `tqf status`, outside any HTTP client's runtime — so a
/// 20-line raw request avoids standing a client up just to ask one
/// question. Returns the whole response (headers included) or `None`.
pub(crate) async fn http_get(addr: SocketAddr, path: &str) -> Option<String> {
    let Ok(Ok(mut stream)) = tokio::time::timeout(PROBE_TIMEOUT, TcpStream::connect(addr)).await
    else {
        return None;
    };

    let request = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await.ok()?;

    let mut buf = Vec::new();
    tokio::time::timeout(PROBE_TIMEOUT, stream.read_to_end(&mut buf))
        .await
        .ok()?
        .ok()?;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

fn generate_api_key() -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open("/dev/urandom")?;
    let mut buf = [0u8; 32];
    file.read_exact(&mut buf)?;
    Ok(buf.iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An explicit `--port` must bind exactly that port. Port 0 asks the
    /// OS for an ephemeral one, which is the only port guaranteed free in
    /// a test, so this asserts the request is honored rather than routed
    /// through the 11434/11435 fallback pair.
    #[tokio::test]
    async fn an_explicit_port_is_bound_exactly_and_never_falls_back() {
        let bound = resolve_and_bind(Some("127.0.0.1"), PortRequest::Explicit(0), false)
            .await
            .expect("binding an ephemeral port must succeed");
        assert_ne!(bound.addr.port(), DEFAULT_PORT);
        assert_ne!(bound.addr.port(), FALLBACK_PORT);
        assert_eq!(
            bound.addr.port(),
            bound.listener.local_addr().unwrap().port()
        );
        assert!(bound.api_key.is_none(), "loopback stays no-auth (spec §69)");
    }

    /// The whole point of the explicit-port rule: a client was told to use
    /// this port, so silently moving the server would break exactly the
    /// caller who was most specific. It must be an error instead.
    #[tokio::test]
    async fn an_occupied_explicit_port_errors_instead_of_relocating() {
        let squatter = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let occupied = squatter.local_addr().unwrap().port();

        let result =
            resolve_and_bind(Some("127.0.0.1"), PortRequest::Explicit(occupied), false).await;

        let error = result.err().expect("an occupied explicit port must fail");
        let message = error.to_string();
        assert!(
            message.contains(&occupied.to_string()),
            "the error must name the port the caller asked for: {message}"
        );
        assert!(
            message.contains("--port"),
            "the error must say how to recover: {message}"
        );
    }

    /// A non-loopback bind mints an API key unless `--insecure`
    /// (spec Part IX section 74).
    #[tokio::test]
    async fn a_non_loopback_bind_requires_a_key_unless_insecure() {
        // 0.0.0.0 is bindable in any environment that can bind loopback.
        let Ok(bound) = resolve_and_bind(Some("0.0.0.0"), PortRequest::Explicit(0), false).await
        else {
            return; // sandbox refuses non-loopback binds; nothing to assert
        };
        assert!(
            bound.api_key.is_some(),
            "a non-loopback bind must require an API key"
        );

        let insecure = resolve_and_bind(Some("0.0.0.0"), PortRequest::Explicit(0), true)
            .await
            .expect("insecure non-loopback bind must succeed");
        assert!(
            insecure.api_key.is_none(),
            "--insecure is the documented opt-out"
        );
    }

    /// Nothing is listening on an ephemeral port that was just closed, so
    /// the occupant probe must not claim otherwise.
    #[tokio::test]
    async fn identifying_an_empty_port_reports_unknown_not_a_false_positive() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        assert_eq!(identify_occupant(addr).await, Occupant::Unknown);
    }

    /// Regression: `--port` was written into the persisted config and
    /// then replayed as an *explicit* request on every later run, so
    /// using the flag once permanently disabled the 11434→11435 fallback
    /// — and turned a busy port into a hard startup failure for a user
    /// who had not asked for that port on this run.
    #[tokio::test]
    async fn a_remembered_port_falls_back_while_an_explicit_one_does_not() {
        let squatter = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let occupied = squatter.local_addr().unwrap().port();

        // Remembered: busy is not fatal, it just moves.
        let preferred =
            resolve_and_bind(Some("127.0.0.1"), PortRequest::Preferred(occupied), false)
                .await
                .expect("a remembered port must fall back rather than fail");
        assert_ne!(preferred.addr.port(), occupied);

        // Explicit: busy is an error, because a client was told this port.
        let explicit =
            resolve_and_bind(Some("127.0.0.1"), PortRequest::Explicit(occupied), false).await;
        assert!(
            explicit.is_err(),
            "an explicitly requested busy port must fail rather than relocate"
        );
    }

    /// A free remembered port is still honored — falling back is only for
    /// when it is unavailable.
    #[tokio::test]
    async fn a_free_remembered_port_is_used_as_is() {
        let probe = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let free = probe.local_addr().unwrap().port();
        drop(probe);

        let bound = resolve_and_bind(Some("127.0.0.1"), PortRequest::Preferred(free), false)
            .await
            .expect("a free remembered port must bind");
        assert_eq!(bound.addr.port(), free);
    }
}
