//! Errors that stay in the library. The CLI maps them to [`crate::Exit`].

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{0}")]
    Raw(#[from] std::io::Error),

    #[error("{0}")]
    BadFile(String),

    #[error("{0}")]
    BadParm(String),

    #[error("unsupported: {0}")]
    Unsupported(String),

    #[error("{0}")]
    Convert(String),
}

impl Error {
    pub fn bad_file(msg: impl Into<String>) -> Self {
        Error::BadFile(msg.into())
    }

    pub fn bad_parm(msg: impl Into<String>) -> Self {
        Error::BadParm(msg.into())
    }

    pub fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }

    pub fn convert(msg: impl Into<String>) -> Self {
        Error::Convert(msg.into())
    }

    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
