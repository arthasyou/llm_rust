// 定义你的错误类型
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Normal Error at --> {message}")]
    Norm { message: String },
    #[error("Io Error")]
    IoError(#[from] std::io::Error),
    #[error("Tensor Error")]
    TensorError(#[from] candle_core::error::Error),
}

pub type Result<T> = core::result::Result<T, Error>;
