use std::fmt;

// 定义你的错误类型
#[derive(Debug)]
pub struct Error {
    pub message: String,
}

// 实现 Display 特质以提供人类可读的错误描述
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

// 实现 Error 特质
impl std::error::Error for Error {}

// 定义一个自定义 Result 类型
pub type Result<T> = core::result::Result<T, Error>;
