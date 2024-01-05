mod error;
mod generation;
mod gpt;
mod train;
mod util;

fn main() {
    train::train().unwrap();
}
