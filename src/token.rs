use candle_core::Tensor;
use tokenizers::Tokenizer;

use crate::util::Batch;

pub fn run() {
    let device = candle_core::Device::Cpu;
    let tokenizer = Tokenizer::from_file("/Users/yousx/models/yi/tokenizer.json").unwrap();
    let text = crate::util::load_txt_files("data/txt").unwrap();
    let tokens = tokenizer.encode(text, true).unwrap();
    // let tokens = tokenizer.encode("天地abc", true).unwrap();
    let ids = tokens.get_ids();

    let t = Tensor::from_slice(tokens.get_ids(), tokens.len(), &device).unwrap();
    println!("{:?}", t);
    let b = Batch::get_batch(&t, 10, 4);
    println!("x: {:?}", b);
}
