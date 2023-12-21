mod error;
mod model;
mod util;

use crate::error::Result;
use crate::model::Bigram;
use crate::util::{
    batch_to_tensor, decode, encode, load_txt_file, sorted_char, tokenization, vec_usize_to_u32,
};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{VarBuilder, VarMap};
use util::{create_block, split_data};

static BLOCK_SIZE: usize = 8;
static BATCH_SIZE: usize = 8;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let text = load_txt_file("data/wizard_of_oz.txt")?;
    let chars: Vec<char> = sorted_char(&text);
    let vacab_size = chars.len();

    let (encoder, decoder) = tokenization(&chars);
    let data = vec_usize_to_u32(encode(&text, &encoder));
    let (train_data, val_data) = split_data(data);
    let train_blocks = create_block(train_data, BLOCK_SIZE);
    let batch = train_blocks.get_batch(BATCH_SIZE, &device);

    // println!("{:?}", batch);

    let model = Bigram::new(vacab_size, vacab_size, device.clone());
    let input = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
    let code = model.generate(&input, 500);
    let output = decode(&code, &decoder);
    println!("{:#?}", output);
    Ok(())
}
