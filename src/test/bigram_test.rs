mod bigram;
mod error;
mod util;

use crate::bigram::Bigram;
use crate::error::Result;
// use crate::models::bigram::Bigram;
use crate::util::{decode, encode, load_txt_file, sorted_char, tokenization};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use util::{create_block, split_data};

static BLOCK_SIZE: usize = 8;
static BATCH_SIZE: usize = 8;

fn main() -> Result<()> {
    let device = Device::Cpu;

    // preparing data
    let text = load_txt_file("data/wizard_of_oz.txt")?;
    let chars: Vec<char> = sorted_char(&text);
    let vacab_size = chars.len();
    let (encoder, decoder) = tokenization(&chars);
    let data = encode(&text, &encoder);
    let (train_data, val_data) = split_data(data);
    let train_blocks = create_block(train_data, BLOCK_SIZE);
    // let batch = train_blocks.get_batch(BATCH_SIZE, &device);

    // println!("{:?}", batch);

    // creating models
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Bigram::new(vb, vacab_size, vacab_size, device.clone());

    // training
    let params = ParamsAdamW {
        lr: 1e-3,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params).unwrap();

    for step in 0..1 {
        let batch = train_blocks.get_batch(BATCH_SIZE, &device);
        let (_ys, loss) = model.forward(&batch.x, Some(&batch.y));
        let loss = loss.unwrap();

        opt.backward_step(&loss).unwrap();
        if step % 100 == 0 {
            println!("{step} {}", loss.to_vec0::<f32>().unwrap());
        }
    }

    println!("{:?}", varmap.all_vars());
    varmap.save("checkpoint.safetensors").unwrap();

    let input = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
    let code = model.generate(&input, 500);
    let output = decode(&code, &decoder);
    println!("{:#?}", output);
    Ok(())
}
