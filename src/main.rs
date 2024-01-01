mod error;
mod gpt;
mod util;

use crate::error::Result;
use crate::gpt::{Config, Gpt};
// use crate::models::bigram::Bigram;
use crate::util::{decode, encode, load_txt_file, sorted_char, tokenization};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use gpt::Cache;
use util::{create_block, split_data};

static BATCH_SIZE: usize = 512;
static BLOCK_SIZE: usize = 128;

static HIDDEN_SIZE: usize = 256;
static INTERMEDIATE_SIZE: usize = 512;
static HIDDEN_LAYER: usize = 4;
static ATTENTION_HEADS: usize = 4;
static KEY_VALUE_HEADS: usize = 4;
static ROPE_THETA: f32 = 100_000.0;
static NORM_EPS: f64 = 1e-6;

fn main() -> Result<()> {
    // let device = Device::Cpu;
    let device = Device::new_cuda(0)?;
    // preparing data
    let text = load_txt_file("data/wizard_of_oz.txt")?;
    let chars: Vec<char> = sorted_char(&text);
    let vocab_size = chars.len();

    let cfg = Config {
        hidden_size: HIDDEN_SIZE,
        intermediate_size: INTERMEDIATE_SIZE,
        vocab_size,
        num_hidden_layers: HIDDEN_LAYER,
        num_attention_heads: ATTENTION_HEADS,
        num_key_value_heads: KEY_VALUE_HEADS,
        use_flash_attn: false,
        rms_norm_eps: NORM_EPS,
        rope_theta: ROPE_THETA,
    };

    let cache = Cache::new(false, &cfg, DType::F32, &device)?;
    // println!("{:?}", cache);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Gpt::init(vb, &cache, &cfg)?;

    let (encoder, decoder) = tokenization(&chars);
    let data = encode(&text, &encoder);
    let (train_data, val_data) = split_data(data);
    let train_blocks = create_block(train_data, BLOCK_SIZE);

    // let batch = train_blocks.get_batch(BATCH_SIZE, &device);

    // let logits = model.forward(&batch.x, 0)?;

    // // let logit = logit.flatten_to(1)?;
    // println!("{:#?}", logits);
    // // println!("{}", batch.x);
    // println!("{}", &batch.y);

    // let loss = candle_nn::loss::cross_entropy(&logits, &batch.y.flatten_to(1)?)?;
    // println!("{:#?}", loss);

    // training
    let params = ParamsAdamW {
        lr: 1e-4,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params).unwrap();

    for step in 0..10000 {
        // println!("Step: {:?}", step);
        let batch = train_blocks.get_batch(BATCH_SIZE, &device);
        // println!("X: {:?}", &batch.x);
        let logits = model.forward(&batch.x, 0)?;
        // println!("Step: ++2{:?}", step);

        let loss = candle_nn::loss::cross_entropy(&logits, &batch.y.flatten_to(1)?)?;
        // println!("Step: ++3{:?}", step);

        opt.backward_step(&loss).unwrap();
        // println!("Step: ++4{:?}", step);
        // println!("{step} {}", loss.to_vec0::<f32>().unwrap());
        if step % 100 == 0 {
            println!("{step} {}", loss.to_vec0::<f32>().unwrap());
        }
    }

    println!("{:?}", varmap.all_vars());
    varmap.save("checkpoint.safetensors").unwrap();
    Ok(())
}
