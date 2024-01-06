use crate::error::{Error, Result};
use crate::gpt::{Config, Gpt};
// use crate::models::bigram::Bigram;
use crate::gpt::Cache;
use crate::util::{decode, encode, load_txt_file, sorted_char, tokenization};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use chrono::Local;
use rayon::prelude::*;
use std::sync::mpsc;
use std::thread;

use crate::util::{create_block, split_data};

static BATCH_SIZE: usize = 1;
static BLOCK_SIZE: usize = 128;

static VOCAB_SIZE: usize = 50254;
static HIDDEN_SIZE: usize = 256;
static INTERMEDIATE_SIZE: usize = 512;
static HIDDEN_LAYER: usize = 2;
static ATTENTION_HEADS: usize = 4;
static KEY_VALUE_HEADS: usize = 4;
static ROPE_THETA: f32 = 100_000.0;
static NORM_EPS: f64 = 1e-6;

pub fn train() -> Result<()> {
    let device = Device::Cpu;
    // let device = Device::new_metal(0)?;
    // let device = Device::new_cuda(0)?;
    // println!("{:?}", &device);

    // preparing data
    let text = load_txt_file("data/wizard_of_oz.txt").map_err(|e| Error::Norm {
        message: e.to_string(),
    })?;
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

    println!("{:?}", model);

    let (encoder, decoder) = tokenization(&chars);
    let data = encode(&text, &encoder);
    let (train_data, val_data) = split_data(data);
    let train_blocks = create_block(train_data, BLOCK_SIZE);

    println!("String traing.........");

    // training
    let params = ParamsAdamW {
        lr: 1e-4,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params).unwrap();
    let (tx, rx) = mpsc::channel();

    let other_thread = thread::spawn(move || {
        for loss in rx {
            println!("recv loss: {:?}", loss);
            opt.backward_step(&loss).expect("Error in backward step");
        }
    });

    println!("String looping.......");
    let background_thread = thread::spawn(move || {
        let num_cpus = num_cpus::get();
        (0..num_cpus - 1).into_par_iter().for_each(|main_step| {
            let tx = tx.clone();
            for step in 0..1000 {
                let batch = train_blocks.get_batch(BATCH_SIZE, &device);

                let logits = model.forward(&batch.x, 0).expect("Error in forward pass");

                let loss = candle_nn::loss::cross_entropy(
                    &logits,
                    &batch.y.flatten_to(1).expect("Error in flattening"),
                )
                .expect("Error in loss calculation");
                // println!("loss: {:?}", loss);
                tx.send(loss).unwrap();
            }
        });
    });

    background_thread.join().unwrap();

    other_thread.join().unwrap();
    println!("{:?}", varmap.all_vars());
    varmap.save("outputs/checkpoint.safetensors").unwrap();

    Ok(())
}
