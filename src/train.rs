use crate::error::{Error, Result};
use crate::models::yi::{Config, Model};
use crate::util::Batch;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use tokenizers::Tokenizer;

static BATCH_SIZE: usize = 4;
static BLOCK_SIZE: usize = 2048;

pub fn run() -> Result<()> {
    // let device = Device::Cpu;
    let device = Device::new_metal(0)?;
    // let device = Device::new_cuda(0)?;
    // println!("{:?}", &device);

    // ================================================================
    // preparing data
    // ================================================================

    println!("preparing data........");

    let tokenizer = Tokenizer::from_file("tokenizers/yi/tokenizer.json").unwrap();

    let train_text = crate::util::load_txt_files("data/txt/train").unwrap();
    let train_tokens = tokenizer.encode(train_text, true).unwrap();
    let train_data =
        Tensor::from_slice(train_tokens.get_ids(), train_tokens.len(), &device).unwrap();

    // println!("ok");

    // let valid_text = crate::util::load_txt_files("data/txt/valid").unwrap();
    // let valid_tokens = tokenizer.encode(valid_text, true).unwrap();
    // let valid_data =
    //     Tensor::from_slice(valid_tokens.get_ids(), valid_tokens.len(), &device).unwrap();

    // ================================================================
    // initialize model
    // ================================================================

    println!("initializing model........");

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = Config::config_6b();
    let mut model = Model::new(&config, vb).unwrap();

    // ================================================================
    // training
    // ================================================================

    println!("starting trainning........");

    let params = ParamsAdamW {
        lr: 1e-4,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params).unwrap();

    for step in 0..1000 {
        let batch = Batch::get_batch(&train_data, BLOCK_SIZE, BATCH_SIZE);
        // println!("X: {:?}", &batch.x);
        let logits = model.forward(&batch.x, 0)?;
        // println!("Step: ++2{:?}", step);

        // println!("logits: {}", &logits);
        // println!("y: {}", &batch.y);

        let loss = candle_nn::loss::cross_entropy(&logits, &batch.y.flatten_to(1)?)?;
        // println!("Step: ++3{:?}", step);

        opt.backward_step(&loss).unwrap();
        // println!("Step: ++4{:?}", step);
        // println!("{step} {}", loss.to_vec0::<f32>().unwrap());
        if step % 5 == 0 {
            println!("{step} {}", loss.to_vec0::<f32>().unwrap());
        }
    }

    println!("{:?}", varmap.all_vars());
    varmap.save("outputs/checkpoint.safetensors").unwrap();
    Ok(())
}
