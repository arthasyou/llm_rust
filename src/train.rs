use crate::error::{Error, Result};
use crate::models::llama::{Cache, Config, Llama};
use crate::util::{self, Batch};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use tokenizers::Tokenizer;
use tqdm::Iter;

static BATCH_SIZE: usize = 2;
static BLOCK_SIZE: usize = 512;

pub fn run() -> Result<()> {
    // let device = Device::Cpu;
    let device = Device::new_metal(0)?;
    // let device = Device::new_cuda(0)?;
    // println!("{:?}", &device);

    // ================================================================
    // preparing data
    // ================================================================

    println!("preparing data........");

    let tokenizer = Tokenizer::from_file("tokenizers/meta_llama/tokenizer.json").unwrap();

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

    let config = Config::config_7b(false);
    let cache = Cache::new(false, &config, DType::F32, &device)?;
    let varmap = VarMap::new();
    let paths = [
        "/Users/you/models/Llama-2-7b-hf/model-00001-of-00002.safetensors",
        "/Users/you/models/Llama-2-7b-hf/model-00002-of-00002.safetensors",
    ];
    let vb = util::from_mmaped_safetensors(&varmap, &paths, DType::F32, &device, false).unwrap();
    // let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = Llama::new(vb, &cache, &config).unwrap();

    // ================================================================
    // training
    // ================================================================

    println!("starting trainning........");

    let params = ParamsAdamW {
        lr: 1e-4,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params).unwrap();

    let steps: std::ops::Range<usize> = 0..100000;

    for step in steps.into_iter().tqdm() {
        let batch = Batch::get_batch(&train_data, BLOCK_SIZE, BATCH_SIZE);
        // println!("X: {:?}", &batch.x);
        let logits = model.forward(&batch.x, 0).unwrap();
        // println!("Step: ++2{:?}", step);

        // println!("logits: {}", &logits);
        // println!("y: {}", &batch.y);

        let loss =
            candle_nn::loss::cross_entropy(&logits, &batch.y.reshape(BATCH_SIZE * BLOCK_SIZE)?)?;
        // println!("Step: ++3{:?}", step);

        opt.backward_step(&loss).unwrap();
        // println!("Step: ++4{:?}", step);
        println!("step: {step} loss: {}", loss.to_vec0::<f32>().unwrap(),);
        if step % 10 == 0 {
            println!("step: {step} loss: {}", loss.to_vec0::<f32>().unwrap(),);
        }

        if step % 500 == 0 {
            println!("saving checkpint: {}", step);
            let file = format!("outputs/checkpoint_{step}.safetensors");
            varmap.save(file).unwrap();
        }
    }

    println!("{:?}", varmap.all_vars());
    varmap.save("outputs/checkpoint.safetensors").unwrap();
    Ok(())
}
