use candle_core::{Device, Module, Tensor};
use candle_nn::{
    embedding,
    loss::cross_entropy,
    ops::{sigmoid, softmax},
    Embedding, VarBuilder,
};
use tracing_subscriber::filter::targets;

use crate::util::{convert_to_u32, create_embedding, get_btc, multinomial};

#[derive(Debug)]
pub struct Bigram {
    pub embed_token: Embedding,
    pub device: Device,
}

impl Bigram {
    pub fn new(vb: VarBuilder, vacab_size: usize, hidden_size: usize, device: Device) -> Self {
        let embed_token = embedding(vacab_size, hidden_size, vb.pp("embedding")).unwrap();

        // let embedding = create_embedding(vacab_size, hidden_size, &device);

        Self {
            embed_token,
            device,
        }
    }

    pub fn forward(&self, index: &Tensor, targets: Option<&Tensor>) -> (Tensor, Option<Tensor>) {
        let logits = self.embed_token.forward(index).unwrap();
        match targets {
            Some(targets) => {
                let (b, t, c) = get_btc(&logits);
                let logits = logits.reshape((b * t, c)).unwrap();
                let targets = targets.reshape(b * t).unwrap();
                let loss = cross_entropy(&logits, &targets).unwrap();
                (logits, Some(loss))
            }
            None => (logits, None),
        }
    }

    pub fn generate(&self, input: &Tensor, max_tokens: usize) -> Vec<usize> {
        let mut results: Vec<usize> = Vec::new();
        let mut iii = input.clone();
        for _ in 0..max_tokens {
            let (logits, _) = self.forward(&iii, None);
            let (b, t, c) = get_btc(&logits);
            let logits = logits.reshape((b * t, c)).unwrap();
            let probs = softmax(&logits, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            let a = multinomial(&probs, 1);
            results.push(*a.get(0).unwrap());
            let a = convert_to_u32(a);

            iii = Tensor::from_vec(a, (1, 1), &self.device).unwrap();
        }
        results
    }

    // cross_entropy(inp, target);
    // sigmoid(xs)
}
