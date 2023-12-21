use crate::error::{Error, Result};
use candle_nn::{embedding, Embedding};
use rand::{seq::SliceRandom, thread_rng};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Read,
};

use rand::Rng;

use candle_core::{Device, Tensor};

pub fn load_txt_file(file_path: &str) -> Result<String> {
    let mut file = File::open(file_path).map_err(|e| Error {
        message: e.to_string(),
    })?;
    let mut text = String::new();

    file.read_to_string(&mut text).map_err(|e| Error {
        message: e.to_string(),
    })?;
    Ok(text)
}

pub fn sorted_char(text: &str) -> Vec<char> {
    let mut chars: Vec<char> = HashSet::<char>::from_iter(text.chars())
        .into_iter()
        .collect();
    chars.sort_unstable();
    chars
}

pub fn tokenization(chars: &Vec<char>) -> (HashMap<char, usize>, HashMap<usize, char>) {
    let string_to_int: HashMap<char, usize> =
        chars.iter().enumerate().map(|(i, &ch)| (ch, i)).collect();
    let int_to_string: HashMap<usize, char> =
        chars.iter().enumerate().map(|(i, &ch)| (i, ch)).collect();
    (string_to_int, int_to_string)
}

pub fn encode(s: &str, mapping: &HashMap<char, usize>) -> Vec<usize> {
    s.chars().filter_map(|c| mapping.get(&c)).cloned().collect()
}

pub fn decode(code: &[usize], mapping: &HashMap<usize, char>) -> String {
    code.iter()
        .filter_map(|&i| mapping.get(&i))
        .cloned()
        .collect()
}

pub fn vec_usize_to_u32(v: Vec<usize>) -> Vec<u32> {
    v.into_iter().map(|x| x as u32).collect()
}

pub fn split_data(data: Vec<u32>) -> (Vec<u32>, Vec<u32>) {
    let len = data.len();

    let n = (0.8 * len as f32) as usize;
    let train_data = &data[..n];
    let val_data = &data[n..];
    let train_data: Vec<u32> = train_data.iter().map(|&x| x).collect();
    let val_data: Vec<u32> = val_data.iter().map(|&x| x).collect();
    (train_data, val_data)
}

#[derive(Debug)]
pub struct Block {
    pub x: Vec<Vec<u32>>,
    pub y: Vec<Vec<u32>>,
    pub size: usize,
}

impl Block {
    pub fn get_batch(&self, batch_size: usize, device: &Device) -> Batch {
        let mut rng = thread_rng();

        // 获取有效索引范围
        let indices: Vec<usize> = (0..self.size).collect();

        // 随机选择索引
        let selected_indices = indices
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect::<Vec<_>>();

        // 根据选择的索引获取 x 和 y 的对应元素
        let selected_x: Vec<Vec<u32>> = selected_indices
            .iter()
            .map(|&i| self.x[i].clone())
            .collect();
        let selected_y: Vec<Vec<u32>> = selected_indices
            .iter()
            .map(|&i| self.y[i].clone())
            .collect();

        Batch {
            x: batch_to_tensor(selected_x, device),
            y: batch_to_tensor(selected_y, device),
        }
    }
}

pub fn create_block(data: Vec<u32>, block_size: usize) -> Block {
    if data.len() < block_size || block_size == 0 {
        return Block {
            x: vec![],
            y: vec![],
            size: 0,
        };
    }

    let windows: Vec<_> = data.windows(block_size).collect();
    let size = windows.len();
    let x: Vec<Vec<u32>> = windows
        .iter()
        .take(size.saturating_sub(1))
        .map(|w| w.to_vec())
        .collect();
    let y: Vec<Vec<u32>> = windows.iter().skip(1).map(|w| w.to_vec()).collect();

    Block {
        x,
        y,
        size: size.saturating_sub(1),
    }
}

fn flatten_vec(nested_vec: Vec<Vec<u32>>) -> Vec<u32> {
    nested_vec.into_iter().flat_map(|inner| inner).collect()
}

#[derive(Debug)]
pub struct Batch {
    pub x: Tensor,
    pub y: Tensor,
}

pub fn batch_to_tensor(data: Vec<Vec<u32>>, device: &Device) -> Tensor {
    let batch_size = data.len();
    let block_size = data[0].len();
    let data = flatten_vec(data);
    Tensor::from_vec(data, (batch_size, block_size), device).unwrap()
}

pub fn create_embedding(vacab_size: usize, hidden_size: usize, device: &Device) -> Embedding {
    let tensor = Tensor::randn(0f32, 1., (vacab_size, hidden_size), &device).unwrap();
    Embedding::new(tensor, hidden_size)
}

pub fn get_btc(embedding: &Tensor) -> (usize, usize, usize) {
    let btc = embedding.dims();
    let b = *btc.get(0).unwrap();
    let t = *btc.get(1).unwrap();
    let c = *btc.get(2).unwrap();
    (b, t, c)
}

// 类似python v[:,-1:,]
pub fn take_vec3_2_last(data: &Vec<Vec<Vec<u32>>>) -> Vec<Vec<u32>> {
    data.iter()
        .map(|outer| outer.last().expect("Outer Vec is empty").clone())
        .collect()
}

// 从一组Vec<f32>随机抽取几个样本
pub fn multinomial(probs: &[f32], num_samples: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let mut samples: Vec<usize> = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let random_value: f32 = rng.gen(); // 生成随机值 [0, 1)
        let mut cumulative_prob = 0.0;

        for (idx, &prob) in probs.iter().enumerate() {
            cumulative_prob += prob;

            if random_value < cumulative_prob {
                samples.push(idx);
                break;
            }
        }
    }

    samples
}

pub fn convert_to_u32(input: Vec<usize>) -> Vec<u32> {
    let result: Vec<u32> = input.iter().map(|&x| x as u32).collect();
    result
}
