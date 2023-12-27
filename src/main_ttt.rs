use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    init::DEFAULT_KAIMING_NORMAL, linear, AdamW, Linear, Module, Optimizer, ParamsAdamW,
    VarBuilder, VarMap,
};
fn main() -> Result<()> {
    // Use backprop to run a linear regression between samples and get the coefficients back.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

    // tr(vb);

    let newvb = vb.pp("embedding");

    // let init_ws = DEFAULT_KAIMING_NORMAL;
    // let ws = vb.get((3, 3), "embedding").unwrap();
    // let ws = vb.get_with_hints((2, 1), "weight", init_ws)?;
    // println!("{:?}", ws.to_vec2::<f32>());
    // println!("{:?}", varmap.all_vars());

    // let model = linear(2, 1, vb.pp("linear"))?;

    // println!("{:?}", model);
    println!("{:?}", varmap.all_vars());
    // println!("{:?}", vb.prefix());
    // println!("{:?}", newvb.prefix());

    // let model = linear(2, 1, vb.pp("linear"))?;
    // let params = ParamsAdamW {
    //     lr: 0.1,
    //     ..Default::default()
    // };
    // let mut opt = AdamW::new(varmap.all_vars(), params)?;
    // for step in 0..10000 {
    //     let ys = model.forward(&sample_xs)?;
    //     let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
    //     opt.backward_step(&loss)?;
    //     println!("{step} {}", loss.to_vec0::<f32>()?);
    // }
    Ok(())
}

fn tr(vb: VarBuilder) {}
