mod error;
mod generation;
mod models;
mod token;
mod train;
mod util;

fn main() {
    // let num_cpus = num_cpus::get();

    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(num_cpus)
    //     .build_global()
    //     .unwrap();

    // train::train().unwrap();
    token::run();
}
