use ndarray::Ix0;

use crate::tensor::Tensor;

mod function;
mod lazy_buffer;
mod ops;
mod tensor;

fn main() {
    let tensor = Tensor::<Ix0>::default();
    println!("{:?}", tensor);
}
