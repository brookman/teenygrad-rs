use ndarray::array;

use crate::tensor::Tensor;

mod tensor;

fn main() {
    let tensor = Tensor::new(array![1, 2, 3]);
    println!("{:?}", tensor);

    let tensor = Tensor::<f32, _>::zeros((2, 2, 3));
    println!("{:?}", tensor);

    let tensor = Tensor::<f32, _>::ones((1, 2, 3, 4));
    println!("{:?}", tensor);

    let tensor = Tensor::<f32, _>::randn((1, 2, 3, 4));
    println!("{:?}", tensor);

    let tensor = Tensor::uniform((2, 2), 0.0, 10.0);
    println!("{:?}", tensor);
}
