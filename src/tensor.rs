use std::fmt::Debug;

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::lazy_buffer::LazyBuffer;

#[derive(Debug, Default)]
pub struct Tensor<D>
where
    D: Dimension,
{
    pub lazy_data: LazyBuffer<D>,
}

#[allow(dead_code)]
impl<D> Tensor<D>
where
    D: Dimension,
{
    pub fn new(array: ArrayBase<OwnedRepr<f32>, D>) -> Self {
        Tensor {
            lazy_data: LazyBuffer::new(array),
        }
    }

    // pub fn full<Sh>(shape: Sh, fill_value: A) -> Self
    // where
    //     Sh: ShapeBuilder<Dim = D>,
    // {
    //     Tensor {
    //         data: Data::Array(Array::from_elem(shape, fill_value)),
    //         ..Default::default()
    //     }
    // }

    // pub fn zeros<Sh>(shape: Sh) -> Self
    // where
    //     A: Zero,
    //     Sh: ShapeBuilder<Dim = D>,
    // {
    //     Tensor {
    //         data: Data::Array(Array::zeros(shape)),
    //         ..Default::default()
    //     }
    // }

    // pub fn ones<Sh>(shape: Sh) -> Self
    // where
    //     A: One,
    //     Sh: ShapeBuilder<Dim = D>,
    // {
    //     Tensor {
    //         data: Data::Array(Array::ones(shape)),
    //         ..Default::default()
    //     }
    // }

    // pub fn full_like<OA>(other: Tensor<OA, D>, fill_value: A) -> Self
    // where
    //     OA: Clone + Debug + Default,
    // {
    //     Tensor::full(other.array.dim(), fill_value)
    // }

    // pub fn zeros_like<OA>(other: Tensor<OA, D>) -> Self
    // where
    //     A: Zero,
    //     OA: Clone + Debug + Default,
    // {
    //     Tensor::zeros(other.array.dim())
    // }

    // pub fn ones_like<OA>(other: Tensor<OA, D>) -> Self
    // where
    //     A: One,
    //     OA: Clone + Debug + Default,
    // {
    //     Tensor::ones(other.array.dim())
    // }

    // pub fn eye(n: Ix) -> Tensor<A, Ix2>
    // where
    //     A: Zero + One,
    // {
    //     Tensor {
    //         data: Data::Array(Array::eye(n)),
    //         ..Default::default()
    //     }
    // }

    // pub fn arange(start: A, stop: A, step: A) -> Tensor<A, Ix1> {
    //     todo!()
    // }

    // pub fn rand<Sh>(shape: Sh) -> Self
    // where
    //     A: Float + SampleUniform,
    //     Sh: ShapeBuilder<Dim = D>,
    // {
    //     Tensor {
    //         array: Array::random(shape, Uniform::new(A::zero(), A::one())),
    //         ..Default::default()
    //     }
    // }

    // pub fn randn<Sh>(shape: Sh) -> Self
    // where
    //     A: Float,
    //     Sh: ShapeBuilder<Dim = D>,
    //     StandardNormal: Distribution<A>,
    // {
    //     Tensor {
    //         array: Array::random(shape, StandardNormal),
    //         ..Default::default()
    //     }
    // }

    // pub fn uniform<Sh>(shape: Sh, low: A, high: A) -> Self
    // where
    //     A: Float + SampleUniform,
    //     Sh: ShapeBuilder<Dim = D>,
    // {
    //     Tensor {
    //         array: Array::random(shape, Uniform::new(low, high)),
    //         ..Default::default()
    //     }
    // }
}
