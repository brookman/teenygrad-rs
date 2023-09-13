use std::fmt::Debug;

use ndarray::{Array, ArrayBase, Dim, Dimension, Ix, Ix1, Ix2, OwnedRepr, ShapeBuilder};
use ndarray_rand::{
    rand_distr::{uniform::SampleUniform, Distribution, StandardNormal, Uniform},
    RandomExt,
};
use num_traits::{Float, One, Zero};

#[derive(Debug)]
pub struct Tensor<A, D>
where
    A: Clone + Debug,
    D: Dimension,
{
    pub array: ArrayBase<OwnedRepr<A>, D>,
}

#[allow(dead_code)]
impl<A, D> Tensor<A, D>
where
    A: Clone + Debug,
    D: Dimension,
{
    pub fn new(array: ArrayBase<OwnedRepr<A>, D>) -> Self {
        Tensor { array }
    }

    pub fn full<Sh>(shape: Sh, fill_value: A) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Tensor {
            array: Array::from_elem(shape, fill_value),
        }
    }

    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        A: Zero,
        Sh: ShapeBuilder<Dim = D>,
    {
        Tensor {
            array: Array::zeros(shape),
        }
    }

    pub fn ones<Sh>(shape: Sh) -> Self
    where
        A: One,
        Sh: ShapeBuilder<Dim = D>,
    {
        Tensor {
            array: Array::ones(shape),
        }
    }

    pub fn full_like<OA>(other: Tensor<OA, D>, fill_value: A) -> Self
    where
        OA: Clone + Debug,
    {
        Tensor::full(other.array.dim(), fill_value)
    }

    pub fn zeros_like<OA>(other: Tensor<OA, D>) -> Self
    where
        A: Zero,
        OA: Clone + Debug,
    {
        Tensor::zeros(other.array.dim())
    }

    pub fn ones_like<OA>(other: Tensor<OA, D>) -> Self
    where
        A: One,
        OA: Clone + Debug,
    {
        Tensor::ones(other.array.dim())
    }

    pub fn eye(n: Ix) -> Tensor<A, Ix2>
    where
        A: Zero + One,
    {
        Tensor {
            array: Array::eye(n),
        }
    }

    pub fn arange(start: A, stop: A, step: A) -> Tensor<A, Ix1> {
        todo!()
    }

    pub fn rand<Sh>(shape: Sh) -> Self
    where
        A: Float + SampleUniform,
        Sh: ShapeBuilder<Dim = D>,
    {
        Tensor {
            array: Array::random(shape, Uniform::new(A::zero(), A::one())),
        }
    }

    pub fn randn<Sh>(shape: Sh) -> Self
    where
        A: Float,
        Sh: ShapeBuilder<Dim = D>,
        StandardNormal: Distribution<A>,
    {
        Tensor {
            array: Array::random(shape, StandardNormal),
        }
    }

    pub fn uniform<Sh>(shape: Sh, low: A, high: A) -> Self
    where
        A: Float + SampleUniform,
        Sh: ShapeBuilder<Dim = D>,
    {
        Tensor {
            array: Array::random(shape, Uniform::new(low, high)),
        }
    }
}
