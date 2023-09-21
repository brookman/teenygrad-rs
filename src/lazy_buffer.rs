#![allow(dead_code)]

use std::{fmt::Debug, vec};

use ndarray::{
    Array, ArrayBase, Axis, Dimension, IntoDimension, Ix, Ix1, Ix2, OwnedRepr, RemoveAxis,
    ShapeBuilder, SliceArg, Zip,
};
use ndarray_rand::{
    rand_distr::{StandardNormal, Uniform},
    RandomExt,
};

use crate::ops::{BinaryOps, Device, ReduceOps, TernaryOps, UnaryOps};

#[derive(Debug, Default, PartialEq, Clone)]
pub struct LazyBuffer<D>
where
    D: Dimension,
{
    pub device: Device,
    pub array: ArrayBase<OwnedRepr<f32>, D>,
}

impl<D> LazyBuffer<D>
where
    D: Dimension,
{
    // Constructors
    pub fn new(array: ArrayBase<OwnedRepr<f32>, D>) -> Self {
        LazyBuffer {
            device: Device::CPU,
            array,
        }
    }

    pub fn full<Sh>(shape: Sh, fill_value: f32) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        LazyBuffer::new(Array::from_elem(shape, fill_value))
    }

    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        LazyBuffer::new(Array::zeros(shape))
    }

    pub fn ones<Sh>(shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        LazyBuffer::new(Array::ones(shape))
    }

    pub fn full_like(other: &LazyBuffer<D>, fill_value: f32) -> Self {
        LazyBuffer::full(other.array.dim(), fill_value)
    }

    pub fn zeros_like(other: &LazyBuffer<D>) -> Self {
        LazyBuffer::zeros(other.array.dim())
    }

    pub fn ones_like(other: &LazyBuffer<D>) -> Self {
        LazyBuffer::ones(other.array.dim())
    }

    pub fn eye(n: Ix) -> LazyBuffer<Ix2> {
        LazyBuffer::new(Array::eye(n))
    }

    pub fn arrange(_start: f32, _stop: f32, _step: f32) -> LazyBuffer<Ix1> {
        todo!()
    }

    pub fn rand<Sh>(shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        LazyBuffer::new(Array::random(shape, Uniform::new(0.0, 1.0)))
    }

    pub fn rand_n<Sh>(shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        LazyBuffer::new(Array::random(shape, StandardNormal))
    }

    pub fn uniform<Sh>(shape: Sh, low: f32, high: f32) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        LazyBuffer::new(Array::random(shape, Uniform::new(low, high)))
    }

    pub fn const_(&self, x: f32) -> Self {
        LazyBuffer::full_like(self, x)
    }

    // Misc
    pub fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    pub fn contiguous(self) -> Self {
        self
    }

    pub fn realize(self) -> Self {
        self
    }

    pub fn from_cpu(array: ArrayBase<OwnedRepr<f32>, D>) -> Self {
        LazyBuffer::new(array)
    }

    pub fn to_cpu(self) -> ArrayBase<OwnedRepr<f32>, D> {
        self.array
    }

    // LoadOps
    pub fn load_rand<Sh>(shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        LazyBuffer::rand(shape)
    }

    pub fn load_const<Sh>(shape: Sh, fill_value: f32) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        LazyBuffer::full(shape, fill_value)
    }

    // MovementOps
    pub fn reshape<E>(self, shape: E) -> LazyBuffer<E::Dim>
    where
        E: IntoDimension,
    {
        LazyBuffer::new(self.array.into_shape(shape).unwrap())
    }

    pub fn expand<E>(self, shape: E) -> LazyBuffer<E::Dim>
    where
        E: IntoDimension,
    {
        LazyBuffer::new(self.array.broadcast(shape).unwrap().to_owned())
    }

    pub fn shrink_or_stride<S>(self, slices: S) -> LazyBuffer<S::OutDim>
    where
        S: SliceArg<D>,
    {
        // TODO: Make sure "slice" can do everything "shrink" and "stride" does.
        LazyBuffer::new(self.array.slice(slices).to_owned())
    }

    pub fn permute<E>(self, permutation: E) -> LazyBuffer<D>
    where
        E: IntoDimension<Dim = D>,
    {
        LazyBuffer::new(self.array.permuted_axes(permutation))
    }

    pub fn pad(self, padding: Vec<(usize, usize)>) -> LazyBuffer<D> {
        // TODO: Optimize by allocating an array of the final size and copying the initial array into it
        let mut padded_array = self.array;

        if padding.len() != padded_array.ndim() {
            panic!("Padding dimension mismatch");
        }

        for i in 0..padded_array.ndim() {
            let (before, after) = padding[i];
            let current_shape = padded_array.shape().to_vec();
            let mut to_concat = vec![];

            if before > 0 {
                let mut before_shape = current_shape.clone();
                before_shape[i] = before;
                to_concat.push(Array::zeros(before_shape));
            }

            to_concat.push(padded_array.into_dyn());

            if after > 0 {
                let mut after_shape = current_shape;
                after_shape[i] = after;
                to_concat.push(Array::zeros(after_shape));
            }

            padded_array = ndarray::concatenate(
                Axis(i),
                &to_concat.iter().map(|a| a.view()).collect::<Vec<_>>(),
            )
            .expect("Error during padding")
            .into_dimensionality::<D>()
            .expect("Error changing dimensionality");
        }

        LazyBuffer::new(padded_array)
    }

    // UnaryOps
    pub fn unary(self, op: UnaryOps) -> LazyBuffer<D> {
        match op {
            UnaryOps::Noop => self,
            UnaryOps::Exp2 => LazyBuffer::new(self.array.mapv(f32::exp2)),
            UnaryOps::Log2 => LazyBuffer::new(self.array.mapv(f32::log2)),
            UnaryOps::Cast => todo!(),
            UnaryOps::Sin => LazyBuffer::new(self.array.mapv(f32::sin)),
            UnaryOps::Sqrt => LazyBuffer::new(self.array.mapv(f32::sqrt)),
            UnaryOps::Recip => todo!(),
            UnaryOps::Neg => LazyBuffer::new(-self.array),
        }
    }

    // BinaryOps
    pub fn binary(self, op: BinaryOps, y: &LazyBuffer<D>) -> Self {
        match op {
            BinaryOps::Add => LazyBuffer::new(self.array + &y.array),
            BinaryOps::Sub => LazyBuffer::new(self.array - &y.array),
            BinaryOps::Mul => LazyBuffer::new(self.array * &y.array),
            BinaryOps::Div => LazyBuffer::new(self.array / &y.array),
            BinaryOps::Max => LazyBuffer::new(
                Zip::from(&self.array)
                    .and(&y.array)
                    .map_collect(|a, b| a.max(*b)),
            ),
            BinaryOps::Mod => todo!(),
            BinaryOps::CmpLt => LazyBuffer::new(
                // TODO: Migrate to bool (currently f32 with true == 1.0 and false == 0.0)
                Zip::from(&self.array)
                    .and(&y.array)
                    .map_collect(|a, b| if a < b { 1.0 } else { 0.0 }),
            ),
        }
    }

    // TernaryOps
    pub fn ternary(self, op: TernaryOps, y: LazyBuffer<D>, z: LazyBuffer<D>) -> Self {
        LazyBuffer::new(match op {
            TernaryOps::MulAcc => todo!(),
            TernaryOps::Where => {
                Zip::from(&self.array) // TODO: Migrate to bool (currently f32 with true != 0.0 and false == 0.0)
                    .and(&y.array)
                    .and(&z.array)
                    .map_collect(|a, b, c| if *a != 0.0 { *b } else { *c })
            }
        })
    }

    // ReduceOps
    pub fn reduce<E>(self, op: ReduceOps, new_shape: E) -> Self
    where
        D: RemoveAxis,
        E: IntoDimension,
    {
        let dimension = new_shape.into_dimension();
        let axes = shape_to_axes(self.array.shape(), dimension.slice());

        let mut reduced_array = self.array.clone().into_dyn();

        for x in axes {
            reduced_array = match op {
                ReduceOps::Sum => reduced_array.sum_axis(x),
                ReduceOps::Max => reduced_array.map_axis(x, |x| {
                    *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
                }),
            };
        }

        LazyBuffer::new(
            reduced_array
                .into_shape(dimension)
                .unwrap()
                .into_dimensionality::<D>()
                .unwrap(),
        )
    }
}

fn shape_to_axes(old_shape: &[usize], new_shape: &[usize]) -> Vec<Axis> {
    if old_shape.len() != new_shape.len() {
        panic!("Reduce shapes must have same dimensions");
    }

    old_shape
        .iter()
        .zip(new_shape.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b) // TODO: ensure that b is 1
        .map(|(i, (_, _))| Axis(i))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use ndarray::{array, s};

    use crate::ops::{BinaryOps, ReduceOps};
    use crate::{lazy_buffer::LazyBuffer, ops::UnaryOps};

    #[test]
    fn load_rand_creates_buffer_with_correct_shape() {
        let b = LazyBuffer::load_rand((2, 3, 1));
        assert_eq!(b.shape(), &[2, 3, 1]);
    }

    #[test]
    fn load_cost_creates_correct_buffer() {
        let b = LazyBuffer::load_const((2, 2), 4.2);
        assert_eq!(b.array, array![[4.2, 4.2], [4.2, 4.2]]);
    }

    #[test]
    fn contiguous_returns_self() {
        let b = LazyBuffer::load_rand((2, 3, 1));
        assert_eq!(b.clone().contiguous(), b);
    }

    #[test]
    fn realize_returns_self() {
        let b = LazyBuffer::load_rand((2, 3, 1));
        assert_eq!(b.clone().realize(), b);
    }

    #[test]
    fn const_returns_full_buffer_of_same_size() {
        let b = LazyBuffer::load_rand((1, 2));
        assert_eq!(b.const_(1.0).array, array![[1.0, 1.0]]);
    }

    #[test]
    fn from_cpu_calls_new() {
        let b = LazyBuffer::from_cpu(array![[1.0, 2.0]]);
        assert_eq!(b.shape(), &[1, 2]);
        assert_eq!(b.array, array![[1.0, 2.0]]);
    }

    #[test]
    fn to_cpu_returns_array() {
        let b = LazyBuffer::load_const((1, 2), 3.0);
        assert_eq!(b.clone().to_cpu(), array![[3.0, 3.0]]);
    }

    #[test]
    fn movement_reshape() {
        let b = LazyBuffer::load_const((2, 2), 3.0);
        assert_eq!(b.reshape((1, 4)).shape(), &[1, 4]);
    }

    #[test]
    fn movement_expand() {
        let b = LazyBuffer::load_const((1, 2), 3.0);
        assert_eq!(b.clone().expand((3, 1, 2)).shape(), &[3, 1, 2]);
        assert_eq!(b.clone().expand((2, 2)).shape(), &[2, 2]);
    }

    #[test]
    fn movement_shrink() {
        let b = LazyBuffer::load_const((4, 5), 3.0);
        assert_eq!(
            b.clone().shrink_or_stride(s![0..4; 1, 0..5; 1]).shape(),
            &[4, 5]
        );
        assert_eq!(
            b.clone().shrink_or_stride(s![0..1; 1, 0..2; 1]).shape(),
            &[1, 2]
        );
    }

    #[test]
    fn movement_permute() {
        let b = LazyBuffer::load_const((2, 3, 4), 3.0);
        assert_eq!(b.clone().permute((0, 1, 2)).shape(), &[2, 3, 4]);
        assert_eq!(b.clone().permute((2, 0, 1)).shape(), &[4, 2, 3]);
    }

    #[test]
    fn movement_pad() {
        let b = LazyBuffer::load_const((2, 2), 3.0);
        assert_eq!(b.clone().pad(vec![(0, 0), (0, 0)]).shape(), &[2, 2]);
        assert_eq!(b.clone().pad(vec![(1, 1), (1, 1)]).shape(), &[4, 4]);
        assert_eq!(
            b.clone().pad(vec![(1, 1), (1, 1)]).array,
            array![
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 3.0, 0.0],
                [0.0, 3.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]
            ]
        );
        assert_eq!(
            b.clone().pad(vec![(0, 1), (1, 1)]).array,
            array![
                [0.0, 3.0, 3.0, 0.0],
                [0.0, 3.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]
            ]
        );
    }

    #[test]
    fn movement_stride() {
        let b = LazyBuffer::load_const((4, 6), 3.0);
        assert_eq!(
            b.clone().shrink_or_stride(s![..; 1, ..; 1]).shape(),
            &[4, 6]
        );
        assert_eq!(
            b.clone().shrink_or_stride(s![..; 2, ..; 2]).shape(),
            &[2, 3]
        );
    }

    #[test]
    fn unary_noop_does_nothing() {
        let b = LazyBuffer::load_const((1, 2), 3.0);
        assert_eq!(b.clone().unary(UnaryOps::Noop), b);
    }

    #[test]
    fn unary_exp2() {
        let b = LazyBuffer::new(array![0.0, 1.0, 2.0]);
        assert_eq!(b.clone().unary(UnaryOps::Exp2).array, array![1.0, 2.0, 4.0]);
    }

    #[test]
    fn unary_log2() {
        let b = LazyBuffer::new(array![1.0, 8.0]);
        assert_eq!(b.clone().unary(UnaryOps::Log2).array, array![0.0, 3.0]);
    }

    #[test]
    fn unary_sin() {
        let b = LazyBuffer::new(array![0.0, PI * 0.5]);
        assert_eq!(b.clone().unary(UnaryOps::Sin).array, array![0.0, 1.0]);
    }

    #[test]
    fn unary_sqrt() {
        let b = LazyBuffer::new(array![1.0, 64.0]);
        assert_eq!(b.clone().unary(UnaryOps::Sqrt).array, array![1.0, 8.0]);
    }

    #[test]
    fn unary_neg() {
        let b = LazyBuffer::new(array![1.0, -2.0]);
        assert_eq!(b.clone().unary(UnaryOps::Neg).array, array![-1.0, 2.0]);
    }

    #[test]
    fn binary_add() {
        let a = LazyBuffer::new(array![1.0, -2.0]);
        let b = LazyBuffer::new(array![-3.0, 4.0]);
        assert_eq!(a.binary(BinaryOps::Add, &b).array, array![-2.0, 2.0]);
    }

    #[test]
    fn binary_sub() {
        let a = LazyBuffer::new(array![1.0, -2.0]);
        let b = LazyBuffer::new(array![-3.0, 4.0]);
        assert_eq!(a.binary(BinaryOps::Sub, &b).array, array![4.0, -6.0]);
    }

    #[test]
    fn binary_mul() {
        let a = LazyBuffer::new(array![1.0, -2.0]);
        let b = LazyBuffer::new(array![-3.0, 4.0]);
        assert_eq!(a.binary(BinaryOps::Mul, &b).array, array![-3.0, -8.0]);
    }

    #[test]
    fn binary_div() {
        let a = LazyBuffer::new(array![1.0, -2.0]);
        let b = LazyBuffer::new(array![-3.0, 4.0]);
        assert_eq!(
            a.binary(BinaryOps::Div, &b).array,
            array![1.0 / -3.0, -2.0 / 4.0]
        );
    }

    #[test]
    fn binary_max() {
        let a = LazyBuffer::new(array![1.0, -2.0]);
        let b = LazyBuffer::new(array![-3.0, 4.0]);
        assert_eq!(a.binary(BinaryOps::Max, &b).array, array![1.0, 4.0]);
    }

    #[test]
    fn binary_cmplt() {
        let a = LazyBuffer::new(array![1.0, -2.0]);
        let b = LazyBuffer::new(array![-3.0, 4.0]);
        assert_eq!(a.binary(BinaryOps::CmpLt, &b).array, array![0.0, 1.0]);
    }

    #[test]
    fn reduce_sum() {
        let b = LazyBuffer::load_const((2, 3, 4), 3.0);
        let r1 = b.clone().reduce(ReduceOps::Sum, (1, 3, 4));
        assert_eq!(r1.shape(), &[1, 3, 4]);
    }

    #[test]
    fn reduce_max() {
        let b = LazyBuffer::load_const((2, 3, 4), 3.0);
        let r1 = b.clone().reduce(ReduceOps::Max, (1, 3, 4));
        assert_eq!(r1.shape(), &[1, 3, 4]);
        assert_eq!(
            r1.array,
            array![[
                [3.0, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0]
            ]]
        );
    }
}
