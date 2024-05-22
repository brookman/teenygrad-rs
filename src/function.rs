use crate::lazy_buffer::LazyBuffer;
use crate::ops::{BinaryOps, Device, UnaryOps};
use crate::tensor::Tensor;
use ndarray::Dimension;
use num_traits::FloatConst;

pub struct BaseFunction {
    pub device: Device,
    pub requires_grad: Option<bool>,
}

pub struct UnaryFunction<D: Dimension> {
    pub base: BaseFunction,
    pub tensor_x: (Tensor<D>, Option<bool>),
}

pub struct BinaryFunction<D: Dimension> {
    pub base: BaseFunction,
    pub tensor_x: (Tensor<D>, Option<bool>),
    pub tensor_y: (Tensor<D>, Option<bool>),
}

pub struct TernaryFunction<D: Dimension> {
    pub base: BaseFunction,
    pub tensor_x: (Tensor<D>, Option<bool>),
    pub tensor_y: (Tensor<D>, Option<bool>),
    pub tensor_z: (Tensor<D>, Option<bool>),
}

pub trait Function1<D: Dimension> {
    fn forward(&mut self, x: LazyBuffer<D>) -> LazyBuffer<D>;
    fn backward(&self, grad_output: LazyBuffer<D>) -> LazyBuffer<D>;

    fn requires_grad(&self) -> bool;
}

pub trait Applicable1<D: Dimension> {
    fn apply(f: impl Function1<D>, x: Tensor<D>) -> Tensor<D>;
}

impl<D: Dimension> Applicable1<D> for UnaryFunction<D> {
    fn apply(mut f: impl Function1<D>, x: Tensor<D>) -> Tensor<D> {
        let ret = Tensor::new(f.forward(x.lazy_data));
        if f.requires_grad() {
            // ret.ctx = f;
        }
        ret
    }
}

impl BaseFunction {
    fn new_unary<D>(device: Device, tensor_x: Tensor<D>) -> UnaryFunction<D>
    where
        D: Dimension,
    {
        let needs_input_grad_x = tensor_x.requires_grad;

        let any_true = needs_input_grad_x == Some(true);
        let any_none = needs_input_grad_x.is_none();

        let requires_grad = if any_true {
            Some(true)
        } else if any_none {
            None
        } else {
            Some(false)
        };

        UnaryFunction {
            base: BaseFunction {
                device,
                requires_grad,
            },
            tensor_x: (tensor_x, needs_input_grad_x),
        }
    }

    fn new_binary<D>(device: Device, tensor_x: Tensor<D>, tensor_y: Tensor<D>) -> BinaryFunction<D>
    where
        D: Dimension,
    {
        let needs_input_grad_x = tensor_x.requires_grad;
        let needs_input_grad_y = tensor_y.requires_grad;

        let any_true = needs_input_grad_x == Some(true) || needs_input_grad_y == Some(true);
        let any_none = needs_input_grad_x.is_none() || needs_input_grad_y.is_none();

        let requires_grad = if any_true {
            Some(true)
        } else if any_none {
            None
        } else {
            Some(false)
        };

        BinaryFunction {
            base: BaseFunction {
                device,
                requires_grad,
            },
            tensor_x: (tensor_x, needs_input_grad_x),
            tensor_y: (tensor_y, needs_input_grad_y),
        }
    }

    fn new_ternary<D>(
        device: Device,
        tensor_x: Tensor<D>,
        tensor_y: Tensor<D>,
        tensor_z: Tensor<D>,
    ) -> TernaryFunction<D>
    where
        D: Dimension,
    {
        let needs_input_grad_x = tensor_x.requires_grad;
        let needs_input_grad_y = tensor_y.requires_grad;
        let needs_input_grad_z = tensor_z.requires_grad;

        let any_true = needs_input_grad_x == Some(true)
            || needs_input_grad_y == Some(true)
            || needs_input_grad_z == Some(true);
        let any_none = needs_input_grad_x.is_none()
            || needs_input_grad_y.is_none()
            || needs_input_grad_z.is_none();

        let requires_grad = if any_true {
            Some(true)
        } else if any_none {
            None
        } else {
            Some(false)
        };

        TernaryFunction {
            base: BaseFunction {
                device,
                requires_grad,
            },
            tensor_x: (tensor_x, needs_input_grad_x),
            tensor_y: (tensor_y, needs_input_grad_y),
            tensor_z: (tensor_z, needs_input_grad_z),
        }
    }
}

// -----------------------------
struct Contiguous;

impl<D: Dimension> Function1<D> for Contiguous {
    fn forward(&mut self, x: LazyBuffer<D>) -> LazyBuffer<D> {
        x.contiguous()
    }

    fn backward(&self, grad_output: LazyBuffer<D>) -> LazyBuffer<D> {
        grad_output.contiguous()
    }

    fn requires_grad(&self) -> bool {
        true
    }
}

struct ContiguousBackward;

impl<D: Dimension> Function1<D> for ContiguousBackward {
    fn forward(&mut self, x: LazyBuffer<D>) -> LazyBuffer<D> {
        x
    }

    fn backward(&self, grad_output: LazyBuffer<D>) -> LazyBuffer<D> {
        grad_output.contiguous()
    }

    fn requires_grad(&self) -> bool {
        true
    }
}

struct Zero;

impl<D: Dimension> Function1<D> for Zero {
    fn forward(&mut self, x: LazyBuffer<D>) -> LazyBuffer<D> {
        x.const_(0.0)
    }

    fn backward(&self, grad_output: LazyBuffer<D>) -> LazyBuffer<D> {
        grad_output.const_(0.0)
    }

    fn requires_grad(&self) -> bool {
        true
    }
}

struct Neg;

impl<D: Dimension> Function1<D> for Neg {
    fn forward(&mut self, x: LazyBuffer<D>) -> LazyBuffer<D> {
        x.unary(UnaryOps::Neg)
    }

    fn backward(&self, grad_output: LazyBuffer<D>) -> LazyBuffer<D> {
        grad_output.unary(UnaryOps::Neg)
    }

    fn requires_grad(&self) -> bool {
        true
    }
}

struct Sin<D: Dimension>(Option<LazyBuffer<D>>);

impl<D: Dimension> Function1<D> for Sin<D> {
    fn forward(&mut self, x: LazyBuffer<D>) -> LazyBuffer<D> {
        self.0 = Some(x.clone());
        x.unary(UnaryOps::Sin)
    }

    fn backward(&self, grad_output: LazyBuffer<D>) -> LazyBuffer<D> {
        let x = self.0.clone().unwrap();
        x.const_(f32::PI() / 2.0)
            .binary(BinaryOps::Sub, &x)
            .unary(UnaryOps::Sin)
            .binary(BinaryOps::Mul, &grad_output)
    }

    fn requires_grad(&self) -> bool {
        true
    }
}

struct Relu<D: Dimension>(Option<LazyBuffer<D>>);

impl<D: Dimension> Function1<D> for Relu<D> {
    fn forward(&mut self, x: LazyBuffer<D>) -> LazyBuffer<D> {
        let ret = x.clone().binary(BinaryOps::Max, &x.const_(0.0));
        self.0 = Some(ret.clone());
        ret
    }

    fn backward(&self, grad_output: LazyBuffer<D>) -> LazyBuffer<D> {
        let x = self.0.clone().unwrap();
        x.clone()
            .const_(0.0)
            .binary(BinaryOps::CmpLt, &x)
            .binary(BinaryOps::Mul, &grad_output)
    }

    fn requires_grad(&self) -> bool {
        true
    }
}
