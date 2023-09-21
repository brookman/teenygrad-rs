// # An instantiation of the Function is the Context
// class Function:
//   def __init__(self, device:str, *tensors:Tensor):
//     self.device = device
//     self.needs_input_grad = [t.requires_grad for t in tensors]
//     self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
//     if self.requires_grad: self.parents = tensors

//   def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
//   def backward(self, *args, **kwargs): raise RuntimeError(f"backward not implemented for {type(self)}")

//   @classmethod
//   def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
//     ctx = fxn(x[0].device, *x)
//     ret = Tensor(ctx.forward(*[t.lazydata for t in x], **kwargs), device=ctx.device, requires_grad=ctx.requires_grad)
//     if ctx.requires_grad and not Tensor.no_grad: ret._ctx = ctx    # used by autograd engine
//     return ret

// use ndarray::ShapeBuilder;

// use crate::{lazy_buffer::LazyBuffer, ops::Device, tensor::Tensor};

// pub trait UnaryFunctionForward {
//     fn forward(&self, x: LazyBuffer) -> LazyBuffer;
// }

// pub trait UnaryFunctionBackward {
//     fn backward(&self, grad_output: LazyBuffer) -> LazyBuffer;
// }

// pub trait ReduceFunctionForward<Sh>
// where
//     Sh: ShapeBuilder,
// {
//     fn forward(&self, x: LazyBuffer, new_shape: Sh) -> LazyBuffer;
// }

// pub trait ReduceFunctionBackward {
//     fn backward(&self, grad_output: LazyBuffer) -> LazyBuffer;
// }

// pub trait BinaryFunctionForward {
//     fn forward(&self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer;
// }

// pub trait BinaryFunctionBackward {
//     fn backward(&self, grad_output: LazyBuffer) -> (Option<LazyBuffer>, Option<LazyBuffer>);
// }

// pub trait TernaryFunctionForward {
//     fn forward(&self, x: LazyBuffer, y: LazyBuffer, z: LazyBuffer) -> LazyBuffer;
// }

// pub trait TernaryFunctionBackward {
//     fn backward(
//         &self,
//         grad_output: LazyBuffer,
//     ) -> (Option<LazyBuffer>, Option<LazyBuffer>, Option<LazyBuffer>);
// }

// pub trait UnaryFunction: UnaryFunctionForward + UnaryFunctionBackward {}
// pub trait BinaryFunction: BinaryFunctionForward + BinaryFunctionBackward {}
// pub trait TernaryFunction: TernaryFunctionForward + TernaryFunctionBackward {}

// pub struct Function {
//     pub device:Device,
//     pub needs_input_grad:bool,
//     pub requires_grad:bool,
//     pub parents:Option<Vec<Tensor<?,?>>>,
// }

// pub struct Function< {
//     pub device:Device,
//     pub needs_input_grad:bool,
//     pub requires_grad:bool,
//     pub parents:Option<Vec<Tensor<?,?>>>,
// }

// impl Function {
//     fn new(device:Device, tensors:Vec<Tensor<?,?>>) -> Self {

//     }
// }
