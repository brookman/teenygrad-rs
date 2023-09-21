#![allow(dead_code)]

#[derive(Debug)]
pub enum Ops {
    Unary(UnaryOps),
    Binary(BinaryOps),
    Ternary(TernaryOps),
    Movement(MovementOps),
    Load(LoadOps),
}

#[derive(Debug)]
pub enum UnaryOps {
    Noop,
    Exp2,
    Log2,
    Cast,
    Sin,
    Sqrt,
    Recip,
    Neg,
}

#[derive(Debug)]
pub enum BinaryOps {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Mod,
    CmpLt,
}

#[derive(Debug)]
pub enum ReduceOps {
    Sum,
    Max,
}

#[derive(Debug)]
pub enum TernaryOps {
    MulAcc,
    Where,
}

#[derive(Debug)]
pub enum MovementOps {
    Reshape,
    Permute,
    Expand,
    Pad,
    Shrink,
    Stride,
}

#[derive(Debug)]
pub enum LoadOps {
    Empty,
    Rand,
    Const,
    From,
    Contiguous,
    Custom,
}

#[derive(Debug, Default, PartialEq, Clone)]
pub enum Device {
    #[default]
    CPU,
}
