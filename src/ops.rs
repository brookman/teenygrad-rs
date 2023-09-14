pub enum Ops {
    Unary(UnaryOps),
    Binary(BinaryOps),
    Ternary(TernaryOps),
    Movement(MovementOps),
    Load(LoadOps),
}

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

pub enum BinaryOps {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Mod,
    CmpLt,
}
pub enum ReduceOps {
    Sum,
    Max,
}
pub enum TernaryOps {
    MulAcc,
    Where,
}
pub enum MovementOps {
    Reshape,
    Permute,
    Expand,
    Pad,
    Shrink,
    Stride,
}
pub enum LoadOps {
    Empty,
    Rand,
    Const,
    From,
    Contiguous,
    Custom,
}

#[derive(Debug, Default)]
pub enum Device {
    #[default]
    CPU,
}
