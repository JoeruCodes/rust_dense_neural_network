use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use ndarray::Array2;
use num_traits::Float;

use super::{
    cache::{CacheKey, CacheManager, NodeId},
    gpu_add, gpu_dot, gpu_elemental_mul, gpu_reshape,
    kernel::cuda_traits::{CudaVectorAdd, SupportedFloat},
    NODE_ID_GENERATOR,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix<T>
where
    T: SupportedFloat + CudaVectorAdd,
{
    pub data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
    _marker: PhantomData<T>,
}

impl<T> Matrix<T>
where
    T: SupportedFloat + CudaVectorAdd,
{
    /// Creates a new Matrix. Asserts that data length matches rows * cols.
    pub fn new(data: Vec<T>, rows: usize, cols: usize) -> Self {
        assert!(
            data.len() == rows * cols,
            "Data length does not match matrix dimensions"
        );
        Matrix {
            data,
            rows,
            cols,
            _marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Nodes<T>
where
    T: SupportedFloat + CudaVectorAdd,
{
    Base {
        id: NodeId,
        matrix: Matrix<T>,
    },
    Add {
        id: NodeId,
        left: Box<Nodes<T>>,
        right: Box<Nodes<T>>,
    },
    Sub {
        id: NodeId,
        left: Box<Nodes<T>>,
        right: Box<Nodes<T>>,
    },
    ElementalMul {
        id: NodeId,
        left: Box<Nodes<T>>,
        right: Box<Nodes<T>>,
    },
    Dot {
        id: NodeId,
        left: Box<Nodes<T>>,
        right: Box<Nodes<T>>,
    },
    Reshape {
        id: NodeId,
        node: Box<Nodes<T>>,
        dims: (usize, usize),
    },
    Transformation {
        id: NodeId,
        tgt: Box<Nodes<T>>,
        transformation: fn(T) -> T,
    },
}
impl<T> Nodes<T>
where
    T: Clone + Debug + Float + SupportedFloat + CudaVectorAdd<CType = T>,
{
    // Constructor methods
    pub fn new_base(matrix: Matrix<T>) -> Self {
        let id = NODE_ID_GENERATOR.next_id();
        Nodes::Base { id, matrix }
    }

    pub fn new_add(left: Nodes<T>, right: Nodes<T>) -> Self {
        let id = NODE_ID_GENERATOR.next_id();
        Nodes::Add {
            id,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn new_sub(left: Nodes<T>, right: Nodes<T>) -> Self {
        let id = NODE_ID_GENERATOR.next_id();
        Nodes::Sub {
            id,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn new_elemental_mul(left: Nodes<T>, right: Nodes<T>) -> Self {
        let id = NODE_ID_GENERATOR.next_id();
        Nodes::ElementalMul {
            id,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn new_dot(left: Nodes<T>, right: Nodes<T>) -> Self {
        let id = NODE_ID_GENERATOR.next_id();
        Nodes::Dot {
            id,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn new_reshape(node: Nodes<T>, dims: (usize, usize)) -> Self {
        let id = NODE_ID_GENERATOR.next_id();
        Nodes::Reshape {
            id,
            node: Box::new(node),
            dims,
        }
    }

    pub fn new_transformation(tgt: Nodes<T>, transformation: fn(T) -> T) -> Self {
        let id = NODE_ID_GENERATOR.next_id();
        Nodes::Transformation {
            id,
            tgt: Box::new(tgt),
            transformation,
        }
    }

    pub fn transform(self, transformation: fn(T) -> T) -> Self {
        Nodes::new_transformation(self, transformation)
    }

    pub fn dot(self, other: Self) -> Self {
        Nodes::new_dot(self, other)
    }

    pub fn reshape(self, dims: (usize, usize)) -> Self {
        Nodes::new_reshape(self, dims)
    }

    pub fn optimize_ast(&self) -> Self {
        self.clone()
    }

    pub fn to_cache_key(&self) -> Option<CacheKey> {
        match self {
            Nodes::Add { left, right, .. } => Some(CacheKey::Add(left.get_id(), right.get_id())),
            Nodes::Sub { left, right, .. } => Some(CacheKey::Sub(left.get_id(), right.get_id())),
            Nodes::ElementalMul { left, right, .. } => {
                Some(CacheKey::ElementalMul(left.get_id(), right.get_id()))
            }
            Nodes::Dot { left, right, .. } => Some(CacheKey::Dot(left.get_id(), right.get_id())),
            Nodes::Reshape { node, dims, .. } => {
                Some(CacheKey::Reshape(node.get_id(), dims.0, dims.1))
            }
            Nodes::Base { .. } => None,
            Nodes::Transformation {
                tgt,
                transformation,
                ..
            } => Some(CacheKey::Transformation(
                tgt.get_id(),
                *transformation as usize,
            )),
        }
    }

    pub fn get_id(&self) -> NodeId {
        match self {
            Nodes::Base { id, .. }
            | Nodes::Add { id, .. }
            | Nodes::Sub { id, .. }
            | Nodes::ElementalMul { id, .. }
            | Nodes::Dot { id, .. }
            | Nodes::Reshape { id, .. }
            | Nodes::Transformation { id, .. } => *id,
        }
    }

    pub fn evaluate(&self, cache_manager: &mut CacheManager<T>) -> Option<Matrix<T>> {
        match self {
            Nodes::Base { matrix, .. } => Some(matrix.clone()),
            Nodes::Add { left, right, .. } => {
                let cache_key = CacheKey::Add(left.get_id(), right.get_id());
                if let Some(cached) = cache_manager.get_value(&cache_key) {
                    return Some(cached.clone());
                }

                let a = left.evaluate(cache_manager)?;
                let b = right.evaluate(cache_manager)?;
                let result = gpu_add(&a, &b);

                cache_manager.insert_value(cache_key, result.clone());

                Some(result)
            }
            Nodes::Sub { left, right, .. } => {
                let cache_key = CacheKey::Sub(left.get_id(), right.get_id());
                if let Some(cached) = cache_manager.get_value(&cache_key) {
                    return Some(cached.clone());
                }

                let a = left.evaluate(cache_manager)?;
                let mut b = right.evaluate(cache_manager)?;

                b.data
                    .iter_mut()
                    .for_each(|d| *d = *d * T::from(-1.0).unwrap());
                let result = gpu_add(&a, &b);

                cache_manager.insert_value(cache_key, result.clone());

                Some(result)
            }
            Nodes::ElementalMul { left, right, .. } => {
                let cache_key = CacheKey::ElementalMul(left.get_id(), right.get_id());
                if let Some(cached) = cache_manager.get_value(&cache_key) {
                    return Some(cached.clone());
                }

                let a = left.evaluate(cache_manager)?;
                let b = right.evaluate(cache_manager)?;
                let result = gpu_elemental_mul(&a, &b);

                cache_manager.insert_value(cache_key, result.clone());

                Some(result)
            }
            Nodes::Dot { left, right, .. } => {
                let cache_key = CacheKey::Dot(left.get_id(), right.get_id());
                if let Some(cached) = cache_manager.get_value(&cache_key) {
                    return Some(cached.clone());
                }

                let a = left.evaluate(cache_manager)?;
                let b = right.evaluate(cache_manager)?;
                let result = gpu_dot(&a, &b);

                cache_manager.insert_value(cache_key, result.clone());

                Some(result)
            }
            Nodes::Reshape { node, dims, .. } => {
                let cache_key = CacheKey::Reshape(node.get_id(), dims.0, dims.1);
                if let Some(cached) = cache_manager.get_value(&cache_key) {
                    return Some(cached.clone());
                }

                let a = node.evaluate(cache_manager)?;
                let result = gpu_reshape(&a, *dims);

                cache_manager.insert_value(cache_key, result.clone());

                Some(result)
            }
            Nodes::Transformation {
                tgt,
                transformation,
                ..
            } => {
                let cache_key = CacheKey::Transformation(tgt.get_id(), *transformation as usize);
                if let Some(cached) = cache_manager.get_value(&cache_key) {
                    return Some(cached.clone());
                }

                let a = tgt.evaluate(cache_manager)?;
                let transformed_data: Vec<T> = a.data.iter().map(|&x| transformation(x)).collect();
                let result = Matrix::new(transformed_data, a.rows, a.cols);

                cache_manager.insert_value(cache_key, result.clone());
                Some(result)
            }
        }
    }

    pub fn get_shape(&self, cache_manager: &mut CacheManager<T>) -> Option<(usize, usize)> {
        if let Some(shape) = cache_manager.get_shape(self.get_id()) {
            return Some(*shape);
        }

        let shape = match self {
            Nodes::Base { matrix, .. } => {
                let rows = matrix.rows;
                let cols = if rows > 0 { matrix.cols } else { 0 };
                (rows, cols)
            }
            Nodes::Add { left, right, .. }
            | Nodes::Sub { left, right, .. }
            | Nodes::ElementalMul { left, right, .. } => {
                let (lrows, lcols) = left.get_shape(cache_manager)?;
                let (rrows, rcols) = right.get_shape(cache_manager)?;
                if lrows == rrows && lcols == rcols {
                    (lrows, lcols)
                } else {
                    return None;
                }
            }
            Nodes::Dot { left, right, .. } => {
                let (lrows, lcols) = left.get_shape(cache_manager)?;
                let (rrows, rcols) = right.get_shape(cache_manager)?;
                if lcols == rrows {
                    (lrows, rcols)
                } else {
                    return None;
                }
            }
            Nodes::Reshape { dims, .. } => *dims,
            Nodes::Transformation { tgt, .. } => {
                // Handle Transformation
                let (rows, cols) = tgt.get_shape(cache_manager)?;
                (rows, cols)
            }
        };

        cache_manager.insert_shape(self.get_id(), shape);
        Some(shape)
    }
}

impl<T: PartialEq + SupportedFloat + CudaVectorAdd> PartialEq for Nodes<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Nodes::Base { matrix: a, .. }, Nodes::Base { matrix: b, .. }) => a == b,
            (
                Nodes::Add {
                    left: a1,
                    right: a2,
                    ..
                },
                Nodes::Add {
                    left: b1,
                    right: b2,
                    ..
                },
            ) => a1 == b1 && a2 == b2,
            (
                Nodes::Sub {
                    left: a1,
                    right: a2,
                    ..
                },
                Nodes::Sub {
                    left: b1,
                    right: b2,
                    ..
                },
            ) => a1 == b1 && a2 == b2,
            (
                Nodes::ElementalMul {
                    left: a1,
                    right: a2,
                    ..
                },
                Nodes::ElementalMul {
                    left: b1,
                    right: b2,
                    ..
                },
            ) => a1 == b1 && a2 == b2,
            (
                Nodes::Dot {
                    left: a1,
                    right: a2,
                    ..
                },
                Nodes::Dot {
                    left: b1,
                    right: b2,
                    ..
                },
            ) => a1 == b1 && a2 == b2,
            (
                Nodes::Reshape {
                    node: a,
                    dims: dims_a,
                    ..
                },
                Nodes::Reshape {
                    node: b,
                    dims: dims_b,
                    ..
                },
            ) => a == b && dims_a == dims_b,
            (
                Nodes::Transformation {
                    tgt: a,
                    transformation: f1,
                    ..
                },
                Nodes::Transformation {
                    tgt: b,
                    transformation: f2,
                    ..
                },
            ) => a == b && (*f1 as usize) == (*f2 as usize),
            _ => false,
        }
    }
}

impl<T> Add for Nodes<T>
where
    T: Float + Debug + SupportedFloat + CudaVectorAdd<CType = T>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Nodes::new_add(self, rhs)
    }
}

impl<T> Sub for Nodes<T>
where
    T: Float + Debug + SupportedFloat + CudaVectorAdd<CType = T>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Nodes::new_sub(self, rhs)
    }
}

impl<T> Mul for Nodes<T>
where
    T: Float + Debug + SupportedFloat + CudaVectorAdd<CType = T>,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Nodes::new_elemental_mul(self, rhs)
    }
}

impl<T: Eq + SupportedFloat + CudaVectorAdd> Eq for Nodes<T> {}

impl<T> From<Array2<T>> for Nodes<T>
where
    T: Clone + Debug + Float + SupportedFloat + CudaVectorAdd<CType = T>,
{
    fn from(value: Array2<T>) -> Self {
        let arr: Vec<Vec<T>> = value.rows().into_iter().map(|row| row.to_vec()).collect();
        Nodes::new_base(arr.into())
    }
}

impl<T: SupportedFloat + CudaVectorAdd<CType = T>> From<Vec<Vec<T>>> for Matrix<T> {
    fn from(value: Vec<Vec<T>>) -> Self {
        let rows = value.len();
        let cols = if rows > 0 { value[0].len() } else { 0 };
        Self::new(value.into_iter().flatten().collect(), rows, cols)
    }
}
