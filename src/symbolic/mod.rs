use std::{
    collections::HashMap,
    fmt::Debug,
    ops::{Add, Mul, Sub},
    sync::atomic::{AtomicU64, Ordering},
};

use num_traits::Float;

// Define a type alias for Node IDs
type NodeId = u64;

// Singleton Node ID generator using AtomicU64 for thread safety
pub struct NodeIdGenerator {
    current: AtomicU64,
}

impl NodeIdGenerator {
    fn new() -> Self {
        NodeIdGenerator {
            current: AtomicU64::new(1), // Start IDs from 1
        }
    }

    fn next_id(&self) -> NodeId {
        self.current.fetch_add(1, Ordering::Relaxed)
    }
}

type Matrix<T> = Vec<Vec<T>>;

// Define CacheKey based on operation and operand NodeIds
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CacheKey {
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    ElementalMul(NodeId, NodeId),
    Dot(NodeId, NodeId),
    Reshape(NodeId, usize, usize),
}

#[derive(Debug, Clone)]
pub enum Nodes<T> {
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
}

impl<T> Nodes<T>
where
    T: Clone + Debug + Float,
{
    // Constructor methods
    fn new_base(matrix: Matrix<T>, generator: &NodeIdGenerator) -> Self {
        Nodes::Base {
            id: generator.next_id(),
            matrix,
        }
    }

    fn new_add(left: Nodes<T>, right: Nodes<T>, generator: &NodeIdGenerator) -> Self {
        Nodes::Add {
            id: generator.next_id(),
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    fn new_sub(left: Nodes<T>, right: Nodes<T>, generator: &NodeIdGenerator) -> Self {
        Nodes::Sub {
            id: generator.next_id(),
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    fn new_elemental_mul(left: Nodes<T>, right: Nodes<T>, generator: &NodeIdGenerator) -> Self {
        Nodes::ElementalMul {
            id: generator.next_id(),
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    fn new_dot(left: Nodes<T>, right: Nodes<T>, generator: &NodeIdGenerator) -> Self {
        Nodes::Dot {
            id: generator.next_id(),
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    fn new_reshape(node: Nodes<T>, dims: (usize, usize), generator: &NodeIdGenerator) -> Self {
        Nodes::Reshape {
            id: generator.next_id(),
            node: Box::new(node),
            dims,
        }
    }

    // Update Add, Sub, Mul trait implementations to use constructors
    pub fn add(self, other: Self, generator: &NodeIdGenerator) -> Self {
        Nodes::new_add(self, other, generator)
    }

    pub fn sub(self, other: Self, generator: &NodeIdGenerator) -> Self {
        Nodes::new_sub(self, other, generator)
    }

    pub fn mul(self, other: Self, generator: &NodeIdGenerator) -> Self {
        Nodes::new_elemental_mul(self, other, generator)
    }

    pub fn dot(self, other: Self, generator: &NodeIdGenerator) -> Self {
        Nodes::new_dot(self, other, generator)
    }

    pub fn reshape(self, dims: (usize, usize), generator: &NodeIdGenerator) -> Self {
        Nodes::new_reshape(self, dims, generator)
    }

    /// Optimizes the AST by eliminating redundant computations using caching.
    fn optimize_ast(&self) -> Self {
        // In this simplified version, we're not modifying the AST.
        // Advanced optimizations like common subexpression elimination can be implemented here.
        self.clone()
    }

    /// Converts the node to a CacheKey based on its operation and operands.
    fn to_cache_key(&self) -> Option<CacheKey> {
        match self {
            Nodes::Add { id, left, right, .. } => Some(CacheKey::Add(left.get_id(), right.get_id())),
            Nodes::Sub { id, left, right, .. } => Some(CacheKey::Sub(left.get_id(), right.get_id())),
            Nodes::ElementalMul { id, left, right, .. } => {
                Some(CacheKey::ElementalMul(left.get_id(), right.get_id()))
            }
            Nodes::Dot { id, left, right, .. } => Some(CacheKey::Dot(left.get_id(), right.get_id())),
            Nodes::Reshape { id, node, dims, .. } => {
                Some(CacheKey::Reshape(node.get_id(), dims.0, dims.1))
            }
            Nodes::Base { .. } => None, // Base nodes don't correspond to operations
        }
    }

    /// Retrieves the NodeId of the node.
    fn get_id(&self) -> NodeId {
        match self {
            Nodes::Base { id, .. }
            | Nodes::Add { id, .. }
            | Nodes::Sub { id, .. }
            | Nodes::ElementalMul { id, .. }
            | Nodes::Dot { id, .. }
            | Nodes::Reshape { id, .. } => *id,
        }
    }

    /// Evaluates the AST using the CacheManager and returns the resulting matrix.
    fn evaluate(&self, cache_manager: &mut CacheManager<T>) -> Option<Matrix<T>> {
        match self {
            Nodes::Base { matrix, .. } => Some(matrix.clone()),
            Nodes::Add { left, right, .. } => {
                let cache_key = CacheKey::Add(left.get_id(), right.get_id());
                if let Some(cached) = cache_manager.get(&cache_key) {
                    return Some(cached.clone());
                }

                let a = left.evaluate(cache_manager)?;
                let b = right.evaluate(cache_manager)?;
                let result = gpu_add(&a, &b)?;

                cache_manager.insert(cache_key, result.clone());

                Some(result)
            }
            Nodes::Sub { left, right, .. } => {
                let cache_key = CacheKey::Sub(left.get_id(), right.get_id());
                if let Some(cached) = cache_manager.get(&cache_key) {
                    return Some(cached.clone());
                }

                let a = left.evaluate(cache_manager)?;
                let b = right.evaluate(cache_manager)?;
                let result = gpu_sub(&a, &b)?;

                cache_manager.insert(cache_key, result.clone());

                Some(result)
            }
            Nodes::ElementalMul { left, right, .. } => {
                let cache_key = CacheKey::ElementalMul(left.get_id(), right.get_id());
                if let Some(cached) = cache_manager.get(&cache_key) {
                    return Some(cached.clone());
                }

                let a = left.evaluate(cache_manager)?;
                let b = right.evaluate(cache_manager)?;
                let result = gpu_elemental_mul(&a, &b)?;

                cache_manager.insert(cache_key, result.clone());

                Some(result)
            }
            Nodes::Dot { left, right, .. } => {
                let cache_key = CacheKey::Dot(left.get_id(), right.get_id());
                if let Some(cached) = cache_manager.get(&cache_key) {
                    return Some(cached.clone());
                }

                let a = left.evaluate(cache_manager)?;
                let b = right.evaluate(cache_manager)?;
                let result = gpu_dot(&a, &b)?;

                cache_manager.insert(cache_key, result.clone());

                Some(result)
            }
            Nodes::Reshape { node, dims, .. } => {
                let cache_key = CacheKey::Reshape(node.get_id(), dims.0, dims.1);
                if let Some(cached) = cache_manager.get(&cache_key) {
                    return Some(cached.clone());
                }

                let a = node.evaluate(cache_manager)?;
                let result = gpu_reshape(&a, *dims)?;

                cache_manager.insert(cache_key, result.clone());

                Some(result)
            }
        }
    }
}

impl<T: PartialEq> PartialEq for Nodes<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Nodes::Base { matrix: a, .. }, Nodes::Base { matrix: b, .. }) => a == b,
            (
                Nodes::Add { left: a1, right: a2, .. },
                Nodes::Add { left: b1, right: b2, .. },
            ) => a1 == b1 && a2 == b2,
            (
                Nodes::Sub { left: a1, right: a2, .. },
                Nodes::Sub { left: b1, right: b2, .. },
            ) => a1 == b1 && a2 == b2,
            (
                Nodes::ElementalMul { left: a1, right: a2, .. },
                Nodes::ElementalMul { left: b1, right: b2, .. },
            ) => a1 == b1 && a2 == b2,
            (
                Nodes::Dot { left: a1, right: a2, .. },
                Nodes::Dot { left: b1, right: b2, .. },
            ) => a1 == b1 && a2 == b2,
            (
                Nodes::Reshape { node: a, dims: dims_a, .. },
                Nodes::Reshape { node: b, dims: dims_b, .. },
            ) => a == b && dims_a == dims_b,
            _ => false,
        }
    }
}

impl<T: Eq> Eq for Nodes<T> {}

type MatrixResult<T> = Option<Matrix<T>>;

// Cache Manager to handle caching logic based on CacheKey
pub struct CacheManager<T> {
    cache: HashMap<CacheKey, Matrix<T>>,
}

impl<T> CacheManager<T> {
    pub fn new() -> Self {
        CacheManager {
            cache: HashMap::new(),
        }
    }

    pub fn get(&self, key: &CacheKey) -> Option<&Matrix<T>> {
        self.cache.get(key)
    }

    pub fn insert(&mut self, key: CacheKey, value: Matrix<T>) {
        self.cache.insert(key, value);
    }

    /// Returns the number of cached entries
    pub fn len(&self) -> usize {
        self.cache.len()
    }
}

fn gpu_add<T>(a: &Matrix<T>, b: &Matrix<T>) -> MatrixResult<T>
where
    T: Clone + Add<Output = T>,
{
    // Implement actual GPU kernel execution here
    // Placeholder implementation:
    if a.len() != b.len() || a.is_empty() || b.is_empty() || a[0].len() != b[0].len() {
        return None; // Dimension mismatch
    }

    Some(
        a.iter()
            .zip(b.iter())
            .map(|(row_a, row_b)| {
                row_a
                    .iter()
                    .zip(row_b.iter())
                    .map(|(x, y)| x.clone() + y.clone())
                    .collect()
            })
            .collect(),
    )
}

/// Simulates GPU execution for matrix subtraction.
fn gpu_sub<T>(a: &Matrix<T>, b: &Matrix<T>) -> MatrixResult<T>
where
    T: Clone + Sub<Output = T>,
{
    // Implement actual GPU kernel execution here
    // Placeholder implementation:
    if a.len() != b.len() || a.is_empty() || b.is_empty() || a[0].len() != b[0].len() {
        return None; // Dimension mismatch
    }

    Some(
        a.iter()
            .zip(b.iter())
            .map(|(row_a, row_b)| {
                row_a
                    .iter()
                    .zip(row_b.iter())
                    .map(|(x, y)| x.clone() - y.clone())
                    .collect()
            })
            .collect(),
    )
}

/// Simulates GPU execution for element-wise multiplication.
fn gpu_elemental_mul<T>(a: &Matrix<T>, b: &Matrix<T>) -> MatrixResult<T>
where
    T: Clone + Mul<Output = T>,
{
    // Implement actual GPU kernel execution here
    // Placeholder implementation:
    if a.len() != b.len() || a.is_empty() || b.is_empty() || a[0].len() != b[0].len() {
        return None; // Dimension mismatch
    }

    Some(
        a.iter()
            .zip(b.iter())
            .map(|(row_a, row_b)| {
                row_a
                    .iter()
                    .zip(row_b.iter())
                    .map(|(x, y)| x.clone() * y.clone())
                    .collect()
            })
            .collect(),
    )
}

/// Simulates GPU execution for matrix dot product.
fn gpu_dot<T>(a: &Matrix<T>, b: &Matrix<T>) -> MatrixResult<T>
where
    T: Clone + Float,
{
    // Implement actual GPU kernel execution here
    // Placeholder implementation:
    let rows_a = a.len();
    let cols_a = if rows_a > 0 { a[0].len() } else { 0 };
    let cols_b = if b.len() > 0 { b[0].len() } else { 0 };

    // Check if the number of columns in A matches the number of rows in B
    if cols_a != b.len() {
        return None; // Dimension mismatch
    }

    let mut result = vec![vec![T::zero(); cols_b]; rows_a];

    for i in 0..rows_a {
        for j in 0..cols_b {
            let mut sum = a[i][0].clone() * b[0][j].clone();
            for k in 1..cols_a {
                sum = sum + (a[i][k].clone() * b[k][j].clone());
            }
            result[i][j] = sum;
        }
    }

    Some(result)
}

/// Simulates GPU execution for matrix reshape.
fn gpu_reshape<T>(a: &Matrix<T>, dims: (usize, usize)) -> MatrixResult<T>
where
    T: Clone,
{
    // Implement actual GPU kernel execution here
    // Placeholder implementation:
    let (new_rows, new_cols) = dims;
    let flat: Vec<T> = a.iter().flat_map(|row| row.iter().cloned()).collect();

    if flat.len() != new_rows * new_cols {
        return None; // Dimension mismatch
    }

    Some(flat.chunks(new_cols).map(|chunk| chunk.to_vec()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_addition() {
        let mut cache_manager = CacheManager::<f64>::new();
        let id_generator = NodeIdGenerator::new();

        // Define two matrices
        let matrix_a = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];

        let matrix_b = vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];

        // Create base nodes
        let node_a = Nodes::new_base(matrix_a, &id_generator);
        let node_b = Nodes::new_base(matrix_b, &id_generator);

        // Build AST: A + B
        let ast_add = node_a.clone().add(node_b.clone(), &id_generator);

        // Optimize AST
        let optimized_ast_add = ast_add.optimize_ast();

        // Evaluate AST
        let result = optimized_ast_add.evaluate(&mut cache_manager).expect("Addition failed");

        // Expected result
        let expected = vec![
            vec![6.0, 8.0],
            vec![10.0, 12.0],
        ];

        assert_eq!(result, expected, "Matrix addition did not produce expected results");

        // Check that the cache has 1 entry
        assert_eq!(cache_manager.len(), 1, "Cache does not contain expected number of entries");
    }

    #[test]
    fn test_matrix_subtraction() {
        let mut cache_manager = CacheManager::<f64>::new();
        let id_generator = NodeIdGenerator::new();

        // Define two matrices
        let matrix_a = vec![
            vec![10.0, 20.0],
            vec![30.0, 40.0],
        ];

        let matrix_b = vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];

        // Create base nodes
        let node_a = Nodes::new_base(matrix_a, &id_generator);
        let node_b = Nodes::new_base(matrix_b, &id_generator);

        // Build AST: A - B
        let ast_sub = node_a.clone().sub(node_b.clone(), &id_generator);

        // Optimize AST
        let optimized_ast_sub = ast_sub.optimize_ast();

        // Evaluate AST
        let result = optimized_ast_sub.evaluate(&mut cache_manager).expect("Subtraction failed");

        // Expected result
        let expected = vec![
            vec![5.0, 14.0],
            vec![23.0, 32.0],
        ];

        assert_eq!(result, expected, "Matrix subtraction did not produce expected results");

        // Check that the cache has 1 entry
        assert_eq!(cache_manager.len(), 1, "Cache does not contain expected number of entries");
    }

    #[test]
    fn test_caching_mechanism() {
        let mut cache_manager = CacheManager::<f64>::new();
        let id_generator = NodeIdGenerator::new();

        // Define two matrices
        let matrix_a = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];

        let matrix_b = vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];

        // Create base nodes
        let node_a = Nodes::new_base(matrix_a, &id_generator);
        let node_b = Nodes::new_base(matrix_b, &id_generator);

        // Build AST: (A + B) * (A + B)
        let ast_add = node_a.clone().add(node_b.clone(), &id_generator);
        let ast_mul = ast_add.clone().mul(ast_add.clone(), &id_generator);

        // Optimize AST
        let optimized_ast_mul = ast_mul.optimize_ast();

        // Evaluate AST
        let result = optimized_ast_mul.evaluate(&mut cache_manager).expect("Multiplication failed");

        // Expected result: (A + B) * (A + B)
        // First compute A + B
        // A + B = [[6, 8], [10, 12]]
        // Then, (A + B) * (A + B) is element-wise multiplication:
        // [[6*6, 8*8], [10*10, 12*12]] = [[36, 64], [100, 144]]

        let expected = vec![
            vec![36.0, 64.0],
            vec![100.0, 144.0],
        ];

        assert_eq!(result, expected, "Matrix multiplication did not produce expected results");

        // Check that the cache has 2 entries:
        // 1. A + B
        // 2. (A + B) * (A + B)
        assert_eq!(cache_manager.len(), 2, "Cache does not contain expected number of entries");
    }

    #[test]
    fn test_matrix_dot_product() {
        let mut cache_manager = CacheManager::<f64>::new();
        let id_generator = NodeIdGenerator::new();

        // Define two matrices suitable for dot product
        let matrix_a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];

        let matrix_b = vec![
            vec![7.0, 8.0],
            vec![9.0, 10.0],
            vec![11.0, 12.0],
        ];

        // Create base nodes
        let node_a = Nodes::new_base(matrix_a, &id_generator);
        let node_b = Nodes::new_base(matrix_b, &id_generator);

        // Build AST: A dot B
        let ast_dot = node_a.clone().dot(node_b.clone(), &id_generator);

        // Optimize AST
        let optimized_ast_dot = ast_dot.optimize_ast();

        // Evaluate AST
        let result = optimized_ast_dot.evaluate(&mut cache_manager).expect("Dot product failed");

        // Expected result:
        // [
        //  [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12],
        //  [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]
        // ]
        // [
        //  [7 + 18 + 33, 8 + 20 + 36] = [58, 64],
        //  [28 + 45 + 66, 32 + 50 + 72] = [139, 154]
        // ]

        let expected = vec![
            vec![58.0, 64.0],
            vec![139.0, 154.0],
        ];

        assert_eq!(result, expected, "Matrix dot product did not produce expected results");

        // Check that the cache has 1 entry
        assert_eq!(cache_manager.len(), 1, "Cache does not contain expected number of entries");
    }

    #[test]
    fn test_matrix_reshape() {
        let mut cache_manager = CacheManager::<f64>::new();
        let id_generator = NodeIdGenerator::new();

        // Define a matrix
        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];

        // Create a base node
        let node = Nodes::new_base(matrix, &id_generator);

        // Build AST: Reshape to (3, 2)
        let ast_reshape = node.clone().reshape((3, 2), &id_generator);

        // Optimize AST
        let optimized_ast_reshape = ast_reshape.optimize_ast();

        // Evaluate AST
        let result = optimized_ast_reshape.evaluate(&mut cache_manager).expect("Reshape failed");

        // Expected result: [[1,2], [3,4], [5,6]]
        let expected = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];

        assert_eq!(result, expected, "Matrix reshape did not produce expected results");

        // Check that the cache has 1 entry
        assert_eq!(cache_manager.len(), 1, "Cache does not contain expected number of entries");
    }

    #[test]
    fn test_multiple_operations_with_caching() {
        let mut cache_manager = CacheManager::<f64>::new();
        let id_generator = NodeIdGenerator::new();

        // Define three matrices
        let matrix_a = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];

        let matrix_b = vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];

        let matrix_c = vec![
            vec![9.0, 10.0],
            vec![11.0, 12.0],
        ];

        // Create base nodes
        let node_a = Nodes::new_base(matrix_a, &id_generator);
        let node_b = Nodes::new_base(matrix_b, &id_generator);
        let node_c = Nodes::new_base(matrix_c, &id_generator);

        // Build AST: (A + B) * (A + B) + C
        let ast_add = node_a.clone().add(node_b.clone(), &id_generator);
        let ast_mul = ast_add.clone().mul(ast_add.clone(), &id_generator);
        let ast_final = ast_mul.clone().add(node_c.clone(), &id_generator);

        // Optimize AST
        let optimized_ast = ast_final.optimize_ast();

        // Evaluate AST
        let result = optimized_ast.evaluate(&mut cache_manager).expect("Complex AST evaluation failed");

        // Compute expected result:
        // A + B = [[6, 8], [10, 12]]
        // (A + B) * (A + B) = [[36, 64], [100, 144]]
        // (A + B) * (A + B) + C = [[36+9, 64+10], [100+11, 144+12]] = [[45, 74], [111, 156]]

        let expected = vec![
            vec![45.0, 74.0],
            vec![111.0, 156.0],
        ];

        assert_eq!(result, expected, "Complex AST evaluation did not produce expected results");

        // Check that the cache has 3 entries:
        // 1. A + B
        // 2. (A + B) * (A + B)
        // 3. (A + B) * (A + B) + C
        assert_eq!(cache_manager.len(), 3, "Cache does not contain expected number of entries");
    }
}
