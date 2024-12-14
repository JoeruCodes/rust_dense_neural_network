use std::ops::{Add, Mul, Sub};

lazy_static::lazy_static! {
    pub static ref NODE_ID_GENERATOR: NodeIdGenerator = NodeIdGenerator::new();
}
use cache::{Matrix, MatrixResult, NodeIdGenerator};
use num_traits::Float;

pub mod cache;
pub mod nodes;

// GPU operation stubs
fn gpu_add<T>(a: &Matrix<T>, b: &Matrix<T>) -> MatrixResult<T>
where
    T: Clone + Add<Output = T>,
{
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

fn gpu_sub<T>(a: &Matrix<T>, b: &Matrix<T>) -> MatrixResult<T>
where
    T: Clone + Sub<Output = T>,
{
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

fn gpu_elemental_mul<T>(a: &Matrix<T>, b: &Matrix<T>) -> MatrixResult<T>
where
    T: Clone + Mul<Output = T>,
{
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

fn gpu_dot<T>(a: &Matrix<T>, b: &Matrix<T>) -> MatrixResult<T>
where
    T: Clone + Float,
{
    let rows_a = a.len();
    let cols_a = if rows_a > 0 { a[0].len() } else { 0 };
    let cols_b = if b.len() > 0 { b[0].len() } else { 0 };

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

fn gpu_reshape<T>(a: &Matrix<T>, dims: (usize, usize)) -> MatrixResult<T>
where
    T: Clone,
{
    let (new_rows, new_cols) = dims;
    let flat: Vec<T> = a.iter().flat_map(|row| row.iter().cloned()).collect();

    if flat.len() != new_rows * new_cols {
        return None; // Dimension mismatch
    }

    Some(flat.chunks(new_cols).map(|chunk| chunk.to_vec()).collect())
}

#[cfg(test)]
mod tests {
    use cache::CacheManager;
    use nodes::Nodes;

    use super::*;

    #[test]
    fn test_matrix_addition() {
        let mut cache_manager = CacheManager::<f64>::new();

        // Define two matrices
        let matrix_a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let matrix_b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        // Create base nodes
        let node_a = Nodes::new_base(matrix_a);
        let node_b = Nodes::new_base(matrix_b);

        // Build AST: A + B
        let ast_add = node_a.clone().add(node_b.clone());

        // Optimize AST
        let optimized_ast_add = ast_add.optimize_ast();

        // Evaluate AST
        let result = optimized_ast_add
            .evaluate(&mut cache_manager)
            .expect("Addition failed");

        // Expected result
        let expected = vec![vec![6.0, 8.0], vec![10.0, 12.0]];

        assert_eq!(
            result, expected,
            "Matrix addition did not produce expected results"
        );

        // Check that the cache has 1 entry
        assert_eq!(
            cache_manager.len(),
            1,
            "Cache does not contain expected number of entries"
        );
    }

    #[test]
    fn test_matrix_subtraction() {
        let mut cache_manager = CacheManager::<f64>::new();

        // Define two matrices
        let matrix_a = vec![vec![10.0, 20.0], vec![30.0, 40.0]];

        let matrix_b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        // Create base nodes
        let node_a = Nodes::new_base(matrix_a);
        let node_b = Nodes::new_base(matrix_b);

        // Build AST: A - B
        let ast_sub = node_a.clone().sub(node_b.clone());

        // Optimize AST
        let optimized_ast_sub = ast_sub.optimize_ast();

        // Evaluate AST
        let result = optimized_ast_sub
            .evaluate(&mut cache_manager)
            .expect("Subtraction failed");

        // Expected result
        let expected = vec![vec![5.0, 14.0], vec![23.0, 32.0]];

        assert_eq!(
            result, expected,
            "Matrix subtraction did not produce expected results"
        );

        // Check that the cache has 1 entry
        assert_eq!(
            cache_manager.len(),
            1,
            "Cache does not contain expected number of entries"
        );
    }

    #[test]
    fn test_caching_mechanism() {
        let mut cache_manager = CacheManager::<f64>::new();

        // Define two matrices
        let matrix_a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let matrix_b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        // Create base nodes
        let node_a = Nodes::new_base(matrix_a);
        let node_b = Nodes::new_base(matrix_b);

        // Build AST: (A + B) * (A + B)
        let ast_add = node_a.clone().add(node_b.clone());
        let ast_mul = ast_add.clone().mul(ast_add.clone());

        // Optimize AST
        let optimized_ast_mul = ast_mul.optimize_ast();

        // Evaluate AST
        let result = optimized_ast_mul
            .evaluate(&mut cache_manager)
            .expect("Multiplication failed");

        // Expected result: (A + B) * (A + B)
        // First compute A + B
        // A + B = [[6, 8], [10, 12]]
        // Then, (A + B) * (A + B) is element-wise multiplication:
        // [[6*6, 8*8], [10*10, 12*12]] = [[36, 64], [100, 144]]

        let expected = vec![vec![36.0, 64.0], vec![100.0, 144.0]];

        assert_eq!(
            result, expected,
            "Matrix multiplication did not produce expected results"
        );

        // Check that the cache has 2 entries:
        // 1. A + B
        // 2. (A + B) * (A + B)
        assert_eq!(
            cache_manager.len(),
            2,
            "Cache does not contain expected number of entries"
        );
    }

    #[test]
    fn test_matrix_dot_product() {
        let mut cache_manager = CacheManager::<f64>::new();

        // Define two matrices suitable for dot product
        let matrix_a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let matrix_b = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];

        // Create base nodes
        let node_a = Nodes::new_base(matrix_a);
        let node_b = Nodes::new_base(matrix_b);

        // Build AST: A dot B
        let ast_dot = node_a.clone().dot(node_b.clone());

        // Optimize AST
        let optimized_ast_dot = ast_dot.optimize_ast();

        // Evaluate AST
        let result = optimized_ast_dot
            .evaluate(&mut cache_manager)
            .expect("Dot product failed");

        // Expected result:
        // [
        //  [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12],
        //  [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]
        // ]
        // [
        //  [7 + 18 + 33, 8 + 20 + 36] = [58, 64],
        //  [28 + 45 + 66, 32 + 50 + 72] = [139, 154]
        // ]

        let expected = vec![vec![58.0, 64.0], vec![139.0, 154.0]];

        assert_eq!(
            result, expected,
            "Matrix dot product did not produce expected results"
        );

        // Check that the cache has 1 entry
        assert_eq!(
            cache_manager.len(),
            1,
            "Cache does not contain expected number of entries"
        );
    }

    #[test]
    fn test_matrix_reshape() {
        let mut cache_manager = CacheManager::<f64>::new();

        // Define a matrix
        let matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        // Create a base node
        let node = Nodes::new_base(matrix);

        // Build AST: Reshape to (3, 2)
        let ast_reshape = node.clone().reshape((3, 2));

        // Optimize AST
        let optimized_ast_reshape = ast_reshape.optimize_ast();

        // Evaluate AST
        let result = optimized_ast_reshape
            .evaluate(&mut cache_manager)
            .expect("Reshape failed");

        // Expected result: [[1,2], [3,4], [5,6]]
        let expected = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        assert_eq!(
            result, expected,
            "Matrix reshape did not produce expected results"
        );

        // Check that the cache has 1 entry
        assert_eq!(
            cache_manager.len(),
            1,
            "Cache does not contain expected number of entries"
        );
    }

    #[test]
    fn test_multiple_operations_with_caching() {
        let mut cache_manager = CacheManager::<f64>::new();

        // Define three matrices
        let matrix_a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let matrix_b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        let matrix_c = vec![vec![9.0, 10.0], vec![11.0, 12.0]];

        // Create base nodes
        let node_a = Nodes::new_base(matrix_a);
        let node_b = Nodes::new_base(matrix_b);
        let node_c = Nodes::new_base(matrix_c);

        // Build AST: (A + B) * (A + B) + C
        let ast_add = node_a.clone().add(node_b.clone());
        let ast_mul = ast_add.clone().mul(ast_add.clone());
        let ast_final = ast_mul.clone().add(node_c.clone());

        // Optimize AST
        let optimized_ast = ast_final.optimize_ast();

        // Evaluate AST
        let result = optimized_ast
            .evaluate(&mut cache_manager)
            .expect("Complex AST evaluation failed");

        // Compute expected result:
        // A + B = [[6, 8], [10, 12]]
        // (A + B) * (A + B) = [[36, 64], [100, 144]]
        // (A + B) * (A + B) + C = [[36+9, 64+10], [100+11, 144+12]] = [[45, 74], [111, 156]]

        let expected = vec![vec![45.0, 74.0], vec![111.0, 156.0]];

        assert_eq!(
            result, expected,
            "Complex AST evaluation did not produce expected results"
        );

        // Check that the cache has 3 entries:
        // 1. A + B
        // 2. (A + B) * (A + B)
        // 3. (A + B) * (A + B) + C
        assert_eq!(
            cache_manager.len(),
            3,
            "Cache does not contain expected number of entries"
        );
    }

    #[test]
    fn test_shape_caching() {
        let mut cache_manager = CacheManager::<f64>::new();

        // Base node
        let matrix_base = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let node_base = Nodes::new_base(matrix_base.clone());
        let shape_base = node_base
            .get_shape(&mut cache_manager)
            .expect("Failed to get shape for base node");
        assert_eq!(shape_base, (2, 3));
        // Shape should be cached
        assert_eq!(cache_manager.get_shape(node_base.get_id()), Some(&(2, 3)));

        // Add node (same shape as its operands)
        let matrix_a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let matrix_b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let node_a = Nodes::new_base(matrix_a);
        let node_b = Nodes::new_base(matrix_b);
        let node_add = node_a.clone() + node_b.clone();

        // Compute shape once
        let shape_add_first = node_add
            .get_shape(&mut cache_manager)
            .expect("Failed to get shape for add node");
        assert_eq!(shape_add_first, (2, 2));

        // Compute shape second time, should use cache and not recompute
        let shape_add_second = node_add
            .get_shape(&mut cache_manager)
            .expect("Failed to get shape for add node from cache");
        assert_eq!(shape_add_second, shape_add_first);
        // Confirm shape is cached
        assert_eq!(cache_manager.get_shape(node_add.get_id()), Some(&(2, 2)));

        // Dot node
        let matrix_c = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let matrix_d = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        let node_c = Nodes::new_base(matrix_c);
        let node_d = Nodes::new_base(matrix_d);
        let node_dot = node_c.dot(node_d);

        let shape_dot = node_dot
            .get_shape(&mut cache_manager)
            .expect("Failed to get shape for dot node");
        assert_eq!(shape_dot, (2, 2));
        // Confirm shape is cached
        assert_eq!(cache_manager.get_shape(node_dot.get_id()), Some(&(2, 2)));

        // Reshape node
        let node_reshape = node_base.reshape((3, 2));
        let shape_reshape = node_reshape
            .get_shape(&mut cache_manager)
            .expect("Failed to get shape for reshape node");
        assert_eq!(shape_reshape, (3, 2));
        // Confirm shape is cached
        assert_eq!(
            cache_manager.get_shape(node_reshape.get_id()),
            Some(&(3, 2))
        );

        println!("All shape tests passed, and caching is verified.");
    }
}
