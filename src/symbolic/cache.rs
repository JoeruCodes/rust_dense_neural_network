use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
};

use super::nodes::Matrix;

pub type NodeId = u64;

pub struct NodeIdGenerator {
    current: AtomicU64,
}

impl NodeIdGenerator {
    pub fn new() -> Self {
        NodeIdGenerator {
            current: AtomicU64::new(1), // Start IDs from 1
        }
    }

    pub fn next_id(&self) -> NodeId {
        self.current.fetch_add(1, Ordering::Relaxed)
    }
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CacheKey {
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    ElementalMul(NodeId, NodeId),
    Dot(NodeId, NodeId),
    Reshape(NodeId, usize, usize),
}

pub type MatrixResult<T> = Option<Matrix<T>>;

// Cache Manager now also holds shape_cache
pub struct CacheManager<T> {
    value_cache: HashMap<CacheKey, Matrix<T>>,
    shape_cache: HashMap<NodeId, (usize, usize)>,
}

impl<T> CacheManager<T> {
    pub fn new() -> Self {
        CacheManager {
            value_cache: HashMap::new(),
            shape_cache: HashMap::new(),
        }
    }

    pub fn get_value(&self, key: &CacheKey) -> Option<&Matrix<T>> {
        self.value_cache.get(key)
    }

    pub fn insert_value(&mut self, key: CacheKey, value: Matrix<T>) {
        self.value_cache.insert(key, value);
    }

    pub fn get_shape(&self, node_id: NodeId) -> Option<&(usize, usize)> {
        self.shape_cache.get(&node_id)
    }

    pub fn insert_shape(&mut self, node_id: NodeId, shape: (usize, usize)) {
        self.shape_cache.insert(node_id, shape);
    }

    /// Returns the number of cached value entries
    pub fn len(&self) -> usize {
        self.value_cache.len()
    }
}
