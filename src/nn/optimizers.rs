use std::collections::HashMap;

use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;

use super::layer::Layer;

pub struct AdamOptimizer<T>
where
    T: Float + From<f64>,
{
    beta1: T,
    beta2: T,
    epsilon: T,
    t: usize,
    m: HashMap<String, (Array2<T>, Array1<T>)>,
    v: HashMap<String, (Array2<T>, Array1<T>)>,
}

impl<T> AdamOptimizer<T>
where
    T: Float + From<f64> + ScalarOperand,
{
    pub fn new(
        weight_dims: Vec<(usize, usize)>,
        bias_dims: Vec<usize>,
        beta1: T,
        beta2: T,
        epsilon: Option<T>,
    ) -> Self {
        let mut m = HashMap::new();
        let mut v = HashMap::new();

        for (idx, (wt_dim, bias_dim)) in weight_dims.iter().zip(bias_dims.iter()).enumerate() {
            m.insert(
                idx.to_string(),
                (Array2::zeros(*wt_dim), Array1::zeros(*bias_dim)),
            );
            v.insert(
                idx.to_string(),
                (Array2::zeros(*wt_dim), Array1::zeros(*bias_dim)),
            );
        }
        Self {
            beta1,
            beta2,
            epsilon: epsilon.unwrap_or(T::epsilon()),
            t: 0,
            m,
            v,
        }
    }

    pub fn update_parameters(&mut self, lr: T, layers: &mut Vec<Layer<T>>) {
        self.t += 1;

        for (idx, layer) in layers.iter_mut().enumerate() {
            let (old_m_w, old_m_b) = self.m.get_mut(&idx.to_string()).unwrap();
            let new_m_w = old_m_w.mapv(|v| v * self.beta1)
                + layer
                    .d_w
                    .clone()
                    .unwrap()
                    .mapv(|v| v * (T::one() - self.beta1));

            let (old_v_w, old_v_b) = self.v.get_mut(&idx.to_string()).unwrap();
            let new_v_w = old_v_w.mapv(|v| v * self.beta2)
                + layer
                    .d_w
                    .clone()
                    .unwrap()
                    .pow2()
                    .mapv(|v| v * (T::one() - self.beta2));

            let m_corrected_w = new_m_w.clone() / (T::one() - self.beta1.powi(self.t as i32));
            let v_corrected_w = new_v_w.clone() / (T::one() - self.beta2.powi(self.t as i32));

            layer.weights = layer.weights.clone()
                - m_corrected_w.mapv(|v| lr * v) / (v_corrected_w.sqrt() + self.epsilon);

            *old_m_w = new_m_w;
            *old_v_w = new_v_w;

            let new_m_b = old_m_b.mapv(|v| v * self.beta1)
                + layer
                    .d_b
                    .clone()
                    .unwrap()
                    .mapv(|v| v * (T::one() - self.beta1));
            let new_v_b = old_v_b.mapv(|v| v * self.beta2)
                + layer
                    .d_b
                    .clone()
                    .unwrap()
                    .pow2()
                    .mapv(|v| v * (T::one() - self.beta2));
            let m_corrected_b = new_m_b.clone() / (T::one() - self.beta1.powi(self.t as i32));
            let v_corrected_b = new_v_b.clone() / (T::one() - self.beta2.powi(self.t as i32));

            layer.bias = layer.bias.clone()
                - m_corrected_b.mapv(|v| lr * v) / (v_corrected_b.sqrt() + self.epsilon);
            *old_m_b = new_m_b;
            *old_v_b = new_v_b;
        }
    }
}
