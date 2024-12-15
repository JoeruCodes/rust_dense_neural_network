use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::Float;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

use super::activation::Activation;

pub struct Layer<T>
where
    T: Float + From<f64>,
{
    pub weights: Array2<T>, // (output_size, input_size)
    pub bias: Array1<T>,    // (output_size)
    pub activation: Box<dyn Activation<T>>,
    pub z: Option<Array2<T>>,        // (output_size, m)
    pub a: Option<Array2<T>>,        // (output_size, m)
    pub d_w: Option<Array2<T>>,      // (output_size, input_size)
    pub d_b: Option<Array1<T>>,      // (output_size)
    pub d_a_prev: Option<Array2<T>>, // (input_size, m)
}

impl<T> Layer<T>
where
    T: Float + From<f64> + 'static + ScalarOperand,
{
    pub fn new(input_size: usize, output_size: usize, activation: Box<dyn Activation<T>>) -> Self {
        let mut rng = thread_rng();
        let std_dev = (1.0 / input_size as f64).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();

        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            normal.sample(&mut rng).into()
        });
        let bias = Array1::zeros(output_size);

        Layer {
            activation,
            weights,
            bias,
            z: None,
            a: None,
            d_w: None, // (output_size, input_size)
            d_b: None, // (output_size)
            d_a_prev: None,
        }
    }

    pub fn forward(&mut self, a_prev: Array2<T>) -> Array2<T> {
        let f = self.weights.dot(&a_prev) + self.bias.clone().insert_axis(Axis(1));
        self.z = Some(f.clone());
        let a = f.mapv(|v| self.activation.call(v));
        self.a = Some(a.clone());
        a
    }

    pub fn backward(&mut self, d_a: Array2<T>, a_prev: Array2<T>) -> Array2<T> {
        let m = a_prev.shape()[1];
        let dz = d_a
            * self
                .a
                .clone()
                .unwrap()
                .mapv(|v| self.activation.call_derivative(v));

        // Correct computation of d_w
        self.d_w = Some(dz.dot(&a_prev.t()) * (T::one() / (m as f64).into()));

        // Correct computation of d_b
        self.d_b = Some(
            dz.sum_axis(Axis(1))
                .mapv(|v| v * (T::one() / (m as f64).into())),
        );

        // Compute da_prev for the previous layer
        let da_p = self.weights.t().dot(&dz);
        self.d_a_prev = Some(da_p.clone());
        da_p
    }
}
