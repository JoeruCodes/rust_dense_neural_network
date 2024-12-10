use std::{collections::HashMap, fmt::Display};

use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, NumCast};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

pub fn sigmoid<T>(z: T) -> T
where
    T: Float + From<f64>,
{
    let exp_neg_z = (-z).exp();
    T::one() / (T::one() + exp_neg_z)
}

pub fn d_sigmoid<T>(a: T) -> T
where
    T: Float + From<f64>,
{
    return a * (T::one() - a);
}

pub fn relu<T>(z: T) -> T
where
    T: Float + From<f64>,
{
    z.max(T::zero())
}

pub fn d_relu<T>(a: T) -> T
where
    T: Float + From<f64>,
{
    if a > T::zero() {
        T::one()
    } else {
        T::zero()
    }
}

pub fn tanh<T>(z: T) -> T
where
    T: Float + From<f64>,
{
    z.tanh()
}

pub fn d_tanh<T>(a: T) -> T
where
    T: Float + From<f64>,
{
    T::one() - a.powf(2.0.into())
}

pub enum ActivationTypes {
    Relu,
    Sigmoid,
    Tanh,
}

impl<T> Activation<T> for ActivationTypes
where
    T: Float + From<f64>,
{
    fn call(&self, x: T) -> T {
        match self {
            ActivationTypes::Relu => relu::<T>(x),
            ActivationTypes::Sigmoid => sigmoid::<T>(x),
            ActivationTypes::Tanh => tanh::<T>(x),
        }
    }

    fn call_derivative(&self, x: T) -> T {
        match self {
            ActivationTypes::Relu => d_relu::<T>(x),
            ActivationTypes::Sigmoid => d_sigmoid::<T>(x),
            ActivationTypes::Tanh => d_tanh::<T>(x),
        }
    }
}

pub trait Activation<T: Float + From<f64>> {
    fn call(&self, x: T) -> T;
    fn call_derivative(&self, x: T) -> T;
}
pub struct Layer<T>
where
    T: Float + From<f64>,
{
    weights: Array2<T>, // (output_size, input_size)
    bias: Array1<T>,    // (output_size)
    activation: Box<dyn Activation<T>>,
    z: Option<Array2<T>>,        // (output_size, m)
    a: Option<Array2<T>>,        // (output_size, m)
    d_w: Option<Array2<T>>,      // (output_size, input_size)
    d_b: Option<Array1<T>>,      // (output_size)
    d_a_prev: Option<Array2<T>>, // (input_size, m)
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

pub struct Sequential<T>
where
    T: Float + From<f64>,
{
    pub layers: Vec<Layer<T>>,
    pub optimizer: Option<AdamOptimizer<T>>,
}

impl<T> Sequential<T>
where
    T: Float + From<f64> + ScalarOperand + Display,
{
    pub fn new() -> Self {
        Self {
            layers: vec![],
            optimizer: None,
        }
    }

    pub fn add_layers(
        &mut self,
        input_size: usize,
        output_size: usize,
        activation: Box<dyn Activation<T>>,
    ) {
        self.layers
            .push(Layer::new(input_size, output_size, activation));
    }

    pub fn with_optimizer(&mut self, beta1: T, beta2: T, epsilon: Option<T>) {
        self.optimizer = Some(AdamOptimizer::new(
            self.layers.iter().map(|d| d.weights.dim()).collect(),
            self.layers.iter().map(|d| d.bias.dim()).collect(),
            beta1,
            beta2,
            epsilon,
        ))
    }

    pub fn forward(&mut self, x: Array2<T>) -> Array2<T> {
        let mut a = x;

        for layer in &mut self.layers {
            a = layer.forward(a);
        }

        a
    }

    pub fn compute_loss(&self, y_hat: Array2<T>, y: Array2<T>, loss: LossType) -> T {
        loss.call(y, y_hat)
    }

    pub fn backward(
        &mut self,
        y: Array2<T>,
        y_hat: Array2<T>,
        loss: LossType,
        x: Option<Array2<T>>,
    ) {
        let mut d_a = loss.call_derivative(y, y_hat);

        let layers = &mut self.layers;
        let num_layers = layers.len();

        let a_prev_values: Vec<Array2<T>> = (0..num_layers)
            .map(|idx| {
                if idx == 0 {
                    x.clone()
                        .expect("Input x must be provided for the first layer")
                } else {
                    layers[idx - 1]
                        .a
                        .clone()
                        .expect("Previous layer activation must exist")
                }
            })
            .collect();

        for idx in (0..num_layers).rev() {
            let a_prev = a_prev_values[idx].clone();
            let layer = &mut layers[idx];
            d_a = layer.backward(d_a, a_prev);
        }
    }

    pub fn update_parameters(&mut self, lr: T) {
        if let Some(optimizer) = &mut self.optimizer {
            optimizer.update_parameters(lr, &mut self.layers);
        } else {
            for layer in &mut self.layers {
                // Update weights
                layer.weights = &layer.weights - &(layer.d_w.clone().unwrap() * lr);

                // Update bias
                layer.bias = &layer.bias - &(layer.d_b.clone().unwrap() * lr);
            }
        }
    }

    pub fn train(
        &mut self,
        x: Array2<T>,
        y: Array2<T>,
        epochs: usize,
        lr: T,
        loss_type: LossType,
        print_loss: bool,
        print_interval: usize,
    ) {
        for epoch in 1..epochs + 1 {
            let y_hat = self.forward(x.clone());
            let loss = self.compute_loss(y_hat.clone(), y.clone(), loss_type.clone());
            self.backward(y.clone(), y_hat.clone(), loss_type.clone(), Some(x.clone()));
            self.update_parameters(lr);

            if print_loss && (epoch % print_interval == 0 || epoch == 1) {
                println!("Epoch {epoch}/{epochs}, loss: {loss}");
            }
        }
    }

    pub fn predict(&mut self, x: Array2<T>) -> Array2<T> {
        self.forward(x)
    }
}

#[derive(Clone)]
pub enum LossType {
    MSE,
    CrossEntropy,
}

impl LossType {
    pub fn call<T: Float + From<f64> + 'static>(&self, y: Array2<T>, y_hat: Array2<T>) -> T {
        let m = y.shape()[1]; // Correctly using batch size
        match self {
            LossType::MSE => {
                ((y_hat - y).pow2()).sum()
                    * (T::one() / (<T as NumCast>::from(2.0 * m as f64).unwrap()))
            }
            LossType::CrossEntropy => {
                let epsilon = T::epsilon();
                -(T::one() / <T as NumCast>::from(m as f64).unwrap())
                    * ((y.clone() * y_hat.mapv(|v| v + epsilon)).ln()
                        + ((y.mapv(|v| T::one() - v))
                            * (y_hat.mapv(|v| T::one() - v)).mapv(|v| v + epsilon))
                        .ln())
                    .sum()
            }
        }
    }

    pub fn call_derivative<T: Float + From<f64> + 'static + ScalarOperand>(
        &self,
        y: Array2<T>,
        y_hat: Array2<T>,
    ) -> Array2<T> {
        let m = y.shape()[1]; // Consistently using batch size
        match self {
            LossType::MSE => (y_hat - y) * (T::one() / (<T as NumCast>::from(m as f64).unwrap())),
            LossType::CrossEntropy => {
                let epsilon = T::epsilon();
                (y_hat.clone() - y) / (y_hat.clone() * (y_hat.mapv(|v| T::one() - v)) + epsilon)
            }
        }
    }
}

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
