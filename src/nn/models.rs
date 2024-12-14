use std::fmt::Display;

use ndarray::{Array2, ScalarOperand};
use num_traits::Float;

use super::{activation::Activation, layer::Layer, loss::LossType, optimizers::AdamOptimizer};

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
