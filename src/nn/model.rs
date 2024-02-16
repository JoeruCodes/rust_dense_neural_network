use super::layer::{Layer, d_sigmoid};

pub struct Model {
    pub layers: Vec<Layer>,
}

impl Model {
    pub fn forward(&self, input: &Vec<f64>, i: usize, breakpoint:usize, use_activation: bool) -> Vec<f64>{
        // assert!(self.layers[0].neurons.iter().all(|neuron| neuron.weights.len() == input.len()));
        if i >= self.layers.len() || i == breakpoint {
            return input.to_vec()
        }
        let output = self.layers[i].forward(&input, use_activation);
        self.forward(&output, i+1, breakpoint, use_activation)
    }

    pub fn compute_loss_grad (&self, input: &Vec<f64>, target: &Vec<f64>) -> Vec<f64>{
        assert!(self.layers[self.layers.len()-1].neurons.len() == target.len());
        let predicted = self.forward(input, 0, self.layers.len(), true);
        let derror_dprediction =  predicted
        .iter()
        .zip(target.iter())
        .map(|(&predicted, &target)| 2.0 * (predicted - target))
        .collect::<Vec<f64>>();

        return derror_dprediction;
    }
    pub fn compute_grads_and_update_params(&mut self, input: &Vec<f64>, b_i: usize, lr: f64, target: &Vec<f64>) -> bool {
        if b_i == 0 || b_i <= 0{
            return true;
        }
        let dprediction_dlayer_i: Vec<f64> = self.forward(input, 0, b_i, false)
            .iter()
            .map(|logit| d_sigmoid(logit))
            .collect();

        let dlayer_dbias_i = 1.0;

        // Compute the gradient of the weights
        let dlayer_dweights_i: Vec<f64> = self.layers[b_i].neurons
            .iter()
            .flat_map(|neuron| 
                neuron.weights
                    .iter()
                    .zip(input.iter())
                    .map(|(&weight, &input)| 0.0 * weight + 1.0 * input)
            )
            .collect();

        let dloss = self.compute_loss_grad(input, target);

        // Compute the gradients for bias and weights in the specified layer
        let derror_dbias: Vec<f64> = dprediction_dlayer_i.iter().zip(dloss.iter()).map(|(&i, &j)| i * j * dlayer_dbias_i).collect();
        let derror_weights: Vec<f64> = dlayer_dweights_i.iter().zip(dprediction_dlayer_i.iter()).zip(dloss.iter()).map(|((&i, &j), &k)| i * j * k).collect();

        // Update the biases and weights of the neurons in the specified layer
        for (neuron, &grad_bias) in self.layers[b_i].neurons.iter_mut().zip(&derror_dbias) {
            neuron.bias -= lr * grad_bias;
        }

        for (neuron, grad_weights) in self.layers[b_i].neurons.iter_mut().zip(derror_weights.chunks(input.len())) {
            for (weight, &grad_weight) in neuron.weights.iter_mut().zip(grad_weights) {
                *weight -= lr * grad_weight;
            }
        }

        self.compute_grads_and_update_params(input, b_i-1, lr, target)
    }
}