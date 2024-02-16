use rand::Rng;

#[derive(Debug)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

#[derive(Debug)]
struct Layer {
    neurons: Vec<Neuron>,
    activation: Option<fn(f64) -> f64>
}

impl Layer {
    fn forward(&self, x: &Vec<f64>, use_activation: bool) -> Vec<f64> {
        assert!(self.neurons[0].weights.len() == x.len());
        self.neurons
            .iter()
            .map(|neuron| {
                let sum: f64 = neuron
                    .weights
                    .iter()
                    .zip(x.iter())
                    .map(|(&weight, &input)| weight * input)
                    .sum();
                let biased_sum = sum + neuron.bias;
                if use_activation == true {
                match &self.activation {
                    Some(activation) => activation(biased_sum),
                    None => biased_sum,
                }} else {
                    biased_sum
                }
            })
            .collect::<Vec<f64>>()
    }
}

fn dense(size: usize, input_size: usize, activation: Option<fn(f64) -> f64>) -> Layer {
    let mut rng = rand::thread_rng();

    let neurons = (0..size)
        .map(|_| Neuron {
            weights: (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            bias: 0.0,
        })
        .collect();
    Layer { neurons, activation }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn d_sigmoid(x:&f64) -> f64 {
    return sigmoid(*x) * (1.0 - sigmoid(*x))
}
struct Model {
    layers: Vec<Layer>,
}

impl Model {
    fn forward(&self, input: &Vec<f64>, i: usize, breakpoint:usize, use_activation: bool) -> Vec<f64>{
        // assert!(self.layers[0].neurons.iter().all(|neuron| neuron.weights.len() == input.len()));
        if i >= self.layers.len() || i == breakpoint {
            return input.to_vec()
        }
        let output = self.layers[i].forward(&input, use_activation);
        self.forward(&output, i+1, breakpoint, use_activation)
    }

    fn compute_loss_grad (&self, input: &Vec<f64>, target: &Vec<f64>) -> Vec<f64>{
        assert!(self.layers[self.layers.len()-1].neurons.len() == target.len());
        let predicted = self.forward(input, 0, self.layers.len(), true);
        let derror_dprediction =  predicted
        .iter()
        .zip(target.iter())
        .map(|(&predicted, &target)| 2.0 * (predicted - target))
        .collect::<Vec<f64>>();

        return derror_dprediction;
    }
    fn compute_grads_and_update_params(&mut self, input: &Vec<f64>, b_i: usize, lr: f64, target: &Vec<f64>) -> bool {
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
fn main() {
    // Define the neural network architecture
    let layer = dense(10, 2, Some(sigmoid));
    let layer1 = dense(2, 10, Some(sigmoid));
    let mut nn = Model { layers: vec![layer, layer1] };

    // Training parameters
    let epochs = 20000;
    let lr = 0.01;

    // Training loop
    for epoch in 0..epochs {
        // Example input and target data
        let input = vec![1.0, 2.0];
        let target = vec![0.0, 1.0];

        // Forward pass
        let predicted_output = nn.forward(&input, 0, nn.layers.len(), true);

        // Compute loss gradient
        let loss_gradient = nn.compute_loss_grad(&input, &target);

        // Backpropagation and parameter update
        nn.compute_grads_and_update_params(&input, nn.layers.len() - 1, lr, &target);

        // Print loss every 100 epochs
        if epoch % 100 == 0 {
            let loss = loss_gradient.iter().fold(0.0, |acc, &x| acc + x.powi(2));
            println!("Epoch {}: Loss = {}", epoch, loss);
        }
    }

    // After training, test the network with new data
    let test_input = vec![0.5, 1.5];
    let test_output = nn.forward(&test_input, 0, nn.layers.len(), true);
    println!("Predicted output for test input: {:?}", test_output);
}