use rand::Rng;

#[derive(Debug)]
pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

#[derive(Debug)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub activation: Option<fn(f64) -> f64>
}

impl Layer {
    pub fn forward(&self, x: &Vec<f64>, use_activation: bool) -> Vec<f64> {
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

pub fn dense(size: usize, input_size: usize, activation: Option<fn(f64) -> f64>) -> Layer {
    let mut rng = rand::thread_rng();

    let neurons = (0..size)
        .map(|_| Neuron {
            weights: (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            bias: 0.0,
        })
        .collect();
    Layer { neurons, activation }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn d_sigmoid(x:&f64) -> f64 {
    return sigmoid(*x) * (1.0 - sigmoid(*x))
}