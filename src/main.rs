mod nn;
use rand::Rng;
fn main() {
    // Define the neural network architecture
    let layer = nn::layer::dense(10, 2, Some(nn::layer::sigmoid));
    let layer1 = nn::layer::dense(2, 10, Some(nn::layer::sigmoid));
    let mut nn = nn::model::Model { layers: vec![layer, layer1] };

    // Training parameters
    let epochs = 200000;
    let lr = 0.01;

    fn foo(x: &Vec<f64>) -> f64 {
        x[0]*x[1] + 1.0
    }
    fn generate_random_vector(size: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        (0..size)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect()
    }
    // Training loop
    for epoch in 0..epochs {

        let input = generate_random_vector(2);
        let target = vec![0.0, foo(&input)];

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
    let test_input = generate_random_vector(2);
    let test_output = nn.forward(&test_input, 0, nn.layers.len(), true);
    println!("Predicted output for test input: {:?}", &test_output);
    println!("GROUND: {:?}", foo(&test_input));
    println!("GROUND: {:?}", &test_input);
}