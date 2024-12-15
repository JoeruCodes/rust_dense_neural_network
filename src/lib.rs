pub mod nn;
pub mod symbolic;
#[test]
fn test() {
    use ndarray::array;
    use nn::{activation::ActivationTypes, loss::LossType, models::Sequential};
    println!("=========================== With Optimizer ======================");
    let mut nn = Sequential::new();

    nn.add_layers(2, 5, Box::new(ActivationTypes::Tanh));
    nn.add_layers(5, 3, Box::new(ActivationTypes::Tanh));
    nn.add_layers(3, 1, Box::new(ActivationTypes::Sigmoid));

    // Use standard Adam optimizer parameters
    nn.with_optimizer(0.9, 0.999, Some(1e-8));

    let x = array![[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]; // Shape: (2, 4)
    let y = array![[0.0, 1.0, 1.0, 0.0]]; // Shape: (1, 4)

    let epochs = 100000;
    let learning_rate = 0.001; // Reduced learning rate
    let loss_type = LossType::MSE;
    let print_loss = true;
    let print_interval = 1000;

    nn.train(
        x.clone(),
        y,
        epochs,
        learning_rate,
        loss_type,
        print_loss,
        print_interval,
    );

    println!("=========================== Without Optimizer ======================");
    let mut nn = Sequential::new();

    nn.add_layers(2, 5, Box::new(ActivationTypes::Tanh));
    nn.add_layers(5, 3, Box::new(ActivationTypes::Tanh));
    nn.add_layers(3, 1, Box::new(ActivationTypes::Sigmoid));

    let x = array![[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]; // Shape: (2, 4)
    let y = array![[0.0, 1.0, 1.0, 0.0]]; // Shape: (1, 4)

    let epochs = 100000;
    let learning_rate = 0.001; // Reduced learning rate
    let loss_type = LossType::MSE;
    let print_loss = true;
    let print_interval = 1000;

    nn.train(
        x.clone(),
        y,
        epochs,
        learning_rate,
        loss_type,
        print_loss,
        print_interval,
    );
}
