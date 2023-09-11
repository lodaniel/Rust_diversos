use rand::Rng;

const LEARNING_RATE: f64 = 0.1;
const EPOCHS: usize = 10000;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_input_hidden: Vec<Vec<f64>>,
    weights_hidden_output: Vec<Vec<f64>>,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weights_input_hidden = (0..input_size)
            .map(|_| (0..hidden_size).map(|_| rng.gen::<f64>() - 0.5).collect())
            .collect();

        let weights_hidden_output = (0..hidden_size)
            .map(|_| (0..output_size).map(|_| rng.gen::<f64>() - 0.5).collect())
            .collect();

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
        }
    }

    fn predict(&self, input: &Vec<f64>) -> Vec<f64> {
        let mut hidden_layer = vec![0.0; self.hidden_size];
        let mut output_layer = vec![0.0; self.output_size];

        for i in 0..self.hidden_size {
            for j in 0..self.input_size {
                hidden_layer[i] += input[j] * self.weights_input_hidden[j][i];
            }
            hidden_layer[i] = sigmoid(hidden_layer[i]);
        }

        for i in 0..self.output_size {
            for j in 0..self.hidden_size {
                output_layer[i] += hidden_layer[j] * self.weights_hidden_output[j][i];
            }
            output_layer[i] = sigmoid(output_layer[i]);
        }

        output_layer
    }

    fn train(&mut self, input: &Vec<f64>, target: &Vec<f64>) {
        // Forward pass
        let mut hidden_layer = vec![0.0; self.hidden_size];
        let mut output_layer = vec![0.0; self.output_size];

        for i in 0..self.hidden_size {
            for j in 0..self.input_size {
                hidden_layer[i] += input[j] * self.weights_input_hidden[j][i];
            }
            hidden_layer[i] = sigmoid(hidden_layer[i]);
        }

        for i in 0..self.output_size {
            for j in 0..self.hidden_size {
                output_layer[i] += hidden_layer[j] * self.weights_hidden_output[j][i];
            }
            output_layer[i] = sigmoid(output_layer[i]);
        }

        // Backpropagation
        let mut output_error = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            output_error[i] = target[i] - output_layer[i];
        }

        let mut hidden_error = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            for j in 0..self.output_size {
                hidden_error[i] += output_error[j] * self.weights_hidden_output[i][j];
            }
        }

        for i in 0..self.output_size {
            for j in 0..self.hidden_size {
                let delta = output_error[i] * sigmoid_derivative(output_layer[i]);
                self.weights_hidden_output[j][i] += hidden_layer[j] * delta * LEARNING_RATE;
            }
        }

        for i in 0..self.hidden_size {
            for j in 0..self.input_size {
                let delta = hidden_error[i] * sigmoid_derivative(hidden_layer[i]);
                self.weights_input_hidden[j][i] += input[j] * delta * LEARNING_RATE;
            }
        }
    }
}

fn main() {
    let mut neural_network = NeuralNetwork::new(2, 4, 1); // Increase hidden_size for better XOR approximation

    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    for _ in 0..EPOCHS {
        for (input, target) in &training_data {
            neural_network.train(input, target);
        }
    }

    for (input, _) in &training_data {
        let prediction = neural_network.predict(input);
        println!("Input: {:?} => Prediction: {:?}", input, prediction);
    }
}
