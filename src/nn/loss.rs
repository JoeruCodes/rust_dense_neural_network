use ndarray::{Array2, ScalarOperand};
use num_traits::{Float, NumCast};

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
