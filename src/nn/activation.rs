use num_traits::Float;

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
