import numpy
import scipy.stats
import pystan

from .cache import model

model_code = """
  data {
    int<lower=1> p; 
    vector[p] y; 
  
}
  parameters {
    vector[p] theta_step; 
    vector[p] lambda_step; 
    vector[p] eta;
    real tau;
  
}
  transformed parameters {
    vector[p] theta; 
    vector[p] lambda; 
    lambda <- (lambda_step .* eta) * tau;
    theta <- ((theta_step .* lambda_step) .* eta) * tau;
  
}
  model {
    tau ~ uniform(0, 1);
    eta ~ cauchy(0, 1);
    lambda_step ~ cauchy(0, 1);
    theta_step ~ normal(0, 1);
    y ~ normal(theta, 1);
}  
"""

def simulate(p, n_causal):
    theta = numpy.zeros(p)
    theta[:n_causal] = 10
    return scipy.stats.multivariate_normal(mean=theta).rvs(1)

if __name__ == '__main__':
    p = 100
    n_causal = 10
    y = simulate(p, n_causal)
    data = {'p': p,
            'y': y
    }
    control = {
        'stepsize': .01
    }
    with model(model_code=model_code) as model:
        fit = model.sampling(data=data, control=control, init='random', iter=1000)
        print(fit)
