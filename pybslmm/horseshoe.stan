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
