data {
  real<lower=0> M_1;
  real<lower=0> M_2;
  real<lower=0> N;
  matrix[M_1,N] X_1;
  matrix[M_2,N] X_2;
  vector[N] y;
}
transformed data {
  matrix[N,N] K <- crossprod(X_1) / M_1 + crossprod(X_2) / M_2
  s_1 <- variance(X_1) / num_elements(X_1)
  s_2 <- variance(X_2) / num_elements(X_2)
  s_u <- trace(K) / (M_1 + M_2)
}
parameters {
  vector[M_1] beta;
  vector[M_2] gamma;
  logpi ~ uniform(log(1 / (M_1 + M_2)), log 1)
  real<lower=0,upper=1> sigma_1;
  real<lower=0,upper=1> sigma_2;
  real<lower=0,upper=1> sigma_u;
  real<lower=0,upper=1> sigma_e;
}
transformed parameters {
  pi <- exp(logpi)
  h_1 <- (pi * M_1 * s_1 * sigma_1 + s_u * sigma_u) / total_var
  h_2 <- (pi * M_2 * s_2 * sigma_2 + s_u * sigma_u) / total_var
  rho_1 <- (pi * M_1 * s_1 * sigma_1) / (total_var - 1)
  rho_2 <- (pi * M_2 * s_2 * sigma_2) / (total_var - 1)
  total_var <- (pi * M_1 * s1 * sigma_1 + pi * M_2 * s2 * sigma_2 + s_u * sigma_u + 1)
}
model {
  h_1 ~ uniform(0, 1)
  h_2 ~ uniform(0, 1)
  rho_1 ~ uniform(0, 1)
  rho_1 ~ uniform(0, 1)
  sigma_e ~ cauchy(0, 1)
  u ~ normal(0, sigma_u * K)
  y ~ normal(beta * X_1 + gamma * X_2 + u, sigma_e)
}
generated quantitites {
  genetic_var <- var(beta * X_1 + gamma * X_2 + u)
    pge_1 <- var(beta * X_1) / genetic_var
    pge_2 <- var(gamma * X_2) / genetic_var
}
