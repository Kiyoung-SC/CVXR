# Quadratic Programming Practice Problem
#
# Problem: Minimize a quadratic objective function with linear constraints
#
# Example: Portfolio Optimization
# Minimize portfolio variance while maintaining expected return
#
# We have 3 assets with different expected returns and risks
# Goal: Find optimal allocation to minimize risk for a target return

library(CVXR)

# Expected returns for 3 assets
mu <- c(0.10, 0.15, 0.08)

# Covariance matrix (risk)
Sigma <- matrix(c(
  0.04, 0.01, 0.02,
  0.01, 0.09, 0.01,
  0.02, 0.01, 0.03
), nrow=3, byrow=TRUE)

# Define variables (portfolio weights)
w <- Variable(3)

# Target return
target_return <- 0.11

# Define objective (minimize variance)
objective <- Minimize(quad_form(w, Sigma))

# Define constraints
constraints <- list(
  sum(w) == 1,              # Weights sum to 1
  t(mu) %*% w >= target_return,  # Meet target return
  w >= 0                     # No short selling
)

# Formulate and solve problem
problem <- Problem(objective, constraints)
result <- solve(problem)

# Print results
cat("Optimal portfolio allocation:\n")
weights <- result$getValue(w)
for(i in 1:length(weights)) {
  cat(sprintf("Asset %d: %.2f%%\n", i, weights[i]*100))
}
cat(sprintf("\nExpected return: %.2f%%\n", sum(mu * weights)*100))
cat(sprintf("Portfolio variance: %.4f\n", result$value))
cat(sprintf("Portfolio std dev: %.2f%%\n", sqrt(result$value)*100))
