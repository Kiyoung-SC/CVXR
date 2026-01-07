# Least Squares Regression Practice Problem
#
# Problem: Find the best fit line for a set of data points
#
# Minimize: ||Ax - b||^2
#
# Example: Simple linear regression
# Fit a line y = mx + c to data points

library(CVXR)

# Generate sample data
set.seed(42)
n <- 50
x_data <- seq(0, 10, length.out=n)
y_data <- 2*x_data + 3 + rnorm(n, sd=2)

# Create design matrix
A <- cbind(x_data, 1)

# Define variables (coefficients)
beta <- Variable(2)

# Define objective (minimize squared error)
objective <- Minimize(sum_squares(A %*% beta - y_data))

# Formulate and solve problem
problem <- Problem(objective)
result <- solve(problem)

# Get coefficients
coeffs <- result$getValue(beta)

# Print results
cat("Linear regression: y = mx + c\n")
cat(sprintf("Slope (m): %.4f\n", coeffs[1]))
cat(sprintf("Intercept (c): %.4f\n", coeffs[2]))
cat(sprintf("Residual sum of squares: %.4f\n", result$value))

# Compare with lm()
lm_fit <- lm(y_data ~ x_data)
cat("\nComparison with lm():\n")
cat(sprintf("lm() slope: %.4f\n", coef(lm_fit)[2]))
cat(sprintf("lm() intercept: %.4f\n", coef(lm_fit)[1]))
