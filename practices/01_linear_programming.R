# Linear Programming Practice Problem
# 
# Problem: Minimize a linear objective function subject to linear constraints
# 
# Minimize: c^T * x
# Subject to: A * x <= b
#             x >= 0
#
# Example: Production Planning
# A company produces two products A and B
# Product A yields $3 profit per unit, Product B yields $5 profit per unit
# Product A requires 2 hours of labor, Product B requires 4 hours
# Available: 40 hours of labor
# Goal: Maximize profit

library(CVXR)

# Define variables
x1 <- Variable(1)  # Units of product A
x2 <- Variable(1)  # Units of product B

# Define objective (maximize profit = minimize negative profit)
objective <- Minimize(-3*x1 - 5*x2)

# Define constraints
constraints <- list(
  2*x1 + 4*x2 <= 40,  # Labor constraint
  x1 >= 0,             # Non-negativity
  x2 >= 0              # Non-negativity
)

# Formulate and solve problem
problem <- Problem(objective, constraints)
result <- solve(problem)

# Print results
cat("Optimal solution:\n")
cat(sprintf("Product A: %.2f units\n", result$getValue(x1)))
cat(sprintf("Product B: %.2f units\n", result$getValue(x2)))
cat(sprintf("Maximum profit: $%.2f\n", -result$value))
