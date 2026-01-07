# CVXR Practice Problems

This directory contains practice problems for learning convex optimization with CVXR in R.

## Prerequisites

Make sure you have CVXR installed:
```r
install.packages("CVXR")
```

## Practice Problems

### 01. Linear Programming
**File:** `01_linear_programming.R`

Learn basic linear programming with a production planning example. This problem demonstrates:
- Setting up decision variables
- Defining linear objectives
- Adding linear constraints
- Maximizing profit subject to resource constraints

**Concepts:** Linear objective, linear constraints, optimization

---

### 02. Quadratic Programming
**File:** `02_quadratic_programming.R`

Portfolio optimization problem that minimizes risk while meeting return targets. This demonstrates:
- Quadratic objective functions
- Working with covariance matrices
- Multi-asset portfolio allocation
- Risk-return tradeoffs

**Concepts:** Quadratic forms, variance minimization, portfolio theory

---

### 03. Least Squares Regression
**File:** `03_least_squares.R`

Classic regression problem fitting a line to noisy data. This shows:
- Formulating regression as optimization
- Minimizing squared errors
- Comparing CVXR results with standard R functions
- Working with design matrices

**Concepts:** Least squares, regression, convex optimization basics

---

## How to Use

1. Open any practice file in R or RStudio
2. Read through the comments to understand the problem
3. Run the code to see the solution
4. Try modifying parameters to experiment with different scenarios
5. Challenge yourself to add new constraints or modify objectives

## Learning Path

**Beginners:** Start with `03_least_squares.R` to understand the basics, then move to `01_linear_programming.R`.

**Intermediate:** Try `02_quadratic_programming.R` and experiment with different risk/return profiles.

**Advanced:** Modify the examples to add new constraints, change objectives, or solve variations of the problems.

## Additional Resources

- [CVXR Documentation](https://cvxr.rbind.io/)
- [CVXR Tutorial Examples](https://cvxr.rbind.io/examples/)
- [Convex Optimization Book by Boyd & Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/)

## Contributing

Feel free to add your own practice problems following the same structure:
- Clear problem statement in comments
- Well-documented code
- Example output or results
- Real-world context when possible
