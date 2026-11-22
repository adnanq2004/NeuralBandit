results <- read.csv("neuralbandit3_original_metrics.csv")

plot(
  results$step,
  results$rolling_reward,
  type = "l",
  xlab = "Iteration",
  ylab = "Rolling Reward",
  main = "NeuralBandit3 Performance Under Concept Drift",
  ylim = c(0, 1)
)

abline(v = c(2000, 4000), col = "red", lty = 2)

legend(
  "bottomright",
  legend = "Concept Drift",
  col = "red",
  lty = 2
)