library(tidyverse)

theme_set(theme_minimal())

##############
# Simulation #
##############

setwd("~/BayesianDlm/")
data = read_csv("data/FirstOrderDlm.csv")

data %>%
  gather(key, value, -time) %>%
  ggplot(aes(x = time, y = value, colour = key)) +
  geom_line() +
  theme(legend.position = "bottom")

ggsave("figures/SimulatedDlm.png")

#############
# Filtering #
#############

filtered = read_csv("data/FirstOrderDlmFiltered.csv")

filtered %>%
  inner_join(data, by = "time") %>%
  filter(time > 200) %>%
  filter(time < 300) %>%
  mutate(upper = qnorm(p = 0.95, mean = state_mean, sd = sqrt(state_variance))) %>%
  mutate(lower = qnorm(p = 0.05, mean = state_mean, sd = sqrt(state_variance))) %>%
  gather(key, value, state_mean, state) %>%
  ggplot(aes(x = time, y = value, colour = key)) +
  geom_line() +
  geom_line(aes(x = time, y = lower), linetype = 3, colour = "#000000") +
  geom_line(aes(x = time, y = upper), linetype = 3, colour = "#000000") +
  theme(legend.position = "bottom") +
  ggtitle("Kalman Filtered")

ggsave("figures/KalmanFilterDlm.png")

#############
# Smoothing #
#############

smoothed = read_csv("data/FirstOrderDlmSmoothed.csv")

smoothed %>%
  filter(time < 100) %>%
  inner_join(data, by = "time") %>%
  mutate(upper_smoothed = qnorm(p = 0.975, mean = smoothed_mean, sd = sqrt(smoothed_variance))) %>%
  mutate(lower_smoothed = qnorm(p = 0.025, mean = smoothed_mean, sd = sqrt(smoothed_variance))) %>%
  gather(key, value, state, smoothed_mean) %>%
  ggplot(aes(x = time, y = value, colour = key)) +
  geom_line() +
  geom_line(aes(x = time, y = lower_smoothed), linetype = 2, colour = "#000000") +
  geom_line(aes(x = time, y = upper_smoothed), linetype = 2, colour = "#000000") +
  theme(legend.position = "bottom") +
  labs(title = "Smoothed State Estimate",
       subtitle = "Actual state and smoothed state, with associated 95% intervals")

ggsave("figures/SmoothedState.png")

#######################
# Metropolis Hastings #
#######################

iters = read_csv("data/FirstOrderDlmIters.csv", col_names = c("V", "W", "m0", "C0", "accepted"))

iters %>%
  select(V, W) %>%
  mutate(iteration = 1:nrow(chain)) %>%
  gather(key = parameter, value, -iteration) %>%
  inner_join(actual_values, by = "parameter") %>%
  ggplot(aes(x = iteration, y = value)) +
  geom_line() +
  geom_hline(aes(yintercept = actual_value), colour = "#ff0000") +
  facet_wrap(~parameter, scales = "free_y")

##################
# Gibbs Sampling #
##################

gibbs_iters = read_csv("data/FirstOrderDlmGibbs.csv")

gibbs_chain = gibbs_iters %>%
  select(V, W)

actual_values = tibble(
  parameter = c("V", "W", "m0", "C0"),
  actual_value = c(3.0, 1.0, 0.0, 1.0)
)

gibbs_chain %>%
  mutate(iteration = 1:nrow(gibbs_chain)) %>%
  gather(key = parameter, value, -iteration) %>%
  inner_join(actual_values, by = "parameter") %>%
  ggplot(aes(x = iteration, y = value)) +
  geom_line() +
  geom_hline(aes(yintercept = actual_value), colour = "#ff0000") +
  facet_wrap(~parameter, scales = "free_y")

# Gibbs Sampling for state

state_gibbs_iters = read_csv("data/FirstOrderDlmStateGibbs.csv", col_names = F)

upper = apply(state_gibbs_iters, 2, function(state) sort(state)[nrow(state_gibbs_iters)*0.995])
lower = apply(state_gibbs_iters, 2, function(state) sort(state)[nrow(state_gibbs_iters)*0.005])

data_frame(
  time = 1:300,
  state_gibbs = colMeans(state_gibbs_iters),
  upper,
  lower
) %>%
  inner_join(data, by = "time") %>%
  filter(time > 250) %>%
  gather(key, value, state, state_gibbs) %>%
  ggplot(aes(x = time, y = value, colour = key)) +
  geom_line() +
  geom_line(aes(x = time, y = lower), linetype = 2, colour = "#000000") +
  geom_line(aes(x = time, y = upper), linetype = 2, colour = "#000000")

## Observation Difference
data %>%
  mutate(differ = (observation - state) * (observation - state)) %>%
  summarise(sum(differ))

## State difference
data %>%
  mutate(differ = c( NA, diff(state))) %>%
  summarise(sum(differ * differ, na.rm = T))

1 / rgamma(10, shape = 151, rate = 278/2)
