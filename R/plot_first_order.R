library(tidyverse)

theme_set(theme_minimal())

##############
# Simulation #
##############

setwd("~/BayesianDlm/")
data = read_csv("data/FirstOrderDlm.csv", col_names = c("time", "observation", "state"))

data %>%
  gather(key, value, -time) %>%
  ggplot(aes(x = time, y = value, colour = key)) +
  geom_line() +
  theme(legend.position = "bottom")

ggsave("figures/SimulatedDlm.png")

#############
# Filtering #
#############

filtered = read_csv("data/FirstOrderDlmFiltered.csv",
                    col_names = c("time", "state_mean", "state_variance",
                                  "one_step_prediction", "prediction_variance"))

filtered %>%
  inner_join(data, by = "time") %>%
  filter(time > 200) %>%
  mutate(upper = qnorm(p = 0.95, mean = state_mean, sd = sqrt(state_variance))) %>%
  mutate(lower = qnorm(p = 0.05, mean = state_mean, sd = sqrt(state_variance))) %>%
  gather(key, value, state_mean, state) %>%
  ggplot(aes(x = time, y = value, colour = key)) +
  geom_line() +
  geom_ribbon(aes(x = time, ymin = lower, ymax = upper), alpha = 0.3, colour = NA) +
  theme(legend.position = "bottom") +
  ggtitle("Kalman Filtered")

ggsave("figures/KalmanFilterDlm.png")

#############
# Smoothing #
#############

smoothed = read_csv("data/FirstOrderDlmSmoothed.csv",
                    col_names = c("time", "smoothed_mean", "smoothed_variance"))

smoothed %>%
  # filter(time < 100) %>%
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

iters = read_csv("data/FirstOrderDlmIters.csv", col_names = c("V", "W", "m0", "C0", "accepted"), skip = 10000)

actual_values = tibble(
  parameter = c("V", "W", "m0", "C0"),
  actual_value = c(3.0, 1.0, 0.0, 1.0)
)

chain = iters %>%
  mutate_at(c("W", "V", "C0"), exp) %>%
  select(-accepted)

chain %>%
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

gibbs_iters = read_csv("data/FirstOrderDlmGibbs.csv", col_names = c("V", "W", "m0", "C0"), skip = 1000)

gibbs_chain = gibbs_iters %>%
  select(V, W)

gibbs_chain %>%
  mutate(iteration = 1:nrow(gibbs_chain)) %>%
  gather(key = parameter, value, -iteration) %>%
  inner_join(actual_values, by = "parameter") %>%
  ggplot(aes(x = iteration, y = value)) +
  geom_line() +
  geom_hline(aes(yintercept = actual_value), colour = "#ff0000") +
  facet_wrap(~parameter, scales = "free_y")
