###############
# Second Order
###############

data = read_csv("data/SecondOrderDlm.csv", col_names = c("time", "observation", paste("state", 1:2, sep = "_")))

data %>%
  filter(time < 100) %>%
  gather(key, value, -time) %>%
  ggplot(aes(x = time, y = value, colour = key)) +
  geom_line() +
  theme(legend.position = "bottom")

##########
# Filtered
##########

filtered = read_csv("data/SecondOrderDlmFiltered.csv",
                    col_names = c("time", paste("state_mean", 1:2, sep = "_"),
                                  paste("state_variance", 1:2, sep = "_"),
                                  "one_step_prediction", "prediction_variance"))

filtered %>%
  inner_join(data, by = "time") %>%
  filter(time > 200) %>%
  mutate(upper = qnorm(p = 0.95, mean = state_mean_2, sd = sqrt(state_variance_2))) %>%
  mutate(lower = qnorm(p = 0.05, mean = state_mean_2, sd = sqrt(state_variance_2))) %>%
  gather(key, value, state_mean_2, state_2) %>%
  ggplot(aes(x = time, y = value, colour = key)) +
  geom_line() +
  geom_line(aes(x = time, y = lower), linetype = 3, colour = "#000000") +
  geom_line(aes(x = time, y = upper), linetype = 3, colour = "#000000") +
  theme(legend.position = "bottom") +
  ggtitle("Kalman Filtered")

###########
# Learn Parameters
##########

gibbs_iters = read_csv("data/SecondOrderDlmGibbs.csv",
                       col_names = c("V", "W1", "W2", "m01", "m02", "C01", "C02"))

gibbs_chain = gibbs_iters %>%
  select(V, W1, W2)

actual_values = tibble(
  parameter = c("V", "W1", "W2"),
  actual_value = c(3.0, 0.0, 1.0)
)

params = gibbs_chain %>%
  mutate(iteration = 1:nrow(gibbs_chain)) %>%
  drop(5000) %>%
  gather(key = parameter, value, -iteration) %>%
  inner_join(actual_values, by = "parameter")

params %>%
  ggplot(aes(x = iteration, y = value)) +
  geom_line() +
  geom_hline(aes(yintercept = actual_value), colour = "#ff0000") +
  facet_wrap(~parameter, scales = "free_y")

params %>%
  group_by(parameter) %>%
  mutate(running_mean = dlm::ergMean(value)) %>%
  ggplot(aes(x = iteration, y = running_mean)) +
  geom_line() +
  geom_hline(aes(yintercept = actual_value), colour = "#ff0000") +
  facet_wrap(~parameter, nrow = 1, scales = "free_y")