# Reproduce Gibbs Sampling for investing data

chain = read_csv("data/secon")

dlmModPoly(3)

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

#############
# Learn Parameters
#############

spain_invest = read_csv(file = "data/invest2.dat", col_names = c("unknown", "spain"))

spain_invest %>%
  mutate(time = seq(from = 1960, by = 1, length.out = nrow(spain_invest))) %>%
  ggplot(aes(x = time, y = spain)) +
  geom_line()

# This is clearly a linear growth trend, perfect for a second order model

chain = read_csv("data/GibbsInvestData.csv", col_names = c("V", "W1", "W2", "m0_1", "m0_2", "C0_1", "C0_2"))

drop = function(df, n) {
  df[-(1:n),]
}

# To Plot
params = chain %>%
  select(V, W1, W2) %>%
  mutate(iteration = 1:nrow(chain)) %>%
  gather(key = parameter, value, -iteration) %>%
  drop(1000)

# Trace plots
p1 = ggplot(params, aes(x = iteration, y = value)) +
  geom_point() +
  facet_wrap(~parameter, nrow = 1, scales = "free_y")

# Running Mean
p2 = params %>%
  group_by(parameter) %>%
  mutate(running_mean = ergMean(value)) %>%
  ggplot(aes(x = iteration, y = running_mean)) +
  geom_line() +
  facet_wrap(~parameter, nrow = 1, scales = "free_y")

gridExtra::grid.arrange(p1, p2, ncol = 1)
