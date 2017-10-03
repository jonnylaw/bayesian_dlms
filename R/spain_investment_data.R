library()

# Reproduce Gibbs Sampling for investing data

###############
# Second Order
###############

data = read_csv("../data/invest2.dat", col_names = c("country1", "spain"))

data %>%
  mutate(time = 1:nrow(data)) %>%
  ggplot(aes(x = time, y = spain)) +
  geom_line()

gibbs_iters = read_csv("../data/gibbs_spain_investment.csv",
                       col_names = c("V", "W1", "W2", "m01", "m02", "C01", "C02"))

gibbs_chain = gibbs_iters %>%
  select(V, W1, W2)

drop = function(df, n) {
  df[-c(1:n),]
}

params = gibbs_chain %>%
  mutate(iteration = 1:nrow(gibbs_chain)) %>%
  drop(12000) %>%
  gather(key = parameter, value, -iteration) %>%
  inner_join(actual_values, by = "parameter")

p1 = params %>%
  ggplot(aes(x = iteration, y = value)) +
  geom_line() +
  facet_wrap(~parameter, scales = "free_y")

p2 = params %>%
  group_by(parameter) %>%
  mutate(running_mean = dlm::ergMean(value)) %>%
  ggplot(aes(x = iteration, y = running_mean)) +
  geom_line() +
  facet_wrap(~parameter, nrow = 1, scales = "free_y")

png(filename = "../figures/invest_gibbs_samples.png")
gridExtra::grid.arrange(p1, p2, ncol = 1)
dev.off()
