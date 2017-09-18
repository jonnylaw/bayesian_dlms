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
  geom_line()

ggsave("figures/SimulatedDlm.png")

#############
# Filtering #
#############

# filtered = read_csv("data/Filtered.csv", col_names = c(""))

######################
# Parameter Learning #
######################

library(coda)

iters = read_csv("data/FirstOrderDlmIters.csv", col_names = c("log_w", "log_v", "m0", "log_c0", "accepted"))

chain = iters %>%
  mutate_at(c("log_w", "log_v", "log_c0"), exp) %>%
  select(-accepted) %>%
  mcmc()

plot(chain)

