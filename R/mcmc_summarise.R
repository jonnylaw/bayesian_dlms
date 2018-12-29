library(tidyverse)
library(coda)
library(ggmcmc)
library(knitr)
library(kableExtra)

## Summarise MCMC
thin = function(df, nth) {
  df[seq(from = 1, to = nrow(df), by = nth),]
}

read_chains = function(files, nth, drop, param_names) {
  chains = files %>%
    map(~read_chain(., nth, drop, param_names)) 
  
  length = chains %>% map_dbl(nrow) %>% purrr::reduce(min)
  
  chains %>%
    map(function(df) mcmc(df[1:length,])) %>% 
    mcmc.list() %>%
    ggs()
}

read_chain = function(file, nth, drop, param_names) {
  read_csv(file, skip = drop, col_names = param_names) %>%
    thin(nth)
}

summary_table = function(chains) {
  chains %>%
    group_by(Parameter) %>%
    summarise(mean = mean(value), median = median(value), 
              upper = quantile(value, probs = 0.95),
              lower = quantile(value, probs = 0.05),
              ESS = effective_size(value),
              )
}

latex_summary_table = function(chains) {
    summary_table(chains) %>%
        kable(digits = 2, booktabs = T, format="latex")
}

effective_size = function(values) {
  n = length(values)
  floor(n / (1 + 2 * sum(acf(x = values, plot = F)$acf)))
}

prior_posterior = function(chain, parameters, prior, limits) {
  chain %>%
    filter(Parameter %in% parameters) %>%
    ggplot(aes(x = value)) +
    geom_line(aes(y = ..density.., colour = 'Posterior'), stat = 'density') +  
    stat_function(fun = prior, aes(colour = 'Prior'), xlim = limits) +
    theme(legend.position = c(0.25, 0.9), title = element_blank()) +
    facet_wrap(~Parameter, ncol = 1)
}

traceplot = function(chains) {
  chains %>%
    ggplot() +
    geom_line(aes(x = Iteration, y = value, colour = as.factor(Chain)), alpha = 0.5) +
    facet_wrap(~Parameter, scales = "free_y", strip.position = "right") +
    theme(legend.position = "none")
}

density_plot = function(chains) {
  chains %>%
    ggplot() +
    geom_density(aes(x = value, fill = as.factor(Chain)), alpha = 0.5) +
    facet_wrap(~Parameter, scales = "free", strip.position = "right") +
    theme(legend.position = "none")
}

plot_density = function(pdf, mode, scale, range = c(-10, 10), title) {
  x = seq(range[1], range[2], length.out = 1000)
  density = pdf(x, mode, scale)
  qplot(x = x, y = density, geom = "line", xlim = range, main = title)
}

plot_diagnostics = function(chains) {
    p1 = traceplot(chains)

    p2 = chains %>%
        filter(Chain == 1) %>%
        ggs_autocorrelation() +
        facet_wrap(~Parameter, ncol = 3) +
        theme(legend.position = "none")

    p3 = density_plot(chains)
    
    p1 / p2 / p3
}

latex_table_sim = function(chains, actual_values) {
    chains %>% 
        drop_na() %>%
        summary_table() %>%
        inner_join(actual_values, by = "Parameter") %>%
        kable(digits = 2, booktabs = T, format="latex") 
}

plot_diagnostics_sim = function(chains, actual_values) {
    p1 = chains %>%
        inner_join(actual_values) %>%
        traceplot() +
        geom_hline(aes(yintercept = actual_value), linetype = "dashed")

    p2 = chains %>%
        filter(Chain == 1) %>%
        ggs_autocorrelation() +
        facet_wrap(~Parameter, ncol = 3) +
        theme(legend.position = "none")

    p3 = chains %>%
        inner_join(actual_values) %>%
        density_plot() +
        geom_vline(aes(xintercept = actual_value), linetype = "dashed")
    
    p1 / p2 / p3
}
