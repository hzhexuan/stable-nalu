rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
source('./_function_task_expand_name.r')

best.range = 100

best.model.step.fn = function (errors) {
  best.step = max(length(errors) - best.range, 0) + which.min(tail(errors, best.range))
  if (length(best.step) == 0) {
    return(length(errors))
  } else {
    return(best.step)
  }
}

first.solved.step = function (steps, errors, threshold) {
  index = first(which(errors < threshold))
  if (is.na(index)) {
    return(NA)
  } else {
    return(steps[index])
  }
}

safe.interval = function (alpha, vec) {
  if (length(vec) <= 1) {
    return(NA)
  }
  
  return(abs(qt((1 - alpha) / 2, length(vec) - 1)) * (sd(vec) / sqrt(length(vec))))
}

eps = read_csv('../results/function_task_static_mse_expectation.csv') %>%
  filter(simple == FALSE & parameter != 'default') %>%
  mutate(
    input.size = as.integer(input.size),
    operation = revalue(operation, operation.full.to.short)
  ) %>%
  select(operation, input.size, overlap.ratio, subset.ratio, extrapolation.range, threshold)

name.parameter = 'hidden.size'
name.label = 'Hidden size'
name.output = '../paper/results/simple_function_static_mul_hidden_size.pdf'

dat = expand.name(
  read_csv('../results/function_task_static_mul_hidden_size.csv')
) %>%
  merge(eps) %>%
  mutate(
    parameter = !!as.name(name.parameter)
  )

dat$model = as.factor(paste0(dat$model, ' (R=', dat$regualizer, ')'))

dat.last = dat %>%
  group_by(name) %>%
  #filter(n() == 201) %>%
  summarise(
    threshold = last(threshold),
    best.model.step = best.model.step.fn(loss.valid.interpolation),
    interpolation.last = loss.valid.interpolation[best.model.step],
    extrapolation.last = loss.valid.extrapolation[best.model.step],
    interpolation.step.solved = first.solved.step(step, loss.valid.interpolation, threshold),
    extrapolation.step.solved = first.solved.step(step, loss.valid.extrapolation, threshold),
    sparse.error.max = sparse.error.max[best.model.step],
    sparse.error.mean = sparse.error.sum[best.model.step] / sparse.error.count[best.model.step],
    solved = replace_na(loss.valid.extrapolation[best.model.step] < threshold, FALSE),
    parameter = last(parameter),
    model = last(model),
    operation = last(operation),
    seed = last(seed),
    size = n()
  )

dat.last.rate = dat.last %>%
  group_by(model, operation, parameter) %>%
  summarise(
    size=n(),
    success.rate.mean = mean(solved) * 100,
    success.rate.upper = NA,
    success.rate.lower = NA,
    
    converged.at.mean = mean(extrapolation.step.solved[solved]),
    converged.at.upper = converged.at.mean + safe.interval(0.95, extrapolation.step.solved[solved]),
    converged.at.lower = converged.at.mean - safe.interval(0.95, extrapolation.step.solved[solved]),
    
    sparse.error.mean = mean(sparse.error.max[solved]),
    sparse.error.upper = sparse.error.mean + safe.interval(0.95, sparse.error.max[solved]),
    sparse.error.lower = sparse.error.mean - safe.interval(0.95, sparse.error.max[solved])
  )

dat.gather.mean = dat.last.rate %>%
  mutate(
    success.rate = success.rate.mean,
    converged.at = converged.at.mean,
    sparse.error = sparse.error.mean
  ) %>%
  select(model, operation, parameter, success.rate, converged.at, sparse.error) %>%
  gather('key', 'mean.value', success.rate, converged.at, sparse.error)

dat.gather.upper = dat.last.rate %>%
  mutate(
    success.rate = success.rate.upper,
    converged.at = converged.at.upper,
    sparse.error = sparse.error.upper
  ) %>%
  select(model, operation, parameter, success.rate, converged.at, sparse.error) %>%
  gather('key', 'upper.value', success.rate, converged.at, sparse.error)

dat.gather.lower = dat.last.rate %>%
  mutate(
    success.rate = success.rate.lower,
    converged.at = converged.at.lower,
    sparse.error = sparse.error.lower
  ) %>%
  select(model, operation, parameter, success.rate, converged.at, sparse.error) %>%
  gather('key', 'lower.value', success.rate, converged.at, sparse.error)

dat.gather = merge(merge(dat.gather.mean, dat.gather.upper), dat.gather.lower) %>%
  mutate(
    model=droplevels(model),
    key = factor(key, levels = c("success.rate", "converged.at", "sparse.error"))
  )

p = ggplot(dat.gather, aes(x = parameter, colour=model)) +
  geom_point(aes(y = mean.value)) +
  geom_line(aes(y = mean.value)) +
  geom_errorbar(aes(ymin = lower.value, ymax = upper.value)) +
  scale_color_discrete(labels = model.to.exp(levels(dat.gather$model))) +
  xlab(name.label) +
  scale_y_continuous(name = element_blank(), limits=c(0,NA)) +
  facet_wrap(~ key, scales='free_y', labeller = labeller(
    key = c(
      success.rate = "Success rate in %",
      converged.at = "Solved at iteration step",
      sparse.error = "Sparsity error"
    )
  )) +
  theme(legend.position="bottom") +
  theme(plot.margin=unit(c(5.5, 10.5, 5.5, 5.5), "points"))
print(p)