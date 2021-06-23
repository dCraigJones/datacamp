[reference](https://campus.datacamp.com/courses/modeling-with-tidymodels-in-r/)
# Machine Learning with Tidymodels

David Svancer

*library(tidymodels)*
<br><br>
## Package Overview

| Package     | Objective					 |
|:------------|:-----------------------------|
| rsample     | Data Sampling				 |
| recipes     | Feature Engineering			 |
| parsnip     | Model Fitting and Prediction |
| tune<br>dials | Model Tuning 				 |
| yardstick   | Model Evaluation			 |

## Supervised Machine Learning
Branch of machine learning (ML) that uses labels data for model fitting as opposed to Unsupervised Machine learning, which...?

Two Types of Supervised Machine Learning:
- Regression (quantitative)
- Classification (Qualitative/Categorical)

Two Types of Variables:
- **outcome variables**, define
- **predictor variables**, define

The outcome variable is used for **stratification** so that its values have a similar range in both datasets. This prevents fitting a model to data that is different from the typical data it will be given in the future. 

`outcome ~ predictor_1 + predictor_2` or to use all available columns `outcome ~ .`


## 1 - Data Resampling

gaurds against overfitting.  Typically, 75% training & 25% testing

```
df_split <- rsample::initial_split(df
				, prop = 0.75
				, strata = outcome_variable
```

Training data is used for *Feature Engineering* and *Model Fitting/Tuning*

```df_train <- df_split %>% training()```

Test data is used for *Model Evaluation*

```df_test <- df_split %>% testing()```

outcome ~ predictor_1 + predictor_2
outcome ~ . #use all available columns

## 2 - Model Fitting and Prediction

### Specify a linear regression model, linear_model
Linear Regression Model *predicting `hwy` using `cty` as predictor*

$\text{hwy} = \Beta_0 + \Beta_1 \cdot \text{cty}$ or in R formula format `hwy ~ cty`


### 2a. Defining a parsnip model object
```
linear_model <- linear_reg() %>%    # Set the model type
  set_engine('lm') %>% 				# Set the model engine
  set_mode('regression') 			# Set the model mode
```

### 2b. Train the model
```
lm_fit <- linear_model %>% 
  fit(selling_price ~ home_age + sqft_living,
      data = home_training)
```

### 2c. Create a model summary tibble
```tidy(lm_fit)```

### 2d. Making Predictions
```
lm_fit %>% 
predict(new_data=mpg_test)
```

### 2e. Combine test data with predictions
```
home_test_results <- home_test %>% 
  select(selling_price, home_age, sqft_living) %>% 
  bind_cols(home_predictions)
```

## Test
 
- **Root-Mean-Squared-Error**
	- `rmse(truth=outcome_variable, estimate=.pred)`

- **R-Squared (coefficient of determination)**
	- *good to find non-linear segments, poorly fit segments*
	- `rsq(truth=outcome_variable, estimate=.pred)`

### Useful ggplot arguments

- `coord_obs_pred()`
	- coordinates x-y axis
- `geom_abline()`
	- draws unity when no parameters are passed? confirm?

### Automating fit

`last_fit()`

```
# Define a linear regression model
linear_model <- linear_reg() %>% 
  set_engine('lm') %>% 
  set_mode('regression')

# Train linear_model with last_fit()
linear_fit <- linear_model %>% 
  last_fit(selling_price ~ ., split = home_split)

# Collect predictions and view results
predictions_df <- linear_fit %>% collect_predictions()
predictions_df
                                        
# Make an R squared plot using predictions_df
ggplot(predictions_df, aes(x = selling_price, y = .pred)) + 
  geom_point(alpha = 0.5) + 
  geom_abline(color = 'blue', linetype = 2) +
  coord_obs_pred() +
  labs(x = 'Actual Home Selling Price', y = 'Predicted Selling Price')
```

similar to `collect_predictions()`, the model can return metrics with `collect_metrics()`

# Classification Models

David Svancer

*library(tidymodels)*
<br><br>

Classifical models predict categorical outcome variables.

**decision boundary**, define?

## 1 - Data Resampling

```
leads_split <- initial_split(leads_df
, prop=0.75
, strata = purchased)

leads_training <- leads_split %>% training()
leads_test <- leads_split %>% testing()
```

## 2 - Model Fitting and Prediction

### 2a. Defining a parsnip model object

```
logistic_model <- logistic_reg() %>%
, set_engine('glm') %>%
, set_mode('classification')
```

### 2b. Train the model

```
logistic_fit <- logistic_model %>%
fit(purchased ~ total_visits + total_time
, data=leads_training)
```

### 2c. Making Predictions

```
class_preds <- logistic_fit %>%
predict(new_data=leads_test
, type='class')
```
*to get estimated probabilities*

```
prob_preds <- logistic_fit %>%
predict(new_data=leads_test
, type='prob')
```

### 2d. Combine test data with predictions

```
leads_results <- leads_test %>%
select(purchased) %>%
bind_cols(class_preds, prob_preds)
```
