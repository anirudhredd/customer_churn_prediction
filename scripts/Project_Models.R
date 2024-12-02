rm(list = ls())

packages <- c("caret", "rpart", "randomForest","corrplot", "gbm", "ROCR", "dplyr", "ggplot2")
new_packages <- packages[!packages %in% installed.packages()[, "Package"]]
if (length(new_packages) > 0) install.packages(new_packages)

lapply(packages, library, character.only = TRUE)

churn_data <- read.csv("C:/Users/aniru/OneDrive/Desktop/spring24/PA/group project/Churn_Modelling.csv")


print(head(churn_data))
print(colnames(churn_data))

# Data preprocessing
# Remove unnecessary columns
churn_data <- subset(churn_data, select = -c(RowNumber, CustomerId, Surname))

# Checking for and summarize missing data
missing_data_summary <- sapply(churn_data, function(x) sum(is.na(x)))
print("Summary of missing data:")
print(missing_data_summary)

# Convert categorical variables to factors with explicit naming for levels
churn_data$Geography <- as.factor(churn_data$Geography)
churn_data$Gender    <- as.factor(churn_data$Gender)
churn_data$Exited    <- factor(churn_data$Exited, levels = c(0, 1), labels = c("No", "Yes"))
print(churn_data$Exited)
# Generate summary statistics for numerical features
summary_stats <- churn_data %>%
  summarise(
    mean_age = mean(Age, na.rm = TRUE),
    median_age = median(Age, na.rm = TRUE),
    sd_age = sd(Age, na.rm = TRUE),
    mean_balance = mean(Balance, na.rm = TRUE),
    median_balance = median(Balance, na.rm = TRUE),
    sd_balance = sd(Balance, na.rm = TRUE)
  )
print(summary_stats)

# EDA 
# Define the color palette
grey_blue <- "#5A92A5"
colors <- c("No" = grey_blue, "Yes" = "darkgrey")

# Histogram for Age Distribution
ggplot(churn_data, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "#5A92A5", color = "black") +
  labs(title = "Age Distribution", x = "Age", y = "Count") +
  theme_minimal()

# Histogram of Churn by Age
ggplot(churn_data, aes(x = Age, fill = Exited)) +
  geom_histogram(bins = 20, position = "stack", alpha = 0.7) +  # Added transparency for visibility
  scale_fill_manual(values = c("No" = "#5A92A5", "Yes" = "darkgrey")) +  # Define your own colors if needed
  labs(title = "Distribution of Churn by Age", x = "Age", y = "Count", fill = "Exited Status") +
  theme_minimal()

# Box Plot for Balance by Geography
ggplot(churn_data, aes(x = Geography, y = Balance)) +
  geom_boxplot(fill = "#5A92A5") +
  labs(title = "Box Plot of Balance by Geography", x = "Geography", y = "Balance") +
  theme_minimal()

# Scatter Plot for Age vs. Balance, colored by Exited
ggplot(churn_data, aes(x = Age, y = Balance, color = Exited)) +
  geom_point(alpha = 0.7) +
  scale_color_manual(values = colors) +
  labs(title = "Age vs. Balance", x = "Age", y = "Balance", color = "Exited") +
  theme_minimal()

# Correlation Heatmap
# Select numeric columns only for correlation analysis
numeric_data <- churn_data %>% 
  select_if(is.numeric)  # Ensures only numeric columns are selected

# Calculate correlation matrix
correlation_matrix <- cor(numeric_data, use = "complete.obs")

# Plot correlation heatmap
corrplot(correlation_matrix, method = "color", type = "upper", tl.col = "black", addCoef.col = "black")


# Data partitioning
set.seed(123)
train_index <- createDataPartition(churn_data$Exited, p = 0.8, list = FALSE)
train_data <- churn_data[train_index, ]
test_data  <- churn_data[-train_index, ]

# Model training and evaluation
# Logistic Regression with Cross-Validation
ctrl <- trainControl(method = "cv", number = 5)
logistic_cv_model <- train(Exited ~ ., data = train_data, method = "glm", family = "binomial", trControl = ctrl)
print(logistic_cv_model)

# Decision Tree Model
cart_model <- rpart(Exited ~ ., data = train_data, method = "class")
cart_preds <- predict(cart_model, newdata = test_data, type = 'class')
summary(cart_model)

# Random Forest Model
rf_model <- randomForest(Exited ~ ., data = train_data)
rf_preds <- predict(rf_model, newdata = test_data)
varImpPlot(rf_model)
print(rf_model)

# Boosted Tree Model
boosted_ctrl <- trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE)
boosted_tuneGrid <- expand.grid(.n.trees = c(100, 500), .interaction.depth = c(1, 3), .shrinkage = c(0.01, 0.1), .n.minobsinnode = c(10, 20))
boosted_model <- train(Exited ~ ., data = train_data, method = "gbm", trControl = boosted_ctrl, tuneGrid = boosted_tuneGrid, metric = "ROC", verbose = FALSE)
boosted_preds <- predict(boosted_model, newdata = test_data, type = "prob")[, "Yes"]
boosted_class <- ifelse(boosted_preds > 0.5, "Yes", "No")
boosted_class <- factor(boosted_class, levels = c("No", "Yes"))
print(boosted_model)
# Function to evaluate model performance
evaluate_model <- function(true_labels, predictions) {
  predictions <- factor(predictions, levels = levels(true_labels))
  confusion <- confusionMatrix(predictions, true_labels)
  roc_pred <- prediction(as.numeric(predictions), as.numeric(true_labels))
  roc_perf <- performance(roc_pred, "tpr", "fpr")
  auc <- performance(roc_pred, "auc")
  auc_value <- auc@y.values[[1]]
  list(Confusion_Matrix = confusion, AUC = auc_value, ROC_Performance = roc_perf)
}

# Evaluate and print AUC scores for all models
logistic_eval <- evaluate_model(test_data$Exited, as.character(predict(logistic_cv_model, newdata = test_data)))
cart_eval <- evaluate_model(test_data$Exited, as.character(cart_preds))
rf_eval <- evaluate_model(test_data$Exited, as.character(rf_preds))
boosted_eval <- evaluate_model(test_data$Exited, boosted_class)

cat("AUC Scores:\n")
cat("Logistic Regression:", logistic_eval$AUC, "\n")
cat("CART:", cart_eval$AUC, "\n")
cat("Random Forest:", rf_eval$AUC, "\n")
cat("Boosted Tree:", boosted_eval$AUC, "\n")
print(boosted_eval$Confusion_Matrix)


# Function to generate ROC curve data
rocCurveData <- function(model, data, positive_class = "Yes") {
  prob <- predict(model, data, type = "prob")[, positive_class]
  predob <- prediction(prob, data$Exited)
  perf <- performance(predob, "tpr", "fpr")
  return (data.frame(tpr = perf@y.values[[1]], fpr = perf@x.values[[1]]))
}

# Create performance data frames for each model
performance.df <- rbind(
  cbind(rocCurveData(logistic_cv_model, test_data, positive_class = "Yes"), model = "Logistic Regression"),
  cbind(rocCurveData(cart_model, test_data, positive_class = "Yes"), model = "CART"),
  cbind(rocCurveData(rf_model, test_data, positive_class = "Yes"), model = "Random Forest"),
  cbind(rocCurveData(boosted_model, test_data, positive_class = "Yes"), model = "Boosted Tree")
)

# Colors for each model
colors <- c("Logistic Regression" = "blue", "CART" = "green", "Random Forest" = "red", "Boosted Tree" = "darkgreen")

# Plotting the ROC curves
ggplot(performance.df, aes(x = fpr, y = tpr, color = model)) +
  geom_line(size = 1) +
  scale_color_manual(values = colors) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), color = "grey", linetype = "dashed") +
  labs(title = "Comparison of ROC Curves",
       x = "1 - Specificity (False Positive Rate)",
       y = "Sensitivity (True Positive Rate)",
       color = "Model") +
  theme_minimal()

