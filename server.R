library(shiny)
library(tidyquant)
library(tidyverse)
library(plotly)
library(shinythemes)
#library(shinycssloaders)
#library(DT)
#library(RColorBrewer)
library(reticulate)

library(shiny)
library(tensorflow)
library(keras)

# reticulate::virtualenv_create("myenv", python="/usr/bin/python3")
# reticulate::use_virtualenv("myenv", required=TRUE)
# 
# if (!is_keras_available()) {
#   install_keras(method="virtualenv", envname="myenv")
#   reticulate::use_virtualenv("myenv", required=TRUE)
#   library(keras)
#   library(reticulate)
# }
# 

PYTHON_DEPENDENCIES = c('pip', 'numpy', 'pandas', 'sklearn')


shinyServer(function(input, output) {
  
  # ------------------ App virtualenv setup (Do not edit) ------------------- #
  
  virtualenv_dir = Sys.getenv('VIRTUALENV_NAME')
  python_path = Sys.getenv('PYTHON_PATH')
  
  # Create virtual env and install dependencies
  reticulate::virtualenv_create(envname = virtualenv_dir, python = python_path)
  reticulate::virtualenv_install(virtualenv_dir, packages = PYTHON_DEPENDENCIES, ignore_installed=FALSE)
  reticulate::use_virtualenv(virtualenv_dir, required = T)
  
  # ------------------ App server logic (Edit anything below) --------------- #
  
  datasetInput <- reactive({
    
    x <- tq_get(input$ticker) %>% select(Date = date, Close = close)
    
    reticulate::source_python("ml_prepare_functions.py")
    
    data1 <- f1(x)
    
    model <- keras_model_sequential()
    model %>%
      layer_lstm(units = 60, return_sequences = TRUE, input_shape = c(60,1)) %>% 
      layer_dropout(rate = 0.2) %>% 
      layer_lstm(units = 60,return_sequences = TRUE) %>% 
      layer_dropout(rate = 0.2) %>% 
      layer_lstm(units = 60,return_sequences = TRUE) %>%
      layer_dropout(rate = 0.2) %>% 
      layer_lstm(units = 60) %>%
      layer_dense(units = 8, activation = "relu") %>% 
      layer_dense(units = 1)
    
    
    model %>% compile(optimizer = "adam", 
                      loss = 'mean_squared_error', 
                      metrics = c('accuracy'))
    
    history <- model %>% fit(data1[[1]], data1[[2]],
                             epochs = 100,
                             batch_size = 32,
                             validation_split = 0.2)
    
    
    data2 <- f2(data1[[3]], data1[[5]], data1[[6]])
    
    y_test <- model %>% predict(data2)
    
    data3 <- f3(y_test, data1[[3]], data1[[4]], data1[[5]], data1[[6]], data1[[7]])
    
    train <- as.data.frame(data3[1])
    train <- train %>% mutate(Date = as_date(rownames(train))) %>% 
      select(Date, `Actual Before` = Close)
    rownames(train) <- NULL
    
    valid <- as.data.frame(data3[2])
    valid <- valid %>% mutate(Date = as_date(rownames(valid))) %>% 
      select(Date, `Actual After` = Close, Prediction = Predictions)
    rownames(valid) <- NULL
    
    df <- full_join(train, valid, by = "Date") %>% 
      gather(key = "variable", value = "value", -Date)
    
    df})
  
  plotInput <- reactive({
    p <- ggplot(datasetInput(), aes(x = Date, y = value)) +
      geom_line(aes(color = variable), size = 0.5) + 
      labs(title = paste(input$ticker, 'Predicted vs. Actual Price'), x = "Date", y = "Price", color = "Closing Prices") + 
      scale_color_manual(values = c("Black", "Red", "Blue")) +
      theme_bw()
    ggplotly(p)
  })
  
  output$graph <- renderPlotly({
    if (input$Compute>0){isolate(plotInput())}
  })
  
})