library(shiny)
library(tidyquant)
library(tidyverse)
library(plotly)
library(shinythemes)
# library(shinycssloaders)
# library(DT)
# library(RColorBrewer)
library(reticulate)
library(keras)
library(tensorflow)

# Begin UI for the R + reticulate example app
ui <- fluidPage(theme = shinytheme("flatly"),
                navbarPage("Machine Learning Stock Price Prediction",
                           tabPanel("Calculator",
                                    titlePanel("Predict the price of any publicly traded stock using a LSTM machine learning model (Computation takes about 2min)"),
                                    titlePanel(""),
                                    sidebarPanel(textInput("ticker", label = "Enter a Stock Ticker", value = "HTZ"),
                                                 actionButton("Compute", 
                                                              "Compute", 
                                                              class = "btn btn-primary")),
                                    mainPanel(plotlyOutput('graph'),
                                              h4("")),
                           )
                )
)