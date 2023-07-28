library(keras)
library(data.table)
library(tensorflow)
library(dplyr)
library(ggplot2)
require(ggpubr)


load("localGLMnet_example.RData")
#n_sample: number of training instances
#x_train: array with size n_samplex10x100 containing the matrices of the past mortality rates related to the past 10 years t-1, t-2, ..., t-10 used for the training 
#y_train: array of size n_samplex1x100 containing the response vectors of the mortality rates at time t used for the training 
#id: list of length n_sample containing information about each training instance 
#all_mort: data frame collecting all the data




eta = 0; alpha = 0
string = "localGLMnet_example"
look_back=10



# localGLMnet architecture
k_clear_session()

rates <- layer_input(shape = c(10,100), dtype = 'float32', name = 'rates')
interim  = rates  %>% layer_reshape(c(10,100,1)) %>% layer_zero_padding_2d(padding =c(2,2))%>%
  layer_locally_connected_2d(filter =1, kernel_size = list(5,5), activation = "sigmoid")%>%layer_reshape(c(10,100), name = 'interim') 

decoded_masked = list(rates, interim) %>% layer_multiply( name = 'decoded_masked')

forecast_rates = decoded_masked %>% layer_permute(dims = c(2,1)) %>% 
  time_distributed(layer_dense(units = 1, activation = "linear", 
                               use_bias = F), trainable = F,  
                   weights = list(array(1, dim = c(10,1)))) %>% 
  layer_permute(dims = c(2,1)) 


Penaltyl1 =interim  %>% layer_lambda(function(x) k_square(x))%>% 
  layer_lambda(function(x) eta*alpha*k_sqrt(x+0.000001))%>% layer_reshape(c(1000,1))%>%
  layer_conv_1d(1, kernel_size = 10, strides = 10, use_bias = F,  trainable = F,  weights = list(array(1, dim = c(10,1,1))))%>%
  layer_reshape(c(1,100))

Penaltyl2 =interim  %>% 
  layer_lambda(function(x) eta*(1-alpha)*k_square(x))%>% layer_reshape(c(1000,1))%>%
  layer_conv_1d(1, kernel_size = 10, strides = 10, use_bias = F,  trainable = F,  weights = list(array(1, dim = c(10,1,1))))%>%
  layer_reshape(c(1,100))

Penalty = list(Penaltyl1, Penaltyl2) %>% layer_add()

Output = list(forecast_rates, Penalty) %>% layer_concatenate(axis = 1)

model <- keras_model(
  inputs = list(rates), 
  outputs = c(Output))
  
  MSEP_regularized <- function (y_true, y_pred){ k_mean((y_true[,1,] - y_pred[,1,])^2 + y_pred[,2,])}
  
  adam = optimizer_nadam()
  
  
  model %>% compile(loss = MSEP_regularized, optimizer = adam)
  
  
  lr_callback = callback_reduce_lr_on_plateau(factor=.90, patience = 50, verbose=1, cooldown = 5, min_lr = 0.00005)
  
  model_callback = callback_model_checkpoint(filepath = paste0(string, "/cp.ckpt"), verbose = 1,save_best_only = TRUE, save_weights_only = T)
  
  
  #Model fitting
  fit <- model %>% keras::fit(x = list(x_train),
                              y= list(y_train),
                              epochs=500,
                              batch_size=16,
                              verbose=2,
                              validation_split = 0.05,
                              callbacks = list(lr_callback,model_callback),
                              shuffle=T)
  


#load the optimized weights

fit = model %>% load_model_weights_tf(filepath = paste0(string, "/cp.ckpt")) 


#Model for attention coefficients and contribution values
model_attention <- keras_model(
  inputs = fit$input, 
  outputs = list( get_layer(fit,  'interim')$output, get_layer(fit,  'decoded_masked')$output))

full_w = model_attention %>% predict(list(x_train))
all_coeff = list()
for (samp in 1:n_sample){
  coeff = full_w[[1]][samp,,] %>% melt() %>% data.table()
  dec = full_w[[2]][samp,,] %>% melt() %>% data.table()
  all_mx = x_train[samp,,] %>% melt() %>% data.table()
  coeff$decoded = dec$value
  coeff$mx = all_mx$value
  st = id[[samp]]
  coeff$Country = strsplit(st, "[.]")[[1]][1]
  coeff$Sex = strsplit(st, "[.]")[[1]][2]
  coeff$for_year = strsplit(st, "[.]")[[1]][3] %>% as.numeric()
  coeff[,Age:=Var2-1]
  coeff[,lag:= look_back-Var1 ]
  coeff$Var1 = NULL
  coeff$Var2 = NULL
  all_coeff[[samp]] = coeff 
  
}

all_coeff = rbindlist(all_coeff)

#Grapical analysis of Attention Coefficients
ggplot(all_coeff[Age %in% c(0,seq(5,99,5), 99)], aes(mx,value, color = factor(lag)))+geom_point(size = 0.5)+
  facet_wrap(~Age, scales = "free",ncol = 7)+
  geom_hline(yintercept=  0, col = "black")+ labs(y = "Regression Attention", x = "mx", color = "lag")+
  theme_pubr() + theme(legend.position = "right", axis.text.x = element_text(angle = 30, vjust = 0.5, hjust=1, size = 8))


#Grapical analysis of Contribution values
ggplot(all_coeff[Age %in% c(0,seq(5,99,5), 99)], aes(mx,decoded, color = factor(lag)))+geom_point(size = 0.5)+facet_wrap(~Age, scales = "free",ncol = 7)+
  facet_wrap(~Age, scales = "free",ncol = 7)+
  geom_hline(yintercept=  0, col = "black")+ labs(y = "Contribution Values", x = "mx", color = "lag")+
  theme_pubr() + theme(legend.position = "right", axis.text.x = element_text(angle = 30, vjust = 0.5, hjust=1, size = 8))


