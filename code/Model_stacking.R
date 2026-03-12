# Osorno Viveros Jorge Proyecto Final acutalizacion (Mejorando los resultados FP Y FN)

# Carga de librerias
library(caret)
library(caretEnsemble)
library(dplyr)
library(randomForest)
library(e1071)
library(xgboost)
library(nnet)
library(glmnet) # Aun necesito instalarla.
library(factoextra) # Analisis PCA
library(ggplot2)
library(reshape2)
library(psych)
library(corrplot)
library(moments)
library(GGally)
library(pROC)  # Para ROC y AUC
library(ROCR)  # Alternativa para ROC y AUC
library(ggplot2)  # Para la visualización


# PASO 1: Extraccion y carga de los datos
# Se hizo uso del conjunto de datos "Spambase", proporcionado por la UCI Machine Learning Repository.
data<-read.csv("spambase.csv")

# PASO 2: Exploracion y Preparacion de los datos

#2.1 Comenzamos observando si se cargaron los datos y como se ven a grandes rasgos.
#Despues, vemos la estructura y resumen de nuestros datos.
head(data,5) # Ver las primeras 5 filas del datase
str(data)# Ver la estructura de los datos


# 2.2 Resumen estadistico de cada variable
summary(data) # Resumen estadístico de todas las columnas

# Ver qué columnas son numéricas
numeric_columns <- sapply(data, is.numeric)

# Mostrar los nombres de las columnas numéricas
names(data)[numeric_columns]

# Estadísticas descriptivas
describe(data[, numeric_columns])


# 2.3 Comprobar valores falantes.
#Limpieza de datos 
# Verificar valores faltantes
sum(is.na(data)) # Busqueda global
colSums(is.na(data)) # Verificar valores faltantes por variable


# 2.4 Dsitribucion de variables numericas

# Transformar los datos
data_long <- melt(data, measure.vars = c("word_freq_make", "word_freq_address", "word_freq_all")) # Elegir las variables que se van a analizar

ggplot(data_long, aes(x = value)) +
  geom_histogram(bins = 30, fill = "red", color = "black") +  # Los bins podemos saber en cuantos intervalos se divide
  facet_wrap(~variable, scales = "free") +
  labs(title = "Histogramas de variables numéricas",
       x = "Valor",
       y = "Frecuencia") +
  theme_minimal()


# 2.5 Exploracion de variables categoricas
# Contar la cantidad de cada clase (Spam vs No Spam)
table(data$class)

# Gráfico de barras para la variable 'spam'
barplot(table(data$class), main="Distribución de Spam vs No Spam", col=c("green", "red")) # Verde=No SPAM, Rojo=Spam


# 2.6 Correlacion de variables numericas
# Crear una matriz de correlación
cor_matrix <- cor(data[, sapply(data, is.numeric)])

# Mapa de calor de la correlación
dev.new()  # Abre un nuevo dispositivo gráfico
heatmap(cor_matrix, main="Mapa de calor de la correlación", col=cm.colors(256))

cor_matrix_melt <- melt(cor_matrix)
echo = TRUE

# Graficar el mapa de calor con ggplot2
ggplot(cor_matrix_melt, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme_minimal() +
  labs(title = "Mapa de calor de la correlación", x = "Variables", y = "Variables")

# Calcular la matriz de correlación
cor_matrix <- cor(data[, numeric_columns])

# Ver la matriz de correlación
print(cor_matrix)
# Visualizar la matriz de correlación usando corrplot
corrplot(cor_matrix, method = "color", addCoef.col = "black", tl.cex = 0.7, tl.col = "black")


# 2.7 Busqueda de Outliers
# Visualiza las mismas variables que las elegidas durante el 2.4 Dsitribucion de variables numericas

# Crear los boxplots para todas las variables
ggplot(data_long, aes(x = variable, y = value)) +
  geom_boxplot(fill = "lightblue", color = "black") +
  theme_minimal() +
  labs(title = "Boxplots de las variables numéricas", x = "Variables", y = "Valores") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotar etiquetas de las variables


# 2.8 DistribuCion de la variable objetivo (SPAM o NO SPAM).
# Distribución de la variable objetivo 'spam'
table(data$class) # Número de correos spam y no spam
prop.table(table(data$class)) # Porcentaje de cada clase


# 2.9 Analisis de relaciones entre varibales
# Relación entre las características numéricas y la variable objetivo
boxplot(word_freq_make ~ class, data = data, main = "Relación entre feature y spam", col = c("red", "green"))


# Un analisis mas profundo.(skewness y Kurtosis)

# Verificar la asimetría (skewness) y curtosis (kurtosis)
skewness(data[, numeric_columns])

kurtosis(data[, numeric_columns])


# Matriz de dispersion

# Esta tarda mucho tiempo en imprimirse y no es muy visible dbeeria hacerse por partes
# Crear una matriz de dispersión
ggpairs(data[, numeric_columns])



#### Analisis PCA

# Crear una matriz de dispersión con las primeras 6 variables numéricas 
ggpairs(data[, numeric_columns][, 1:6],  # Modificar las variables que se quieren mostrar
        aes(color = as.factor(data$class)))  # Colorear por la clase (Spam vs No Spam)


# Primer analisis de PCA
# Escalar datos
data_scaled <- scale(data[, numeric_columns])

# Detectar outliers
library(robustHD)
library(robustbase)

# Calcular distancia de Mahalanobis robusta
cov_robust <- covMcd(data_scaled)
mahal_dist <- mahalanobis(data_scaled, center = cov_robust$center, cov = cov_robust$cov)

# Determinar un umbral para outliers (distribución chi-cuadrado)
threshold <- qchisq(0.975, df = ncol(data_scaled))
outlier_indices <- which(mahal_dist > threshold)

# Imputar outliers con la mediana
data_imputed <- data  # Hacemos copia del dataset original
for (i in outlier_indices) {
  for (j in which(numeric_columns)) {
    var_name <- names(data)[j]
    data_imputed[i, var_name] <- median(data[[var_name]], na.rm = TRUE)
  }
}

# Escalar nuevamente después de imputar
data_scaled_imputed <- scale(data_imputed[, numeric_columns])

# Eliminar columnas con varianza cero o casi cero
nzv <- nearZeroVar(data_scaled_imputed, saveMetrics = TRUE)
data_scaled_imputed <- data_scaled_imputed[, !nzv$zeroVar]

# PCA sobre datos imputados
pca <- prcomp(data_scaled_imputed, center = TRUE, scale. = TRUE)

# Seleccionar número de componentes que expliquen >= 70% de la varianza
library(factoextra)
eig_vals <- get_eigenvalue(pca)
num_components <- which(eig_vals$cumulative.variance.percent >= 70)[1]

# Crear dataset con los componentes principales
pca_data <- as.data.frame(pca$x[, 1:num_components])

# Añadir la variable objetivo (class)
pca_data$class <- data$class  # Usamos `data` o `data_imputed`, ya que las filas son iguales


# Tenemos un nuevo dataset con solo 27 variables

# Normalizar los datos (recomendable para PCA)
#data_scaled <- scale(data[, numeric_columns])

# Aplicar PCA
#pca <- prcomp(data_scaled, center = TRUE, scale. = TRUE)

# Resumen del PCA para ver la varianza explicada por cada componente
summary(pca)


# Visualizar la varianza explicada por cada componente
fviz_eig(pca)

# Visualizar los primeros dos componentes principales
fviz_pca_ind(pca, 
             geom.ind = "point", 
             col.ind = as.factor(data$class), 
             palette = c("blue", "red"), 
             addEllipses = TRUE,
             title = "PCA: Proyección de los datos en los dos primeros componentes")


# Segundo Analisis de componentes por filtro de Correlacion
# Filtro de reduccion

# Calcular la matriz de correlación para las variables numéricas
cor_matrix <- cor(data[, numeric_columns])

# Establecer un umbral de correlación (por ejemplo, 0.9)
threshold <- 0.9

# Encontrar los pares de variables con alta correlación
highly_correlated <- findCorrelation(cor_matrix, cutoff = threshold)

# Eliminar las variables correlacionadas
data_reduced <- data[, -highly_correlated]

# Ver las variables eliminadas
removed_variables <- colnames(data)[highly_correlated]
print(paste("Variables eliminadas debido a alta correlación:", paste(removed_variables, collapse = ", ")))


# Para SVM Redes neuronales
# Escalar los datos (por ejemplo, estandarizar)
preprocess <- preProcess(data[, -ncol(data)], method = c("center", "scale"))
data_scaled <- predict(preprocess, newdata = data[, -ncol(data)])

# Agregar la columna `class` de vuelta
data_scaled$class <- data$class


# Limpiamos las variable creada anteriormente y liberamos memoria (para mejor orden)
rm(data_long) # Eliminar una vairbale
rm(cor_matrix_melt) 
rm(data_scaled_imputed)
rm(cov_robust)
rm(mahal_dist)
rm(nzv)
rm(eig_vals) # Quiza esta se use en el futuro
# rm(pca)
rm(data_imputed)

gc() # Liberar memoria



# PASO 3: Division y entrenamiento

# Division y entrenamiento de datos (Modificar al dataset que se quiera DATA_SCALED, RATA_REDUCED o PCA_DATA son las opciones actuales)
set.seed(777) # Para reproducibilidad
trainIndex <- createDataPartition(data_scaled$class, p = 0.7, list = FALSE)
train_data <- data_scaled[trainIndex, ]
test_data <- data_scaled[-trainIndex, ]


# PASO 1: Preprocesamiento de los datos
# Convertir `class` en un factor en ambos conjuntos de datos (entrenamiento y prueba)
train_data$class <- factor(train_data$class)
test_data$class <- factor(test_data$class)

# Asegurarse de que los niveles de `class` sean los mismos en ambos conjuntos
levels(test_data$class) <- levels(train_data$class)


# Naive Bayes
# Entrenar un modelo Naive Bayes
model_nb <- naiveBayes(class ~ ., data = train_data)

# Realizar predicciones
predictions_nb <- predict(model_nb, newdata = test_data)

# Evaluar el modelo
confusionMatrix(predictions_nb, test_data$class)


# SVM
# Entrenar un modelo SVM
model_svm <- svm(class ~ ., data = train_data)

# Realizar predicciones
predictions_svm <- predict(model_svm, newdata = test_data)

# Evaluar el modelo
confusionMatrix(predictions_svm, test_data$class)


# Random Forest
model_rf <- randomForest(class ~ ., data = train_data)
predictions_rf <- predict(model_rf, newdata = test_data) # Realizar predicciones

# Evaluar el modelo
confusionMatrix(predictions_rf, test_data$class)



# Entrenar un modelo de regresión logística
model_lr <- glmnet(as.matrix(train_data[, -ncol(train_data)]), train_data$class, family = "binomial")

# Realizar predicciones (probabilidades)
predictions_lr <- predict(model_lr, newx = as.matrix(test_data[, -ncol(test_data)]), type = "response")

# Seleccionar las probabilidades para la clase positiva (1)
prob_class_1 <- predictions_lr[, 2]

# Convertir las probabilidades en clases (usamos un umbral de 0.5)
predictions_lr_class <- ifelse(prob_class_1 > 0.5, 1, 0)

# Convertir las predicciones a un factor con los niveles adecuados
predictions_lr_class <- factor(predictions_lr_class, levels = levels(test_data$class))

# Evaluar el modelo
confusionMatrix(predictions_lr_class, test_data$class)


# Redes neuronales
library(nnet)
library(NeuralNetTools)

# Entrenar un modelo de red neuronal (MLP)
model_nn <- nnet(class ~ ., data = train_data, size = 10, linout = FALSE)

# Realizar predicciones
predictions_nn <- predict(model_nn, newdata = test_data, type = "class")

# Convertir las predicciones a factor y asegurarse de que los niveles coincidan
predictions_nn <- factor(predictions_nn, levels = levels(test_data$class))

# Evaluar el modelo
confusionMatrix(predictions_nn, test_data$class)

plotnet(model_nn) # Imprimimos la red Neuronal

print(plotnet(model_nn))


# Entrenando otra red neuronal (para mejora)
library(h2o)
h2o.init()

# Convertir a H2OFrame
train_h2o <- as.h2o(train_data)
test_h2o <- as.h2o(test_data)

# Especificar la variable respuesta y las predictoras
y <- "class"
x <- setdiff(names(train_data), y)  # Todas las columnas excepto "class"

# Asegurarse de que la variable respuesta sea categórica
train_h2o[, y] <- as.factor(train_h2o[, y])
test_h2o[, y] <- as.factor(test_h2o[, y])


model_h2o <- h2o.deeplearning(
  x = x,
  y = y,
  training_frame = train_h2o,
  activation = "Rectifier",         # Función de activación (puede ser "Tanh", "Rectifier", etc.)
  hidden = c(10, 5),                # Dos capas ocultas: 10 y 5 neuronas
  epochs = 50,                      # Número de iteraciones
  seed = 123                        # Reproducibilidad
)

# Realizar predicciones
predictions_h2o <- h2o.predict(model_h2o, test_h2o)

# Convertir predicciones y etiquetas verdaderas a vectores
pred <- as.vector(predictions_h2o$predict)
true <- as.vector(test_h2o[, y])

# Comparacion library(caret)

confusionMatrix(factor(pred, levels = levels(test_data$class)), test_data$class)

summary(model_h2o)


# PASO 4: Comparacion de los modelos
# F1 score
# Función para calcular F1 Score
f1_score <- function(conf_matrix) {
  precision <- conf_matrix$byClass['Pos Pred Value']
  recall <- conf_matrix$byClass['Sensitivity']
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(f1)
}

# Crear una función para obtener las métricas
get_metrics <- function(predictions, true_values) {
  cm <- confusionMatrix(predictions, true_values)
  
  # Métricas de la matriz de confusión
  accuracy <- cm$overall['Accuracy']
  kappa <- cm$overall['Kappa']
  precision <- cm$byClass['Pos Pred Value']
  recall <- cm$byClass['Sensitivity']
  specificity <- cm$byClass['Specificity']
  f1 <- f1_score(cm)
  balanced_accuracy <- (recall + specificity) / 2  # Balanced Accuracy
  
  # Devuelvo las métricas en un data.frame
  return(data.frame(
    Accuracy = accuracy,
    Kappa = kappa,
    Precision = precision,
    Recall = recall,
    Specificity = specificity,
    F1 = f1,
    BalancedAccuracy = balanced_accuracy
  ))
}

# Metricas de cada modelo

# Obtener las métricas para Naive Bayes
metrics_nb <- get_metrics(predictions_nb, test_data$class)

# Obtener las métricas para SVM
metrics_svm <- get_metrics(predictions_svm, test_data$class)

# Obtener las métricas para Random Forest
metrics_rf <- get_metrics(predictions_rf, test_data$class)

# Obtener las métricas para Regresión Logística
predictions_lr_class <- factor(predictions_lr_class, levels = levels(test_data$class))
metrics_lr <- get_metrics(predictions_lr_class, test_data$class)

# Obtener las métricas para Redes Neuronales
metrics_nn <- get_metrics(predictions_nn, test_data$class)

# Combinar todas las métricas en un solo data.frame
all_metrics <- rbind(
  cbind(Model = 'Naive Bayes', metrics_nb),
  cbind(Model = 'SVM', metrics_svm),
  cbind(Model = 'Random Forest', metrics_rf),
  cbind(Model = 'Logistic Regression', metrics_lr),
  cbind(Model = 'Neural Network', metrics_nn)
)

# Mostrar las métricas
print(all_metrics)


# Función para crear la curva ROC y calcular el AUC
plot_roc_curve <- function(model, predictions, true_values) {
  # Para obtener probabilidades, si no es un modelo de clasificación binaria directamente
  if (model == "Logistic Regression") {
    prob <- predictions_lr[, 2]  # Probabilidades para la clase positiva (1)
  } else if (model == "Neural Network") {
    prob <- as.numeric(predictions_nn)  # Predicciones como probabilidad
  } else {
    prob <- as.numeric(predictions)  # Para Naive Bayes, SVM y Random Forest
  }
  
  # Crear la curva ROC
  roc_curve <- roc(true_values, prob)
  
  # Graficar la curva ROC
  plot(roc_curve, col = "blue", main = paste(model, "ROC Curve"))
  auc_value <- auc(roc_curve)
  return(auc_value)
}

# Graficar las curvas ROC y calcular el AUC para cada modelo
auc_nb <- plot_roc_curve("Naive Bayes", predictions_nb, test_data$class)
auc_svm <- plot_roc_curve("SVM", predictions_svm, test_data$class)
auc_rf <- plot_roc_curve("Random Forest", predictions_rf, test_data$class)
auc_lr <- plot_roc_curve("Logistic Regression", predictions_lr_class, test_data$class)
auc_nn <- plot_roc_curve("Neural Network", predictions_nn, test_data$class)

# Mostrar los AUCs
print(paste("AUC Naive Bayes:", auc_nb))
print(paste("AUC SVM:", auc_svm))
print(paste("AUC Random Forest:", auc_rf))
print(paste("AUC Logistic Regression:", auc_lr))
print(paste("AUC Neural Network:", auc_nn))



# Limpiamos las variable creada anteriormente y liberamos memoria (para mejor orden)
rm(auc_nb) 
rm(auc_svm) 
rm(auc_rf)
rm(auc_lr)
rm(auc_nn)

gc() # Liberar memoria
```

# Visualizaciones

# Gráfico de comparación de Accuracy entre modelos
ggplot(all_metrics, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Comparación de Accuracy entre Modelos", x = "Modelo", y = "Accuracy") +
  theme_minimal()


# Comparacion de F1

# Gráfico de comparación de F1 Score entre modelos
ggplot(all_metrics, aes(x = Model, y = F1, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Comparación de F1 Score entre Modelos", x = "Modelo", y = "F1 Score") +
  theme_minimal()

# PASO 5: Mejora del modelo stacking

# Ensamblamos el modelo con nuestros modelos anteriores
# Predicciones de los modelos base 
rf_pred <- predict(model_rf, newdata = test_data)
svm_pred <- predict(model_svm, newdata = test_data)
ann_pred <- predict(model_nn, newdata = test_data)

# Combinar estas predicciones en un dataframe
stacking_data <- data.frame(rf = rf_pred, svm = svm_pred, ann = ann_pred)

# Añadir la columna 'class' al stacking_data
stacking_data$class <- test_data$class


# Crear el modelo meta usando, por ejemplo, Random Forest
meta_model <- train(
  class ~ .,  # donde class es la variable objetivo
  data = stacking_data,
  method = "rf",  # o el método que prefieras para el meta-modelo
  trControl = trainControl(method = "cv", number = 5)
)

# Hacer predicciones con el modelo meta
final_pred <- predict(meta_model, newdata = stacking_data)

# Evaluar el rendimiento
conf_matrix <- confusionMatrix(final_pred, test_data$class, positive = "0")  
print(conf_matrix)


# Se logro una mejora en el modelo stacking 96%
 
# Ver la importancia de las predicciones de los modelos base en el meta-modelo
varImp(meta_model)

#Observamos que este modelo no hace uso de la ANN


# Ultimo modelo con pesos Ponderados

# Predicciones de los modelos base 
rf_pred <- predict(model_rf, newdata = test_data)
svm_pred <- predict(model_svm, newdata = test_data)
#ann_pred <- predict(model_nn, newdata = test_data)

# Combinar estas predicciones en un dataframe
stacking_data <- data.frame(rf = rf_pred, svm = svm_pred) #, ann = ann_pred)

# Añadir la columna 'class' al stacking_data
stacking_data$class <- test_data$class

rf_pred_numeric <- as.numeric(rf_pred) - 1  # "0" se convierte en 0, "1" se convierte en 1
svm_pred_numeric <- as.numeric(svm_pred) - 1
#ann_pred_numeric <- as.numeric(ann_pred) - 1

# Corroborar de que la columna class está presente
names(stacking_data)

# Verificar si tiene valores NA
sum(is.na(stacking_data$class))
# Verificar si es un factor con dos niveles
str(stacking_data$class)


class ~ rf_weighted + svm_weighted #+ ann_weighted
class ~ rf_weighted + svm_weighted #+ ann_weighted

# Asegura de que las predicciones tengan la misma longitud
length(rf_pred_numeric) == nrow(test_data)  # Debe ser TRUE

####
# Asignar pesos a los modelos base
rf_weight <- 0.7
svm_weight <- 0.2
ann_weight <- 0.1
####


# Crear nuevo dataframe de stacking
stacking_data <- data.frame(
  class = test_data$class,  # Aquí agregamos la variable objetivo
  rf_weighted = rf_pred_numeric * rf_weight,
  svm_weighted = svm_pred_numeric * svm_weight #,
  #ann_weighted = ann_pred_numeric * ann_weight
)

# Confirmar que no hay NA ahora
sum(is.na(stacking_data))  # Debe ser 0

# Entrenar modelo meta
meta_model_en <- train(
  class ~ rf_weighted + svm_weighted, #+ ann_weighted,
  data = stacking_data,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5)
)

# Hacer predicciones con el modelo meta
final_pred <- predict(meta_model_en, newdata = stacking_data)

# Evaluar el rendimiento
conf_matrix <- confusionMatrix(final_pred, test_data$class, positive = "1")  
print(conf_matrix)


# Si eliminamos el ANN vemos una disminucion del rendimiento por lo que se conlcuye que si influye en el modelo.