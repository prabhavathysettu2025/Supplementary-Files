TITLE : 
An Automated deep learning-based soil moisture estimates using Markov Optimal Multispectral Retrieval and Soil Moisture Active Passive observations.



DESCRIPTION : 
This automated soil moisture estimation framework integrates satellite data, geospatial processing, probabilistic modeling, and deep learning. 
The process begins with data acquisition, where SMAP satellite observations, multispectral imagery (e.g., MODIS, Sentinel-2), and ground-truth moisture readings are collected. 
Preprocessing includes aligning spatial grids, resampling data, extracting vegetation and surface features (e.g., NDVI, LST), and applying normalization. 
The Markov Optimal Multispectral Retrieval (MOMR) step models soil moisture as a hidden state inferred from spectral observations using Bayesian and EM/MDP methods. 
These features are then passed to a hybrid CNN-LSTM model, which captures spatial and temporal patterns to estimate moisture content.
The model is trained and validated using ground-truth data and standard loss functions like MSE. 
Finally, predictions are deployed for real-time estimation and are evaluated against benchmark methods for accuracy and robustness, enhancing precision agriculture and environmental monitoring.



DATASET INFORMATION :
The dataset employed in this research originates from NASA’s Soil Moisture Active Passive (SMAP) mission, which provides global-scale observations of soil moisture using a combination of passive microwave radiometry and active radar sensing. 
The SMAP dataset captures volumetric soil moisture content at various depths and across diverse geographic regions, offering crucial insights into surface hydrological conditions. 
It includes temporally and spatially rich measurements with high accuracy, making it ideal for deep learning-based modeling. The dataset is extensively used for climate modeling, drought monitoring, agricultural planning, and environmental management. 
In this study, SMAP observations serve as the ground-truth reference for training and validating the proposed SGAN-SMAP framework, ensuring the reliability of soil moisture estimation. 
The dataset is further enhanced with auxiliary multispectral imagery (e.g., vegetation indices, surface temperature) and preprocessed through spatial alignment, normalization, and quality control procedures to ensure consistency and readiness for model training



STEP-BY-STEP ALGORITHM FOR AUTOMATED DEEP LEARNING BASED SOIL MOISTURE ESTIMATION :
Step 1: Data Acquisition
Obtain Soil Moisture Active Passive (SMAP) satellite observations (L-band radiometer & radar data).
Collect Multispectral Data from sources like MODIS, Sentinel-2, or Landsat.
Gather Ground-Truth Soil Moisture Measurements from field sensors for validation.

Step 2: Data Preprocessing
Geospatial Alignment: Match multispectral and SMAP data to a common grid.
Resampling & Interpolation: Ensure uniform spatial and temporal resolution.
Feature Extraction: Compute vegetation indices (NDVI, EVI), surface temperature, soil brightness.
Normalization & Scaling: Apply Min-Max scaling or Standardization.

Step 3: Markov Optimal Multispectral Retrieval (MOMR)
Define a Markov model where soil moisture states transition based on multispectral data.
Formulate Bayesian inference for soil moisture retrieval from observed spectra.
Optimize retrieval using Expectation-Maximization (EM) or Markov Decision Process (MDP).

Step 4: Deep Learning Model
Design a CNN-LSTM Hybrid Network to capture spatial and temporal soil moisture patterns.
Input: Multispectral features + SMAP observations + MOMR retrieval.
Output: Soil Moisture Estimates.
Train using Mean Squared Error (MSE) loss and Adam optimizer.

Step 5: Model Training & Validation
Split Dataset: 80% training, 20% validation.
Train using TensorFlow/PyTorch, with early stopping and dropout regularization.
Validate against ground-truth measurements using RMSE and R² metrics.

Step 6: Prediction & Evaluation
Deploy the trained model for real-time soil moisture estimation.
Evaluate model predictions against independent datasets.
Compare with classical retrieval methods (e.g., Soil Water Balance models).




CODE IMPLEMENTATION :
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LSTM, Dense, Flatten, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Step 1: Load & Preprocess Data (Example Synthetic Data)
def load_data():
    smap_data = np.random.rand(1000, 10, 10, 1)  # Simulated SMAP Data (1000 samples)
    multispectral_data = np.random.rand(1000, 10, 10, 7)  # 7 multispectral bands
    ground_truth = np.random.rand(1000, 1)  # Ground truth soil moisture
    return smap_data, multispectral_data, ground_truth

smap_data, multispectral_data, ground_truth = load_data()

# Step 2: Normalize Data
scaler = MinMaxScaler()
ground_truth_scaled = scaler.fit_transform(ground_truth)

# Step 3: Feature Engineering & Data Preparation
X = np.concatenate([smap_data, multispectral_data], axis=-1)  # Combine inputs
X_train, X_test, y_train, y_test = train_test_split(X, ground_truth_scaled, test_size=0.2, random_state=42)

# Step 4: Define Deep Learning Model (CNN-LSTM)
model = Sequential([
    TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same'), input_shape=(10, 10, 10, 8)),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Output Soil Moisture Prediction
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Step 5: Train Model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16)

# Step 6: Prediction
predictions = model.predict(X_test)
predicted_soil_moisture = scaler.inverse_transform(predictions)  # Convert back to original scale

print("Soil Moisture Predictions:", predicted_soil_moisture[:10])



### Install and import the following packages
# This step is best run on a virtual machine so that files in the Google Cloud bucket can be used directly
# Use gcsfuse or gcsfuse --implicit-dirs to import bucket folders
library(data.table)
library(rgdal)
library(raster)
library(caret)
library(randomForest)
library(quantregForest)

### Prepare site files and import model
# Make sure to set the correct path to read the file
# Set prediction depth, site, and time periods for prediction
depth = 7.5 
ROI = "Field" 
doy_frame <- c(as.Date("2002-01-01"), as.Date("2019-12-31")) 
dates <- seq.Date(doy_frame[1],doy_frame[2], 1)
# Import model derived from the regionalized or full quality-controlled model
#model_full <- readRDS("moisture_model_RAP_QA_15_5.rds") # Regional model
model_full <- readRDS("moisture_model_full_QA_15_5.rds") # Full model
# Define site parameters based on resolution and extents
res <- c(0.000269494585235856472, 0.000269494585235856472) # 30 meters
x <- raster(xmn = -110, xmx = -109, # Specify the spatial extent of the boundary
            ymn = 38, ymx = 39, 
            res=res, crs="+proj=longlat +datum=WGS84")
# Select files within the defined time frame
startingDate <- as.POSIXct(doy_frame[1], format="%Y-%m-%d")
endingDate <- as.POSIXct(doy_frame[2], format="%Y-%m-%d")
# Customize the pathways based on where the covariate files are stored
list_of_all_cov_files <- list.files(path = paste0("./",ROI,"/Dynamic_cov"), recursive = TRUE,pattern = "^cov.*tif", full.names = TRUE)
fileCovDates <- as.POSIXct(strptime(substr(list_of_all_cov_files,(nchar(ROI)+20),(nchar(ROI)+30)), format="%Y-%m-%d"))
list_of_selected_cov_files <- list.files(path = paste0("./",ROI,"/Dynamic_cov"), pattern = "^cov.*tif", full.names = TRUE)[fileCovDates >= startingDate & fileCovDates <= endingDate]
list_of_all_nldas_files <- list.files(path = paste0("./",ROI,"/Dynamic_cov"), recursive = TRUE, pattern = "^nldas.*tif", full.names = TRUE)
fileNldasDates <- as.POSIXct(strptime(substr(list_of_all_nldas_files,(nchar(ROI)+22),(nchar(ROI)+32)), format="%Y-%m-%d"))
list_of_selected_nldas_files <- list.files(path = paste0("./",ROI,"/Dynamic_cov"), pattern = "^nldas.*tif", full.names = TRUE)[fileNldasDates >= startingDate & fileNldasDates <= endingDate]
# Import constant covariates generated in GEE
predict.raster <- paste0("./",ROI,"/constant.tif")
raster <- new("GDALReadOnlyDataset", predict.raster)
width <- dim(raster)[2]
height <- dim(raster)[1]
imagedata <- data.frame(getRasterTable(raster))
names(imagedata) <- c('x', 'y', 'EL', 'SL', 'AS', 'TWI', 'surfrough', 'mcurv', 'hcurv', 'vcurv', 
                      'SOC5', 'BD5', 'Clay5', 'Sand5', 'SOC15', 'BD15', 'Clay15', 'Sand15', 'SOC30', 'BD30',
                      'Clay30', 'Sand30', 'SOC60', 'BD60', 'Clay60', 'Sand60', 'SOC100', 'BD100','Clay100','Sand100',
                      'landcover2001','landcover2004','landcover2006', 'landcover2008', 'landcover2011', 'landcover2013', 
                      'landcover2016', 'landcover2019', 'Depth')
imagedata$Depth <- depth
# Automatically extract covariate values corresponding to the defined predictive depth
imagedataSOC <- imagedataSOC5
imagedataBD <- imagedataBD5
imagedataClay <- imagedataClay5
imagedataSand <- imagedataSand5
if (depth > 10) {
  if (depth <= 22.5) {
    imagedataSOC <- imagedataSOC15
    imagedataBD<- imagedataBD15
    imagedataClay <- imagedataClay15
    imagedataSand <- imagedataSand15
  } 
  else if (depth <= 45) {
    imagedataSOC <- imagedataSOC30
    imagedataBD<- imagedataBD30
    imagedataClay <- imagedataClay30
    imagedataSand <- imagedataSand30
  } 
  else if (depth <= 80) {
    imagedataSOC <- imagedataSOC60
    imagedataBD<- imagedataBD60
    imagedataClay <- imagedataClay60
    imagedataSand <- imagedataSand60
  }  
  else{
    imagedataSOC <- imagedataSOC100
    imagedataBD<- imagedataBD100
    imagedataClay <- imagedataClay100
    imagedataSand <- imagedataSand100
  }
}
points <- SpatialPoints(imagedata[, c('x', 'y')], 
                        proj4string=CRS('+proj=longlat +datum=WGS84'))
pts <- spTransform(points, CRS("+proj=lcc +lat_1=60 +lat_2=25 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0"))
pts_nldas <- spTransform(points, CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))

### Model run
# Use loop to automatically export predicted soil moisture for selected dates
for (i in 1:length(dates)){
  cov.raster <- stack(list_of_selected_cov_files[i])
  rasValue=extract(cov.raster, pts)
  colnames(rasValue) <- c("ppt","Tavg","VPD","GPP","EVI","NDWI","LST","Tree","TREE","AFGC","PFGC","SHR","BG","LTR")
  pts_cov <- cbind(imagedata,rasValue)
  nldas.raster <- stack(list_of_selected_nldas_files[i])
  nldasValue=extract(nldas.raster, pts_nldas)
  colnames(nldasValue) <- c("d5","d25","d70") 
  pts_all <- cbind(pts_cov,nldasValue)
  pts_allSM <- pts_alld5
  if (depth > 15) {
    if (depth <= 47.5) {
      pts_allSM <- pts_alld25/3
    }
    else{
      pts_allSM <- pts_alld70/6
    }
  }
  year = substr(dates[[1]],1,4)
  pts_allLULC = pts_alllandcover2001
  if ((year == "2003")|(year == "2004")) {pts_allLULC = pts_alllandcover2004}
  if ((year == "2005")|(year == "2006")) {pts_allLULC = pts_alllandcover2006}
  if ((year == "2007")|(year == "2008")|(year == "2009")) {pts_allLULC = pts_alllandcover2008}
  if ((year == "2010")|(year == "2011")) {pts_allLULC = pts_alllandcover2011}
  if ((year == "2012")|(year == "2013")|(year == "2014")) {pts_allLULC = pts_alllandcover2013}
  if ((year == "2015")|(year == "2016")|(year == "2017")) {pts_allLULC = pts_alllandcover2016}
  if ((year == "2018")|(year == "2019")|(year == "2020")|(year == "2021")|(year == "2022")) {pts_allLULC = pts_alllandcover2019}
  pts_allLULC[pts_allLULC == 21] <- 71
  pts_allLULC[pts_allLULC == 22] <- 31
  pts_allLULC[pts_allLULC == 23] <- 31
  pts_allLULC[pts_allLULC == 24] <- 31
  pts_allLULC[pts_allLULC == 90] <- 43
  pts_allLULC[pts_allLULC == 95] <- 71
  pts_alllandcover[pts_allLULC==31] <- "Barren"
  pts_alllandcover[pts_allLULC==41|pts_allLULC==42|pts_allLULC==43] <- "Forest"
  pts_alllandcover[pts_allLULC==52] <- "Shrub"
  pts_alllandcover[pts_allLULC==71|pts_allLULC==72] <- "Grassland"
  pts_alllandcover[pts_allLULC==81] <- "Pasture"
  pts_alllandcover[pts_allLULC==82] <- "Crop"
  pts_allSM[pts_allSM > 60] <- 60
  pts_allBD <- pts_allBD/1000
  # pts_all <- pts_all[,c("x","y","landcover","EL","SL","AS","TWI","mcurv","hcurv","vcurv","surfrough",
  #                       "Clay","Sand","BD","SOC","SM","LST","GPP","EVI","NDWI","TREE","AFGC","PFGC","SHR","BG","LTR","ppt","Tavg","VPD","Depth")] # Regional model
  pts_all <- pts_all[,c("x","y","landcover","EL","SL","AS","TWI","mcurv","hcurv","vcurv","surfrough",
                        "Clay","Sand","BD","SOC","SM","LST","GPP","EVI","NDWI","Tree","ppt","Tavg","VPD","Depth")]
  pts_all <- na.omit(pts_all)
  pts_alllandcover <- as.factor(pts_alllandcover)
  levels(pts_alllandcover) <- c("Barren","Crop","Forest","Shrub","Grassland","Pasture")
  # Define model covariates
  # index_full <-names(pts_all) %in%  c("landcover", 
  #                                     "EL", "SL", "AS", "TWI", "mcurv", "hcurv", "vcurv", "surfrough",
  #                                     "Clay", "Sand", "BD", "SOC", 
  #                                     "SM",
  #                                     "LST", 
  #                                     "GPP", "EVI", "NDWI", 
  #                                     "TREE", "AFGC", "PFGC", "SHR", "BG", "LTR",
  #                                     "ppt", "Tavg", "VPD",
  #                                     "Depth") # Regional model
  index_full <-names(pts_all) %in%  c("landcover", 
                                      "EL", "SL", "AS", "TWI", "mcurv", "hcurv", "vcurv", "surfrough",
                                      "Clay", "Sand", "BD", "SOC", 
                                      "SM",
                                      "LST", 
                                      "GPP", "EVI", "NDWI", "Tree",
                                      "ppt", "Tavg", "VPD",
                                      "Depth")
  # Model prediction
  soil_pred_mean = predict(model_full, newdata = pts_all[, index_full], what = mean) 
  result <- cbind(pts_all[,c("x","y")],soil_pred_mean)
  filedate <- gsub("\\.","_",substr(names(cov.raster)[1],5,14))
  mois_map30 <- rasterize(result[, c('x', 'y')], x, result[, 'soil_pred_mean'], fun=mean, na.rm = TRUE)
  # Define pathway for eports; make sure to create the path in the Cloud Bucket first
  filepath <- paste0("./",ROI,"/",depth[[1]],"/SM_",depth[[1]],"_",filedate[[1]]) 
  writeRaster(mois_map30, filepath,format="GTiff",overwrite=TRUE)
  rm("cov.raster","nldas.raster","mois_map30","nldasValue","pts_all","pts_cov","rasValue","result","index_full","soil_pred_mean")
}



REQUIREMENTS :
This section offers an extensive presentation summary and the experimental data of our proposed method. 
In this work, we trained Super Generative Adversarial Networks (SGANs) for soil sample identification using version 6.1.1 of the Deep Learning GPU Training System (DIGITS), created by NVIDIA Corporation, based in Santa Clara, California. 
A 64-bit Ubuntu 16.04 workstation with an NVIDIA GeForce GTX 1660 Ti GPU was used for the experimentation. 
DIGITS facilitated the import of the image database after the image segmentation stage. 
We then utilized the Caffe backend of DIGITS to manage the training of the soil identification models. 




CITATION :
https://www.kaggle.com/datasets/amirmohammdjalili/soil-moisture-dataset




