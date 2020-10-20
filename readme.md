https://github.com/nyok0/a1-472-2020-fall

Victor Soledad
#21297627

Datasets need to be placed in the /dataset/ folder

Each file needs to load a functions file called a1_functions.py located at the root of the folder

Each model runs on a separate file and will output two files,
one with the csv of the predicted values
and one with the Classification report and the Confusion matrix


GNB

py a1_dataset1_gnb.py

py a1_dataset2_gnb.py


Base DT

py a1_dataset1_base_dt.py

py a1_dataset2_base_dt.py


Best DT

py a1_dataset1_best_dt.py

py a1_dataset2_best_dt.py


PER

py a1_dataset1_per.py

py a1_dataset2_per.py


Base MLP

py a1_dataset1_base_mlp.py

py a1_dataset2_base_mlp.py


Best MLP

py a1_dataset1_best_mlp.py

py a1_dataset2_best_mlp.py





Each model follows closely the tutorial for recognizing hand written digits at scikit-learn.org

https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html