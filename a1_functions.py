import csv
import sys
import numpy
from sklearn import datasets, svm, metrics

def a1_init():
    nimages = 15
    return nimages

def csvParser(trainfile):
    csvdata = []
    csvdataParsed = []
    csvresult = []
    with open(trainfile, newline='') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for row in csvreader:
            xrow = []
            xresult = 0
            for i, rcol in enumerate(row):
                try:
                    if(i < 1024):
                        val = int(rcol)
                        xrow.append(val)
                    else:
                        val = int(rcol)
                        xresult = val
                except:
                    if(i < 1024):
                        xrow.append(0)
                    else:
                        xresult.append(0)
            # print(xrow)
            csvdata.append(xrow)
            csvresult.append(xresult)
        # print(xresult)

    # print(csvdata)

    for xcol in csvdata:
        oDataRow = []
        oData = []
        # cind = 0
        i = 0
        for xval in xcol:
            # nrow = int(i / 32)
            if(i >= 32):
                if len(oData) != 0:
                    oDataRow.append(oData)
                i = 0
                oData = []
            # cind = nrow
            oData.append(xval)
            i = i + 1
        
        oDataRow.append(oData)

        # print("oDataRow")
        # print(oDataRow)
        csvdataParsed.append(oDataRow)


    csvdataParsed = numpy.array(csvdataParsed)

    return csvdataParsed, csvresult


def csvsave(data, outfile):
    with open(outfile, mode='w', newline='', encoding='utf-8') as outcsv:
        csvwriter = csv.writer(outcsv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for i, drow in enumerate(data):
            drow = [drow]
            drow.insert(0, i)
            csvwriter.writerow(drow)

def showTraining(dataTrain, dataTrainResult, axes, nimages, letters, plt): 
    images_and_labels = list(zip(dataTrain, dataTrainResult))
    for ax, (image, label) in zip(axes[0, :], images_and_labels[:nimages]):
        # print(label)
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        xlabel = letters[label]
        ax.set_title('Tra: %s' % xlabel)

def showPrediction(dataTest, predicted, axes, nimages, letters, plt):
    images_and_predictions = list(zip( dataTest , predicted))
    for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:nimages]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        
        xlabel = letters[prediction]
        ax.set_title('Pre: %s' % xlabel)

# CONFUSION MATRIX
def reportResults(classifier, X_test, y_test, predicted, fileout):

    original_stdout = sys.stdout
    with open(fileout, 'w') as f:
        sys.stdout = f

        print("Classification report for classifier %s:\n%s\n" 
            % (classifier, metrics.classification_report(y_test, predicted)))
        disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
        disp.figure_.suptitle("Confusion Matrix")
        print("Confusion matrix:\n%s" % disp.confusion_matrix)
        
        sys.stdout = original_stdout
