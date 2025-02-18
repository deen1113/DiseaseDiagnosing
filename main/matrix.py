from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import preprocessing

'''

'''

def confMatrix(model):
    # get predictions
    y_pred = model.predict(preprocessing.test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)  # converts probabilities to class labels 

    # get true labels
    y_true = preprocessing.test_generator.classes

    # compute matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    print(cm)

    # visualize matrix
    class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()