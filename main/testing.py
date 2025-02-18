import preprocessing

'''
Here, the model uses the test generator to run its predictions against the test images.
The function is called by the model, which prints three informative percentages:
Test Accuracy: how many samples are correctly identified by the model.
Test Precision: how many samples are correctly identified as positive out of all positive predictions.
Test Recall: how many samples are correctly identified out of all samples identified the same.
'''

def testModel(model):
    test_loss, test_acc, test_precision, test_recall = model.evaluate(preprocessing.test_generator)
    print(f"Test Accuracy: {test_acc:.2f}")
    print(f"Test Precision: {test_precision:.2f}")
    print(f"Test Recall: {test_recall:.2f}")