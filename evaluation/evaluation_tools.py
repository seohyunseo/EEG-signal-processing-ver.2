from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pylab as plt

def plot_confusion_matrix(y_test, y_pred, labels=None):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()

def evaluate_with_tsne(X, y, feature_name=None):

    print("[Result of t-SNE]")
    y = y.reshape(-1)
    X_embedded = TSNE(n_components=2, n_iter=1000, random_state=42).fit_transform(X)
    plt.title(f't-SNE Result: {feature_name}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.clf()
    plt.scatter(X_embedded[:, 0],X_embedded[:, 1], 15, y)
    plt.show()

def evaluate_with_randomforest(X, y, feature_name=None, show=True):
    y = y.reshape([y.shape[0]])
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train the classifier
    clf.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    # Evaluate the model

    test_accuracy = accuracy_score(y_test, y_pred)
    
    if show:
        print("[Result of Random Forest]")
        print(f"{feature_name} Test Accuracy: {test_accuracy:.2f}")
        plot_confusion_matrix(y_test, y_pred, clf.classes_)
    else:
        return test_accuracy
    
def evaluate_feature(X, y, feature_name=None, show=True):
    
    evaluate_with_tsne(X, y, feature_name=feature_name)

    evaluate_with_randomforest(X, y, feature_name=feature_name, show=show)