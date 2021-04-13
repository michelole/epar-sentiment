from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
from lime.lime_text import LimeTextExplainer


def plot_confusion_matrix(y, y_pred, title):
    conf_matrix = confusion_matrix(y, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='OrRd')
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


def explain(clf, X_train, y, instance, name, method):
    clf.fit(X_train, y)
    explainer = LimeTextExplainer(class_names=[-1, 0, 1])
    exp = explainer.explain_instance(instance, method, top_labels=1, num_features=10)
    exp.show_in_notebook()
    exp.save_to_file(f"../{name}_explanation.html")
