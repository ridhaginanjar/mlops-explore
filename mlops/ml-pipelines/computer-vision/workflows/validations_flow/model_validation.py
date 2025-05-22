import seaborn as sns
import matplotlib.pyplot as plt
import mlflow

from prefect import task
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


@task(name='classification report')
def classif_report(y_true, y_pred, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

    NORMAL_class = report['NORMAL']
    PNEUMONIA_class = report['PNEUMONIA']
    prediction_accuracy = report['accuracy']

    return {
        "NORMAL_report": NORMAL_class,
        "PNEUMONIA_report": PNEUMONIA_class,
        "predict_acc":prediction_accuracy
    }


def visualize_conf_mat(cm, test_gen):
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_gen.class_indices.keys(), yticklabels=test_gen.class_indices.keys(), cmap='Blues')

    plt.title("Confussion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    filename = "confussion_matrix.png"
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close()



@task(name='confusion_matrix')
def conf_matrix(y_true, y_pred, test_gen):
    cm = confusion_matrix(y_true, y_pred)
    TN,FP,FN,TP = cm.ravel()

    # Visualize data
    visualize_conf_mat(cm, test_gen)

    return {
        'true-negatif': TN,
        'false-positif': FP, 
        'false-negatif': FN,
        'true-positif': TP 
    }

@task(name='generate-ROC-AUC')
def roc_auc_generator(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    return {
        'false-positif-rate':fpr,
        'true-positif-rate': tpr,
        'area-under-the-curve': auc
    }
