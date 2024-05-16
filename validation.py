import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path
from detector import recognize_faces

def validate(model: str = "hog"):
    y_true = []
    y_pred = []

    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            true_label = filepath.parent.name  # Extract true label from directory name
            results = recognize_faces(image_location=str(filepath.absolute()), model=model)
            if results:
                for _, predicted_label in results:
                    y_true.append(true_label)
                    y_pred.append(predicted_label if predicted_label != "Unknown" else "Unknown")
            else:
                # No faces found in the image
                y_true.append(true_label)
                y_pred.append("No Face Found")

    # Compute confusion matrix and accuracy
    labels = sorted(set(y_true)) + ["Unknown", "No Face Found"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

validate()
