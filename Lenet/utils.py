import torch 
"""
    Precision = TP / (TP + FN)
    Recall = TP / (TP + FP)
"""


def precision_recall_f1(y_true, y_predict, num_classes):
    true_positives = torch.zeros(num_classes)
    false_positives = torch.zeros(num_classes)
    false_negatives = torch.zeros(num_classes)

    for idx in range(num_classes):
        true_positives[idx] = torch.sum((y_true == idx) & (y_predict == idx)).item()
        false_positives[idx] = torch.sum((y_true != idx) & (y_predict == idx)).item()
        false_negatives[idx] = torch.sum((y_true == idx) & (y_predict != idx)).item()
        
    precision = true_positives / (false_negatives + true_positives)
    recall = true_positives / (false_positives + true_positives)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1 

def confusion_matrix(y_true, y_pred, num_classes):
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    
    for t, p in zip(y_true, y_pred):
        matrix[t.long(), p.long()] += 1
    
    return matrix