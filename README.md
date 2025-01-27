# varroa-ai

This project aims to train a model to detect varrao mites on bee images

## Training the model
```
python src/varroa_classificator_train --train_dir train/ --validation_dir val/
```

```
python src/varroa_detector_train --train_dir train/videos/ --labels_dir train/labels/
```

### Understanding the Output

    - bbox_loss: This measures how well the model predicts the bounding boxes around the objects in the image. Lower values generally indicate better performance, as it means the predicted boxes are closer to the actual object locations.
    - classification_accuracy: This metric shows the percentage of correctly classified objects within the predicted bounding boxes. Higher values are better, indicating that the model is accurately identifying the objects.
    - classification_loss: This represents the loss associated with the object classification task. Similar to bbox_loss, lower values are better, indicating that the model is making more accurate classifications.
    - loss: This is the overall loss function, which typically combines bbox_loss and classification_loss (and potentially other losses). A decreasing loss value generally indicates that the model is improving.
    - val_bbox_loss, val_classification_accuracy, val_classification_loss, val_loss: These are the same metrics as above, but calculated on a separate "validation" dataset that the model has not been trained on. These metrics provide an unbiased estimate of the model's performance on unseen data and are crucial for preventing overfitting.
    - learning_rate: This value controls how much the model's weights are adjusted during each training step. A well-chosen learning rate is important for efficient training.

### Interpreting Progress

    Decreasing Loss:
        loss: You generally want to see the loss decreasing over time. This indicates that the model is learning and improving its predictions.
        val_loss: The val_loss should also ideally decrease, but it's more important that it doesn't start increasing significantly after a certain point. An increasing val_loss while the loss continues to decrease is a strong sign of overfitting.

    Increasing Accuracy:
        classification_accuracy: You expect to see this metric increasing over time, indicating that the model is getting better at correctly identifying objects.
        val_classification_accuracy: Similar to val_loss, a significant drop in val_classification_accuracy while classification_accuracy continues to increase suggests overfitting.

    Monitoring bbox_loss and val_bbox_loss:
        Observe the trends of these metrics. Decreasing values generally indicate that the model is improving its ability to localize objects accurately.

### Important Considerations

    Overfitting: Closely monitor the validation metrics (val_loss, val_classification_accuracy, val_bbox_loss) to detect overfitting. If these metrics start to degrade while the training metrics continue to improve, the model is likely overfitting the training data and will not generalize well to new, unseen data.
    Early Stopping: Implement early stopping to prevent overfitting. This technique stops the training process when the val_loss starts to increase, even if the loss is still decreasing.
    Learning Rate: Experiment with different learning rates to find the optimal value. A learning rate that is too high can cause the training to diverge, while a learning rate that is too low can make the training process very slow.

By carefully monitoring these metrics and using techniques like early stopping, you can effectively train your object detection model and achieve the best possible performance.