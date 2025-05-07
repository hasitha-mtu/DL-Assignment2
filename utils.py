import matplotlib.pyplot as plt
import os

def plot_accuracy(model_result, file_name = "model_result", save=False):
  accuracy = model_result.history["accuracy"]
  val_accuracy = model_result.history["val_accuracy"]
  loss = model_result.history["loss"]
  val_loss = model_result.history["val_loss"]
  epochs = range(1, len(accuracy) + 1)

  fig, ax1 = plt.subplots(figsize=(10, 6))

  ax1.set_xlabel("Epochs")
  ax1.set_ylabel("Accuracy", color="tab:blue")
  ax1.plot(epochs, accuracy, label="Training Accuracy", color="tab:blue", linestyle="-")
  ax1.plot(epochs, val_accuracy, label="Validation Accuracy", color="tab:blue", linestyle="--")
  ax1.tick_params(axis="y", labelcolor="tab:blue")

  ax2 = ax1.twinx()
  ax2.set_ylabel("Loss", color="tab:red")
  ax2.plot(epochs, loss, label="Training Loss", color="tab:red", linestyle="-")
  ax2.plot(epochs, val_loss, label="Validation Loss", color="tab:red", linestyle="--")
  ax2.tick_params(axis="y", labelcolor="tab:red")


  lines_1, labels_1 = ax1.get_legend_handles_labels()
  lines_2, labels_2 = ax2.get_legend_handles_labels()
  ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

  plt.title("Training and Validation Accuracy & Loss")
  plt.grid(True)
  plt.tight_layout()
  if save:
    file_path = f"output/{file_name}.png"
    plt.savefig(file_path)
  else:
    plt.show()

