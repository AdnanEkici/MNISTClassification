import numpy as np
import cv2
import mnist_classifier_app.utils as utils
import torch
import pandas as pd
from torchvision import transforms
import argparse
import yaml

def predict(image , device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    image = transform(image).float().to(device).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    prediction = output.argmax().item()
    return prediction

def inference(dataframe):
    for index, row in dataframe.iterrows():
        label = row["label"]
        pixel_values = row.drop("label").values.astype(float)
        image = np.reshape(pixel_values, (28, 28))
        predicted_class = predict(image=image)
        img_with_padding = cv2.copyMakeBorder(image.squeeze(), 28, 28, 28,28, cv2.BORDER_CONSTANT, value=0)
        img_display = cv2.resize(img_with_padding, (200, 200))  # Resize for display
        cv2.putText(img_display, f"Actual: {label}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img_display, f"Predicted: {predicted_class}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1)
        cv2.imshow('MNIST Image', img_display)
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path of th config file." , default="F:\Git_Repos\CURRENT_MNIST\configuration_files\configuration_file.yml")
    args = parser.parse_args()
    config_file = args.config

    with open(config_file, "r") as config:
        config_data = yaml.safe_load(config)
        try:
            inference_config = config_data.get("inference")
        except yaml.YAMLError as exc:
            utils.get_logger().critical(exc)

    device = inference_config["device"]
    test_dataset_path = inference_config["test_dataset_path"]
    model_path = inference_config["model_path"]

    model = utils.get_model(model_path=model_path)
    dataframe = pd.read_csv(test_dataset_path)
    inference(dataframe=dataframe)