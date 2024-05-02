import yaml

import torch 
from sklearn.datasets import load_iris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import mlflow 
import mlflow.pytorch

# datetime 
from datetime import datetime
now = datetime.now()


def load_iris_data(test_size):
    iris = load_iris()
    X = iris.data
    y = iris.target

    # normalize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # convert to torch tensors
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    return X_train, X_test, y_train, y_test

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)
    

def validate(model, X_test, y_test, criterion):
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        outputs = model(X_test)
        loss = criterion(outputs, y_test)
        accuracy = accuracy_score(y_test.cpu(), outputs.argmax(1).cpu())
    return loss, accuracy


def train_model(X_train, y_train, X_test, y_test, params):
    epochs = params["epochs"]
    lr = params["learning-rate"]
    batch_size = params["batch-size"]
    validation_steps = params["validation-steps"]
    input_dim = X_train.shape[1]
    output_dim = len(torch.unique(y_train))

    model = Model(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size)

    with mlflow.start_run() as run:
        mlflow.set_tag("mlflow.runName", f"iris-classification-pytorch_{now.strftime('%Y-%m-%d_%H-%M-%S')}")
        for epoch in range(epochs):
            total_loss = 0
            total_accuracy = 0.0
            for i, (inputs, targets) in enumerate(train_dl):
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_accuracy += accuracy_score(targets.cpu(), outputs.argmax(1).cpu())

                # validate
                if i % validation_steps == 0:
                    val_loss, val_accuracy = validate(model, X_test, y_test, criterion)
                    mlflow.log_metric("val_loss", val_loss)
                    mlflow.log_metric("val_accuracy", val_accuracy)

            mlflow.log_metric("train_loss", total_loss / len(train_dl))
            mlflow.log_metric("train_accuracy", total_accuracy / len(train_dl))
        
        # log model
        mlflow.pytorch.log_model(model, "model")
        
        # log params
        mlflow.log_params(params)

        # log model code
        mlflow.log_artifact("iris-training-params.yaml")
        mlflow.log_artifact("ml-flow-pytorch.py")

        print("Run ID: {}".format(run.info.run_uuid))
        print("Artifact URI: {}".format(run.info.artifact_uri))
        print("Training completed successfully!")



if __name__ == "__main__":
    # load yaml file
    with open("iris-training-params.yaml") as f:
        params = yaml.safe_load(f)   


    # load data
    X_train, X_test, y_train, y_test = load_iris_data(params["test-size"])

    # print data shape
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    mlflow.set_experiment("iris_classification_pytorch")

    # train modelt
    train_model(X_train, y_train, X_test, y_test, params)



    