from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 8, 7),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG("aircraft_detection_training_dag", default_args=default_args, schedule_interval="@daily", catchup=False) as dag:
    import torch
    import cv2
    import torchvision
    import mlflow
    import mlflow.pytorch

    import pandas as pd
    import numpy as np

    from torch.utils.data import DataLoader, Dataset

    from sklearn.model_selection import train_test_split

    from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

    from sklearn.metrics import average_precision_score

    # Disable beta transforms warning
    torchvision.disable_beta_transforms_warning()
    import torchvision.transforms.v2 as transforms

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    # Set experiment name
    mlflow.set_experiment("aircraft_detection")


    class AircraftDataset(Dataset):
        def __init__(self, annotations: pd.DataFrame, images_dir: str, transforms=None):
            super().__init__()

            self.image_ids = annotations["image_id"].unique()
            self.annotations = annotations
            self.images_dir = images_dir
            self.transforms = transforms

        def __getitem__(self, index: int):
            # Get image id and corresponding records
            image_id = self.image_ids[index]
            records = self.annotations[self.annotations["image_id"] == image_id]
            # Load and preprocess image
            image = cv2.imread(f"{self.images_dir}/{image_id}", cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(float)
            image /= 255.0
            # Extract bounding box information
            boxes = records[["x", "y", "w", "h"]].values
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Calculate area for each bounding box
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            area = torch.as_tensor(area, dtype=torch.float32)
            # Set labels for all instances
            labels = torch.ones((records.shape[0],), dtype=torch.int64)
            # Assume instances are not crowd
            iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
            # Prepare target dictionary
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd
            # Apply data transformations
            if self.transforms:
                image = self.transforms(image)

            return image, target

        def __len__(self) -> int:
            return len(self.image_ids)

    def get_transform():
        return transforms.Compose([
            transforms.ToImageTensor(), 
            transforms.ConvertImageDtype(torch.float32)
            ])

    def calculate_mAP(model, data_loader, device):
        model.eval()
        gt_labels = []
        pred_scores = []

        with torch.no_grad():
            for images, targets in data_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(images)
                
                for output in outputs:
                    scores = output["scores"].cpu().numpy()
                    labels = output["labels"].cpu().numpy()

                    gt_labels.append(labels)
                    pred_scores.append(scores)
                
                gt_labels = np.concatenate(gt_labels)

        pred_scores = np.concatenate(pred_scores)
        mAP = average_precision_score(gt_labels, pred_scores)

        return mAP

    # Define directories
    images_dir = "/home/airflow/data/images"
    annotations_dir = "/home/airflow/data/annotations.csv"
    # Load annotations
    annotations = pd.read_csv(annotations_dir, index_col=0, header=0)
    # Split dataframe into train and test
    train_ids, test_ids = train_test_split(annotations["image_id"].unique(), test_size=0.1)
    # Create datasets for training and test
    train_aircraft_dataset = AircraftDataset(
        annotations[annotations["image_id"].isin(train_ids)],
        images_dir,
        get_transform(),
    )
    test_aircraft_dataset = AircraftDataset(
        annotations[annotations["image_id"].isin(test_ids)], 
        images_dir, 
        get_transform(),
    )
    # Create data loaders
    train_dataloader = DataLoader(
        train_aircraft_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
    )
    test_dataloader = DataLoader(
        test_aircraft_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
    )
    # Define device
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    # Load model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    # Replace the classifier with a new one for the number of classes (aircraft + background)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )

    def train_task():
        # Define optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        # Define learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        # Move model to the right device
        model.to(device)
        # Define number of epochs
        num_epochs = 10
        # Create an MLflow experiment
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_param("batch_size", train_dataloader.batch_size)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("lr", optimizer.param_groups[0]['lr'])
            mlflow.log_param("momentum", optimizer.param_groups[0]['momentum'])
            mlflow.log_param("weight_decay", optimizer.param_groups[0]['weight_decay'])

            # Training loop
            for _ in range(num_epochs):
                total_loss = 0
                for images, targets in train_dataloader:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    model.train()
                    loss_dict = model(images, targets)

                    losses = sum(loss for loss in loss_dict.values())
                    loss_value = losses.item()

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    total_loss += loss_value

                # Calculate and log custom metrics
                average_loss = total_loss / len(train_dataloader)
                mAP = calculate_mAP(model, test_dataloader, device)

                mlflow.log_metric("total_loss", total_loss)
                mlflow.log_metric("average_loss", average_loss)
                mlflow.log_metric("mAP", mAP)

                # Log model artifacts
                mlflow.log_artifact("aircraft_detection_training_dag.py")

                # Update the learning rate
                if lr_scheduler is not None:
                    lr_scheduler.step()

    def evaluate_task():
        # Calculate mAP in test dataset
        mAP = calculate_mAP(model, test_dataloader, device)
        # If mAP is greater than 0.5, save the model
        if mAP > 0.5:
            # Log the model to MLFlow registry
            with mlflow.start_run() as run:
                mlflow.log_metric("mAP", mAP)
                mlflow.pytorch.log_model(
                    model, registered_model_name="fasterrcnn_resnet50_fpn"
                )
            print("Model saved to MLFlow registry.")
        else:
            print("mAP is not greater than 0.5. Model not saved.")

    train_operator = PythonOperator(
        task_id="train_task",
        python_callable=train_task,
        dag=dag,
    )

    evaluate_operator = PythonOperator(
        task_id="evaluate_task",
        python_callable=evaluate_task,
        dag=dag,
    )

    train_operator >> evaluate_operator