"""
Training Script
Train YOLOv11, YOLO-NAS, or RT-DETR on custom dataset
"""

import argparse
from pathlib import Path
from loguru import logger
import yaml


def train_yolov11(data_yaml: str, config: dict):
    """Train YOLOv11 model"""
    from ultralytics import YOLO
    
    model_config = config['model']['yolov11']
    train_config = config.get('training', {})
    
    # Initialize model
    variant = model_config['variant']
    model = YOLO(f"{variant}.pt")
    
    logger.info(f"Training {variant} on {data_yaml}")
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=train_config.get('epochs', 300),
        imgsz=train_config.get('img_size', 640),
        batch=train_config.get('batch_size', 16),
        lr0=train_config.get('learning_rate', 0.01),
        optimizer=train_config.get('optimizer', 'SGD'),
        device=config['performance']['device'],
        patience=50,
        save=True,
        project='runs/train',
        name=f'yolov11_{variant}',
        exist_ok=True,
        pretrained=True,
        verbose=True
    )
    
    logger.info(f"Training complete! Best model: {results.save_dir}/weights/best.pt")
    return results


def train_yolo_nas(data_yaml: str, config: dict):
    """Train YOLO-NAS model"""
    from super_gradients.training import Trainer, models
    from super_gradients.training.dataloaders.dataloaders import (
        coco_detection_yolo_format_train,
        coco_detection_yolo_format_val
    )
    from super_gradients.training.losses import PPYoloELoss
    from super_gradients.training.metrics import DetectionMetrics_050
    
    model_config = config['model']['yolo_nas']
    train_config = config.get('training', {})
    
    # Load data config
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Setup dataloaders
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': Path(data_yaml).parent,
            'images_dir': 'train/images',
            'labels_dir': 'train/labels',
            'classes': data_config['names']
        },
        dataloader_params={
            'batch_size': train_config.get('batch_size', 16),
            'num_workers': 4
        }
    )
    
    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': Path(data_yaml).parent,
            'images_dir': 'valid/images',
            'labels_dir': 'valid/labels',
            'classes': data_config['names']
        },
        dataloader_params={
            'batch_size': train_config.get('batch_size', 16),
            'num_workers': 4
        }
    )
    
    # Initialize model
    variant = model_config['variant']
    model = models.get(variant, num_classes=len(data_config['names']))
    
    # Setup trainer
    trainer = Trainer(experiment_name='yolo_nas_training', ckpt_root_dir='runs/train')
    
    # Training parameters
    train_params = {
        'silent_mode': False,
        'max_epochs': train_config.get('epochs', 300),
        'lr_mode': 'cosine',
        'initial_lr': train_config.get('learning_rate', 0.01),
        'optimizer': train_config.get('optimizer', 'SGD'),
        'loss': PPYoloELoss(),
        'valid_metrics_list': [DetectionMetrics_050(
            score_thres=0.1,
            num_cls=len(data_config['names']),
            post_prediction_callback=None
        )],
        'metric_to_watch': 'mAP@0.50'
    }
    
    # Train
    logger.info(f"Training {variant}")
    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_data,
        valid_loader=val_data
    )
    
    logger.info("Training complete!")


def train_rtdetr(data_yaml: str, config: dict):
    """Train RT-DETR model"""
    from ultralytics import RTDETR
    
    model_config = config['model']['rt_detr']
    train_config = config.get('training', {})
    
    # Initialize model
    variant = model_config['variant']
    model = RTDETR(f"{variant}.pt")
    
    logger.info(f"Training {variant} on {data_yaml}")
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=train_config.get('epochs', 100),
        imgsz=train_config.get('img_size', 640),
        batch=train_config.get('batch_size', 16),
        lr0=train_config.get('learning_rate', 0.01),
        device=config['performance']['device'],
        patience=50,
        save=True,
        project='runs/train',
        name=f'rtdetr_{variant}',
        exist_ok=True,
        verbose=True
    )
    
    logger.info(f"Training complete! Best model: {results.save_dir}/weights/best.pt")
    return results


def main():
    parser = argparse.ArgumentParser(description='Train detection model')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['yolov11', 'yolo-nas', 'rt-detr'],
                       help='Model type to train')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data.yaml')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config.yaml')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train model
    if args.model == 'yolov11':
        train_yolov11(args.data, config)
    elif args.model == 'yolo-nas':
        train_yolo_nas(args.data, config)
    elif args.model == 'rt-detr':
        train_rtdetr(args.data, config)


if __name__ == '__main__':
    main()
