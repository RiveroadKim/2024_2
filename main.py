import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse

from model import AcousticSceneClassifier
from dataloader import get_dataloader, TUTAcousticSceneDataset

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 하이퍼파라미터 설정
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DATA_DIR = "data"

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="훈련 중"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    average_loss = total_loss / len(train_loader)
    return average_loss, accuracy

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="평가 중"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    average_loss = total_loss / len(val_loader)
    return average_loss, accuracy

def main(mode):
    # 데이터 로드
    full_dataset = TUTAcousticSceneDataset(DATA_DIR, mode='train')
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(DATA_DIR, mode='test', batch_size=BATCH_SIZE, shuffle=False)

    # 모델 초기화
    model = AcousticSceneClassifier().to(device)

    if mode == 'train':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 학습
        best_val_accuracy = 0
        for epoch in range(EPOCHS):
            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            
            print(f"에포크 {epoch+1}/{EPOCHS}")
            print(f"훈련 손실: {train_loss:.4f}, 훈련 정확도: {train_accuracy:.2f}%")
            print(f"검증 손실: {val_loss:.4f}, 검증 정확도: {val_accuracy:.2f}%")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), 'best_model.pth')
                print("최고 모델 저장됨!")

    elif mode == 'test':
        # 테스트 데이터 추론
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()

        predictions = []
        with torch.no_grad():
            for inputs in tqdm(test_loader, desc="테스트 중"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())

        # 레이블 디코딩
        label_decoder = {v: k for k, v in full_dataset.label_encoder.items()}
        decoded_predictions = [label_decoder[pred] for pred in predictions]

        # 결과 저장
        results_df = pd.DataFrame({
            'Id': range(len(decoded_predictions)),
            'Scene_label': decoded_predictions
        })
        results_df.to_csv('y_pred.csv', index=False)
        print("예측 결과가 y_pred.csv 파일로 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="음향 장면 분류 모델 학습 및 추론")
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help="실행 모드 선택 (train 또는 test)")
    args = parser.parse_args()

    main(args.mode)
