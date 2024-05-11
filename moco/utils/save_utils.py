# -*- coding: utf-8 -*-
import os 
import torch
import json
import csv


def save_metadata(metadata: dict):
    file_path = metadata.get('experiment_path')
    if file_path is None:
        raise KeyError('metadata не содержит путь для сохранения <experiment_path>')
    
    # Создаем каталог, если он не существует
    os.makedirs(file_path, exist_ok=True)
    
    file_path = os.path.join(file_path, 'metadata.json')
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)


def save_loss(file_path, loss, batch_number, time_cutoff):
    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Если файл пустой, записываем заголовки
        if csv_file.tell() == 0:
            writer.writerow(['Epoch', 'Loss', 'Time_cutoff'])

        writer.writerow([batch_number, round(float(loss), 4), int(time_cutoff)])


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir='./checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filepath = os.path.join(checkpoint_dir, 'model_ep%04d.pt' % epoch)
    
    torch.save(
        {'epoch': epoch,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         'loss': loss
        },
        checkpoint_filepath
    )
