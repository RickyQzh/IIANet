#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预提取视频特征脚本
将所有npz文件通过预训练的视频编码器，生成对应的_embedding.npz文件
这样可以在训练过程中直接加载预提取的特征，避免重复的视频编码计算
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import yaml

# 添加项目根目录到路径（脚本在子目录中，需要找到上一级目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # 上一级目录是项目根目录
sys.path.insert(0, project_root)

import look2hear.videomodels
from look2hear.datas.transform import get_preprocessing_pipelines


def parse_args():
    parser = argparse.ArgumentParser(description='预提取视频编码特征')
    parser.add_argument(
        '--conf_dir',
        type=str,
        default='configs/LRS2-IIANet.yml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--data_split',
        type=str,
        default='all',
        choices=['train', 'val', 'test', 'all'],
        help='要处理的数据集分割 (train/val/test/all)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='批处理大小'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='使用的设备'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='是否覆盖已存在的embedding文件'
    )
    return parser.parse_args()


def load_video_model(config, device):
    """加载预训练的视频模型"""
    print(f"正在加载视频模型: {config['videonet']['videonet_name']}")
    
    # 处理预训练权重路径
    video_config = config['videonet']['videonet_config'].copy()
    if 'pretrain' in video_config and video_config['pretrain']:
        pretrain_path = video_config['pretrain']
        if not os.path.isabs(pretrain_path):
            pretrain_path = os.path.join(project_root, pretrain_path)
        video_config['pretrain'] = pretrain_path
        print(f"预训练权重路径: {pretrain_path}")
    
    video_model = getattr(
        look2hear.videomodels,
        config['videonet']['videonet_name']
    )(**video_config)
    
    video_model = video_model.to(device)
    video_model.eval()  # 设置为评估模式
    
    print(f"视频模型已加载到设备: {device}")
    return video_model


def collect_npz_files(json_dir):
    """从json文件中收集所有的npz文件路径"""
    npz_files = set()
    
    # 读取s1.json和s2.json
    for source_name in ['s1', 's2']:
        json_path = os.path.join(json_dir, f'{source_name}.json')
        if not os.path.exists(json_path):
            print(f"警告: {json_path} 不存在")
            continue
            
        with open(json_path, 'r') as f:
            source_infos = json.load(f)
        
        # 每个source_info是一个列表 [audio_path, npz_path]
        for info in source_infos:
            if len(info) >= 2:
                npz_path = info[1]
                if npz_path.endswith('.npz'):
                    npz_files.add(npz_path)
    
    return list(npz_files)


def process_single_npz(npz_path, video_model, preprocessing_func, device, overwrite=False):
    """处理单个npz文件，生成embedding"""
    # 生成输出路径
    output_path = npz_path.replace('.npz', '_embedding.npz')
    
    # 检查是否已存在
    if os.path.exists(output_path) and not overwrite:
        return output_path, 'skipped'
    
    try:
        # 加载原始数据
        data = np.load(npz_path)
        video_frames = data['data']  # Shape: (T, H, W)
        
        # 应用预处理
        video_frames_processed = preprocessing_func(video_frames)  # (T, H, W)
        
        # 转换为tensor并添加batch维度: (1, 1, T, H, W)
        video_tensor = torch.from_numpy(video_frames_processed).unsqueeze(0).unsqueeze(0)
        video_tensor = video_tensor.to(device).float()
        
        # 通过视频模型提取特征
        with torch.no_grad():
            embedding = video_model(video_tensor)  # (1, C, T')
        
        # 转换回numpy并移除batch维度
        embedding_np = embedding.squeeze(0).cpu().numpy()  # (C, T')
        
        # 保存embedding
        np.savez_compressed(output_path, embedding=embedding_np)
        
        return output_path, 'success'
        
    except Exception as e:
        print(f"\n处理 {npz_path} 时出错: {str(e)}")
        return output_path, f'error: {str(e)}'


def process_batch_npz(npz_paths, video_model, preprocessing_func, device, max_frames=50):
    """批量处理npz文件"""
    batch_data = []
    valid_paths = []
    
    for npz_path in npz_paths:
        try:
            data = np.load(npz_path)
            video_frames = data['data']
            
            # 限制帧数到max_frames
            if video_frames.shape[0] > max_frames:
                video_frames = video_frames[:max_frames]
            
            video_frames_processed = preprocessing_func(video_frames)
            batch_data.append(torch.from_numpy(video_frames_processed))
            valid_paths.append(npz_path)
        except Exception as e:
            print(f"\n加载 {npz_path} 时出错: {str(e)}")
            continue
    
    if not batch_data:
        return []
    
    # 填充到相同长度
    max_len = max(x.shape[0] for x in batch_data)
    padded_batch = []
    for data in batch_data:
        if data.shape[0] < max_len:
            pad = torch.zeros(max_len - data.shape[0], *data.shape[1:])
            data = torch.cat([data, pad], dim=0)
        padded_batch.append(data)
    
    # 堆叠成批次: (B, T, H, W) -> (B, 1, T, H, W)
    batch_tensor = torch.stack(padded_batch).unsqueeze(1).to(device).float()
    
    # 提取特征
    with torch.no_grad():
        embeddings = video_model(batch_tensor)  # (B, C, T')
    
    # 保存每个embedding
    results = []
    for i, npz_path in enumerate(valid_paths):
        output_path = npz_path.replace('.npz', '_embedding.npz')
        try:
            embedding_np = embeddings[i].cpu().numpy()  # (C, T')
            np.savez_compressed(output_path, embedding=embedding_np)
            results.append((output_path, 'success'))
        except Exception as e:
            results.append((output_path, f'error: {str(e)}'))
    
    return results


def main():
    args = parse_args()
    
    # 处理配置文件路径（如果是相对路径，相对于项目根目录）
    conf_path = args.conf_dir
    if not os.path.isabs(conf_path):
        conf_path = os.path.join(project_root, conf_path)
    
    # 加载配置
    print(f"正在加载配置文件: {conf_path}")
    with open(conf_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 确定要处理的数据集（处理相对路径）
    data_dirs = {}
    if args.data_split in ['train', 'all']:
        train_dir = config['datamodule']['data_config']['train_dir']
        if not os.path.isabs(train_dir):
            train_dir = os.path.join(project_root, train_dir)
        data_dirs['train'] = train_dir
    if args.data_split in ['val', 'all']:
        val_dir = config['datamodule']['data_config']['valid_dir']
        if not os.path.isabs(val_dir):
            val_dir = os.path.join(project_root, val_dir)
        data_dirs['val'] = val_dir
    if args.data_split in ['test', 'all']:
        test_dir = config['datamodule']['data_config']['test_dir']
        if not os.path.isabs(test_dir):
            test_dir = os.path.join(project_root, test_dir)
        data_dirs['test'] = test_dir
    
    # 加载视频模型
    device = torch.device(args.device)
    video_model = load_video_model(config, device)
    
    # 使用验证模式的预处理（不包含随机性）
    preprocessing_func = get_preprocessing_pipelines()['val']
    
    print("\n" + "="*80)
    print("开始预提取视频特征")
    print("="*80)
    
    # 处理每个数据集分割
    for split_name, data_dir in data_dirs.items():
        print(f"\n处理数据集: {split_name}")
        print(f"数据目录: {data_dir}")
        
        # 收集所有npz文件
        npz_files = collect_npz_files(data_dir)
        
        if not npz_files:
            print(f"警告: 在 {data_dir} 中没有找到npz文件")
            continue
        
        print(f"找到 {len(npz_files)} 个npz文件")
        
        # 过滤已存在的文件
        if not args.overwrite:
            npz_files_to_process = []
            for npz_path in npz_files:
                output_path = npz_path.replace('.npz', '_embedding.npz')
                if not os.path.exists(output_path):
                    npz_files_to_process.append(npz_path)
            
            print(f"需要处理 {len(npz_files_to_process)} 个文件 (跳过 {len(npz_files) - len(npz_files_to_process)} 个已存在的文件)")
            npz_files = npz_files_to_process
        
        if not npz_files:
            print(f"没有需要处理的文件")
            continue
        
        # 处理文件
        success_count = 0
        error_count = 0
        
        with tqdm(total=len(npz_files), desc=f"处理 {split_name}") as pbar:
            for npz_path in npz_files:
                output_path, status = process_single_npz(
                    npz_path, video_model, preprocessing_func, device, args.overwrite
                )
                
                if status == 'success':
                    success_count += 1
                elif status.startswith('error'):
                    error_count += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'success': success_count,
                    'error': error_count
                })
        
        print(f"\n{split_name} 完成:")
        print(f"  - 成功: {success_count}")
        print(f"  - 错误: {error_count}")
    
    print("\n" + "="*80)
    print("预提取完成！")
    print("="*80)


if __name__ == '__main__':
    main()

