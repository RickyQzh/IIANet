#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试预提取视频特征功能
"""

import os
import sys
import yaml
import numpy as np
import torch

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def test_imports():
    """测试导入"""
    print("测试1: 检查模块导入...")
    try:
        import look2hear.videomodels
        import look2hear.datas
        from look2hear.datas.transform import get_preprocessing_pipelines
        print("✓ 模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False


def test_video_model_loading():
    """测试视频模型加载"""
    print("\n测试2: 检查视频模型加载...")
    try:
        import look2hear.videomodels
        
        config_path = os.path.join(project_root, "configs/LRS2-IIANet.yml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        video_model = getattr(
            look2hear.videomodels,
            config['videonet']['videonet_name']
        )(**config['videonet']['videonet_config'])
        
        video_model.eval()
        print(f"✓ 视频模型加载成功: {config['videonet']['videonet_name']}")
        print(f"  - 预训练权重: {config['videonet']['videonet_config']['pretrain']}")
        return True
    except Exception as e:
        print(f"✗ 视频模型加载失败: {e}")
        return False


def test_preprocessing():
    """测试预处理函数"""
    print("\n测试3: 检查预处理函数...")
    try:
        from look2hear.datas.transform import get_preprocessing_pipelines
        
        preprocessing = get_preprocessing_pipelines()
        train_func = preprocessing['train']
        val_func = preprocessing['val']
        
        print("✓ 预处理函数获取成功")
        print(f"  - 训练模式: {train_func}")
        print(f"  - 验证模式: {val_func}")
        return True
    except Exception as e:
        print(f"✗ 预处理函数获取失败: {e}")
        return False


def test_dataset_with_precompute():
    """测试数据集是否支持use_precomputed_embeddings参数"""
    print("\n测试4: 检查数据集参数...")
    try:
        import look2hear.datas
        from inspect import signature
        
        # 检查AVSpeechDataset
        dataset_class = look2hear.datas.AVSpeechDataset
        sig = signature(dataset_class.__init__)
        params = list(sig.parameters.keys())
        
        if 'use_precomputed_embeddings' in params:
            print("✓ AVSpeechDataset 支持 use_precomputed_embeddings 参数")
        else:
            print("✗ AVSpeechDataset 不支持 use_precomputed_embeddings 参数")
            return False
        
        # 检查AVSpeechDynamicDataset
        dataset_class = look2hear.datas.AVSpeechDynamicDataset
        sig = signature(dataset_class.__init__)
        params = list(sig.parameters.keys())
        
        if 'use_precomputed_embeddings' in params:
            print("✓ AVSpeechDynamicDataset 支持 use_precomputed_embeddings 参数")
        else:
            print("✗ AVSpeechDynamicDataset 不支持 use_precomputed_embeddings 参数")
            return False
        
        # 检查DataModule
        datamodule_class = look2hear.datas.AVSpeechDyanmicDataModule
        sig = signature(datamodule_class.__init__)
        params = list(sig.parameters.keys())
        
        if 'use_precomputed_embeddings' in params:
            print("✓ AVSpeechDyanmicDataModule 支持 use_precomputed_embeddings 参数")
        else:
            print("✗ AVSpeechDyanmicDataModule 不支持 use_precomputed_embeddings 参数")
            return False
        
        return True
    except Exception as e:
        print(f"✗ 数据集参数检查失败: {e}")
        return False


def test_config_file():
    """测试配置文件"""
    print("\n测试5: 检查配置文件...")
    try:
        config_path = os.path.join(project_root, "configs/LRS2-IIANet.yml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 检查是否有use_precomputed_embeddings参数
        if 'use_precomputed_embeddings' in config['datamodule']['data_config']:
            value = config['datamodule']['data_config']['use_precomputed_embeddings']
            print(f"✓ 配置文件包含 use_precomputed_embeddings 参数")
            print(f"  - 当前值: {value}")
        else:
            print("✗ 配置文件缺少 use_precomputed_embeddings 参数")
            return False
        
        return True
    except Exception as e:
        print(f"✗ 配置文件检查失败: {e}")
        return False


def test_npz_file_example():
    """测试能否找到npz文件示例"""
    print("\n测试6: 检查数据文件...")
    try:
        # 从配置文件读取数据路径
        config_path = os.path.join(project_root, "configs/LRS2-IIANet.yml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        train_dir = config['datamodule']['data_config']['train_dir']
        s1_json = os.path.join(train_dir, 's1.json')
        
        if os.path.exists(s1_json):
            import json
            with open(s1_json, 'r') as f:
                sources = json.load(f)
            
            if len(sources) > 0:
                example_npz = sources[0][1] if len(sources[0]) > 1 else None
                if example_npz and example_npz.endswith('.npz'):
                    print(f"✓ 找到npz文件示例")
                    print(f"  - 示例路径: {example_npz}")
                    
                    if os.path.exists(example_npz):
                        data = np.load(example_npz)
                        print(f"  - 数据键: {list(data.keys())}")
                        if 'data' in data:
                            print(f"  - 数据形状: {data['data'].shape}")
                    else:
                        print(f"  - 警告: 文件不存在")
                else:
                    print("✗ JSON中没有找到npz文件路径")
                    return False
            else:
                print("✗ JSON文件为空")
                return False
        else:
            print(f"✗ 找不到数据JSON文件: {s1_json}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ 数据文件检查失败: {e}")
        return False


def test_precompute_script():
    """测试预提取脚本是否存在"""
    print("\n测试7: 检查预提取脚本...")
    try:
        script_path = os.path.join(project_root, "video_embedding_precompute", "precompute_video_embeddings.py")
        if os.path.exists(script_path):
            print(f"✓ 预提取脚本存在: {script_path}")
            # 检查脚本是否可执行
            with open(script_path, 'r') as f:
                content = f.read()
                if 'def main()' in content:
                    print("  - 脚本包含main函数")
                if 'process_single_npz' in content:
                    print("  - 脚本包含处理函数")
            return True
        else:
            print(f"✗ 预提取脚本不存在: {script_path}")
            return False
    except Exception as e:
        print(f"✗ 预提取脚本检查失败: {e}")
        return False


def main():
    print("="*80)
    print("视频特征预提取功能测试")
    print("="*80)
    
    tests = [
        test_imports,
        test_video_model_loading,
        test_preprocessing,
        test_dataset_with_precompute,
        test_config_file,
        test_npz_file_example,
        test_precompute_script,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ 测试异常: {e}")
            results.append(False)
    
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("\n✓ 所有测试通过！功能已就绪。")
        print("\n下一步操作:")
        print("1. 运行预提取脚本:")
        print("   python video_embedding_precompute/precompute_video_embeddings.py --conf_dir configs/LRS2-IIANet.yml --data_split all")
        print("\n2. 使用预提取特征训练:")
        print("   python train.py --conf_dir configs/LRS2-IIANet.yml --use_precomputed_embeddings")
    else:
        print("\n✗ 部分测试失败，请检查上述错误信息。")
    
    print("="*80)


if __name__ == '__main__':
    main()

