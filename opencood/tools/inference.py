# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
torch.multiprocessing.set_sharing_strategy('file_system')

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="102.4,102.4",
                        help="detection range is [-102.4, +102.4, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    if 'heter' in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    
    infer_info = opt.fusion_method + opt.note


    for i, batch_data in enumerate(data_loader):
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                # 获取详细的推理结果，包括原始模型输出
                output_dict = OrderedDict()
                cav_content = batch_data['ego']
                output_dict['ego'] = model(cav_content)
                
                # 获取后处理结果
                pred_box_tensor, pred_score, gt_box_tensor = \
                    opencood_dataset.post_process(batch_data, output_dict)
                
                infer_result = {"pred_box_tensor": pred_box_tensor, 
                               "pred_score": pred_score, 
                               "gt_box_tensor": gt_box_tensor}
                
                # 保存原始模型输出用于热力图分析
                raw_output = output_dict['ego']
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            
            # 继续原有的评估流程
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "agent_modality_list": agent_modality_list})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                # BEV Heatmap 可视化功能 - 与可视化保持相同频率
                try:
                    # 直接调用模型获取原始输出
                    cav_content = batch_data['ego']
                    raw_output = model(cav_content)
                    
                    # 提取 cls_preds - 检查并修正可能的问题
                    if 'cls_preds' in raw_output:
                        cls_preds = raw_output['cls_preds']
                        
                        # 获取 cls_preds 的实际维度
                        batch_size, num_classes, height, width = cls_preds.shape
                        print(f"Raw cls_preds shape: {cls_preds.shape}")
                        print(f"Raw cls_preds value range: [{cls_preds.min().item():.4f}, {cls_preds.max().item():.4f}]")
                        
                        # 对 cls_preds 做 sigmoid 激活（这是正确的，所有后处理都是这样做的）
                        cls_preds_sigmoid = torch.sigmoid(cls_preds)
                        
                        # 检查sigmoid后的值范围
                        print(f"After sigmoid - value range: [{cls_preds_sigmoid.min().item():.4f}, {cls_preds_sigmoid.max().item():.4f}]")
                        
                        # 转换为 numpy 并提取第一个batch和第一个类别（第0类）
                        cls_heatmap = cls_preds_sigmoid[0, 0, :, :].detach().cpu().numpy()
                        
                        # 创建更详细的分类热力图可视化
                        plt.figure(figsize=(15, 10))
                        
                        # 1. 原始cls热力图
                        plt.subplot(2, 3, 1)
                        plt.imshow(cls_heatmap, cmap='viridis', origin='upper')
                        plt.colorbar(label='Class Prediction Probability')
                        plt.title(f'BEV Heatmap - Frame {i:04d} - Class 0\nShape: {cls_preds.shape}')
                        plt.xlabel('X (Grid Cells)')
                        plt.ylabel('Y (Grid Cells)')
                        
                        # 2. 高置信度区域 (> 0.5)
                        plt.subplot(2, 3, 2)
                        high_conf_mask = cls_heatmap > 0.5
                        plt.imshow(high_conf_mask.astype(float), cmap='Reds', origin='upper')
                        plt.colorbar(label='High Confidence (>0.5)')
                        plt.title(f'High Confidence Regions - Frame {i:04d}')
                        plt.xlabel('X (Grid Cells)')
                        plt.ylabel('Y (Grid Cells)')
                        
                        # 3. 分类概率分布直方图
                        plt.subplot(2, 3, 3)
                        plt.hist(cls_heatmap.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
                        plt.xlabel('Classification Probability')
                        plt.ylabel('Frequency')
                        plt.title(f'Probability Distribution - Frame {i:04d}')
                        plt.grid(True, alpha=0.3)
                        plt.yscale('log')  # 使用对数尺度更好地显示分布
                        
                        # 4. X方向分类强度轮廓
                        cls_profile_x = np.mean(cls_heatmap, axis=0)
                        plt.subplot(2, 3, 4)
                        plt.plot(np.arange(width), cls_profile_x, 'b-', linewidth=2)
                        plt.xlabel('X Position (Grid Cells)')
                        plt.ylabel('Average Classification Probability')
                        plt.title(f'Classification Profile (X-axis) - Frame {i:04d}')
                        plt.grid(True, alpha=0.3)
                        
                        # 5. Y方向分类强度轮廓
                        cls_profile_y = np.mean(cls_heatmap, axis=1)
                        plt.subplot(2, 3, 5)
                        plt.plot(cls_profile_y, np.arange(height), 'g-', linewidth=2)
                        plt.ylabel('Y Position (Grid Cells)')
                        plt.xlabel('Average Classification Probability')
                        plt.title(f'Classification Profile (Y-axis) - Frame {i:04d}')
                        plt.grid(True, alpha=0.3)
                        
                        # 6. 多类别对比（如果有多个类别）
                        plt.subplot(2, 3, 6)
                        if num_classes > 1:
                            for cls_idx in range(min(3, num_classes)):  # 最多显示3个类别
                                cls_data = cls_preds_sigmoid[0, cls_idx, :, :].detach().cpu().numpy()
                                cls_mean_profile = np.mean(cls_data, axis=0)
                                plt.plot(np.arange(width), cls_mean_profile, linewidth=2, 
                                        label=f'Class {cls_idx}', alpha=0.8)
                            plt.xlabel('X Position (Grid Cells)')
                            plt.ylabel('Average Classification Probability')
                            plt.title(f'Multi-Class Comparison - Frame {i:04d}')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                        else:
                            # 如果只有一个类别，显示阈值分析
                            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
                            coverage = []
                            for thresh in thresholds:
                                coverage.append(np.mean(cls_heatmap > thresh) * 100)
                            plt.bar(range(len(thresholds)), coverage, alpha=0.7, color='orange')
                            plt.xlabel('Threshold Index')
                            plt.ylabel('Coverage Percentage (%)')
                            plt.title(f'Threshold Coverage Analysis - Frame {i:04d}')
                            plt.xticks(range(len(thresholds)), [f'{t:.1f}' for t in thresholds])
                            plt.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        
                        # 保存增强的分类热力图
                        heatmap_save_path = os.path.join(vis_save_path_root, 'cls_analysis_%05d.png' % i)
                        plt.savefig(heatmap_save_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"Saved enhanced cls analysis to: {heatmap_save_path}")
                        print(f"cls_preds shape: {cls_preds.shape}")
                        print(f"Final heatmap - min: {cls_heatmap.min():.4f}, max: {cls_heatmap.max():.4f}, mean: {cls_heatmap.mean():.4f}")
                        print(f"High confidence pixels (>0.5): {np.sum(cls_heatmap > 0.5)} / {cls_heatmap.size} ({np.mean(cls_heatmap > 0.5)*100:.2f}%)")
                        
                        # 额外检查：对比后处理中的处理方式
                        try:
                            # 按照后处理的方式处理 cls_preds
                            cls_processed = torch.sigmoid(cls_preds.permute(0, 2, 3, 1).contiguous())
                            cls_reshaped = cls_processed.reshape(1, -1)
                            
                            # 应用阈值（通常是0.1或更低）
                            threshold = 0.1  # 根据后处理代码中的典型阈值
                            mask = torch.gt(cls_reshaped, threshold)
                            high_conf_count = mask.sum().item()
                            total_count = cls_reshaped.numel()
                            
                            print(f"Post-processing style check:")
                            print(f"  Threshold {threshold}: {high_conf_count}/{total_count} pixels ({high_conf_count/total_count*100:.2f}%)")
                            print(f"  Max confidence in flattened: {cls_reshaped.max().item():.4f}")
                            
                        except Exception as postprocess_error:
                            print(f"Post-processing style check failed: {str(postprocess_error)}")
                        
                        
                    # 生成 reg 回归热度图（框的长宽分布）- 正确的解码方式
                    if 'reg_preds' in raw_output:
                        reg_preds = raw_output['reg_preds']
                        
                        # reg_preds 通常包含多个回归参数
                        # 维度为 [batch_size, num_params, height, width]
                        batch_size, num_params, height, width = reg_preds.shape
                        
                        # 根据CenterPoint的解码逻辑，正确提取长宽信息
                        if num_params >= 6:
                            # 获取模型的参数用于解码
                            try:
                                # 尝试获取模型的解码参数
                                if hasattr(model, 'out_size_factor'):
                                    out_size_factor = model.out_size_factor
                                else:
                                    out_size_factor = 1.0  # 默认值
                                
                                if hasattr(model, 'voxel_size'):
                                    voxel_size = model.voxel_size
                                else:
                                    voxel_size = [0.4, 0.4, 4.0]  # 默认值
                                
                                # 获取anchor box参数用于正确的anchor-based解码
                                anchor_box = None
                                use_anchor_based = False
                                
                                if 'anchor_box' in raw_output:
                                    anchor_box = raw_output['anchor_box']
                                    print(f"Found anchor_box in raw_output with shape: {anchor_box.shape}")
                                else:
                                    # 尝试从post_processor获取anchor box
                                    if hasattr(opencood_dataset, 'post_processor') and hasattr(opencood_dataset.post_processor, 'anchor_box'):
                                        anchor_box = opencood_dataset.post_processor.anchor_box
                                        print(f"Found anchor_box from post_processor with shape: {anchor_box.shape}")
                                    elif hasattr(opencood_dataset, 'anchor_box'):
                                        anchor_box = opencood_dataset.anchor_box
                                        print(f"Found anchor_box from dataset with shape: {anchor_box.shape}")
                                
                                # 获取参数顺序（hwl vs lhw）
                                param_order = 'hwl'  # 默认
                                if hasattr(opencood_dataset, 'post_processor') and hasattr(opencood_dataset.post_processor, 'params'):
                                    param_order = opencood_dataset.post_processor.params.get('order', 'hwl')
                                print(f"Parameter order: {param_order}")
                                
                                # 按照参数顺序提取回归偏移量
                                if param_order == 'hwl':
                                    # hwl order: [x, y, z, h, w, l, yaw]
                                    h_offset = reg_preds[0, 3, :, :].detach().cpu().numpy()  # 高度偏移
                                    w_offset = reg_preds[0, 4, :, :].detach().cpu().numpy()  # 宽度偏移  
                                    l_offset = reg_preds[0, 5, :, :].detach().cpu().numpy()  # 长度偏移
                                else:
                                    # lhw order: [x, y, z, l, h, w, yaw]
                                    l_offset = reg_preds[0, 3, :, :].detach().cpu().numpy()  # 长度偏移
                                    h_offset = reg_preds[0, 4, :, :].detach().cpu().numpy()  # 高度偏移
                                    w_offset = reg_preds[0, 5, :, :].detach().cpu().numpy()  # 宽度偏移
                                
                                # Anchor-based解码：pred = exp(offset) * anchor
                                if anchor_box is not None:
                                    # 将anchor_box转换为numpy数组（如果还不是的话）
                                    if isinstance(anchor_box, torch.Tensor):
                                        anchor_box_np = anchor_box.detach().cpu().numpy()
                                    else:
                                        anchor_box_np = anchor_box
                                    
                                    if len(anchor_box_np.shape) == 4:
                                        # 形状为 (H, W, num_anchors, 7)
                                        anchor_h, anchor_w, num_anchors, _ = anchor_box_np.shape
                                        if anchor_h == height and anchor_w == width:
                                            # 取第一个anchor (通常有2个anchor对应不同方向)
                                            if param_order == 'hwl':
                                                anchor_height = anchor_box_np[:, :, 0, 3]  # anchor高度
                                                anchor_width = anchor_box_np[:, :, 0, 4]   # anchor宽度
                                                anchor_length = anchor_box_np[:, :, 0, 5]  # anchor长度
                                            else:
                                                anchor_length = anchor_box_np[:, :, 0, 3]  # anchor长度
                                                anchor_height = anchor_box_np[:, :, 0, 4]  # anchor高度
                                                anchor_width = anchor_box_np[:, :, 0, 5]   # anchor宽度
                                            
                                            # 正确的anchor-based解码
                                            h_decoded = np.exp(h_offset) * anchor_height
                                            w_decoded = np.exp(w_offset) * anchor_width
                                            l_decoded = np.exp(l_offset) * anchor_length
                                            
                                            print(f"Anchor-based decoding success - Frame {i:04d}:")
                                            print(f"  Anchor sizes - H: [{anchor_height.min():.2f}, {anchor_height.max():.2f}], W: [{anchor_width.min():.2f}, {anchor_width.max():.2f}], L: [{anchor_length.min():.2f}, {anchor_length.max():.2f}]")
                                            print(f"  Offset ranges - H: [{h_offset.min():.4f}, {h_offset.max():.4f}], W: [{w_offset.min():.4f}, {w_offset.max():.4f}], L: [{l_offset.min():.4f}, {l_offset.max():.4f}]")
                                            print(f"  Decoded ranges - H: [{h_decoded.min():.2f}, {h_decoded.max():.2f}], W: [{w_decoded.min():.2f}, {w_decoded.max():.2f}], L: [{l_decoded.min():.2f}, {l_decoded.max():.2f}]")
                                            
                                            use_anchor_based = True
                                        else:
                                            print(f"Warning: Anchor box spatial dimensions ({anchor_h}x{anchor_w}) don't match reg_preds spatial dimensions ({height}x{width})")
                                            use_anchor_based = False
                                    else:
                                        print(f"Warning: Unexpected anchor_box shape: {anchor_box_np.shape}")
                                        use_anchor_based = False
                                else:
                                    print("Warning: No anchor_box found")
                                    use_anchor_based = False
                                
                                # 如果无法使用anchor-based解码，回退到简单的exp解码
                                if not use_anchor_based:
                                    h_decoded = np.exp(h_offset)
                                    w_decoded = np.exp(w_offset)
                                    l_decoded = np.exp(l_offset)
                                    print(f"Fallback exp decoding - Frame {i:04d}:")
                                    print(f"  Offset ranges - H: [{h_offset.min():.4f}, {h_offset.max():.4f}], W: [{w_offset.min():.4f}, {w_offset.max():.4f}], L: [{l_offset.min():.4f}, {l_offset.max():.4f}]")
                                    print(f"  Decoded ranges - H: [{h_decoded.min():.2f}, {h_decoded.max():.2f}], W: [{w_decoded.min():.2f}, {w_decoded.max():.2f}], L: [{l_decoded.min():.2f}, {l_decoded.max():.2f}]")
                                
                                # 创建解码后的长宽分布图
                                plt.figure(figsize=(15, 10))
                                
                                # 1. 宽度分布轮廓图
                                width_profile_x = np.mean(w_decoded, axis=0)  # 沿Y轴平均
                                width_profile_y = np.mean(w_decoded, axis=1)  # 沿X轴平均
                                
                                plt.subplot(2, 3, 1)
                                plt.plot(np.arange(width), width_profile_x, 'b-', linewidth=2)
                                plt.xlabel('X Position (Grid Cells)')
                                plt.ylabel('Decoded Width (meters)')
                                plt.title(f'Width Distribution (X-axis) - Frame {i:04d}')
                                plt.grid(True, alpha=0.3)
                                
                                plt.subplot(2, 3, 2)
                                plt.plot(width_profile_y, np.arange(height), 'g-', linewidth=2)
                                plt.ylabel('Y Position (Grid Cells)')
                                plt.xlabel('Decoded Width (meters)')
                                plt.title(f'Width Distribution (Y-axis) - Frame {i:04d}')
                                plt.grid(True, alpha=0.3)
                                
                                # 2. 长度分布轮廓图
                                length_profile_x = np.mean(l_decoded, axis=0)
                                length_profile_y = np.mean(l_decoded, axis=1)
                                
                                plt.subplot(2, 3, 3)
                                plt.plot(np.arange(width), length_profile_x, 'r-', linewidth=2)
                                plt.xlabel('X Position (Grid Cells)')
                                plt.ylabel('Decoded Length (meters)')
                                plt.title(f'Length Distribution (X-axis) - Frame {i:04d}')
                                plt.grid(True, alpha=0.3)
                                
                                plt.subplot(2, 3, 4)
                                plt.plot(length_profile_y, np.arange(height), 'm-', linewidth=2)
                                plt.ylabel('Y Position (Grid Cells)')
                                plt.xlabel('Decoded Length (meters)')
                                plt.title(f'Length Distribution (Y-axis) - Frame {i:04d}')
                                plt.grid(True, alpha=0.3)
                                
                                # 3. 长宽比分布
                                aspect_ratio = l_decoded / w_decoded
                                aspect_ratio_flat = aspect_ratio.flatten()
                                aspect_ratio_flat = aspect_ratio_flat[np.isfinite(aspect_ratio_flat)]  # 移除inf和nan
                                
                                plt.subplot(2, 3, 5)
                                plt.hist(aspect_ratio_flat, bins=50, alpha=0.7, color='orange', edgecolor='black')
                                plt.xlabel('Length/Width Ratio')
                                plt.ylabel('Frequency')
                                plt.title(f'Aspect Ratio Distribution - Frame {i:04d}')
                                plt.grid(True, alpha=0.3)
                                
                                # 4. 长宽联合分布 (2D散点图)
                                w_flat = w_decoded.flatten()
                                l_flat = l_decoded.flatten()
                                
                                plt.subplot(2, 3, 6)
                                plt.scatter(w_flat[::100], l_flat[::100], alpha=0.5, s=1)  # 采样显示
                                plt.xlabel('Decoded Width (meters)')
                                plt.ylabel('Decoded Length (meters)')
                                plt.title(f'Width vs Length - Frame {i:04d}')
                                plt.grid(True, alpha=0.3)
                                
                                plt.tight_layout()
                                reg_profile_save_path = os.path.join(vis_save_path_root, 'reg_decoded_profiles_%05d.png' % i)
                                plt.savefig(reg_profile_save_path, dpi=300, bbox_inches='tight')
                                plt.close()
                                
                                # 创建原始回归值和解码后值的对比直方图
                                plt.figure(figsize=(15, 6))
                                
                                plt.subplot(2, 3, 1)
                                plt.hist(w_offset.flatten(), bins=50, alpha=0.7, color='lightblue', edgecolor='black')
                                plt.xlabel('Raw Width Regression Value')
                                plt.ylabel('Frequency')
                                plt.title(f'Raw Width Regression - Frame {i:04d}')
                                plt.grid(True, alpha=0.3)
                                
                                plt.subplot(2, 3, 2)
                                plt.hist(w_decoded.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
                                plt.xlabel('Decoded Width (meters)')
                                plt.ylabel('Frequency')
                                plt.title(f'Decoded Width Distribution - Frame {i:04d}')
                                plt.grid(True, alpha=0.3)
                                
                                plt.subplot(2, 3, 3)
                                plt.hist(l_offset.flatten(), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
                                plt.xlabel('Raw Length Regression Value')
                                plt.ylabel('Frequency')
                                plt.title(f'Raw Length Regression - Frame {i:04d}')
                                plt.grid(True, alpha=0.3)
                                
                                plt.subplot(2, 3, 4)
                                plt.hist(l_decoded.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
                                plt.xlabel('Decoded Length (meters)')
                                plt.ylabel('Frequency')
                                plt.title(f'Decoded Length Distribution - Frame {i:04d}')
                                plt.grid(True, alpha=0.3)
                                
                                plt.subplot(2, 3, 5)
                                plt.hist(h_offset.flatten(), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
                                plt.xlabel('Raw Height Regression Value')
                                plt.ylabel('Frequency')
                                plt.title(f'Raw Height Regression - Frame {i:04d}')
                                plt.grid(True, alpha=0.3)
                                
                                plt.subplot(2, 3, 6)
                                plt.hist(h_decoded.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
                                plt.xlabel('Decoded Height (meters)')
                                plt.ylabel('Frequency')
                                plt.title(f'Decoded Height Distribution - Frame {i:04d}')
                                plt.grid(True, alpha=0.3)
                                
                                plt.tight_layout()
                                reg_hist_save_path = os.path.join(vis_save_path_root, 'reg_comparison_%05d.png' % i)
                                plt.savefig(reg_hist_save_path, dpi=300, bbox_inches='tight')
                                plt.close()
                                
                                print(f"Saved reg decoded profiles to: {reg_profile_save_path}")
                                print(f"Saved reg comparison to: {reg_hist_save_path}")
                                print(f"reg_preds shape: {reg_preds.shape}")
                                print(f"Raw values - w: [{w_offset.min():.4f}, {w_offset.max():.4f}], l: [{l_offset.min():.4f}, {l_offset.max():.4f}], h: [{h_offset.min():.4f}, {h_offset.max():.4f}]")
                                print(f"Decoded values - w: [{w_decoded.min():.4f}, {w_decoded.max():.4f}]m, l: [{l_decoded.min():.4f}, {l_decoded.max():.4f}]m, h: [{h_decoded.min():.4f}, {h_decoded.max():.4f}]m")
                                
                            except Exception as decode_error:
                                print(f"Error in decoding regression values: {str(decode_error)}")
                                # 如果解码失败，回退到原来的方法
                                dw_preds = reg_preds[0, 3, :, :].detach().cpu().numpy()  # 维度3
                                dl_preds = reg_preds[0, 4, :, :].detach().cpu().numpy()  # 维度4
                                print(f"Fallback: reg_preds shape: {reg_preds.shape}, dw range: [{dw_preds.min():.4f}, {dw_preds.max():.4f}], dl range: [{dl_preds.min():.4f}, {dl_preds.max():.4f}]")
                        
                except Exception as e:
                    print(f"Error creating BEV heatmap for frame {i}: {str(e)}")
                    # 继续执行，不中断推理流程

                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)
        torch.cuda.empty_cache()

    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, infer_info)

if __name__ == '__main__':
    main()
