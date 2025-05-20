import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import timm
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
# 导入 MobileNetV2
# from torchvision.models import mobilenet_v2
from torchvision.models import mobilenet_v2, resnet18, shufflenet_v2_x1_0, efficientnet_b0, mobilenet_v3_small


# 消融适应模块
class AblationModule(nn.Module):
    def __init__(self, module, enable=True):
        super(AblationModule, self).__init__()
        self.module = module
        self.enable = enable

    def forward(self, *args, **kwargs):
        if self.enable:
            return self.module(*args, **kwargs)
        else:
            # 根据模块的输入和期望的输出类型，返回适当的默认值
            if isinstance(args[0], list):
                # 如果输入是列表，直接返回输入列表
                return args[0]
            elif len(args) == 2:
                # 如果有两个输入，返回它们的绝对差值
                return torch.abs(args[0] - args[1])
            else:
                # 对于其他情况，直接返回输入
                return args[0]


# 定义高效多尺度注意力（EMA）模块
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(
            channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0
        )
        self.conv3x3 = nn.Conv2d(
            channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


# 修改后的多尺度特征增强模块（MSFE）
class MultiScaleFeatureEnhancement(nn.Module):
    def __init__(self, in_channels_list, out_channels=64):
        super(MultiScaleFeatureEnhancement, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        # 定义每个分支的输出通道数，确保所有分支的输出通道数相同
        self.num_branches = 4
        base_channel = out_channels // self.num_branches
        self.branch_out_channels = [base_channel] * self.num_branches
        # 调整最后一个分支的通道数，使总和等于 out_channels
        current_sum = sum(self.branch_out_channels)
        difference = out_channels - current_sum
        self.branch_out_channels[-1] += difference

        assert sum(self.branch_out_channels) == self.out_channels, "Sum of branch channels must equal out_channels"

        self.conv_blocks = nn.ModuleList(
            [self._make_conv_block(in_ch) for in_ch in in_channels_list]
        )

        # 融合卷积层
        self.fusion_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels_list
            ]
        )

        # 残差连接，调整通道数以匹配输出
        self.residual_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False) for in_ch in in_channels_list]
        )

        # 使用 EMA 模块作为注意力机制
        self.attention_blocks = nn.ModuleList([EMA(out_channels) for _ in in_channels_list])

        # 融合权重参数，初始化为全1，可学习参数
        self.weights = nn.Parameter(torch.ones(self.num_branches), requires_grad=True)

    def _make_conv_block(self, in_ch):
        branches = nn.ModuleList()

        # 分支1：1x1卷积（逐点卷积）
        branch1 = nn.Sequential(
            nn.Conv2d(in_ch, self.branch_out_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(self.branch_out_channels[0]),
            nn.ReLU(inplace=True),
        )
        branches.append(branch1)

        # 分支2：3x3深度可分离卷积
        branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),  # 深度卷积
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, self.branch_out_channels[1], kernel_size=1, bias=False),  # 逐点卷积
            nn.BatchNorm2d(self.branch_out_channels[1]),
            nn.ReLU(inplace=True),
        )
        branches.append(branch2)
        
        # 分支3：3x3膨胀卷积（dilation=2）
        branch3 = nn.Sequential(
            nn.Conv2d(in_ch, self.branch_out_channels[2], kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(self.branch_out_channels[2]),
            nn.ReLU(inplace=True),
        )
        branches.append(branch3)

        # 分支4：5x5深度可分离卷积
        branch4 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch, bias=False),  # 深度卷积
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, self.branch_out_channels[3], kernel_size=1, bias=False),  # 逐点卷积
            nn.BatchNorm2d(self.branch_out_channels[3]),
            nn.ReLU(inplace=True),
        )
        branches.append(branch4)

        return branches

    def forward(self, features):
        processed_features = []

        for idx, x in enumerate(features):
            # print(f"Feature {idx}: {x.size()}")
            branches = self.conv_blocks[idx]
            branch_outputs = []
            for i, branch in enumerate(branches):
                out = branch(x)
                # print(f"    Branch {i} output: {out.size()}")
                # 如果输出尺寸与输入尺寸不同，进行上采样
                if out.size(2) != x.size(2) or out.size(3) != x.size(3):
                    out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=False)
                branch_outputs.append(out)

            # 动态融合权重，进行归一化
            weights = F.softmax(self.weights, dim=0)

            # 对每个分支的输出进行加权
            weighted_branch_outputs = []
            for w, b in zip(weights, branch_outputs):
                w = w.view(1, 1, 1, 1)
                weighted_branch_outputs.append(b * w)

            # 融合分支输出
            fused = torch.cat(weighted_branch_outputs, dim=1)

            # 融合卷积
            fused = self.fusion_convs[idx](fused)

            # 残差连接
            residual = self.residual_convs[idx](x)
            fused = fused + residual

            # 应用 EMA 注意力机制
            fused = self.attention_blocks[idx](fused)

            processed_features.append(fused)
        return processed_features


# 定义 SE 块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        se_weight = self.se(x)
        return x * se_weight


# 示例：替换 CBAM 为更高效的注意力模块（如 Efficient Attention）
class EfficientAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(EfficientAttention, self).__init__()
        self.channel_att = SEBlock(channels, reduction)
        # 添加其他高效注意力机制，如 Spatial Attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.channel_att(x)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        return x * spatial_att


class GroupFusion(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(GroupFusion, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # 替换为 EfficientAttention
        self.efficient_att = EfficientAttention(out_channels, reduction=reduction)
        # 添加 1x1 卷积调整通道数，以匹配残差连接
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, feat1, feat2):
        # 计算特征的绝对差异
        diff = torch.abs(feat1 - feat2)
        # 对差异特征进行卷积处理
        out = self.conv(diff)
        out = self.efficient_att(out)
        # 残差连接
        residual = self.residual_conv(diff)
        out += residual
        return out


class FeatureDifferenceFusion(nn.Module):
    def __init__(self, in_channels, out_channels=64, reduction=16):
        super(FeatureDifferenceFusion, self).__init__()
        # 使用优化后的 GroupFusion 模块
        self.group_fusion = GroupFusion(in_channels, out_channels, reduction=reduction)

    def forward(self, feat1, feat2):
        # 调用 GroupFusion 模块处理特征差异
        out = self.group_fusion(feat1, feat2)
        return out


# 修改后的全局上下文注意力模块（GCA）- 引入渐进式融合策略
class GlobalContextAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, kernel_size=7):
        super(GlobalContextAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 渐进式融合层次数
        self.num_stages = 2

        # 定义每个阶段的卷积层
        self.fusion_convs = nn.ModuleList()
        for _ in range(self.num_stages):
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # 最后的 1x1 卷积调整通道数
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        # 输出卷积，调整通道数
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 残差连接，如果输入和输出通道不一致，则使用 1x1 卷积调整
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x_high, x_low):
        # 上采样 x_high 以匹配 x_low 的尺寸
        x_high = F.interpolate(x_high, size=x_low.size()[2:], mode='bilinear', align_corners=False)

        # 渐进式融合
        x = x_low
        for fusion_conv in self.fusion_convs:
            x = x + x_high
            x = fusion_conv(x)

        # 1x1 卷积调整通道数
        x = self.conv1x1(x)

        # 通道注意力
        channel_att = self.channel_attention(x)
        x = x * channel_att

        # 空间注意力
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
        x = x * spatial_att

        # 调整残差连接
        residual = self.residual_conv(x_low)
        x = self.out_conv(x) + residual

        return x


def visualize_features(tensor, title, img1=None, img2=None, save_dir="visualizations", filename_prefix="feature",
                       is_fft_features=False, is_mag_phase=False, is_concat=False, is_freq_conv=False, freq_dim=16):
    """
    多模式可视化特征函数，支持FFT复数特征，幅度/相位，拼接特征，频率卷积后特征。
    特定模式（FFT、幅度/相位、拼接特征）使用3D表面图可视化，每个通道单独保存。
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from datetime import datetime

    os.makedirs(save_dir, exist_ok=True)

    def tensor_to_np(t):
        t = t.detach().cpu().numpy()
        return t

    # 输入图像辅助展示函数（左侧，使用2D）
    def show_input_images():
        fig_2d = plt.figure(figsize=(8, 8))
        gs_2d = fig_2d.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]

        for i, (img, ax, txt) in enumerate(zip([img1, img2],
                                              [fig_2d.add_subplot(gs_2d[0, 0]), fig_2d.add_subplot(gs_2d[1, 0])],
                                              ["Input Image 1", "Input Image 2"])):
            if img is None:
                ax.text(0.5, 0.5, f"No {txt} Provided", fontsize=10, ha='center', va='center')
                ax.axis('off')
                continue
            img_np = tensor_to_np(img)[0]
            if img_np.shape[0] == 3:
                img_disp = img_np * std + mean
                img_disp = np.clip(img_disp, 0, 1)
                img_disp = np.transpose(img_disp, (1, 2, 0))
                ax.imshow(img_disp)
            else:
                ax.imshow(img_np[0], cmap='gray')
            ax.set_title(txt)
            ax.axis('off')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"{filename_prefix}_input_images_{timestamp}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig_2d)
        print(f"[Info] Saved input images: {save_path}")

    show_input_images()

    # 3D表面图绘制函数
    def plot_3d_surface(data, title, cmap, save_path):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        H, W = data.shape
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        Z = data
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none')
        ax.set_title(title)
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_zlabel('Value')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"[Info] Saved 3D visualization: {save_path}")

    # FFT复数特征显示：实部和虚部（3D表面图）
    if is_fft_features:
        fft_np = tensor_to_np(tensor)[0]  # [C, H, W], complex64/128 numpy array
        num_show_ch = min(8, fft_np.shape[0])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i in range(num_show_ch):
            # Real part
            real_map = np.real(fft_np[i])
            real_map = (real_map - real_map.min()) / (real_map.max() - real_map.min() + 1e-8)
            save_path = os.path.join(save_dir, f"{filename_prefix}_fft_real_ch{i}_{title.replace(' ', '_')}_{timestamp}.png")
            plot_3d_surface(real_map, f"FFT Real Channel {i}", 'viridis', save_path)

            # Imaginary part
            imag_map = np.imag(fft_np[i])
            imag_map = (imag_map - imag_map.min()) / (imag_map.max() - imag_map.min() + 1e-8)
            save_path = os.path.join(save_dir, f"{filename_prefix}_fft_imag_ch{i}_{title.replace(' ', '_')}_{timestamp}.png")
            plot_3d_surface(imag_map, f"FFT Imag Channel {i}", 'plasma', save_path)

    # 幅度和相位：tensor为tuple (magnitude, phase)（3D表面图）
    elif is_mag_phase:
        magnitude, phase = tensor
        mag_np = tensor_to_np(magnitude)[0]
        phase_np = tensor_to_np(phase)[0]
        max_channels = min(8, mag_np.shape[0])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i in range(max_channels):
            # Magnitude
            mag_map = mag_np[i]
            mag_map = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min() + 1e-8)
            save_path = os.path.join(save_dir, f"{filename_prefix}_magnitude_ch{i}_{title.replace(' ', '_')}_{timestamp}.png")
            plot_3d_surface(mag_map, f"Magnitude Channel {i}", 'inferno', save_path)

            # Phase
            phase_map = phase_np[i]
            phase_map_norm = (phase_map + np.pi) / (2 * np.pi)
            save_path = os.path.join(save_dir, f"{filename_prefix}_phase_ch{i}_{title.replace(' ', '_')}_{timestamp}.png")
            plot_3d_surface(phase_map_norm, f"Phase Channel {i}", 'twilight', save_path)

    # 拼接后的magnitude+phase特征：tensor形状[B, 2*C, H, W]（3D表面图）
    elif is_concat:
        concat_np = tensor_to_np(tensor)[0]
        C = concat_np.shape[0] // 2
        mag_np = concat_np[:C]
        phase_np = concat_np[C:]
        max_channels = min(8, C)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i in range(max_channels):
            # Magnitude
            mag_map = mag_np[i]
            mag_map = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min() + 1e-8)
            save_path = os.path.join(save_dir, f"{filename_prefix}_concat_mag_ch{i}_{title.replace(' ', '_')}_{timestamp}.png")
            plot_3d_surface(mag_map, f"Concat Magnitude Channel {i}", 'inferno', save_path)

            # Phase
            phase_map = phase_np[i]
            phase_map_norm = (phase_map + np.pi) / (2 * np.pi)
            save_path = os.path.join(save_dir, f"{filename_prefix}_concat_phase_ch{i}_{title.replace(' ', '_')}_{timestamp}.png")
            plot_3d_surface(phase_map_norm, f"Concat Phase Channel {i}", 'twilight', save_path)

    # 频率卷积降维后的特征图（保留2D热力图）
    elif is_freq_conv:
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(4, 5, width_ratios=[1] * 5, height_ratios=[1] * 4, wspace=0.3, hspace=0.4)
        freq_np = tensor_to_np(tensor)[0]
        max_channels = min(16, freq_np.shape[0])
        for i in range(max_channels):
            row = i // 4
            col = 1 + (i % 4)
            ax = fig.add_subplot(gs[row, col])
            fmap = freq_np[i]
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            ax.imshow(fmap, cmap='cividis')
            ax.set_title(f"FreqConv Channel {i}")
            ax.axis('off')

        fig.suptitle(f"{title} Visualization", fontsize=16)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"{filename_prefix}_{title.replace(' ', '_')}_{timestamp}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"[Info] Saved visualization: {save_path}")

    # 默认普通特征展示（保留2D热力图）
    else:
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(4, 5, width_ratios=[1] * 5, height_ratios=[1] * 4, wspace=0.3, hspace=0.4)
        feat_np = tensor_to_np(tensor)[0]
        max_channels = min(16, feat_np.shape[0])
        for i in range(max_channels):
            row = i // 4
            col = 1 + (i % 4)
            ax = fig.add_subplot(gs[row, col])
            fmap = feat_np[i]
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            ax.imshow(fmap, cmap='plasma')
            ax.set_title(f"Feature Channel {i}")
            ax.axis('off')

        fig.suptitle(f"{title} Visualization", fontsize=16)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"{filename_prefix}_{title.replace(' ', '_')}_{timestamp}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"[Info] Saved visualization: {save_path}")




class HybridGCNModule(nn.Module):
    def __init__(self, in_channels, reduction=16, freq_dim=16):
        super(HybridGCNModule, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.freq_dim = freq_dim

        # Spatial attention convolutions
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # Frequency feature processing
        self.freq_conv = nn.Conv2d(2 * in_channels, freq_dim, kernel_size=1, bias=False)
        # Learnable weighting for frequency features (to emphasize global context)
        self.freq_weight = nn.Parameter(torch.ones(1, freq_dim, 1, 1))  # Shape [1, freq_dim, 1, 1]

        # Output convolution for feature fusion
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels + freq_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Convolution to adjust difference map channels
        self.diff_conv = nn.Conv2d(1, in_channels, kernel_size=1, bias=False)

    def forward(self, x, img1=None, img2=None):
        device = next(self.parameters()).device
        x = x.to(device, dtype=torch.float32)
        B, C, H, W = x.size()

        # Spatial attention
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, H*W, C//reduction]
        key = self.key_conv(x).view(B, -1, H * W)  # [B, C//reduction, H*W]
        attention = F.softmax(torch.matmul(query, key), dim=-1)  # [B, H*W, H*W]
        value = self.value_conv(x).view(B, -1, H * W)  # [B, C, H*W]
        spatial_out = torch.matmul(attention, value.permute(0, 2, 1)).view(B, C, H, W)  # [B, C, H, W]

        # Frequency feature extraction
        fft_features = torch.fft.fft2(x, dim=(-2, -1))  # [B, C, H, W], complex
        magnitude = torch.abs(fft_features)  # [B, C, H, W]
        phase = torch.angle(fft_features)  # [B, C, H, W]

        # Concatenate magnitude and phase
        freq_features = torch.cat([magnitude, phase], dim=1)  # [B, 2*C, H, W]
        freq_features = self.freq_conv(freq_features)  # [B, freq_dim, H, W]

        # Apply learnable weighting to frequency features
        freq_features = freq_features * self.freq_weight  # [B, freq_dim, H, W]

        # Feature fusion
        combined_features = torch.cat([spatial_out, freq_features], dim=1)  # [B, C + freq_dim, H, W]
        out = self.out_conv(combined_features) + x  # Residual connection
        """
        # Optional: Process img1 and img2 for change detection context
        if img1 is not None and img2 is not None:
            # Compute difference map
            diff = torch.abs(img1 - img2)  # [B, 3, H_img, W_img]
            diff = diff.mean(dim=1, keepdim=True)  # [B, 1, H_img, W_img]
            # Downsample diff to match out's spatial dimensions
            diff = F.interpolate(diff, size=(H, W), mode='bilinear', align_corners=False)  # [B, 1, H, W]
            # Adjust channel dimension to match out
            diff = self.diff_conv(diff)  # [B, C, H, W]
            out = out + 0.1 * diff  # Weight the difference map lightly
"""
        return out




# 修改后的解码器，使用深度可分离卷积
class EfficientDecoder(nn.Module):
    def __init__(self, channels=64, fusion_alpha=0.3):
        super(EfficientDecoder, self).__init__()
        self.channels = channels
        self.fusion_alpha = fusion_alpha  # 融合比例

        # 定义 MSAM 模块
        self.msams = nn.ModuleList([MultiScaleAttentionModule(channels) for _ in range(4)])

        # 最终的分类层，使用深度可分离卷积
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),  # 深度卷积
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),  # 逐点卷积
        )

    def forward(self, features):
        c1, c2, c3, c4, c5 = features  # c1 是最高分辨率

        # 使用 c5 作为全局特征
        gc = c5

        # 从高层到低层逐步解码
        masks = []
        x = c5  # 初始特征
        for idx, msam in enumerate(self.msams):
            # 获取对应的低层特征
            if idx == 0:
                low_feat = c4
            elif idx == 1:
                low_feat = c3
            elif idx == 2:
                low_feat = c2
            elif idx == 3:
                low_feat = c1

            # 应用 MSAM 模块
            x, mask = msam(low_feat, x, gc)

            masks.append(mask)

        # 最终输出
        output = self.final_conv(x)

        # 将 masks 调整到与 output 相同的尺寸
        masks_upsampled = [
            F.interpolate(mask, size=output.size()[2:], mode='bilinear', align_corners=False) for mask in masks
        ]

        # 融合所有的 masks，可以使用平均或加权平均，这里采用平均
        combined_mask = torch.mean(torch.stack(masks_upsampled), dim=0)

        # 对 output 和 combined_mask 应用 sigmoid 函数
        # output_sigmoid = torch.sigmoid(output)
        # combined_mask_sigmoid = torch.sigmoid(combined_mask)

        # 按照融合比例进行加权融合
        # fused_output = output_sigmoid * (1 - self.fusion_alpha) + combined_mask_sigmoid * self.fusion_alpha
        fused_output = output * (1 - self.fusion_alpha) + combined_mask * self.fusion_alpha
        # 返回最终的融合输出
        return fused_output, masks


# 定义新的 Multi-Scale Attention Module (MSAM)
class MultiScaleAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleAttentionModule, self).__init__()
        self.in_channels = in_channels

        # 通道注意力部分
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )

        # 特征融合卷积，使用深度可分离卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # 预测掩码的卷积层
        self.cls = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, low_feat, high_feat, global_feat):
        # 上采样高层特征到低层特征的尺寸
        high_feat_up = F.interpolate(high_feat, size=low_feat.size()[2:], mode='bilinear', align_corners=False)
        global_feat_up = F.interpolate(global_feat, size=low_feat.size()[2:], mode='bilinear', align_corners=False)

        # 特征融合
        fused_feat = low_feat + high_feat_up + global_feat_up

        # 通道注意力
        low_pool = self.global_pool(low_feat)
        high_pool = self.global_pool(high_feat_up)
        global_pool = self.global_pool(global_feat_up)

        # 拼接池化后的特征
        concat_pool = torch.cat([low_pool, high_pool, global_pool], dim=1)

        # 计算注意力权重
        channel_att = self.channel_attention(concat_pool)
        fused_feat = fused_feat * channel_att

        # 特征融合卷积
        fused_feat = self.fusion_conv(fused_feat)

        # 生成掩码
        mask = self.cls(fused_feat)

        return fused_feat, mask


# EFANet 架构，使用 MobileNetV2 作为编码器，并集成消融适应模块
class EMFINet(nn.Module):
    def __init__(self, num_classes=1, backbone='mobilenet_v2', use_msfe=True, use_fdf=True, use_gca=True):
        super(EMFINet, self).__init__()

        # 选择骨干网络
        if backbone == 'resnet18':
            self.encoder = resnet18(pretrained=True)
            # 五层特征通道依次是 [64, 64, 128, 256, 512]
            self.in_channels_list = [64, 64, 128, 256, 512]
            self.encoder_features = self.encoder_features_resnet18
        elif backbone == 'shufflenet':
            self.encoder = shufflenet_v2_x1_0(pretrained=True)
            # 这里选择 5 个特征层
            self.in_channels_list = [24, 116, 232, 464, 1024]
            self.encoder_features = self.encoder_features_shufflenet
        elif backbone == 'efficientnet_b0':
            self.encoder = efficientnet_b0(pretrained=True)
            self.in_channels_list = [16, 24, 40, 112, 320]
            self.encoder_features = self.encoder_features_efficientnet_b0
        elif backbone == 'mobilenet_v3':
            self.encoder = mobilenet_v3_small(pretrained=True)
            # 这里指定 5 个特征层：24, 40, 48, 96, 160
            self.in_channels_list = [24, 40, 40, 96, 160]
            self.encoder_features = self.encoder_features_mobilenetv3
        elif backbone == 'mobilevit':
            self.encoder = timm.create_model('mobilevit_s', pretrained=True)
            self.encoder_features = self.encoder_features_mobilevit
            self.in_channels_list = [16, 32, 64, 96, 128]  # MobileViT-S 的输出通道数
            # model = timm.create_model('mobilevit_s', pretrained=True)
            # print(model)
        else:
            # 默认使用 MobileNetV2
            self.encoder = mobilenet_v2(pretrained=True)
            self.encoder_features = self.encoder_features_mobilenetv2
            self.in_channels_list = [16, 24, 32, 96, 320]

        # MSFE
        if use_msfe:
            self.msfe = AblationModule(
                MultiScaleFeatureEnhancement(in_channels_list=self.in_channels_list, out_channels=64), enable=True
            )
        else:
            self.msfe = AblationModule(nn.Identity(), enable=False)

        # FDF
        if use_fdf:
            self.feature_diff_fusion = nn.ModuleList(
                [
                    AblationModule(FeatureDifferenceFusion(in_channels=64), enable=True)
                    for _ in range(len(self.in_channels_list))
                ]
            )
        else:
            self.feature_diff_fusion = nn.ModuleList(
                [AblationModule(nn.Identity(), enable=False) for _ in range(len(self.in_channels_list))]
            )

        # GCA 应用于多个层级

        if use_gca:
            self.gca_modules = nn.ModuleList(
                [
                    AblationModule(GlobalContextAttention(in_channels=64, out_channels=64), enable=True)
                    for _ in range(len(self.in_channels_list) - 1)
                ]
            )
        else:
            self.gca_modules = nn.ModuleList(
                [AblationModule(nn.Identity(), enable=False) for _ in range(len(self.in_channels_list) - 1)]
            )

        # 仅在较高层次应用 Dynamic GCN 模块（假设应用于最后两个层次）
        self.gcn_modules = nn.ModuleList([
            HybridGCNModule(in_channels=64) for _ in range(1)  # 仅应用于最后两个层次
        ])

        # 使用新的解码器
        self.decoder = EfficientDecoder(channels=64)

    def encoder_features_mobilevit(self, x):
        features = []
        x = self.encoder.stem(x)
        features.append(x)  # 通道数：16

        for stage in self.encoder.stages:
            x = stage(x)
            features.append(x)  # 通道数逐步增加
        return features[:5]  # [16, 32, 64, 96, 128]
        # return features

    def forward(self, img1, img2):
        # 提取特征
        features1 = self.encoder_features(img1)
        features2 = self.encoder_features(img2)
        # 打印每个特征层的通道数
        # for idx, feat in enumerate(features1):
        # print(f"Feature {idx} channels: {feat.shape[1]}")
        # MSFE
        enhanced_features1 = self.msfe(features1)
        enhanced_features2 = self.msfe(features2)

        # FDF
        fused_features = []
        for idx, (f1, f2) in enumerate(zip(enhanced_features1, enhanced_features2)):
            fused_feat = self.feature_diff_fusion[idx](f1, f2)
            fused_features.append(fused_feat)

        # 应用 GCA 到多个层级特征
        if len(self.gca_modules) > 0:
            for i in range(len(fused_features) - 1, 0, -1):
                fused_features[i] = self.gca_modules[i - 1](fused_features[i], fused_features[i - 1])

        # 仅在较高层次应用 Dynamic GCN 模块
        # 假设 fused_features[-1] 是最低分辨率的特征（最高层次）
        #fused_features[-1] = self.gcn_modules[0](fused_features[-1])
        fused_features[-1] = self.gcn_modules[0](fused_features[-1], img1=img1, img2=img2)
        # fused_features[-2] = self.gcn_modules[1](fused_features[-2])
        # fused_features[-3] = self.gcn_modules[2](fused_features[-3])
        # fused_features[-4] = self.gcn_modules[2](fused_features[-4])

        # 构建特征列表，从高分辨率到低分辨率
        features = fused_features  # 已经是从高到低分辨率

        # 解码器
        output, masks = self.decoder(features)

        # 获取输入尺寸
        input_size = img1.size()[2:]

        # 上采样输出和掩码到输入尺寸
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
        masks = [F.interpolate(mask, size=input_size, mode='bilinear', align_corners=False) for mask in masks]

        return output, masks

    def encoder_features_mobilenetv2(self, x):
        features = []
        # MobileNetV2 特征提取
        x = self.encoder.features[0](x)  # ConvBNReLU(3,32,stride=2)
        x = self.encoder.features[1](x)  # InvertedResidual(32,16)
        features.append(x)  # 通道数：16

        x = self.encoder.features[2](x)  # InvertedResidual(16,24)
        x = self.encoder.features[3](x)  # InvertedResidual(24,24)
        features.append(x)  # 通道数：24

        x = self.encoder.features[4](x)  # InvertedResidual(24,32)
        x = self.encoder.features[5](x)  # InvertedResidual(32,32)
        x = self.encoder.features[6](x)  # InvertedResidual(32,32)
        features.append(x)  # 通道数：32

        x = self.encoder.features[7](x)  # InvertedResidual(32,64)
        x = self.encoder.features[8](x)  # InvertedResidual(64,64)
        x = self.encoder.features[9](x)  # InvertedResidual(64,64)
        x = self.encoder.features[10](x)  # InvertedResidual(64,64)
        x = self.encoder.features[11](x)  # InvertedResidual(64,96)
        x = self.encoder.features[12](x)  # InvertedResidual(96,96)
        x = self.encoder.features[13](x)  # InvertedResidual(96,96)
        features.append(x)  # 通道数：96

        x = self.encoder.features[14](x)  # InvertedResidual(96,160)
        x = self.encoder.features[15](x)  # InvertedResidual(160,160)
        x = self.encoder.features[16](x)  # InvertedResidual(160,160)
        x = self.encoder.features[17](x)  # InvertedResidual(160,320)
        features.append(x)  # 通道数：320

        return features

    # ResNet18 的特征提取函数
    def encoder_features_resnet18(self, x):
        """
        返回 5 个特征:
          c1: 经过 (conv1 + bn1 + relu) 后的输出
          c2: layer1 的输出
          c3: layer2 的输出
          c4: layer3 的输出
          c5: layer4 的输出
        """

        # 1) 先经过最前面的 7x7 卷积 + BN + ReLU
        x = self.encoder.conv1(x)  # shape: [B, 64, H/2, W/2]
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        c1 = x  # 第1层特征 (通道=64, 尺寸=H/2)

        # 2) maxpool 进一步下采样
        x = self.encoder.maxpool(x)  # [B, 64, H/4, W/4]
        # 3) layer1 -> 通道依然 64
        c2 = self.encoder.layer1(x)  # [B, 64, H/4, W/4]
        # 4) layer2 -> 通道变成 128
        c3 = self.encoder.layer2(c2)  # [B,128, H/8,  W/8 ]
        # 5) layer3 -> 通道 256
        c4 = self.encoder.layer3(c3)  # [B,256, H/16, W/16]
        # 6) layer4 -> 通道 512
        c5 = self.encoder.layer4(c4)  # [B,512, H/32, W/32]

        return [c1, c2, c3, c4, c5]

    # ShuffleNet 的特征提取函数
    def encoder_features_shufflenet(self, x):
        # 先过第一个卷积和 maxpool，输出通道 24
        x = self.encoder.conv1(x)  # [B, 24,  H/2,  W/2 ]
        x = self.encoder.maxpool(x)  # [B, 24,  H/4,  W/4 ]

        c1 = x  # 24 通道
        c2 = self.encoder.stage2(c1)  # 116 通道
        c3 = self.encoder.stage3(c2)  # 232 通道
        c4 = self.encoder.stage4(c3)  # 464 通道
        c5 = self.encoder.conv5(c4)  # 1024 通道

        return [c1, c2, c3, c4, c5]

    # EfficientNet-B0 的特征提取函数
    def encoder_features_efficientnet_b0(self, x):
        # features[0] -> 32 通道
        x = self.encoder.features[0](x)

        # 1) c1 = 16 通道
        c1 = self.encoder.features[1](x)

        # 2) c2 = 24 通道
        c2 = self.encoder.features[2](c1)

        # 3) c3 = 40 通道
        c3 = self.encoder.features[3](c2)
        # 也可以把 features[4](c3)->80, 先保留在中间
        tmp = self.encoder.features[4](c3)  # 80 通道

        # 4) c4 = 112 通道
        c4 = self.encoder.features[5](tmp)

        # (6) -> 192, (7) -> 320
        tmp = self.encoder.features[6](c4)
        c5 = self.encoder.features[7](tmp)  # 320 通道

        return [c1, c2, c3, c4, c5]

    # MobileViT 的特征提取函数

    def encoder_features_mobilenetv3(self, x):
        # 通过 MobileNetV3 的特征提取层提取特征
        x0 = self.encoder.features[0](x)  # [B, 16, H/2, W/2]
        x1 = self.encoder.features[1](x0)  # [B, 16, H/2, W/2]
        x2 = self.encoder.features[2](x1)  # [B, 24, H/4, W/4]
        x3 = self.encoder.features[3](x2)  # [B, 24, H/4, W/4]
        x4 = self.encoder.features[4](x3)  # [B, 40, H/8, W/8]
        x5 = self.encoder.features[5](x4)  # [B, 40, H/8, W/8]
        x6 = self.encoder.features[6](x5)  # [B, 48, H/16, W/16]
        x7 = self.encoder.features[7](x6)  # [B, 48, H/16, W/16]
        x8 = self.encoder.features[8](x7)  # [B, 96, H/32, W/32]
        x9 = self.encoder.features[9](x8)  # [B, 96, H/32, W/32]
        x10 = self.encoder.features[10](x9)  # [B, 160, H/32, W/32]

        # 返回 5 个特征层
        return [x2, x4, x6, x8, x10]  # [24, 40, 48, 96, 160] 通道数
