import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .srm_filter_kernel import all_normalized_hpf_list
from .npr_feature import build_npr_feature_extractor
import numpy as np


class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        all_hpf_list_5x5 = []
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
            all_hpf_list_5x5.append(hpf_item)

        hpf_weight = torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5).contiguous()
        hpf_weight = torch.nn.Parameter(hpf_weight.repeat(1, 3, 1, 1), requires_grad=False)

        self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = self.hpf(input)
        return output


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class AIDE_Model(nn.Module):
    """
    AIDE detector: 2-branch baseline or 3-branch (+ NPR) variant.

    Fusion architecture (use_npr=True):
        HPF (2048) ─┐
                    ├─→ hpf_sem_fuse: Linear(2304→1024)+GELU ─┐
        SEM (256)  ─┘                                          ├─→ Linear(1152→512)+GELU → Linear(512→2)
                                          NPR (512→128 proj) ──┘

    Fusion architecture (use_npr=False) — identical to AIDE_2BRANCH:
        HPF (2048) ─┐
                    ├─→ Mlp(2304→1024→2)
        SEM (256)  ─┘
    """

    def __init__(
        self,
        resnet_path,
        convnext_path,
        use_npr=False,
        npr_path=None,
        fusion_type='concat',
        freeze_npr=True,
        npr_input_size=224,
        npr_proj_dim=128,
        npr_branch_dropout=0.0,
        openclip_model=None,
        hpf_branch_dropout=0.0,
        manifold_mixup=False,
        manifold_mixup_alpha=0.2,
        skip_pretrained=False,
        zero_npr_at_eval=False,
        use_gating=False,
        npr_residual_alpha_init=0.1,
    ):
        super(AIDE_Model, self).__init__()
        self.hpf = HPF()
        self.model_min = ResNet(Bottleneck, [3, 4, 6, 3])
        self.model_max = ResNet(Bottleneck, [3, 4, 6, 3])

        if resnet_path is not None:
            pretrained_dict = torch.load(resnet_path, map_location='cpu')
            model_min_dict = self.model_min.state_dict()
            model_max_dict = self.model_max.state_dict()

            for k in pretrained_dict.keys():
                if k in model_min_dict and pretrained_dict[k].size() == model_min_dict[k].size():
                    model_min_dict[k] = pretrained_dict[k]
                    model_max_dict[k] = pretrained_dict[k]
                else:
                    print(f"Skipping layer {k} because of size mismatch")

            self.model_min.load_state_dict(model_min_dict, strict=False)
            self.model_max.load_state_dict(model_max_dict, strict=False)

        print('build model with convnext_xxl')
        if openclip_model is None:
            if convnext_path is None:
                raise ValueError('convnext_path is required when openclip_model is not provided')
            import open_clip
            openclip_model, _, _ = open_clip.create_model_and_transforms(
                'convnext_xxlarge', pretrained=convnext_path
            )

        self.openclip_convnext_xxl = openclip_model.visual.trunk
        if hasattr(self.openclip_convnext_xxl, 'head'):
            if hasattr(self.openclip_convnext_xxl.head, 'global_pool'):
                self.openclip_convnext_xxl.head.global_pool = nn.Identity()
            if hasattr(self.openclip_convnext_xxl.head, 'flatten'):
                self.openclip_convnext_xxl.head.flatten = nn.Identity()

        self.openclip_convnext_xxl.eval()
        for param in self.openclip_convnext_xxl.parameters():
            param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.convnext_proj = nn.Linear(3072, 256)

        self.use_npr = use_npr
        self.freeze_npr = freeze_npr
        self.npr_input_size = npr_input_size
        self.fusion_type = fusion_type
        self.npr_proj_dim = npr_proj_dim
        self.npr_branch_dropout = npr_branch_dropout
        self.hpf_branch_dropout = hpf_branch_dropout
        self.manifold_mixup = manifold_mixup
        self.manifold_mixup_alpha = manifold_mixup_alpha
        self.zero_npr_at_eval = zero_npr_at_eval
        self.use_gating = use_gating

        if fusion_type not in ('concat', 'residual'):
            raise ValueError(
                f'Unsupported fusion_type: {fusion_type}. Supported: concat, residual.'
            )

        # Stage-1 fusion: HPF + Semantic → 1024-dim shared representation.
        # This sub-graph is identical to the AIDE_2BRANCH fusion projection,
        # ensuring that the 2-branch path is exactly reproduced when use_npr=False.
        self.hpf_sem_fuse = nn.Sequential(
            nn.Linear(2048 + 256, 1024),
            nn.GELU(),
        )

        if self.use_npr:
            # NPR branch + its own 512→npr_proj_dim linear projection
            self.npr_branch = build_npr_feature_extractor(checkpoint_path=npr_path, freeze=freeze_npr, skip_pretrained=skip_pretrained)
            self.npr_proj = nn.Linear(512, npr_proj_dim)

            if self.use_gating:
                self.npr_gate = nn.Sequential(
                    nn.Linear(1024 + npr_proj_dim, npr_proj_dim),
                    nn.Sigmoid(),
                )

            # Stage-2 classifier: [1024 + npr_proj_dim] → 2
            if self.fusion_type == 'concat':
                self.classifier = Mlp(1024 + npr_proj_dim, 512, 2)
            else:
                self.classifier = nn.Linear(1024, 2)
                self.npr_classifier = nn.Sequential(
                    nn.LayerNorm(npr_proj_dim, elementwise_affine=False),
                    nn.Linear(npr_proj_dim, 2, bias=False),
                )
                init_alpha = min(max(float(npr_residual_alpha_init), 1e-6), 1.0 - 1e-6)
                self.npr_alpha_logit = nn.Parameter(
                    torch.tensor(math.log(init_alpha / (1.0 - init_alpha)), dtype=torch.float32)
                )
        else:
            # 2-branch path: direct classifier on 1024-dim fused features.
            # Mlp(1024, 1024, 2) with hidden=1024 is the same as Linear(1024→1024)+GELU+Linear(1024→2).
            # To be fully identical to AIDE_2BRANCH (Mlp(2304,1024,2) = Linear+GELU+Linear)
            # we use a single linear layer here because hpf_sem_fuse already applies
            # the Linear(2304→1024)+GELU step.
            self.classifier = nn.Linear(1024, 2)

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _extract_convnext_feature(self, tokens):
        """Extract frozen ConvNeXt-XXL features and apply projection (both frozen via no_grad)."""
        with torch.no_grad():
            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to(tokens, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens, non_blocking=True).view(3, 1, 1)

            local_convnext_image_feats = self.openclip_convnext_xxl(
                tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            )
            local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)
            return self.convnext_proj(local_convnext_image_feats)

    def _extract_npr_feature(self, tokens):
        """Run NPR backbone (optionally frozen) then apply trainable projection."""
        npr_tokens = tokens
        if self.npr_input_size is not None and npr_tokens.shape[-1] != self.npr_input_size:
            npr_tokens = F.interpolate(
                npr_tokens,
                size=(self.npr_input_size, self.npr_input_size),
                mode='bilinear',
                align_corners=False,
            )

        if self.freeze_npr:
            with torch.no_grad():
                feat = self.npr_branch(npr_tokens)
        else:
            feat = self.npr_branch(npr_tokens)

        # npr_proj is always trainable
        return self.npr_proj(feat)

    # ------------------------------------------------------------------
    # Regularisation helpers (training-only)
    # ------------------------------------------------------------------

    def _apply_npr_branch_dropout(self, npr_feat):
        if (not self.training) or self.npr_branch_dropout <= 0:
            return npr_feat

        keep_prob = 1.0 - self.npr_branch_dropout
        if keep_prob <= 0:
            return torch.zeros_like(npr_feat)

        mask = (torch.rand(npr_feat.size(0), 1, device=npr_feat.device) < keep_prob).float()
        return npr_feat * mask / keep_prob

    def _apply_hpf_branch_dropout(self, hpf_feat):
        if (not self.training) or self.hpf_branch_dropout <= 0:
            return hpf_feat

        keep_prob = 1.0 - self.hpf_branch_dropout
        if keep_prob <= 0:
            return torch.zeros_like(hpf_feat)

        mask = (torch.rand(hpf_feat.size(0), 1, device=hpf_feat.device) < keep_prob).float()
        return hpf_feat * mask / keep_prob

    def _apply_manifold_mixup(self, feat, targets=None):
        if (not self.training) or (not self.manifold_mixup) or targets is None:
            return feat, targets

        batch_size = feat.size(0)
        lam = np.random.beta(self.manifold_mixup_alpha, self.manifold_mixup_alpha)
        index = torch.randperm(batch_size, device=feat.device)

        mixed_feat = lam * feat + (1 - lam) * feat[index]

        if targets.dim() == 1:
            targets_a, targets_b = targets, targets[index]
            mixed_targets = (
                lam * F.one_hot(targets_a, num_classes=2).float()
                + (1 - lam) * F.one_hot(targets_b, num_classes=2).float()
            )
        else:
            mixed_targets = lam * targets + (1 - lam) * targets[index]

        return mixed_feat, mixed_targets

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, targets=None):
        x_minmin  = x[:, 0]
        x_maxmax  = x[:, 1]
        x_minmin1 = x[:, 2]
        x_maxmax1 = x[:, 3]
        tokens    = x[:, 4]

        # HPF branch
        x_minmin  = self.hpf(x_minmin)
        x_maxmax  = self.hpf(x_maxmax)
        x_minmin1 = self.hpf(x_minmin1)
        x_maxmax1 = self.hpf(x_maxmax1)

        x_min  = self.model_min(x_minmin)
        x_max  = self.model_max(x_maxmax)
        x_min1 = self.model_min(x_minmin1)
        x_max1 = self.model_max(x_maxmax1)
        x_hpf  = (x_min + x_max + x_min1 + x_max1) / 4          # (B, 2048)
        x_hpf  = self._apply_hpf_branch_dropout(x_hpf)

        # Semantic branch
        x_sem = self._extract_convnext_feature(tokens)            # (B, 256)

        # Stage-1 fusion: HPF + Semantic → (B, 1024)
        x_fused = self.hpf_sem_fuse(torch.cat([x_sem, x_hpf], dim=1))

        if self.use_npr:
            # NPR branch → (B, npr_proj_dim)
            x_npr = self._extract_npr_feature(tokens)
            x_npr = self._apply_npr_branch_dropout(x_npr)

            if self.zero_npr_at_eval and not self.training:
                x_npr = torch.zeros_like(x_npr)

            if self.use_gating:
                gate = self.npr_gate(torch.cat([x_fused, x_npr], dim=1))
                x_npr = x_npr * gate

            # Stage-2 fusion: [fused, npr] → (B, 1024+npr_proj_dim)
            if self.fusion_type == 'concat':
                x_cat = torch.cat([x_fused, x_npr], dim=1)
                x_cat, targets = self._apply_manifold_mixup(x_cat, targets)
                out = self.classifier(x_cat)
            else:
                x_cat = torch.cat([x_fused, x_npr], dim=1)
                x_cat, targets = self._apply_manifold_mixup(x_cat, targets)
                x_fused = x_cat[:, :1024]
                x_npr = x_cat[:, 1024:]

                base_logits = self.classifier(x_fused)
                npr_logits = self.npr_classifier(x_npr)
                npr_alpha = torch.sigmoid(self.npr_alpha_logit)
                out = base_logits + npr_alpha * npr_logits
        else:
            x_fused, targets = self._apply_manifold_mixup(x_fused, targets)
            out = self.classifier(x_fused)

        if targets is not None and targets.dim() > 1:
            return {'logits': out, 'targets': targets}
        return out


def AIDE(
    resnet_path,
    convnext_path,
    use_npr=False,
    npr_path=None,
    fusion_type='concat',
    freeze_npr=True,
    npr_input_size=224,
    npr_proj_dim=128,
    npr_branch_dropout=0.0,
    openclip_model=None,
    hpf_branch_dropout=0.0,
    manifold_mixup=False,
    manifold_mixup_alpha=0.2,
    skip_pretrained=False,
    zero_npr_at_eval=False,
    use_gating=False,
    npr_residual_alpha_init=0.1,
):
    return AIDE_Model(
        resnet_path=resnet_path,
        convnext_path=convnext_path,
        use_npr=use_npr,
        npr_path=npr_path,
        fusion_type=fusion_type,
        freeze_npr=freeze_npr,
        npr_input_size=npr_input_size,
        npr_proj_dim=npr_proj_dim,
        npr_branch_dropout=npr_branch_dropout,
        openclip_model=openclip_model,
        hpf_branch_dropout=hpf_branch_dropout,
        manifold_mixup=manifold_mixup,
        manifold_mixup_alpha=manifold_mixup_alpha,
        skip_pretrained=skip_pretrained,
        zero_npr_at_eval=zero_npr_at_eval,
        use_gating=use_gating,
        npr_residual_alpha_init=npr_residual_alpha_init,
    )
