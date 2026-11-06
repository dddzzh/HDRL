# SAC 动作一致性修复备份

日期：2026-11-05。本目录不参与训练导入，也不包含模型或实验数据的移动/删除。

## 哪份是旧代码

- **修复前 SAC 源码**：`pre_fix/src/sac/hierarchical_low_sac.py`。
- 对应原样 Git 存档：`pre_fix_sac_git_HEAD.zip`。
- 来源提交：`cd8cd30756805f195f6d63423af5c5a7096c6a2d`。
- 修复前文件 SHA256：`917A85E7E030CF806668E849C2D6F86884DA8F623A972370EC90F63169315BB3`。

上一轮修改前，此 SAC 文件没有工作区改动；本备份从该提交恢复其原始内容。其行为还与修复后 legacy_clip 路径做过逐位对照（3 种状态模式），通过。

存档解压后的文件使用 Windows 换行；按仓库路径的 Git 换行规则计算内容对象为 `dac67497b5d139a0a8ee9ddf72e7db33d37a9266`，与 HEAD 中该文件对象完全相同。上述 SHA256 是可直接恢复使用的备份文件字节哈希。

## 当前快照

`current_snapshot/` 是此次备份时的文件快照：含修复后的 SAC、原 SM 配置、新 SM 配置、相关训练入口、reward 文件与自检脚本，共 12 个文件。复制后逐文件验证 SHA256 与源文件相同。

其中 `current_snapshot/src/sac/hierarchical_low_sac.py` 是**修复后**版本，不要误当旧代码。其 SHA256 为 `ACFE50C5DA0C158BC4693E9D4552B81BDECE8C12ACC84E83AA7E620184103BD2`。

这不是整个仓库的全量备份；原先其他未提交修改没有被回滚。当前源码仍保留，未执行恢复覆盖。

## 本轮训练

在项目根目录、py310 环境执行：

```powershell
python -u tools/train_coupled_hierarchical_d3qn_sac.py --config configs/sac_action_fix_sm74_20261105.json --episodes 1200 --seed 42
```

独立输出根目录：`results/sac_action_consistency_sm74_20261105`。不传 `--resume-checkpoint`，不加载旧低层权重。保留旧 reward，只启用 server_affine 动作修复；完整上下层从头训练。此次备份操作没有启动训练。
