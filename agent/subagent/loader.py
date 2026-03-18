"""SubAgent 配置加载器"""

import re
from pathlib import Path
from typing import Any


def load_subagent_configs(base_dir: str = "agent/subagent") -> list[dict[str, Any]]:
    """加载所有 subagent 配置

    Args:
        base_dir: subagent 配置目录路径

    Returns:
        SubAgent 配置字典列表
    """
    configs = []
    base_path = Path(base_dir)

    if not base_path.exists():
        return configs

    for subdir in base_path.iterdir():
        if not subdir.is_dir() or subdir.name.startswith("__"):
            continue

        config_file = subdir / "agent.md"
        if not config_file.exists():
            continue

        try:
            config = _parse_agent_md(config_file)
            configs.append(config)
        except Exception:
            continue

    return configs


def _parse_agent_md(file_path: Path) -> dict[str, Any]:
    """解析 agent.md 文件

    Returns:
        SubAgent 配置字典，包含 name, description, system_prompt, model (可选)
    """
    content = file_path.read_text(encoding="utf-8")

    # 解析 YAML front matter
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
    if not match:
        raise ValueError("Invalid agent.md format")

    front_matter = match.group(1)
    system_prompt = match.group(2).strip()

    # 简单解析 YAML（只支持 key: value 格式）
    metadata = {}
    for line in front_matter.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    # 构建 SubAgent 配置字典
    config: dict[str, Any] = {
        "name": metadata.get("name", ""),
        "description": metadata.get("description", ""),
        "system_prompt": system_prompt,
    }

    # 如果指定了 model，添加到配置中（否则使用默认模型）
    if metadata.get("model"):
        config["model"] = metadata["model"]

    return config
