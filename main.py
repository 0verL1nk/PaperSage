from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent

def run_app() -> None:
    navigation = st.navigation(
        [
            st.Page(
                str(REPO_ROOT / "pages/0_agent_center.py"),
                title="Agent中心",
                icon="🤖",
                default=True,
            ),
            st.Page(
                str(REPO_ROOT / "pages/1_file_center.py"),
                title="文件中心",
                icon="📁",
            ),
            st.Page(
                str(REPO_ROOT / "pages/2_settings.py"),
                title="设置中心",
                icon="⚙️",
            ),
            st.Page(
                str(REPO_ROOT / "pages/3_project_center.py"),
                title="项目中心",
                icon="🗂️",
            ),
        ],
        position="sidebar",
    )

    navigation.run()


def _cli_entry() -> None:
    """CLI entry point: `paper-sage` launches the Streamlit app."""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", __file__],
        check=False,
    )


if __name__ == "__main__":
    run_app()
