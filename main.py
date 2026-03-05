from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent

navigation = st.navigation(
    [
        st.Page(
            str(REPO_ROOT / "pages/0_🤖_Agent中心.py"),
            title="Agent中心",
            icon="🤖",
            default=True,
        ),
        st.Page(
            str(REPO_ROOT / "pages/1_📁_文件中心.py"),
            title="文件中心",
            icon="📁",
        ),
        st.Page(
            str(REPO_ROOT / "pages/2_⚙️_设置中心.py"),
            title="设置中心",
            icon="⚙️",
        ),
        st.Page(
            str(REPO_ROOT / "pages/3_🗂️_项目中心.py"),
            title="项目中心",
            icon="🗂️",
        ),
    ],
    position="sidebar",
)

navigation.run()
