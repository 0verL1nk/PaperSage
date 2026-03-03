import streamlit as st


def inject_global_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink-900: #0f2f52;
            --ink-700: #1d4e89;
            --ink-500: #2b6cb0;
            --sky-50: #f4f8ff;
            --sky-100: #e7f0ff;
            --sky-200: #d3e5ff;
            --line-soft: #d9e8ff;
            --paper: #ffffff;
            --success: #166534;
            --warn: #92400e;
            --badge-high-bg: #ecfdf3;
            --badge-high-line: #bbf7d0;
            --badge-mid-bg: #fff7ed;
            --badge-mid-line: #fed7aa;
            --badge-low-bg: #eff6ff;
            --badge-low-line: #bfdbfe;
            --workspace-grad-start: #f8fbff;
            --workspace-grad-end: #eef5ff;
            --workspace-title: #133b70;
            --workspace-chip-bg: #ffffff;
            --workspace-chip-line: #cfe2ff;
            --workspace-hint-bg: #f5f9ff;
            --workspace-card-title: #113861;
            --workspace-card-sub: #27527f;
            --trace-item-bg: #f8fbff;
            --trace-item-line: #dbeafe;
            --trace-meta-bg: #eef5ff;
            --trace-content: #163a66;
            --ctx-card-bg: #f8fbff;
            --ctx-card-line: #d9e8ff;
            --ctx-kpi-bg: #ffffff;
            --ctx-kpi-line: #dbeafe;
            --ctx-bar-bg: #e9f1ff;
            --ctx-bar-fill-start: #2b6cb0;
            --ctx-bar-fill-end: #60a5fa;
            --ctx-bar-free-start: #16a34a;
            --ctx-bar-free-end: #86efac;
            --ctx-cat-1: #2563eb;
            --ctx-cat-2: #0891b2;
            --ctx-cat-3: #7c3aed;
            --ctx-cat-4: #ea580c;
            --ctx-cat-5: #dc2626;
            --ctx-cat-6: #0d9488;
            --ctx-cat-7: #16a34a;
            --ctx-cat-8: #d97706;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --ink-900: #dbeafe;
                --ink-700: #bfdbfe;
                --ink-500: #93c5fd;
                --sky-50: #0b1425;
                --sky-100: #122038;
                --sky-200: #213654;
                --line-soft: #2b3f60;
                --paper: #0a1221;
                --success: #86efac;
                --warn: #fdba74;
                --badge-high-bg: #052e1a;
                --badge-high-line: #166534;
                --badge-mid-bg: #422006;
                --badge-mid-line: #92400e;
                --badge-low-bg: #0c2142;
                --badge-low-line: #1d4e89;
                --workspace-grad-start: #0f1a2d;
                --workspace-grad-end: #172640;
                --workspace-title: #dbeafe;
                --workspace-chip-bg: #0a1221;
                --workspace-chip-line: #365077;
                --workspace-hint-bg: #0f1d34;
                --workspace-card-title: #e2edff;
                --workspace-card-sub: #9fbee8;
                --trace-item-bg: #0f1d34;
                --trace-item-line: #2f476a;
                --trace-meta-bg: #1a2d49;
                --trace-content: #d2e6ff;
                --ctx-card-bg: #0f1d34;
                --ctx-card-line: #2f476a;
                --ctx-kpi-bg: #0a1221;
                --ctx-kpi-line: #2f476a;
                --ctx-bar-bg: #1c2f4c;
                --ctx-bar-fill-start: #60a5fa;
                --ctx-bar-fill-end: #93c5fd;
                --ctx-bar-free-start: #22c55e;
                --ctx-bar-free-end: #86efac;
                --ctx-cat-1: #60a5fa;
                --ctx-cat-2: #22d3ee;
                --ctx-cat-3: #a78bfa;
                --ctx-cat-4: #fb923c;
                --ctx-cat-5: #f87171;
                --ctx-cat-6: #2dd4bf;
                --ctx-cat-7: #86efac;
                --ctx-cat-8: #fbbf24;
            }
        }
        .llm-section-card {
            background: var(--paper);
            border: 1px solid var(--line-soft);
            border-radius: 14px;
            padding: 12px 14px;
            margin: 8px 0 12px 0;
        }
        .llm-section-card h3, .llm-section-card h4 {
            color: var(--ink-900);
            margin-top: 0;
        }
        .llm-muted {
            color: var(--ink-700);
            font-size: 13px;
        }
        .llm-chip-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .llm-chip {
            background: var(--sky-50);
            border: 1px solid var(--sky-200);
            color: var(--ink-700);
            border-radius: 999px;
            padding: 3px 10px;
            font-size: 12px;
            line-height: 1.4;
        }
        .llm-badge-high {
            background: var(--badge-high-bg);
            border: 1px solid var(--badge-high-line);
            color: var(--success);
            border-radius: 999px;
            padding: 2px 8px;
            font-size: 11px;
            font-weight: 600;
        }
        .llm-badge-mid {
            background: var(--badge-mid-bg);
            border: 1px solid var(--badge-mid-line);
            color: var(--warn);
            border-radius: 999px;
            padding: 2px 8px;
            font-size: 11px;
            font-weight: 600;
        }
        .llm-badge-low {
            background: var(--badge-low-bg);
            border: 1px solid var(--badge-low-line);
            color: var(--ink-700);
            border-radius: 999px;
            padding: 2px 8px;
            font-size: 11px;
            font-weight: 600;
        }
        .llm-trace-scroll {
            max-height: min(48vh, 460px);
            overflow-y: auto;
            padding-right: 4px;
        }
        .llm-trace-item {
            background: var(--trace-item-bg);
            border: 1px solid var(--trace-item-line);
            border-radius: 10px;
            padding: 8px 10px;
            margin-bottom: 8px;
        }
        .llm-trace-item-head {
            background: var(--trace-meta-bg);
            border-radius: 999px;
            display: inline-block;
            font-size: 12px;
            padding: 2px 10px;
            margin-bottom: 6px;
            color: var(--ink-700);
        }
        .llm-trace-item-content {
            color: var(--trace-content);
            font-size: 13px;
            line-height: 1.45;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .llm-context-card {
            background: var(--ctx-card-bg);
            border: 1px solid var(--ctx-card-line);
            border-radius: 12px;
            padding: 10px 12px;
            margin: 6px 0 10px 0;
        }
        .llm-context-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }
        .llm-context-title {
            color: var(--ink-900);
            font-weight: 700;
            font-size: 13px;
        }
        .llm-context-health {
            border-radius: 999px;
            padding: 2px 9px;
            font-size: 11px;
            border: 1px solid var(--ctx-kpi-line);
            color: var(--ink-700);
            background: var(--ctx-kpi-bg);
            font-weight: 600;
            white-space: nowrap;
        }
        .llm-context-health-warn {
            border-color: var(--badge-mid-line);
            color: var(--warn);
            background: var(--badge-mid-bg);
        }
        .llm-context-health-risk {
            border-color: #ef4444;
            color: #dc2626;
            background: #fee2e2;
        }
        .llm-context-kpi-row {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
        }
        .llm-context-kpi {
            background: var(--ctx-kpi-bg);
            border: 1px solid var(--ctx-kpi-line);
            border-radius: 10px;
            padding: 6px 8px;
        }
        .llm-context-kpi .k {
            color: var(--ink-700);
            font-size: 11px;
        }
        .llm-context-kpi .v {
            color: var(--ink-900);
            font-size: 13px;
            font-weight: 700;
            margin-top: 2px;
        }
        .llm-context-item {
            margin-bottom: 10px;
        }
        .llm-context-item-head {
            display: flex;
            justify-content: space-between;
            gap: 8px;
            font-size: 12px;
            color: var(--ink-700);
            margin-bottom: 4px;
        }
        .llm-context-item-head .name {
            color: var(--ink-900);
            font-weight: 600;
        }
        .llm-context-matrix {
            border-radius: 10px;
            background: var(--ctx-bar-bg);
            border: 1px solid var(--ctx-kpi-line);
            padding: 6px;
            display: grid;
            grid-template-columns: repeat(20, minmax(0, 1fr));
            gap: 2px;
            margin-bottom: 10px;
        }
        .llm-context-cell {
            aspect-ratio: 1 / 1;
            border-radius: 2px;
            background: transparent;
        }
        .llm-context-cell.empty {
            box-shadow: inset 0 0 0 1px var(--ctx-kpi-line);
        }
        .llm-context-legend-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 6px;
            font-size: 12px;
            color: var(--ink-700);
        }
        .llm-context-legend-left {
            display: flex;
            align-items: center;
            gap: 6px;
            min-width: 0;
        }
        .llm-context-dot {
            width: 9px;
            height: 9px;
            border-radius: 999px;
            flex-shrink: 0;
        }
        .llm-context-legend-label {
            color: var(--ink-900);
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .llm-context-legend-value {
            white-space: nowrap;
        }
        @media (max-width: 768px) {
            .llm-section-card {
                padding: 10px 10px;
                border-radius: 10px;
            }
            .llm-chip {
                font-size: 11px;
                padding: 2px 8px;
            }
            .llm-trace-scroll {
                max-height: min(42vh, 360px);
            }
            .llm-trace-item-content {
                font-size: 12px;
            }
            .llm-context-kpi-row {
                grid-template-columns: 1fr;
            }
            .llm-context-matrix {
                grid-template-columns: repeat(15, minmax(0, 1fr));
            }
            .llm-context-legend-item {
                font-size: 11px;
            }
        }
        </style>
    """,
        unsafe_allow_html=True,
    )
