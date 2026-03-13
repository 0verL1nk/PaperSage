package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html/template"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type node struct {
	Name     string  `json:"name"`
	Children []*node `json:"children"`
}

type themeConfig struct {
	Name            string
	BodyStart       string
	BodyEnd         string
	PanelBackground string
	PanelBorder     string
	TitleColor      string
	SubTitleColor   string
	LinkColor       string
	NodeBorder      string
	GridGlow        string
	Palette         []string
}

type cliConfig struct {
	InputPath  string
	OutputPath string
	Title      string
	Width      int
	Height     int
	Theme      string
	EchartsURL string
}

type pageData struct {
	Title            string
	Width            int
	Height           int
	EchartsURL       string
	GeneratedAt      string
	Theme            themeConfig
	CSS              cssTheme
	PaletteJSON      template.JS
	MindmapDataJSON  template.JS
	InitialTreeDepth int
}

type cssTheme struct {
	BodyStart       template.CSS
	BodyEnd         template.CSS
	PanelBackground template.CSS
	PanelBorder     template.CSS
	TitleColor      template.CSS
	SubTitleColor   template.CSS
	GridGlow        template.CSS
}

func main() {
	cfg := parseFlags()
	if err := run(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func parseFlags() cliConfig {
	var cfg cliConfig
	flag.StringVar(&cfg.InputPath, "in", "-", "mindmap JSON input path; use '-' for stdin")
	flag.StringVar(&cfg.OutputPath, "out", "mindmap.html", "HTML output path; use '-' for stdout")
	flag.StringVar(&cfg.Title, "title", "Mind Map", "chart title shown in HTML")
	flag.IntVar(&cfg.Width, "width", 1360, "canvas width in pixels")
	flag.IntVar(&cfg.Height, "height", 860, "canvas height in pixels")
	flag.StringVar(&cfg.Theme, "theme", "ocean", "theme: ocean | sunrise | forest")
	flag.StringVar(
		&cfg.EchartsURL,
		"echarts-url",
		"https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js",
		"ECharts JS URL",
	)
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Mindmap Renderer CLI\n\n")
		fmt.Fprintf(flag.CommandLine.Output(), "Render skill-generated mindmap JSON into interactive HTML.\n\n")
		fmt.Fprintf(flag.CommandLine.Output(), "Usage:\n")
		fmt.Fprintf(flag.CommandLine.Output(), "  mindmap-cli -in data.json -out mindmap.html\n")
		fmt.Fprintf(flag.CommandLine.Output(), "  cat data.json | mindmap-cli -in - -out mindmap.html -theme forest\n\n")
		fmt.Fprintf(flag.CommandLine.Output(), "Flags:\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	return cfg
}

func run(cfg cliConfig) error {
	if cfg.Width < 480 || cfg.Height < 360 {
		return errors.New("width/height are too small; use at least 480x360")
	}

	theme, err := loadTheme(cfg.Theme)
	if err != nil {
		return err
	}

	inputBytes, err := readInput(cfg.InputPath)
	if err != nil {
		return err
	}
	root, err := parseMindmapJSON(inputBytes)
	if err != nil {
		return err
	}

	rootJSON, err := json.Marshal(root)
	if err != nil {
		return fmt.Errorf("marshal mindmap: %w", err)
	}
	paletteJSON, err := json.Marshal(theme.Palette)
	if err != nil {
		return fmt.Errorf("marshal palette: %w", err)
	}

	payload := pageData{
		Title:       cfg.Title,
		Width:       cfg.Width,
		Height:      cfg.Height,
		EchartsURL:  cfg.EchartsURL,
		GeneratedAt: time.Now().Format("2006-01-02 15:04:05"),
		Theme:       theme,
		CSS: cssTheme{
			BodyStart:       template.CSS(theme.BodyStart),
			BodyEnd:         template.CSS(theme.BodyEnd),
			PanelBackground: template.CSS(theme.PanelBackground),
			PanelBorder:     template.CSS(theme.PanelBorder),
			TitleColor:      template.CSS(theme.TitleColor),
			SubTitleColor:   template.CSS(theme.SubTitleColor),
			GridGlow:        template.CSS(theme.GridGlow),
		},
		PaletteJSON:      template.JS(paletteJSON),
		MindmapDataJSON:  template.JS(rootJSON),
		InitialTreeDepth: 3,
	}

	htmlBytes, err := renderPage(payload)
	if err != nil {
		return err
	}
	if err := writeOutput(cfg.OutputPath, htmlBytes); err != nil {
		return err
	}

	return nil
}

func loadTheme(name string) (themeConfig, error) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "", "ocean":
		return themeConfig{
			Name:            "ocean",
			BodyStart:       "#0f172a",
			BodyEnd:         "#0b1120",
			PanelBackground: "rgba(15, 23, 42, 0.72)",
			PanelBorder:     "rgba(56, 189, 248, 0.32)",
			TitleColor:      "#e2e8f0",
			SubTitleColor:   "#94a3b8",
			LinkColor:       "#38bdf8",
			NodeBorder:      "#7dd3fc",
			GridGlow:        "0 0 0 1px rgba(56, 189, 248, 0.2), 0 30px 60px rgba(2, 6, 23, 0.55)",
			Palette:         []string{"#38bdf8", "#22d3ee", "#34d399", "#a3e635", "#facc15", "#f97316"},
		}, nil
	case "sunrise":
		return themeConfig{
			Name:            "sunrise",
			BodyStart:       "#2a1a1f",
			BodyEnd:         "#1f1420",
			PanelBackground: "rgba(42, 26, 31, 0.76)",
			PanelBorder:     "rgba(251, 146, 60, 0.35)",
			TitleColor:      "#fff7ed",
			SubTitleColor:   "#fed7aa",
			LinkColor:       "#fb923c",
			NodeBorder:      "#fdba74",
			GridGlow:        "0 0 0 1px rgba(251, 146, 60, 0.2), 0 30px 60px rgba(17, 24, 39, 0.55)",
			Palette:         []string{"#fb923c", "#f97316", "#ef4444", "#f59e0b", "#facc15", "#22c55e"},
		}, nil
	case "forest":
		return themeConfig{
			Name:            "forest",
			BodyStart:       "#0b2219",
			BodyEnd:         "#071712",
			PanelBackground: "rgba(11, 34, 25, 0.76)",
			PanelBorder:     "rgba(74, 222, 128, 0.32)",
			TitleColor:      "#ecfdf5",
			SubTitleColor:   "#86efac",
			LinkColor:       "#4ade80",
			NodeBorder:      "#86efac",
			GridGlow:        "0 0 0 1px rgba(74, 222, 128, 0.2), 0 30px 60px rgba(3, 7, 18, 0.6)",
			Palette:         []string{"#4ade80", "#34d399", "#2dd4bf", "#22c55e", "#84cc16", "#facc15"},
		}, nil
	default:
		return themeConfig{}, fmt.Errorf("unsupported theme %q (use ocean|sunrise|forest)", name)
	}
}

func readInput(path string) ([]byte, error) {
	if strings.TrimSpace(path) == "" || path == "-" {
		payload, err := io.ReadAll(os.Stdin)
		if err != nil {
			return nil, fmt.Errorf("read stdin: %w", err)
		}
		return payload, nil
	}

	payload, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read input file %q: %w", path, err)
	}
	return payload, nil
}

func parseMindmapJSON(payload []byte) (*node, error) {
	payload = bytes.TrimSpace(payload)
	if len(payload) == 0 {
		return nil, errors.New("input JSON is empty")
	}

	var root node
	if err := json.Unmarshal(payload, &root); err != nil {
		return nil, fmt.Errorf("parse input JSON: %w", err)
	}
	if err := normalizeNode(&root, "root"); err != nil {
		return nil, err
	}
	return &root, nil
}

func normalizeNode(n *node, path string) error {
	if n == nil {
		return fmt.Errorf("%s is null", path)
	}
	n.Name = strings.TrimSpace(n.Name)
	if n.Name == "" {
		return fmt.Errorf("%s.name is empty", path)
	}
	if n.Children == nil {
		n.Children = []*node{}
	}
	for idx, child := range n.Children {
		childPath := fmt.Sprintf("%s.children[%d]", path, idx)
		if err := normalizeNode(child, childPath); err != nil {
			return err
		}
	}
	return nil
}

func writeOutput(path string, payload []byte) error {
	if strings.TrimSpace(path) == "" || path == "-" {
		_, err := os.Stdout.Write(payload)
		return err
	}

	dir := filepath.Dir(path)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create output directory %q: %w", dir, err)
		}
	}
	if err := os.WriteFile(path, payload, 0o644); err != nil {
		return fmt.Errorf("write output file %q: %w", path, err)
	}
	return nil
}

func renderPage(data pageData) ([]byte, error) {
	tmpl, err := template.New("mindmap").Parse(pageTemplate)
	if err != nil {
		return nil, fmt.Errorf("parse HTML template: %w", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return nil, fmt.Errorf("render HTML template: %w", err)
	}
	return buf.Bytes(), nil
}

const pageTemplate = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ .Title }}</title>
  <script src="{{ .EchartsURL }}"></script>
  <style>
    :root {
      --body-start: {{ .CSS.BodyStart }};
      --body-end: {{ .CSS.BodyEnd }};
      --panel-bg: {{ .CSS.PanelBackground }};
      --panel-border: {{ .CSS.PanelBorder }};
      --title-color: {{ .CSS.TitleColor }};
      --subtitle-color: {{ .CSS.SubTitleColor }};
      --grid-glow: {{ .CSS.GridGlow }};
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
      color: var(--title-color);
      background:
        radial-gradient(1200px 600px at 20% -20%, rgba(255, 255, 255, 0.07), transparent 60%),
        radial-gradient(900px 500px at 90% 120%, rgba(255, 255, 255, 0.06), transparent 60%),
        linear-gradient(140deg, var(--body-start), var(--body-end));
      font-family: "Segoe UI", "Noto Sans SC", "Microsoft YaHei", sans-serif;
    }
    .panel {
      width: min(96vw, {{ .Width }}px);
      border-radius: 18px;
      border: 1px solid var(--panel-border);
      background: var(--panel-bg);
      box-shadow: var(--grid-glow);
      backdrop-filter: blur(8px);
      overflow: hidden;
    }
    .panel-header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      padding: 14px 18px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.22);
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0));
    }
    .panel-title {
      margin: 0;
      font-size: 19px;
      letter-spacing: 0.2px;
      color: var(--title-color);
    }
    .panel-subtitle {
      margin: 2px 0 0;
      color: var(--subtitle-color);
      font-size: 12px;
    }
    #mindmap {
      width: 100%;
      height: {{ .Height }}px;
    }
    .pill {
      font-size: 11px;
      letter-spacing: 0.4px;
      text-transform: uppercase;
      color: var(--subtitle-color);
      border: 1px solid rgba(148, 163, 184, 0.4);
      border-radius: 999px;
      padding: 4px 8px;
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .fullscreen-btn {
      border: 1px solid rgba(148, 163, 184, 0.45);
      border-radius: 8px;
      padding: 4px 10px;
      font-size: 12px;
      color: var(--title-color);
      background: rgba(148, 163, 184, 0.15);
      cursor: pointer;
    }
    .fullscreen-btn:hover {
      background: rgba(148, 163, 184, 0.24);
    }
    .panel:fullscreen {
      width: 100vw;
      height: 100vh;
      border-radius: 0;
      border: none;
    }
    .panel:fullscreen #mindmap {
      height: calc(100vh - 58px);
    }
    @media (prefers-color-scheme: light) {
      body {
        color: #0f172a;
        background:
          radial-gradient(1200px 600px at 20% -20%, rgba(99, 102, 241, 0.08), transparent 60%),
          radial-gradient(900px 500px at 90% 120%, rgba(14, 165, 233, 0.08), transparent 60%),
          linear-gradient(140deg, #f8fafc, #e2e8f0);
      }
      .panel {
        background: rgba(255, 255, 255, 0.92);
        border-color: rgba(99, 102, 241, 0.22);
        box-shadow: 0 0 0 1px rgba(99, 102, 241, 0.12), 0 30px 60px rgba(15, 23, 42, 0.12);
      }
      .panel-title {
        color: #0f172a;
      }
      .panel-subtitle,
      .pill {
        color: #475569;
      }
      .pill {
        border-color: rgba(99, 102, 241, 0.35);
      }
    }
  </style>
</head>
<body>
  <section class="panel">
    <header class="panel-header">
      <div>
        <h1 class="panel-title">{{ .Title }}</h1>
        <p class="panel-subtitle">Drag to pan, wheel to zoom, click nodes to expand/collapse.</p>
      </div>
      <div class="header-actions">
        <span class="pill">{{ .Theme.Name }}</span>
        <button id="fullscreen-toggle" type="button" class="fullscreen-btn">全屏</button>
      </div>
    </header>
    <div id="mindmap"></div>
  </section>

  <script>
    const root = {{ .MindmapDataJSON }};
    const palette = {{ .PaletteJSON }};

    function styleTree(node, depth) {
      const color = palette[depth % palette.length];
      node.itemStyle = {
        color: color,
        borderColor: "{{ .Theme.NodeBorder }}",
        borderWidth: depth === 0 ? 2 : 1
      };
      node.lineStyle = {
        color: "{{ .Theme.LinkColor }}",
        width: Math.max(1.2, 3.2 - depth * 0.5),
        curveness: 0.25
      };
      node.label = {
        color: "{{ .Theme.TitleColor }}",
        fontSize: depth === 0 ? 16 : 12,
        backgroundColor: "rgba(15, 23, 42, 0.32)",
        borderRadius: 6,
        padding: [4, 7]
      };
      if (Array.isArray(node.children)) {
        node.children.forEach((child) => styleTree(child, depth + 1));
      }
      return node;
    }

    const chartRoot = document.getElementById("mindmap");
    const chart = echarts.init(chartRoot, null, { renderer: "canvas" });
    const panel = document.querySelector(".panel");
    const fullscreenBtn = document.getElementById("fullscreen-toggle");
    const colorSchemeQuery = window.matchMedia("(prefers-color-scheme: dark)");

    function cloneRootData() {
      if (typeof structuredClone === "function") {
        return structuredClone(root);
      }
      return JSON.parse(JSON.stringify(root));
    }

    function buildOption() {
      const isDark = colorSchemeQuery.matches;
      const labelColor = isDark ? "{{ .Theme.TitleColor }}" : "#0f172a";
      const labelBg = isDark ? "rgba(15, 23, 42, 0.32)" : "rgba(255, 255, 255, 0.85)";
      const linkColor = isDark ? "{{ .Theme.LinkColor }}" : "#6366f1";

      function themedTree(node, depth) {
        const styled = styleTree(node, depth);
        styled.lineStyle = {
          color: linkColor,
          width: Math.max(1.2, 3.2 - depth * 0.5),
          curveness: 0.25,
        };
        styled.label = {
          color: labelColor,
          fontSize: depth === 0 ? 16 : 12,
          backgroundColor: labelBg,
          borderRadius: 6,
          padding: [4, 7],
        };
        return styled;
      }

      return {
        animationDuration: 650,
        animationDurationUpdate: 420,
        tooltip: {
          trigger: "item",
          triggerOn: "mousemove",
          confine: true,
          formatter: (params) => params?.data?.name || "",
        },
        toolbox: {
          show: true,
          orient: "horizontal",
          right: 18,
          top: 12,
          itemSize: 16,
          feature: {
            restore: {},
            saveAsImage: { title: "Save PNG", pixelRatio: 2 },
          },
        },
        series: [
          {
            type: "tree",
            data: [themedTree(cloneRootData(), 0)],
            top: "8%",
            bottom: "5%",
            left: "4%",
            right: "30%",
            orient: "LR",
            layout: "orthogonal",
            edgeForkPosition: "10%",
            symbol: "roundRect",
            symbolSize: 10,
            roam: true,
            expandAndCollapse: true,
            initialTreeDepth: {{ .InitialTreeDepth }},
            emphasis: {
              focus: "descendant",
            },
            lineStyle: {
              color: linkColor,
              width: 2,
            },
            label: {
              position: "right",
              verticalAlign: "middle",
              align: "left",
            },
            leaves: {
              label: {
                position: "right",
                verticalAlign: "middle",
                align: "left",
              },
            },
          },
        ],
      };
    }

    function renderChart() {
      chart.setOption(buildOption(), true);
      chart.resize();
    }

    function toggleFullscreen() {
      if (!document.fullscreenElement) {
        panel.requestFullscreen?.();
      } else {
        document.exitFullscreen?.();
      }
    }

    function syncFullscreenButton() {
      const isFull = document.fullscreenElement === panel;
      fullscreenBtn.textContent = isFull ? "退出全屏" : "全屏";
      chart.resize();
    }

    fullscreenBtn.addEventListener("click", toggleFullscreen);
    document.addEventListener("fullscreenchange", syncFullscreenButton);
    colorSchemeQuery.addEventListener("change", renderChart);
    renderChart();
    window.addEventListener("resize", () => chart.resize());
    console.info("mindmap rendered at {{ .GeneratedAt }}");
  </script>
</body>
</html>
`
