# mindmap-cli

A small Go CLI that renders skill-generated mindmap JSON into an interactive HTML page.

## Input JSON contract

The CLI expects the same structure produced by the `mindmap` skill:

```json
{
  "name": "Root",
  "children": [
    {
      "name": "Branch",
      "children": [
        {"name": "Leaf", "children": []}
      ]
    }
  ]
}
```

## Build binary

```bash
cd tools/mindmap-cli
go build -o ../../bin/mindmap-cli .
```

Optional cross-build examples:

```bash
GOOS=linux GOARCH=amd64 go build -o ../../bin/mindmap-cli-linux-amd64 .
GOOS=windows GOARCH=amd64 go build -o ../../bin/mindmap-cli-windows-amd64.exe .
```

## Usage

From file:

```bash
./bin/mindmap-cli -in ./mindmap.json -out ./mindmap.html -title "Paper Mind Map"
```

From stdin:

```bash
cat ./mindmap.json | ./bin/mindmap-cli -in - -out ./mindmap.html -theme forest
```

Open `mindmap.html` in browser or embed it in a WebView component.

## Flags

- `-in`: JSON input path, `-` means stdin
- `-out`: HTML output path, `-` means stdout
- `-title`: header title in HTML
- `-width`: panel width
- `-height`: chart height
- `-theme`: `ocean | sunrise | forest`
- `-echarts-url`: ECharts script URL
