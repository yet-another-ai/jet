"""Render the documented MMLU measurements as a paired native-TikZ dot plot."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = [r for r in json.loads((ROOT / 'mmlu-results.json').read_text())
        if r.get('show_in_figure', False)]


def throughput(row):
    if row.get('wall_s') is not None:
        return row['requests'] / row['wall_s']
    return row.get('req_s')


rates = [throughput(r) or 0 for r in rows]
rate_max = max(12, 4 * math.ceil(max(rates) / 4))
lines = [r'\begin{figure}[H]', r'\centering',
         r'\begin{tikzpicture}[x=1cm,y=1cm,font=\footnotesize]',
         r'\node[anchor=west,font=\small\bfseries] at (0,0.8) {Model / deployment};',
         r'\node[font=\small\bfseries] at (8.3,0.8) {MMLU accuracy (\%)};',
         r'\node[font=\small\bfseries] at (13.1,0.8) {Throughput (req/s)};']
for value in (0, 20, 40, 60, 80, 100):
    x = 6.3 + 3.6 * value / 100
    lines.append(rf'\node[font=\scriptsize,text=black!65] at ({x},0.3) {{{value}}};')
for value in range(0, rate_max + 1, 4):
    x = 11.5 + 3.1 * value / rate_max
    lines.append(rf'\node[font=\scriptsize,text=black!65] at ({x},0.3) {{{value}}};')
y = -0.2
last_group = None
for r in rows:
    if r['group'] != last_group:
        if last_group is not None:
            y = round(y - 0.23, 2)
        lines.append(rf'\node[anchor=west,font=\footnotesize\bfseries] at (0,{y}) {{{r["group"]}}};')
        last_group = r['group']
        y = round(y - 0.43, 2)
    color = ('green!55!black' if r.get('fine_tuned') else
             'blue!65!black' if r.get('deployment') == 'a770' else
             'violet!75!black' if r.get('backend') == 'CUDA' else
             {'mmlu': 'teal!75!black', 'api': 'orange!80!black'}[r['scope']])
    deployment_label = r['figure_label']
    if r['scope'] == 'mmlu':
        deployment_label += f" ({r['backend']})"
    lines.append(rf'\node[anchor=west] at (0.1,{y}) {{{deployment_label}}};')
    lines.append(rf'\draw[black!12] (6.3,{y}) -- (9.9,{y});')
    lines.append(rf'\draw[black!12] (11.5,{y}) -- (14.6,{y});')
    marker = 'draw' if r.get('fine_tuned') else 'fill'
    size = '2.8pt' if r.get('fine_tuned') else '2pt'
    if r['accuracy'] is not None:
        x = 6.3 + 3.6 * r['accuracy'] / 100
        lines.append(rf'\{marker}[{color}] ({x:.4f},{y}) circle ({size});')
        accuracy_label = f'{r["accuracy"]:.2f}'
    else:
        accuracy_label = 'pending'
    lines.append(rf'\node[anchor=east] at (10.8,{y}) {{{accuracy_label}}};')
    rate = throughput(r)
    if rate is not None:
        x = 11.5 + 3.1 * rate / rate_max
        lines.append(rf'\{marker}[{color}] ({x:.4f},{y}) circle ({size});')
        label = ('\\(\\sim\\)' if r.get('approximate') else '') + f'{rate:.2f}'
    else:
        label = 'pending' if r.get('status') == 'pending' else '---'
    lines.append(rf'\node[anchor=east] at (15.5,{y}) {{{label}}};')
    y = round(y - 0.43, 2)
lines += [r'\end{tikzpicture}',
          r'\caption{Full-MMLU accuracy and throughput with reasoning disabled and mean token log-probability scoring for JET. Teal: RTX 4090 (Vulkan); violet: RTX 4090 (CUDA); green outlined: Qwen3.5-4B LoRA (CUDA); orange: Jev 1.13 hosted API. Each JET point combines accuracy and complete-process throughput from the same completed 14,042-question run. Arc A770 reruns remain pending and are excluded. Jev uses full-MMLU accuracy and a separately reported API rate. Each metric uses a common horizontal scale.}',
          r'\label{fig:mmlu-throughput}', r'\end{figure}']
(ROOT / 'mmlu-throughput.tex').write_text('\n'.join(lines) + '\n')
