"""Render the documented MMLU measurements as a paired native-TikZ dot plot."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = [r for r in json.loads((ROOT / 'mmlu-results.json').read_text())
        if r.get('show_in_figure', False)]
lines = [r'\begin{figure}[H]', r'\centering',
         r'\begin{tikzpicture}[x=1cm,y=1cm,font=\footnotesize]',
         r'\node[anchor=west,font=\small\bfseries] at (0,0.8) {Model / deployment};',
         r'\node[font=\small\bfseries] at (8.3,0.8) {MMLU accuracy (\%)};',
         r'\node[font=\small\bfseries] at (13.1,0.8) {Throughput (req/s)};']
for value in (0, 20, 40, 60, 80, 100):
    x = 6.3 + 3.6 * value / 100
    lines.append(rf'\node[font=\scriptsize,text=black!65] at ({x},0.3) {{{value}}};')
for value in (0, 2, 4, 6, 8):
    x = 11.5 + 3.1 * value / 8
    lines.append(rf'\node[font=\scriptsize,text=black!65] at ({x},0.3) {{{value}}};')
y = -0.2
last_group = None
for r in rows:
    if r['group'] != last_group:
        if last_group is not None:
            y -= 0.23
        lines.append(rf'\node[anchor=west,font=\footnotesize\bfseries] at (0,{y}) {{{r["group"]}}};')
        last_group = r['group']
        y -= 0.43
    color = {'mixed': 'blue!65!black', 'mmlu': 'teal!75!black', 'api': 'orange!80!black'}[r['scope']]
    lines.append(rf'\node[anchor=west] at (0.1,{y}) {{{r["figure_label"]}}};')
    lines.append(rf'\draw[black!12] (6.3,{y}) -- (9.9,{y});')
    lines.append(rf'\draw[black!12] (11.5,{y}) -- (14.6,{y});')
    x = 6.3 + 3.6 * r['accuracy'] / 100
    lines.append(rf'\fill[{color}] ({x:.4f},{y}) circle (2pt);')
    lines.append(rf'\node[anchor=east] at (10.8,{y}) {{{r["accuracy"]:.2f}}};')
    rate = r.get('req_s')
    if rate is None and r.get('wall_s') is not None:
        rate = r['requests'] / r['wall_s']
    if rate is not None:
        x = 11.5 + 3.1 * rate / 8
        lines.append(rf'\fill[{color}] ({x:.4f},{y}) circle (2pt);')
        label = ('\\(\\sim\\)' if r.get('approximate') else '') + f'{rate:.2f}'
    else:
        label = '---'
    lines.append(rf'\node[anchor=east] at (15.5,{y}) {{{label}}};')
    y -= 0.43
lines += [r'\end{tikzpicture}',
          r'\caption{MMLU accuracy and throughput for the best-performing evaluated JET configuration per model and deployment, with reasoning disabled, and Jev 1.13. Blue: CPU/Arc A770. Teal: RTX 4090. Orange: hosted API. A770 expert offloading executes some model experts on CPU. JET CPU/A770 accuracy uses an MMLU subset; throughput covers the mixed BoolQ/MMLU workload. RTX 4090 accuracy uses full MMLU, while its throughput comes from a separate 285-question run of the same configuration. Jev uses full-MMLU accuracy and a separately reported API rate. Each metric uses a common horizontal scale.}',
          r'\label{fig:mmlu-throughput}', r'\end{figure}']
(ROOT / 'mmlu-throughput.tex').write_text('\n'.join(lines) + '\n')
