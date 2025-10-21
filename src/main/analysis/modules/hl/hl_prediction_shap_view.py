# pip install shap html2image  # 画像化まで行う場合
import numpy as np
import shap
import json

from src.modules.helper.helper import Helper

dir = Helper.ROOT / "output" / "shap" / "hl" / "EXP2" / "basic_version_version_6"
with open(dir / "shapv_schwanhausser_580.json", mode="r") as f:
    result = json.load(f)

html_items: list[str] = []

maxv = max(result["shap_values"])
minv = min(result["shap_values"])


def get_color(x):
    w = 90
    if x > 0:
        s = min(x / maxv, 1) * 100
        return (100, 100 - s, 100 - s)  # 赤成分を強く
    elif x < 0:
        s = min(abs(x) / abs(minv), 1) * 100
        return (100 - s, 100 - s, 100)  # 青成分を強く
    else:
        return (100, 100, 100)  # 白（ゼロ）


for token, sv in zip(result["tokens"], result["shap_values"]):
    clr = get_color(sv)
    item = f"<span class='char' style='background-color: rgb({clr[0]}% {clr[1]}% {clr[2]}% / 70%);'>{token}</span>"
    html_items.append(item)

item = "".join(html_items)
HTML = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
        <link rel="stylesheet" href="test.css">
    </head>
    <body>
        <div class='view'>
            {item}
        </div>
    </body>
    </html>
"""

with open("test.html", mode="w") as f:
    f.write(HTML)
