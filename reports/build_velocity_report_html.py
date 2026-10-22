"""Convert the verified report to a self-contained, offline HTML document."""
from base64 import b64encode
from html import escape
from pathlib import Path
import re

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent
source = ROOT / 'ADAPT_velocity_sensitivity_report_2026-10-21.docx'
output = source.with_suffix('.html')
doc = Document(source)
parts, navigation = [], []
image_count = 0
table_count = 0
equation = '''<div class="equation" role="img" aria-label="평면 속도 RMSE는 전진 오차 제곱과 좌우 오차 제곱의 합을 시간 평균한 값의 제곱근입니다">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
<msub><mi mathvariant="normal">RMSE</mi><mi>xy</mi></msub><mo>=</mo><msqrt>
<mfrac><mn>1</mn><mi>N</mi></mfrac>
<munderover><mo>∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover>
<mrow><mo>[</mo><msup><mrow><msub><mi>e</mi><mi>x</mi></msub><mo>(</mo><mi>t</mi><mo>)</mo></mrow><mn>2</mn></msup>
<mo>+</mo><msup><mrow><msub><mi>e</mi><mi>y</mi></msub><mo>(</mo><mi>t</mi><mo>)</mo></mrow><mn>2</mn></msup><mo>]</mo></mrow>
</msqrt></math></div>'''

for element in doc.element.body:
    if element.tag == qn('w:p'):
        para = Paragraph(element, doc)
        if element.xpath('.//m:oMath'):
            parts.append(equation)
            continue
        blips = element.xpath('.//a:blip')
        if blips:
            for blip in blips:
                image = doc.part.related_parts[blip.get(qn('r:embed'))]
                encoded = b64encode(image.blob).decode('ascii')
                parts.append(f'<figure><img src="data:{image.content_type};base64,{encoded}" '
                             'alt="선속도 노이즈와 지연에 따른 평면 속도 벡터 RMSE 및 yaw rate RMSE 그래프. 오차 막대는 trial 간 표준편차입니다."></figure>')
                image_count += 1
            continue
        text = para.text.strip()
        if not text:
            continue
        content = escape(text).replace('\n', '<br>')
        style = para.style.name
        if style == 'Title':
            parts.append(f'<h1>{content}</h1>')
        elif style == 'Heading 1':
            anchor = 'section-' + str(len(navigation) + 1)
            navigation.append((anchor, text))
            parts.append(f'<h2 id="{anchor}">{content}</h2>')
        elif style == 'Heading 2':
            parts.append(f'<h3>{content}</h3>')
        elif text.startswith('CHECKPOINT='):
            parts.append(f'<pre aria-label="평가 재현 명령"><code>{escape(text)}</code></pre>')
        else:
            css = {'Subtitle': 'subtitle', 'Caption': 'caption'}.get(style, '')
            if para.runs and all(r.bold for r in para.runs if r.text.strip()):
                content = f'<strong>{content}</strong>'
            parts.append(f'<p class="{css}">{content}</p>')
    elif element.tag == qn('w:tbl'):
        table_count += 1
        table = Table(element, doc)
        body = []
        for index, row in enumerate(table.rows):
            tag = 'th' if index == 0 else 'td'
            scope = ' scope="col"' if index == 0 else ''
            cells = ''.join(f'<{tag}{scope}>{escape(cell.text).replace(chr(10), "<br>")}</{tag}>' for cell in row.cells)
            body.append(f'<tr>{cells}</tr>')
        cls = 'settings' if table_count == 1 else 'numeric'
        parts.append(f'<div class="table-wrap" tabindex="0" role="region" aria-label="보고서 데이터 표 {table_count}">'
                     f'<table class="{cls}"><thead>{body[0]}</thead><tbody>{"".join(body[1:])}</tbody></table></div>')

assert image_count == 1 and table_count == 4 and len(navigation) == 7
nav = ''.join(f'<a href="#{anchor}">{escape(title)}</a>' for anchor, title in navigation)
template = '''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="ADAPT 선속도 입력 노이즈와 지연에 대한 750회 평가 보고서. actor와 observer의 공유 오차 및 평면 속도 벡터 RMSE 분석.">
<title>ADAPT 선속도 입력 노이즈와 지연에 따른 보행 성능 평가</title>
<style>
:root{color-scheme:light;--ink:#17212c;--muted:#536171;--line:#d9e1e8;--blue:#17557c}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:32px}
body{margin:0;background:#f4f6f8;color:var(--ink);font-family:"Noto Sans CJK KR","Noto Sans KR","Malgun Gothic",sans-serif;font-size:16px;line-height:1.85;word-break:keep-all}
a{color:var(--blue)}.layout{max-width:1330px;margin:auto;display:grid;grid-template-columns:238px minmax(0,1fr);gap:34px;padding:42px 28px}
aside{align-self:start;position:sticky;top:30px;padding:8px 0}.brand{font-size:18px;font-weight:750;margin-bottom:4px}.small{font-size:12px;color:var(--muted)}
nav{margin:26px 0}nav a{display:block;padding:8px 0;font-size:13px;text-decoration:none;line-height:1.5}nav a:hover{text-decoration:underline}
button{font:inherit;font-size:13px;padding:9px 16px;background:white;border:1px solid var(--line);border-radius:6px;color:var(--ink);cursor:pointer}
button:hover{border-color:var(--blue)}a:focus-visible,button:focus-visible,.table-wrap:focus-visible{outline:3px solid #8ac5e6;outline-offset:4px}
main{background:#fff;min-width:0;padding:48px 48px 40px;border:1px solid #e4e9ee;border-radius:8px}
h1,h2,h3{color:#111;line-height:1.35;letter-spacing:-.035em}h1{font-size:32px;margin:0 0 18px;font-weight:750}h2{font-size:23px;margin:42px 0 18px}h3{font-size:19px;margin:28px 0 12px}
p{margin:0 0 17px;overflow-wrap:anywhere}strong{font-weight:700}.subtitle{font-size:18px;line-height:1.65;margin-bottom:16px;color:#3f4e5c}
.caption{font-size:12.5px;color:var(--muted);line-height:1.7;margin:12px 0 20px}.table-wrap{overflow:auto;margin:18px 0 10px;border:1px solid var(--line);border-radius:5px}
table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.65;font-variant-numeric:tabular-nums}th{background:#e8f0f6;color:#132e40;font-weight:650}
td,th{padding:11px 12px;border-bottom:1px solid var(--line);border-right:1px solid var(--line);vertical-align:middle;text-align:center}td:last-child,th:last-child{border-right:0}tbody tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#f8fafc}
.settings td:first-child{width:95px;white-space:nowrap}.settings td:nth-child(2){text-align:left}.numeric{min-width:550px}.numeric td{white-space:nowrap}
figure{margin:20px -10px 10px}img{width:100%;height:auto;display:block}.equation{padding:20px 0;margin:20px 0;font-size:22px;overflow:auto}math{font-family:"STIX Two Math","Cambria Math",serif}
pre{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;padding:18px;background:#f4f6f8;border:1px solid var(--line);border-radius:5px;font-size:12px;line-height:1.7}code{font-family:"DejaVu Sans Mono",Consolas,monospace}
footer{margin-top:32px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
@media(max-width:1050px){.layout{grid-template-columns:180px minmax(0,1fr);gap:20px;padding:26px 18px}main{padding:32px}h1{font-size:27px}}
@media(max-width:760px){.layout{display:block;padding:12px}aside{position:static;padding:12px 6px 22px}nav{display:none}.brand{display:inline-block;margin-right:14px}.small{display:inline}aside button{display:block;margin-top:12px}main{padding:25px 18px}h1{font-size:25px}h1 br{display:none}h2{font-size:21px}body{font-size:15px}.caption{font-size:12px}figure{margin-left:0;margin-right:0}}
@media print{@page{size:A4;margin:17mm}html{scroll-behavior:auto}body{background:white;font-size:10pt;line-height:1.6}.layout{display:block;padding:0;max-width:none}aside{display:none}main{padding:0;border:0;border-radius:0}h1{font-size:23pt}h2{font-size:16pt;margin-top:24pt;break-after:avoid}h3{font-size:12pt;break-after:avoid}p{orphans:3;widows:3}.subtitle{font-size:12pt}.caption{font-size:8pt}.table-wrap{overflow:visible;border-radius:0;break-inside:avoid}table{font-size:8pt;line-height:1.4}.numeric{min-width:0}td,th{padding:6pt}thead{display:table-header-group}tr,figure,.equation,pre{break-inside:avoid}figure{margin:12pt 0}.equation{font-size:16pt}pre{font-size:8pt}a{color:inherit;text-decoration:none}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
</style></head><body>
<div class="layout"><aside aria-label="보고서 목차"><div class="brand">ADAPT</div><div class="small">실험 보고서 · 2026.10.21</div>
<nav>__NAV__</nav><button type="button" onclick="window.print()">인쇄 및 PDF 저장</button></aside>
<main id="report">__BODY__<footer>저장된 750회 평가 결과를 재집계한 보고서입니다. 그래프를 포함한 모든 내용은 이 HTML 파일 안에 저장되어 있습니다.</footer></main></div>
</body></html>'''
result = template.replace('__NAV__', nav).replace('__BODY__', '\n'.join(parts))
assert not re.search(r'(?:src|href)=["\']https?://', result)
output.write_text(result, encoding='utf-8')
print(output)
