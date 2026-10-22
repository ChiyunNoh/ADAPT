"""Publish the Appendix D re-evaluation as a standalone Korean HTML report."""
from base64 import b64encode
from html import escape
from pathlib import Path
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'logs/evaluation/velocity_sensitivity_paper_metrics_2026-10-21'
OUT = ROOT / 'reports/ADAPT_velocity_sensitivity_report_2026-10-21.html'
OLD = OUT.with_name(OUT.stem + '_planar.html')
if not OLD.exists():
    shutil.copy2(OUT, OLD)
css = re.search(r'<style>(.*?)</style>', OLD.read_text(), re.S).group(1)
rows = json.loads((DATA / 'summary.json').read_text())
meta = json.loads((DATA / 'metadata.json').read_text())
by = {(r['noise_mps'], r['delay_ms']): r for r in rows}
keys = ['vx_rmse_mps', 'lateral_drift_m', 'yaw_drift_rad']
labels = ['Velocity error (m/s)', 'Lateral drift (m)', 'Yaw drift (rad)']


def table(headers, records):
    head = ''.join(f'<th scope="col">{escape(x)}</th>' for x in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(str(x))}</td>' for x in record) + '</tr>' for record in records)
    return f'<div class="table-wrap" tabindex="0" role="region" aria-label="평가 결과 표"><table class="numeric"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def image(name, alt):
    encoded = b64encode((DATA / name).read_bytes()).decode('ascii')
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{escape(alt)}"></figure>'


def fmt(row, key):
    val = row[key]
    return f"{val['mean']:.4f} ± {val['std']:.4f}"


selected = table(['지연 (ms)'] + labels, [
    [d] + [fmt(by[.05, d], k) for k in keys] for d in meta['delays_ms']])
full = table(['노이즈 ± (m/s)', '지연 (ms)'] + labels, [
    [r['noise_mps'], r['delay_ms']] + [fmt(r, k) for k in keys] for r in rows])
comparison = table(['항목', '이번 평가', '논문 Figure 3'], [
    ['변화시키는 조건', '선속도 입력 노이즈와 고정 지연', '몸통에 가하는 외력의 크기와 방향'],
    ['시간 구성', '30초 보행 중 첫 2초 제외', '1초 정지 안정화 후 10초 보행'],
    ['명령', 'vx = 0.8 m/s, vy = 0, yaw rate = 0', '전진 속도 0.5, 0.8, 1.1 m/s 등'],
    ['보고 지표', '전진 속도 RMSE, 횡방향 변위, wrapped yaw 차이', '동일한 지표 정의'],
])
v0 = by[.05, 0]
v40 = by[.05, 40]
v120 = by[.05, 120]
changes = [100 * (v40[k]['mean'] / v0[k]['mean'] - 1) for k in keys]
parts = [
    '<h1>ADAPT 선속도 입력 노이즈와 지연에 따른 보행 성능 평가</h1>',
    '<p class="subtitle">논문의 세 지표로 재평가한 결과</p>',
    '<p class="caption">2026년 10월 21일 · actor와 observer에 동일한 선속도 오차 적용 · 25조건 × 30회</p>',
    '<h2 id="summary">1 핵심 결과</h2>',
    '<p>논문 Appendix D의 정의에 따라 <strong>Velocity error, Lateral drift, Yaw drift</strong>를 계산했다. '
    '기존 750회 기록을 재사용했으며, 시뮬레이션이나 학습을 새로 실행하지 않았다. '
    '세 지표 모두 첫 평가 샘플인 <strong>2.02초부터 마지막 샘플인 30.00초까지</strong>의 구간으로 통일했다.</p>',
    f'<p><strong>노이즈 ±0.05 m/s에서 40 ms 지연을 주면 전진 속도 오차는 {abs(changes[0]):.1f}% 감소하지만, '
    f'횡방향 드리프트는 {changes[1]:.1f}%, yaw 드리프트는 {changes[2]:.1f}% 증가한다.</strong> '
    '따라서 전진 속도 오차의 감소를 전체 보행 성능의 개선으로 해석할 수 없다. 드리프트의 trial 간 변동도 크므로, 이 차이를 통계적으로 유의한 개선이나 악화로 단정하지 않는다.</p>',
    '<p>기존 750회 시험의 낙상은 모두 0회였다. 그러나 선속도 무오차·무지연 기준에서도 '
    f"평균 횡방향 드리프트가 {by[0,0]['lateral_drift_m']['mean']:.2f} m, yaw 드리프트가 {by[0,0]['yaw_drift_rad']['mean']:.3f} rad로 남아 있다. "
    '낙상률과 경로 유지 성능은 별도로 판단해야 한다.</p>',
    '<h2 id="definitions">2 논문의 지표 정의</h2>',
    '<p>논문 <a href="https://arxiv.org/html/2606.16542v1#A4">Appendix D의 식 14–16</a>을 사용한다. '
    '모든 지표는 trial별로 먼저 계산한 다음, 조건별 30회 결과의 평균과 표준편차를 구한다.</p>',
    '<h3>Velocity error</h3>',
    '<div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mi>e</mi><mi>vx</mi></msub><mo>=</mo><msqrt><mfrac><mn>1</mn><mi>N</mi></mfrac><munderover><mo>∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><msup><mrow><mo>(</mo><msubsup><mi>v</mi><mi>x,k</mi><mi>b</mi></msubsup><mo>−</mo><msubsup><mi>v</mi><mi>x</mi><mi>cmd</mi></msubsup><mo>)</mo></mrow><mn>2</mn></msup></msqrt></math></div>',
    '<p>참값 root의 몸체 좌표 전진 속도와 명령 속도의 RMSE이며 단위는 m/s이다. '
    '이번 명령은 0.8 m/s다. 전진·좌우를 합친 평면 RMSE는 이번 세 지표에 포함하지 않고, summary.json에 보조 지표로 보존했다.</p>',
    '<h3>Lateral drift</h3>',
    '<div class="equation">|d<sub>y</sub>| = |y<sub>end</sub> − y<sub>start</sub>|</div>',
    '<p>동일한 평가 구간의 시작·끝에서 측정한 세계 좌표 y 위치 차이의 절댓값이며 단위는 m이다. '
    '누적 횡방향 이동거리나 최대 이탈량은 아니다.</p>',
    '<h3>Yaw drift</h3>',
    '<div class="equation">|d<sub>ψ</sub>| = |wrap<sub>[−π, π]</sub>(ψ<sub>end</sub> − ψ<sub>start</sub>)|</div>',
    '<p>같은 구간의 시작·끝 heading 차이를 −π~π로 감싼 뒤 절댓값을 취한다. 단위는 rad이고 범위는 0~π다. '
    'yaw-rate RMSE나 unwrapped 각도 변화와는 다르다. 각 trial을 먼저 wrap하므로, trial 평균 각도를 wrap하는 계산도 사용하지 않는다.</p>',
    '<h2 id="conditions">3 실험 조건과 재계산 방법</h2>',
    '<p>Stage 2의 model_34998.pt를 고정해 평가한 기록을 사용했다. 물리 간격 1 ms, 정책 및 측정 주기 20 ms, '
    '몸체 기준 명령 [0.8 m/s, 0, 0], 외부 하중 없는 평지 조건이다. 선속도 xyz 각 축의 균등 노이즈 반폭은 '
    '0, 0.05, 0.1, 0.2, 0.5 m/s이며 고정 지연은 0, 20, 40, 80, 120 ms다. '
    '각 조건에서 seed 42, 142, 242별로 10회씩 실행해 총 30회, 전체 750회다.</p>',
    '<p>하나의 노이즈 샘플과 FIFO 지연 신호를 actor와 observer가 공유했다. observer에는 센서 위치 오프셋을 보정하고 '
    '세계 좌표로 변환한 동일한 오차를 동역학 및 운동량 계산 전에 반영했다. '
    'actor 각속도 노이즈 ±0.2 rad/s, observer 각속도 노이즈 0 등 나머지 조건은 그대로다.</p>',
    '<p>전진 RMSE는 50 Hz 제어 주기에서 계산해 저장한 trial별 통계값을 그대로 사용했다. '
    '이는 2.02, 2.04, …, 30.00초의 1,400개 샘플로 계산한 값이다. 위치·heading 기록에는 '
    '2.02초와 30.00초의 실제 샘플이 모두 있어 드리프트를 보간 없이 계산할 수 있었다. '
    '누적 heading 기록의 두 끝점 차이에 atan2(sin Δψ, cos Δψ)를 적용했다. '
    '750건 모두의 생존 상태, 샘플 수, 조건 순서와 시작·끝 시각을 확인했다.</p>',
    '<p>기존 보고서의 드리프트는 시작 자세부터 계산했지만, 이번에는 전진 RMSE와 동일한 구간으로 통일했다. '
    '따라서 이전 드리프트와 수치가 달라진 것은 평가 구간 및 각도 처리의 변경에 따른 것이다.</p>',
    '<h2 id="results">4 노이즈와 지연에 따른 결과</h2>',
    image('paper_metrics_curves.png', '논문 정의의 전진 속도 오차, 횡방향 드리프트, yaw 드리프트를 5개 노이즈 수준과 5개 지연 조건에서 비교한 그래프'),
    '<p class="caption">그림 1 세 지표의 조건별 평균 ± 1 표준편차. 표준편차는 신뢰구간이 아니다. 작을수록 좋다.</p>',
    '<h3>노이즈 ±0.05 m/s에서의 상세 비교</h3>',
    selected,
    '<p>20–40 ms에서 전진 오차의 평균은 낮아지지만, 이 조건의 두 드리프트 평균은 무지연보다 높다. '
    '120 ms에서는 세 지표가 모두 무지연보다 커진다. '
    '큰 노이즈와 지연에서 전진 오차가 커지는 경향은 확인되지만, 드리프트는 단조 증가하지 않는다.</p>',
    '<h3>전체 25개 조건</h3>',
    full,
    '<p class="caption">모든 값은 trial별 지표의 평균 ± 표준편차이며 각 조건 30회다. 모든 조건에서 낙상은 0/30회다.</p>',
    '<h2 id="limits">5 해석의 한계</h2>',
    comparison,
    '<p>이번 결과는 논문의 <strong>지표 정의를 적용한 센서 입력 민감도 평가</strong>다. '
    'Figure 3의 외력 실험 자체를 재현한 결과가 아니므로, 논문 그래프와 수치를 직접 비교해 우열을 판단할 수 없다. '
    '드리프트는 평가 시간에 영향을 받으므로 다른 길이의 시험과 비교할 때도 주의해야 한다.</p>',
    '<p>Wrapped yaw drift는 한 바퀴 회전 후 같은 방향으로 돌아온 경우 작은 값이 될 수 있고, '
    '끝점 횡방향 드리프트도 이동 중의 최대 이탈을 드러내지 않는다. '
    '이 한계는 논문 정의를 맞추기 위해 유지했으며, unwrapped_yaw_delta_rad와 원래 궤적 기록을 함께 보존했다.</p>',
    '<p>하나의 학습 모델을 반복 실행한 결과이며, 반복 trial의 변동과 독립 학습 모델 사이의 변동은 다르다. '
    '학습 물리 간격 5 ms와 평가 1 ms의 차이, 실제 센서의 bias·시간 상관·가변 지연, 각속도 오차 공유 및 하중 효과는 추가 검증이 필요하다. '
    '지연이 좋아 보이는 원인이나 통계적 유의성을 입증한 실험은 수행하지 않았다.</p>',
    '<h2 id="reproduce">6 결과 위치와 재현</h2>',
    f'<p>결과 디렉터리<br><code>{escape(str(DATA))}</code></p>',
    '<p><code>summary.json</code>은 조건별 평균과 표준편차, <code>trials.json</code>은 750개 trial의 지표와 끝점, '
    '<code>metadata.json</code>은 정의와 조건, <code>source_hashes.json</code>은 입력 원자료의 SHA256을 담고 있다.</p>',
    '<p>다음 명령은 ADAPT 작업 디렉터리에서 저장된 원자료를 다시 집계한다. output은 새 경로여야 한다.</p>',
    '<pre><code>'+escape('.venv/bin/python -m scripts.evaluate_paper_metrics \\\n  logs/evaluation/velocity_sensitivity_shared_2026-10-21_seed42 \\\n  logs/evaluation/velocity_sensitivity_shared_2026-10-21_seed142 \\\n  logs/evaluation/velocity_sensitivity_shared_2026-10-21_seed242 \\\n  --output logs/evaluation/paper_metrics_repeat')+'</code></pre>',
    '<p class="caption">기준 논문: ADAPT Analytical Disturbance-Aware Policy Training for Humanoid Locomotion, '
    '<a href="https://arxiv.org/html/2606.16542v1#A4">Appendix D</a>. '
    '실험 시간 및 외력 조건은 <a href="https://arxiv.org/html/2606.16542v1#A3">Appendix C</a>에 기술되어 있다.</p>',
]
navigation = [('summary','1 핵심 결과'), ('definitions','2 논문의 지표 정의'), ('conditions','3 실험 조건과 재계산'),
              ('results','4 평가 결과'), ('limits','5 해석의 한계'), ('reproduce','6 결과 위치와 재현')]
if (DATA / 's3_style_trajectories.png').exists():
    insert_at = parts.index('<h2 id="limits">5 해석의 한계</h2>')
    parts[insert_at:insert_at] = [
        '<h3 id="paths">Figure S3 형식의 보행 궤적 비교</h3>',
        '<p>가로축은 전진 거리, 세로축은 부호가 있는 횡방향 드리프트다. '
        '각 열은 지연 0, 20, 40, 80, 120 ms이고, 대표 그림의 두 행은 노이즈 ±0.05와 ±0.5 m/s다. '
        '파란 점선은 같은 정책의 무노이즈·무지연 기준 조건이며, 빨간 실선은 각 행·열의 시험 조건이다. '
        '별도의 baseline 정책과 ADAPT를 비교한 그림은 아니다.</p>',
        image('s3_style_trajectories.png', 'Figure S3 형식의 전진 거리와 횡방향 드리프트 그래프. 두 행은 선속도 노이즈 ±0.05 및 ±0.5 m/s, 다섯 열은 지연 0부터 120 ms까지.'),
        '<p class="caption">그림 2 조건별 평균 곡선과 ±1 표준편차 음영. 모든 곡선은 30개 trial을 포함한다. '
        '왼쪽과 오른쪽 이탈의 부호가 반대이므로 평균이 0에 가까워도 이탈이 작다는 뜻은 아니다.</p>',
        '<p>위치 원점은 기존 평가 구간의 시작인 2.02초로 맞췄다. 각 trial이 정해진 전진 거리를 '
        '처음 통과하는 시점의 횡방향 위치를 저장된 약 5 Hz 궤적에서 선형 보간해 비교했다. '
        '되돌아오는 경로를 x 순서로 재정렬하지 않았고, 외삽하거나 일부 trial을 제외하지 않았다. '
        '모든 조건의 30개 trial이 도달한 공통 범위인 0~6 m를 사용한다. '
        '전진 거리는 world x 변위이며, 이동 경로의 누적 길이가 아니다.</p>',
        '<p>이 그림은 같은 전진 거리에서의 부호 있는 횡방향 위치를 비교한다. '
        '앞의 표에서 사용한 고정 시간 구간의 끝점 절댓값 지표와 집계 방식이 다르므로, 두 수치를 직접 대조하지 않는다.</p>',
        '<details><summary>전체 25개 조건의 궤적 그림 펼치기</summary>' +
        image('s3_style_trajectories_all_conditions.png', '노이즈 5수준과 지연 5수준의 전체 25개 조건에 대한 전진 거리와 횡방향 드리프트 그래프') + '</details>',
        '<p class="caption">원자료와 계산 방식은 s3_style_curves.npz 및 s3_style_metadata.json에 저장했다. '
        '그림은 PNG와 SVG 형식으로 결과 디렉터리에 저장되어 있다.</p>',
    ]
    navigation.insert(4, ('paths', '4 보행 궤적 비교'))
nav = ''.join(f'<a href="#{key}">{title}</a>' for key,title in navigation)
html = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ADAPT 논문 지표 기반 선속도 입력 민감도 평가</title><style>{css}
.numeric{{min-width:640px}}.numeric td{{white-space:normal}}figure{{margin-left:0;margin-right:0}}
@media print{{.numeric{{min-width:0}}figure{{break-inside:auto}}}}</style></head><body><div class="layout">
<aside aria-label="보고서 목차"><div class="brand">ADAPT</div><div class="small">논문 지표 재평가 · 2026.10.21</div>
<nav>{nav}</nav><button onclick="window.print()" type="button">인쇄 및 PDF 저장</button></aside>
<main>{''.join(parts)}<footer>750회 원자료 재평가 · 세 지표의 공통 구간 2.02–30.00초 · 그래프가 포함된 단일 HTML 파일</footer></main>
</div></body></html>'''
OUT.write_text(html)
versioned = OUT.with_name(OUT.stem + '_paper_metrics.html')
versioned.write_text(html)
shutil.copy2(OUT, DATA / 'report.html')
for src in [Path(__file__), ROOT / 'scripts/evaluate_paper_metrics.py', ROOT / 'scripts/plot_s3_style_trajectories.py']:
    dst = DATA / 'analysis_source' / src.relative_to(ROOT)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
hashes = {str(p.relative_to(DATA / 'analysis_source')): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in (DATA / 'analysis_source').rglob('*.py')}
(DATA / 'analysis_source_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
print(OUT)
