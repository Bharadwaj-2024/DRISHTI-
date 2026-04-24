path = r'c:\Users\bhara\deepfake-detection-ai--1\Django Application\ml_app\templates\predict.html'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

latency_html = (
    '\n'
    '    <!-- Detection Latency Counter -->\n'
    '    <div class="panel-soft p-4 mb-4" style="border: 1px solid rgba(216,177,90,0.2);">\n'
    '        <div class="row g-3 align-items-center text-center">\n'
    '            <div class="col-md-4">\n'
    '                <div class="muted mb-1" style="font-size:0.78rem;text-transform:uppercase;letter-spacing:.08em;">DRISHTI Detection Time</div>\n'
    '                <div style="font-size:2.2rem;font-weight:900;color:#d8b15a;font-family:monospace;">{{ detection_time }}s</div>\n'
    '            </div>\n'
    '            <div class="col-md-4">\n'
    '                <div class="muted mb-1" style="font-size:0.78rem;text-transform:uppercase;letter-spacing:.08em;">Manual PIB Baseline</div>\n'
    '                <div style="font-size:2.2rem;font-weight:900;color:#888;font-family:monospace;">{{ manual_baseline_min }}m</div>\n'
    '            </div>\n'
    '            <div class="col-md-4">\n'
    '                <div class="muted mb-1" style="font-size:0.78rem;text-transform:uppercase;letter-spacing:.08em;">Speedup Factor</div>\n'
    '                <div style="font-size:2.2rem;font-weight:900;color:#4ade80;font-family:monospace;">{{ speedup }}x faster</div>\n'
    '            </div>\n'
    '        </div>\n'
    '    </div>\n'
    '\n'
)

# Insert the latency block before the first "row g-4 mb-4" that contains the signal stack
marker = '    <div class="row g-4 mb-4">\n        <div class="col-xl-7">\n            <div class="panel p-4 p-lg-5 h-100">\n                <div class="section-title">'
if marker in content:
    content = content.replace(marker, latency_html + marker, 1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS - latency block inserted')
else:
    print('MARKER NOT FOUND')
    # Show what the file has around that area
    lines = content.split('\n')
    for i, l in enumerate(lines[100:115], start=101):
        print(f'{i}: {repr(l)}')
