"""
Web应用 - 可视化采集结果和视频播放
支持：
- 场景/Route/Frame筛选
- RGB图像预览
- 分类标签展示
- 视频播放（支持进度条）
"""

import json
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request, send_file
from PIL import Image
import io
import base64
import os
import lzma
import pickle
from collector import SCENARIO_TO_ROAD_STRUCTURE, SCENARIO_TO_FINE_EVENTS, load_pickle_file

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 配置路径：默认跟随当前仓库，必要时可用环境变量覆盖。
KEYFRAME_FILTER_DIR = Path(__file__).resolve().parent
AUTOMOT_ROOT = KEYFRAME_FILTER_DIR.parent
LEAD_DATA_ROOT = Path(os.environ.get("LEAD_DATA_ROOT", AUTOMOT_ROOT / "lead_data"))
LEAD_VIDEO_ROOT = Path(os.environ.get("LEAD_VIDEO_ROOT", AUTOMOT_ROOT / "lead_video"))
COLLECTION_OUTPUT = Path(
    os.environ.get("KEYFRAME_COLLECTION_OUTPUT", KEYFRAME_FILTER_DIR / "collection_output")
)


# ============================================================================
# API 端点
# ============================================================================

@app.route('/api/scenarios')
def get_scenarios():
    """获取所有场景列表"""
    scenarios = sorted(SCENARIO_TO_ROAD_STRUCTURE.keys())
    return jsonify({
        "total": len(scenarios),
        "scenarios": scenarios
    })


@app.route('/api/scenario/<scenario>/info')
def get_scenario_info(scenario):
    """获取场景信息"""
    if scenario not in SCENARIO_TO_ROAD_STRUCTURE:
        return jsonify({"error": "Scenario not found"}), 404

    # 尝试加载采集结果
    result_file = COLLECTION_OUTPUT / f"{scenario}_result.json"
    result = None
    if result_file.exists():
        with open(result_file, 'r') as f:
            result = json.load(f)

    return jsonify({
        "scenario": scenario,
        "road_structures": [rs.value for rs in SCENARIO_TO_ROAD_STRUCTURE[scenario]],
        "events": [ev.value for ev in SCENARIO_TO_FINE_EVENTS.get(scenario, [])],
        "collected": result is not None,
        "total_frames": result.get('total_frames', 0) if result else 0,
        "num_routes": len(result.get('routes', [])) if result else 0,
    })


@app.route('/api/scenario/<scenario>/routes')
def get_scenario_routes(scenario):
    """获取scenario的routes"""
    scenario_dir = LEAD_DATA_ROOT / scenario
    if not scenario_dir.exists():
        return jsonify({"error": "Scenario directory not found"}), 404

    routes = sorted([d.name for d in scenario_dir.iterdir() if d.is_dir()])
    return jsonify({"scenario": scenario, "routes": routes})


@app.route('/api/scenario/<scenario>/route/<route_id>/frames')
def get_route_frames(scenario, route_id):
    """获取route的frames"""
    # 从采集结果文件获取
    result_file = COLLECTION_OUTPUT / f"{scenario}_result.json"
    if not result_file.exists():
        return jsonify({"error": "No collection result"}), 404

    with open(result_file, 'r') as f:
        result = json.load(f)

    route_data = None
    for route in result.get('routes', []):
        if route.get('route_id') == route_id:
            route_data = route
            break

    if not route_data:
        return jsonify({"error": "Route not found"}), 404

    frames = route_data.get('annotations', [])
    return jsonify({
        "scenario": scenario,
        "route_id": route_id,
        "total_frames": len(frames),
        "frames": [f['frame_id'] for f in frames]
    })


@app.route('/api/scenario/<scenario>/route/<route_id>/annotations')
def get_route_annotations(scenario, route_id):
    """获取route全部标注，供前端按帧联动显示。"""
    result_file = COLLECTION_OUTPUT / f"{scenario}_result.json"
    if not result_file.exists():
        return jsonify({"error": "No collection result"}), 404

    with open(result_file, 'r') as f:
        result = json.load(f)

    route_data = None
    for route in result.get('routes', []):
        if route.get('route_id') == route_id:
            route_data = route
            break

    if not route_data:
        return jsonify({"error": "Route not found"}), 404

    annotations = route_data.get('annotations', [])
    frame_ids = [ann.get('frame_id') for ann in annotations if 'frame_id' in ann]

    return jsonify({
        "scenario": scenario,
        "route_id": route_id,
        "total_frames": len(frame_ids),
        "frames": frame_ids,
        "annotations": annotations,
    })


@app.route('/api/frame/<scenario>/<route_id>/<int:frame_id>')
def get_frame_data(scenario, route_id, frame_id):
    """获取单帧的图像和标注"""
    rgb_dir = LEAD_DATA_ROOT / scenario / route_id / "rgb"
    jpg_file = rgb_dir / f"{frame_id:04d}.jpg"
    png_file = rgb_dir / f"{frame_id:04d}.png"
    pkl_file = rgb_dir / f"{frame_id:04d}.pkl"

    if not rgb_dir.exists():
        return jsonify({"error": f"RGB目录不存在: {rgb_dir}"}), 404

    try:
        # 优先读取 jpg/png；若不存在再回退读取 pkl
        if jpg_file.exists() or png_file.exists():
            image_file = jpg_file if jpg_file.exists() else png_file
            with open(image_file, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode()
            mime = "jpeg" if image_file.suffix.lower() == ".jpg" else "png"
        else:
            if not pkl_file.exists():
                return jsonify({
                    "error": f"Frame文件不存在: {frame_id:04d}.jpg/.png/.pkl"
                }), 404

            frame_data = load_pickle_file(pkl_file)

            # 将numpy数组转为图像
            if isinstance(frame_data, dict) and 'data' in frame_data:
                img_array = frame_data['data']
            else:
                img_array = frame_data

            # 转换为PIL Image
            from PIL import Image as PILImage
            import numpy as np

            if isinstance(img_array, np.ndarray) and len(img_array.shape) == 3:
                img = PILImage.fromarray(img_array.astype('uint8'))
            else:
                return jsonify({"error": "无效的图像数据格式"}), 400

            # 转为base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            mime = "jpeg"

        # 获取标注
        result_file = COLLECTION_OUTPUT / f"{scenario}_result.json"
        annotation = None
        if result_file.exists():
            with open(result_file, 'r') as f:
                result = json.load(f)
            for route in result.get('routes', []):
                if route.get('route_id') == route_id:
                    for ann in route.get('annotations', []):
                        if ann['frame_id'] == frame_id:
                            annotation = ann
                            break

        return jsonify({
            "frame_id": frame_id,
            "image_base64": f"data:image/{mime};base64,{img_base64}",
            "annotation": annotation,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/video/<scenario>/<route_id>')
def get_video_path(scenario, route_id):
    """获取视频路径"""
    scenario_video_dir = LEAD_VIDEO_ROOT / scenario / route_id
    candidates = [
        scenario_video_dir / "input.mp4",
        scenario_video_dir / "output.mp4",
    ]

    video_file = None
    for cand in candidates:
        if cand.exists():
            video_file = cand
            break

    if video_file is None:
        alt_matches = list(LEAD_VIDEO_ROOT.glob(f"{scenario}/{route_id}/*.mp4"))
        if alt_matches:
            video_file = alt_matches[0]

    if video_file is None:
        return jsonify({"error": "Video not found"}), 404

    return jsonify({
        "video_url": f"/api/video/file/{scenario}/{route_id}",
        "video_name": video_file.name,
    })


@app.route('/api/video/file/<scenario>/<route_id>')
def stream_video_file(scenario, route_id):
    """返回视频文件，供前端video标签播放。"""
    scenario_video_dir = LEAD_VIDEO_ROOT / scenario / route_id
    candidates = [
        scenario_video_dir / "input.mp4",
        scenario_video_dir / "output.mp4",
    ]

    video_file = None
    for cand in candidates:
        if cand.exists():
            video_file = cand
            break

    if video_file is None:
        alt_matches = list(LEAD_VIDEO_ROOT.glob(f"{scenario}/{route_id}/*.mp4"))
        if alt_matches:
            video_file = alt_matches[0]

    if video_file is None:
        return jsonify({"error": "Video file not found"}), 404

    return send_file(video_file, mimetype='video/mp4', as_attachment=False)


# ============================================================================
# HTML 前端
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>场景事件采集可视化</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        h1 { font-size: 28px; margin-bottom: 5px; }
        .subtitle { font-size: 14px; opacity: 0.9; }

        .layout {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 20px;
        }

        .panel {
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .control-panel h2 { font-size: 18px; margin-bottom: 15px; color: #333; }

        .control-group {
            margin-bottom: 20px;
        }
        .control-group label {
            display: block;
            font-size: 13px;
            font-weight: bold;
            margin-bottom: 5px;
            color: #666;
        }
        .control-group select, .control-group input {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 12px;
        }
        .control-group select:focus, .control-group input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 5px rgba(102, 126, 234, 0.3);
        }

        .btn {
            width: 100%;
            padding: 10px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: bold;
            transition: 0.3s;
        }
        .btn:hover { background: #764ba2; }

        .content-panel {
            display: grid;
            grid-template-rows: auto 1fr auto;
            gap: 15px;
        }

        .tabs {
            display: flex;
            gap: 10px;
            border-bottom: 2px solid #eee;
        }
        .tab {
            padding: 10px 15px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            color: #666;
            font-weight: bold;
            transition: 0.3s;
        }
        .tab.active {
            border-bottom-color: #667eea;
            color: #667eea;
        }
        .tab:hover { color: #667eea; }

        .tab-content {
            display: none;
            height: 600px;
            overflow-y: auto;
        }
        .tab-content.active { display: block; }

        .media-and-info {
            display: grid;
            grid-template-rows: auto auto;
            gap: 12px;
        }

        .image-container {
            text-align: center;
            background: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .image-container img {
            max-width: 100%;
            max-height: 500px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }

        .info-panel {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            font-size: 12px;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        .info-row:last-child { border-bottom: none; }
        .info-label { font-weight: bold; color: #666; }
        .info-value { color: #333; }

        .tag {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            margin: 2px;
            font-size: 11px;
        }
        .tag.event { background: #f093fb; }
        .tag.primary { background: #27ae60; font-size: 14px; padding: 5px 10px; }
        .tag.secondary { background: #8e44ad; }
        .tag.review { background: #e67e22; }
        .tag.ok { background: #27ae60; }

        .annotation-card {
            background: #fff;
            border: 1px solid #e8e8e8;
            border-radius: 8px;
            padding: 12px;
            margin: 12px 0;
        }
        .annotation-card h3 {
            font-size: 14px;
            margin-bottom: 10px;
            color: #333;
        }
        .annotation-main {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 10px;
        }
        .annotation-box {
            background: #f6f8ff;
            border-radius: 6px;
            padding: 10px;
        }
        .annotation-box .small-label {
            display: block;
            font-weight: bold;
            color: #666;
            margin-bottom: 6px;
        }
        .evidence-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px 14px;
            margin-top: 8px;
        }
        .evidence-item {
            border-bottom: 1px dashed #e1e1e1;
            padding-bottom: 5px;
        }
        .mono {
            font-family: Consolas, Monaco, monospace;
            word-break: break-all;
        }

        .video-container {
            background: #000;
            border-radius: 8px;
            overflow: hidden;
        }
        .video-container video {
            width: 100%;
            height: auto;
        }

        .status { text-align: center; color: #666; font-size: 12px; padding: 10px; }
        .status.loading { color: #667eea; }
        .status.error { color: #e74c3c; }
        .status.success { color: #27ae60; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎬 场景事件采集可视化系统</h1>
            <p class="subtitle">选择场景、路线、帧号查看RGB图像和分类标签 | 支持视频播放</p>
        </header>

        <div class="layout">
            <div class="panel control-panel">
                <h2>控制面板</h2>

                <div class="control-group">
                    <label>场景选择</label>
                    <select id="scenarioSelect" onchange="onScenarioChange()">
                        <option value="">-- 选择场景 --</option>
                    </select>
                </div>

                <div class="control-group">
                    <label>路线ID</label>
                    <select id="routeSelect" onchange="onRouteChange()">
                        <option value="">-- 选择路线 --</option>
                    </select>
                </div>

                <div class="control-group">
                    <label>帧号</label>
                    <select id="frameSelect" onchange="onFrameChange()">
                        <option value="">-- 选择帧 --</option>
                    </select>
                </div>

                <div class="control-group">
                    <label>帧号（数字输入）</label>
                    <input type="number" id="frameInput" min="0" onchange="onFrameInputChange()" placeholder="直接输入帧号">
                </div>

                <button class="btn" onclick="loadFrame()">加载帧数据</button>
                <button class="btn" onclick="loadVideo()" style="background: #f093fb; margin-top: 10px;">加载视频</button>

                <div style="margin-top: 30px; padding: 15px; background: #f0f4ff; border-radius: 8px;">
                    <h3 style="font-size: 14px; margin-bottom: 10px;">ℹ️ 场景信息</h3>
                    <div id="scenarioInfo" style="font-size: 12px; line-height: 1.6;">
                        <p>选择场景以查看详情</p>
                    </div>
                </div>
            </div>

            <div class="panel content-panel">
                <div class="tabs">
                    <div class="tab active" onclick="switchTab('image', event)">📷 RGB图像</div>
                    <div class="tab" onclick="switchTab('video', event)">🎥 视频</div>
                </div>

                <div class="media-and-info">
                    <div id="imageTab" class="tab-content active">
                        <div class="image-container">
                            <img id="frameImage" src="" alt="加载图像..." style="display:none;">
                            <p id="imagePlaceholder" style="color: #999;">选择帧后点击"加载帧数据"显示图像</p>
                        </div>
                    </div>

                    <div id="videoTab" class="tab-content">
                        <div class="video-container">
                            <video id="videoPlayer" controls style="width:100%; height:auto;">
                                <source id="videoSource" src="" type="video/mp4">
                                您的浏览器不支持视频播放
                            </video>
                        </div>
                    </div>

                    <div class="info-panel">
                        <div class="info-row">
                            <span class="info-label">同步来源</span>
                            <span class="info-value" id="infoSource">-</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">场景</span>
                            <span class="info-value" id="infoScenario">-</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">路线</span>
                            <span class="info-value" id="infoRoute">-</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">帧号</span>
                            <span class="info-value" id="infoFrame">-</span>
                        </div>
                        <div class="annotation-card">
                            <h3>✅ 本帧逐帧 RS 标注结果</h3>
                            <div class="annotation-main">
                                <div class="annotation-box">
                                    <span class="small-label">本帧最终标签</span>
                                    <div id="primaryRoadStructure"><span class="tag primary">-</span></div>
                                </div>
                                <div class="annotation-box">
                                    <span class="small-label">本帧 secondary</span>
                                    <div id="secondaryRoadStructures">-</div>
                                </div>
                                <div class="annotation-box">
                                    <span class="small-label">本帧标签置信度</span>
                                    <div id="infoConfidence">-</div>
                                </div>
                            </div>
                            <div class="info-row">
                                <span class="info-label">标注解释</span>
                                <span class="info-value" id="infoReason">-</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">复核状态</span>
                                <span class="info-value" id="reviewStatus">-</span>
                            </div>
                        </div>

                        <div style="margin: 15px 0;">
                            <span class="info-label">该场景全部候选 RS（不是本帧最终标签）</span>
                            <div id="roadStructures"></div>
                        </div>
                        <div style="margin: 15px 0;">
                            <span class="info-label">事件 (Events)</span>
                            <div id="events"></div>
                        </div>
                        <div class="annotation-card">
                            <h3>🧭 证据归因：XODR / XML / LEAD meta</h3>
                            <div class="evidence-grid">
                                <div class="evidence-item"><strong>决策来源:</strong> <span id="decisionSource">-</span></div>
                                <div class="evidence-item"><strong>规则类型:</strong> <span id="ruleKind">-</span></div>
                                <div class="evidence-item"><strong>命中规则:</strong> <span id="rulesFired">-</span></div>
                                <div class="evidence-item"><strong>使用输入:</strong> <span id="usedInputs">-</span></div>
                                <div class="evidence-item"><strong>XML/route:</strong> <span id="xmlEvidence">-</span></div>
                                <div class="evidence-item"><strong>LEAD meta:</strong> <span id="metaEvidence">-</span></div>
                                <div class="evidence-item"><strong>XODR:</strong> <span id="xodrEvidence">-</span></div>
                                <div class="evidence-item"><strong>时序去抖:</strong> <span id="temporalSmoothingEvidence">-</span></div>
                                <div class="evidence-item"><strong>弱/缺失输入:</strong> <span id="weakInputs">-</span></div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="status" id="status"></div>
            </div>
        </div>
    </div>

    <script>
        let routeFrames = [];
        let routeAnnotationsByFrame = {};

        // 初始化
        loadScenarios();
        bindVideoEvents();

        function loadScenarios() {
            fetch('/api/scenarios')
                .then(r => r.json())
                .then(data => {
                    const select = document.getElementById('scenarioSelect');
                    data.scenarios.forEach(scenario => {
                        const option = document.createElement('option');
                        option.value = scenario;
                        option.textContent = scenario;
                        select.appendChild(option);
                    });
                });
        }

        function onScenarioChange() {
            const scenario = document.getElementById('scenarioSelect').value;
            if (!scenario) return;

            // 更新场景信息
            fetch(`/api/scenario/${scenario}/info`)
                .then(r => r.json())
                .then(data => {
                    const info = document.getElementById('scenarioInfo');
                    info.innerHTML = `
                        <p><strong>道路结构:</strong> ${data.road_structures.join(', ')}</p>
                        <p><strong>事件候选:</strong> ${data.events.slice(0, 3).join(', ')}...</p>
                        <p><strong>已采集:</strong> ${data.collected ? '✓' : '✗'}</p>
                        <p><strong>帧数:</strong> ${data.total_frames}</p>
                        <p><strong>Route数:</strong> ${data.num_routes}</p>
                    `;
                });

            // 加载routes
            fetch(`/api/scenario/${scenario}/routes`)
                .then(r => r.json())
                .then(data => {
                    const select = document.getElementById('routeSelect');
                    select.innerHTML = '<option value="">-- 选择路线 --</option>';
                    data.routes.forEach(route => {
                        const option = document.createElement('option');
                        option.value = route;
                        option.textContent = route.substring(0, 50);
                        select.appendChild(option);
                    });
                });
        }

        function onRouteChange() {
            const scenario = document.getElementById('scenarioSelect').value;
            const route = document.getElementById('routeSelect').value;
            if (!scenario || !route) return;

            routeFrames = [];
            routeAnnotationsByFrame = {};

            fetch(`/api/scenario/${scenario}/route/${route}/frames`)
                .then(r => r.json().then(data => ({ ok: r.ok, data })))
                .then(({ ok, data }) => {
                    if (!ok || data.error) {
                        throw new Error(data.error || '加载帧列表失败');
                    }

                    const select = document.getElementById('frameSelect');
                    select.innerHTML = '<option value="">-- 选择帧 --</option>';
                    data.frames.forEach(frame => {
                        const option = document.createElement('option');
                        option.value = frame;
                        option.textContent = `帧 ${frame}`;
                        select.appendChild(option);
                    });

                    routeFrames = data.frames;
                    return fetch(`/api/scenario/${scenario}/route/${route}/annotations`);
                })
                .then(r => {
                    if (!r) return null;
                    return r.json().then(data => ({ ok: r.ok, data }));
                })
                .then(payload => {
                    if (!payload) return;

                    const { ok, data } = payload;
                    if (!ok || data.error) {
                        throw new Error(data.error || '加载标注失败');
                    }

                    routeAnnotationsByFrame = {};
                    (data.annotations || []).forEach(ann => {
                        routeAnnotationsByFrame[String(ann.frame_id)] = ann;
                    });
                })
                .catch(e => setStatus('错误: ' + e.message, 'error'));
        }

        function onFrameChange() {
            const frame = document.getElementById('frameSelect').value;
            document.getElementById('frameInput').value = frame;
            updateAnnotationFromCurrentFrame('手动选择帧');
        }

        function onFrameInputChange() {
            const frame = document.getElementById('frameInput').value;
            if (frame !== '') {
                document.getElementById('frameSelect').value = frame;
            }
            updateAnnotationFromCurrentFrame('手动输入帧号');
        }

        function loadFrame() {
            const scenario = document.getElementById('scenarioSelect').value;
            const route = document.getElementById('routeSelect').value;
            const frameSelectValue = document.getElementById('frameSelect').value;
            const frameInputValue = document.getElementById('frameInput').value;
            const frame = frameInputValue !== '' ? frameInputValue : frameSelectValue;

            if (!scenario || !route || frame === '') {
                setStatus('请选择场景、路线和帧', 'error');
                return;
            }

            setStatus('加载中...', 'loading');
            fetch(`/api/frame/${scenario}/${route}/${frame}`)
                .then(r => r.json().then(data => ({ ok: r.ok, data })))
                .then(({ ok, data }) => {
                    if (!ok || data.error) {
                        throw new Error(data.error || '加载帧失败');
                    }

                    // 显示图像
                    const img = document.getElementById('frameImage');
                    img.src = data.image_base64;
                    img.style.display = 'block';
                    document.getElementById('imagePlaceholder').style.display = 'none';

                    // 显示标注
                    const ann = data.annotation;
                    renderAnnotation(scenario, route, frame, ann, '图像帧加载');

                    if (ann) {
                        routeAnnotationsByFrame[String(frame)] = ann;
                    }

                    setStatus('✓ 加载成功', 'success');
                })
                .catch(e => setStatus('错误: ' + e.message, 'error'));
        }

        function loadVideo() {
            const scenario = document.getElementById('scenarioSelect').value;
            const route = document.getElementById('routeSelect').value;

            if (!scenario || !route) {
                setStatus('请选择场景和路线', 'error');
                return;
            }

            setStatus('加载视频中...', 'loading');
            fetch(`/api/video/${scenario}/${route}`)
                .then(r => r.json().then(data => ({ ok: r.ok, data })))
                .then(({ ok, data }) => {
                    if (!ok || data.error) {
                        throw new Error(data.error || '加载视频失败');
                    }

                    document.getElementById('videoSource').src = data.video_url;
                    document.getElementById('videoPlayer').load();
                    switchTab('video', null);
                    setStatus('✓ 视频已加载', 'success');
                })
                .catch(e => setStatus('错误: ' + e.message, 'error'));
        }

        function bindVideoEvents() {
            const videoPlayer = document.getElementById('videoPlayer');
            videoPlayer.addEventListener('timeupdate', syncAnnotationWithVideo);
            videoPlayer.addEventListener('seeked', syncAnnotationWithVideo);
            videoPlayer.addEventListener('loadedmetadata', syncAnnotationWithVideo);
        }

        function syncAnnotationWithVideo() {
            const videoPlayer = document.getElementById('videoPlayer');
            const scenario = document.getElementById('scenarioSelect').value;
            const route = document.getElementById('routeSelect').value;

            if (!scenario || !route || routeFrames.length === 0) {
                return;
            }

            const duration = videoPlayer.duration;
            if (!duration || !Number.isFinite(duration)) {
                return;
            }

            const ratio = Math.max(0, Math.min(1, videoPlayer.currentTime / duration));
            const frameIndex = Math.round(ratio * (routeFrames.length - 1));
            const frameId = routeFrames[frameIndex];

            document.getElementById('frameInput').value = frameId;
            document.getElementById('frameSelect').value = String(frameId);

            const ann = routeAnnotationsByFrame[String(frameId)] || null;
            renderAnnotation(
                scenario,
                route,
                frameId,
                ann,
                `视频时间 ${videoPlayer.currentTime.toFixed(2)}s`
            );
        }

        function updateAnnotationFromCurrentFrame(source) {
            const scenario = document.getElementById('scenarioSelect').value;
            const route = document.getElementById('routeSelect').value;
            const frameSelectValue = document.getElementById('frameSelect').value;
            const frameInputValue = document.getElementById('frameInput').value;
            const frame = frameInputValue !== '' ? frameInputValue : frameSelectValue;

            if (!scenario || !route || frame === '') {
                return;
            }

            const ann = routeAnnotationsByFrame[String(frame)] || null;
            renderAnnotation(scenario, route, frame, ann, source);
        }

        function escapeHtml(value) {
            return String(value ?? '-')
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;')
                .replaceAll('"', '&quot;')
                .replaceAll("'", '&#39;');
        }

        function formatScalar(value) {
            if (value === null || value === undefined || value === '') {
                return '-';
            }
            if (typeof value === 'number') {
                return Number.isInteger(value) ? String(value) : value.toFixed(3);
            }
            if (typeof value === 'boolean') {
                return value ? 'true' : 'false';
            }
            return escapeHtml(value);
        }

        function formatList(values, className) {
            if (!values || values.length === 0) {
                return '<span class="tag">无</span>';
            }
            return values.map(v => `<span class="tag ${className || ''}">${escapeHtml(v)}</span>`).join('');
        }

        function normalizeFrameRs(ann) {
            const frameRs = ann.frame_rs_annotation || {};
            const evidence = ann.evidence || {};
            const diagnostic = evidence.diagnostic_attribution || {};
            return {
                label: frameRs.label || ann.primary_road_structure || '-',
                secondary: frameRs.secondary || ann.secondary_road_structures || [],
                confidence: frameRs.confidence ?? ann.confidence ?? '-',
                comment: frameRs.comment || ann.annotation_comment || ann.reason || '-',
                ruleKind: frameRs.rule_kind || evidence.rule_kind || '-',
                rulesFired: frameRs.rules_fired || evidence.rules_fired || [],
                decisionSource: frameRs.decision_source || diagnostic.decision_source || '-',
                reviewRequired: frameRs.review_required ?? evidence.review_required ?? false,
                reviewReasons: frameRs.review_reasons || evidence.review_reasons || [],
                metrics: frameRs.metrics || {},
                xodr: frameRs.xodr_summary || evidence.xodr || {},
                evidence,
            };
        }

        function renderAnnotation(scenario, route, frame, ann, source) {
            document.getElementById('infoSource').textContent = source || '-';
            document.getElementById('infoScenario').textContent = scenario || '-';
            document.getElementById('infoRoute').textContent = route ? route.substring(0, 30) : '-';
            document.getElementById('infoFrame').textContent = frame;

            if (ann) {
                const frameRs = normalizeFrameRs(ann);

                document.getElementById('primaryRoadStructure').innerHTML =
                    `<span class="tag primary">${escapeHtml(frameRs.label)}</span>`;
                document.getElementById('secondaryRoadStructures').innerHTML =
                    formatList(frameRs.secondary, 'secondary');
                document.getElementById('infoConfidence').textContent =
                    `${formatScalar(frameRs.confidence)}（对应本帧最终标签 ${frameRs.label}）`;
                document.getElementById('infoReason').textContent = frameRs.comment;

                const reviewText = frameRs.reviewRequired
                    ? `需要复核：${(frameRs.reviewReasons || []).join('; ') || '未给出原因'}`
                    : '无需复核';
                const reviewClass = frameRs.reviewRequired ? 'review' : 'ok';
                document.getElementById('reviewStatus').innerHTML =
                    `<span class="tag ${reviewClass}">${escapeHtml(reviewText)}</span>`;

                document.getElementById('roadStructures').innerHTML =
                    formatList(ann.road_structures || [], '');
                document.getElementById('events').innerHTML =
                    formatList(ann.events || [], 'event');

                const metrics = frameRs.metrics || {};
                const xodr = frameRs.xodr || {};
                const evidence = frameRs.evidence || {};
                const xmlPieces = [
                    `route_s=${formatScalar(metrics.route_progress_m ?? evidence.route_progress_m)}m`,
                    `proj_err=${formatScalar(metrics.route_projection_error_m ?? evidence.route_projection_error_m)}m`,
                    `trigger_dist=${formatScalar(metrics.trigger_distance_m ?? evidence.trigger_distance_m)}m`,
                    evidence.xml_route_path ? `xml=${evidence.xml_route_path}` : '',
                ].filter(Boolean);
                const metaPieces = [
                    `traffic_light=${formatScalar(metrics.traffic_light_state ?? evidence.traffic_light_state)}`,
                    `active=${formatScalar(metrics.active_scenario ?? evidence.current_active_scenario_type)}`,
                ];
                const xodrPieces = [
                    `available=${formatScalar(xodr.available ?? xodr.xodr_available)}`,
                    `trusted=${formatScalar(xodr.trusted ?? xodr.xodr_topology_trusted)}`,
                    `source=${formatScalar(xodr.source ?? xodr.xodr_source)}`,
                    `road=${formatScalar(xodr.road_id ?? xodr.map_road_id)}`,
                    `lane=${formatScalar(xodr.lane_id ?? xodr.map_lane_id)}`,
                    `junction=${formatScalar(xodr.is_junction ?? xodr.map_is_junction)}`,
                    `roundabout=${formatScalar(xodr.is_roundabout ?? xodr.map_is_roundabout)}`,
                    `opposite=${formatScalar(xodr.opposite_lane ?? xodr.has_opposite_driving_lane)}`,
                    `parking=${formatScalar(xodr.parking_or_shoulder ?? xodr.has_parking_or_shoulder_nearby)}`,
                    `merge=${formatScalar(xodr.merge_split_hint ?? xodr.ramp_merge_split_hint)}`,
                ];

                document.getElementById('decisionSource').textContent = frameRs.decisionSource;
                document.getElementById('ruleKind').textContent = frameRs.ruleKind;
                document.getElementById('rulesFired').innerHTML = formatList(frameRs.rulesFired, '');
                document.getElementById('usedInputs').textContent =
                    (evidence.used_inputs || evidence.inputs_used || []).join(', ') || '见下方 XML/meta/XODR 指标';
                document.getElementById('xmlEvidence').innerHTML = `<span class="mono">${escapeHtml(xmlPieces.join(' | ') || '-')}</span>`;
                document.getElementById('metaEvidence').innerHTML = `<span class="mono">${escapeHtml(metaPieces.join(' | ') || '-')}</span>`;
                document.getElementById('xodrEvidence').innerHTML = `<span class="mono">${escapeHtml(xodrPieces.join(' | ') || '-')}</span>`;
                const smoothing = evidence.temporal_smoothing || [];
                const smoothingText = smoothing.map(item =>
                    `${item.from}->${item.to} (${item.reason}, ${item.inherited_from})`
                ).join(' | ');
                document.getElementById('temporalSmoothingEvidence').innerHTML =
                    `<span class="mono">${escapeHtml(smoothingText || '-')}</span>`;
                document.getElementById('weakInputs').textContent =
                    (evidence.weak_or_missing_inputs || evidence.review_reasons || frameRs.reviewReasons || []).join(', ') || '-';
            } else {
                document.getElementById('primaryRoadStructure').innerHTML = '<span class="tag primary">无标注</span>';
                document.getElementById('secondaryRoadStructures').textContent = '-';
                document.getElementById('roadStructures').innerHTML = '<span class="tag">无标注</span>';
                document.getElementById('events').innerHTML = '<span class="tag event">无标注</span>';
                document.getElementById('infoConfidence').textContent = '-';
                document.getElementById('infoReason').textContent = '该帧未找到标注';
                document.getElementById('reviewStatus').textContent = '-';
                document.getElementById('decisionSource').textContent = '-';
                document.getElementById('ruleKind').textContent = '-';
                document.getElementById('rulesFired').textContent = '-';
                document.getElementById('usedInputs').textContent = '-';
                document.getElementById('xmlEvidence').textContent = '-';
                document.getElementById('metaEvidence').textContent = '-';
                document.getElementById('xodrEvidence').textContent = '-';
                document.getElementById('temporalSmoothingEvidence').textContent = '-';
                document.getElementById('weakInputs').textContent = '-';
            }
        }

        function switchTab(tab, eventObj) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            if (eventObj && eventObj.target) {
                eventObj.target.classList.add('active');
            } else {
                const tabMap = { image: 0, video: 1 };
                const tabIndex = tabMap[tab];
                const tabElement = document.querySelectorAll('.tab')[tabIndex];
                if (tabElement) {
                    tabElement.classList.add('active');
                }
            }

            document.getElementById(tab + 'Tab').classList.add('active');
        }

        function setStatus(msg, type) {
            const status = document.getElementById('status');
            status.textContent = msg;
            status.className = 'status ' + type;
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)


# ============================================================================
# 启动
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Web应用已启动")
    print("="*60)
    print("\n访问地址: http://localhost:5000")
    print("\n功能:")
    print("  1. 场景/Route/Frame筛选")
    print("  2. RGB图像预览")
    print("  3. 分类标签展示")
    print("  4. 视频播放（支持进度条）")
    print("\n" + "="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
