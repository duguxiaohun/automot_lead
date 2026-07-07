"""
快速启动脚本 - 采集、分析、Web应用

支持4种正式采集 + 逐帧 RS/EVENT 标注模式:
  1. 单场景全部采集 + 逐帧标注 - 采集某场景的所有routes
  2. 单场景指定数目采集 + 逐帧标注 - 采集某场景的N个routes
  3. 多场景采集 + 逐帧标注 - 同时采集多个指定场景
  4. 全部采集 + 逐帧标注 - 采集所有43个场景

第9项只保留为小范围 smoke / 参数闭环调试入口，不再是唯一逐帧标注入口。
"""

import sys
import argparse
from pathlib import Path
import socket
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple
from collector import (
    EVENT_LABELS,
    ROAD_STRUCTURE_LABELS,
    ScenarioCollector,
    SCENARIO_TO_ROAD_STRUCTURE,
    SCENARIO_TO_FINE_EVENTS,
    SCENARIO_RULE_KIND,
    SCENARIO_RULE_CONFIG,
    RouteXmlIndex,
    load_pickle_file,
    _DEFAULT_XML_ROOT,
    _DEFAULT_CARLA_ROOT,
    _DEFAULT_LEAD_DATA_ROOT,
    _extract_route_num,
)
from analyzer import StructureAnalyzer, quick_analysis
import json

from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route


_POLICY_LOGIC_BY_KIND: Dict[str, Dict[str, Any]] = {
    "same_direction_obstacle": {
        "primary_rules": [
            "有效 traffic_light_state/light_hazard 或同源受控路口窗口 -> R4",
            "同向事故/施工/停放障碍只作为事件证据，不改变道路结构 -> R1",
        ],
        "xml_xodr_usage": [
            "XML trigger/distance/speed 只记录障碍上下文与路口近邻窗口",
            "XODR 只用于确认是否进入受控 junction，不用 parking/opposite hint 升级",
        ],
    },
    "twoways_obstacle": {
        "primary_rules": [
            "有效灯态/受控路口窗口优先 -> R4，R2 仅作为 secondary",
            "TwoWays trigger/distance/active 窗口 + XODR 对向 driving lane + 同向车道不足 + 核心障碍交互 -> R2",
            "XODR 不可用时也必须有近距离障碍、stuck、vehicle_hazard 或 lane-change 核心证据；普通 layout-prior 只给弱候选",
            "过最近障碍点后持续远离且无 stuck/vehicle_hazard 时，route 级裁剪回 R1/R4",
        ],
        "xml_xodr_usage": [
            "XML distance/front/behind/frequency 只召回绕障核心窗口，不延长整段 TwoWays",
            "XODR lane_id 符号反转与同向车道数决定是否确实需要对向参与",
        ],
    },
    "invading_turn": {
        "primary_rules": [
            "灯态/受控路口优先 -> R4",
            "trigger/distance/offset 窗口 + 对向 driving lane/narrow road -> R2",
            "窗口外或路网不支持对向交会 -> R1/review",
        ],
        "xml_xodr_usage": [
            "XML offset/distance 定义被动侵入影响区",
            "XODR opposite lane 用于区分被动对向侵占与普通弯道",
        ],
    },
    "signalized_junction": {
        "primary_rules": [
            "有效 Red/Yellow/Green 或 light_hazard -> R4",
            "scenario trigger 对应受控 junction 窗口 -> R4 medium/high",
            "路口外跟车/离开背景 -> R1",
        ],
        "xml_xodr_usage": [
            "XML trigger/waypoints 定位 stopline/junction approach",
            "XODR signal/controller/junction 支持 signalized 证据；灯色仍以 meta 为准",
        ],
    },
    "nonsignalized_junction": {
        "primary_rules": [
            "无有效正常灯态 + trigger/junction/stop-yield 窗口 -> R5",
            "出现连续有效灯态时降级 review，不直接 high R5",
            "路口外 -> R1",
        ],
        "xml_xodr_usage": [
            "XML trigger/flow/source_dist_interval 定义接近与冲突流窗口",
            "XODR junction 且缺少 signal controller、存在 stop/yield hint 时增强 R5",
        ],
    },
    "defect_junction": {
        "primary_rules": [
            "trigger 对应 junction 前后窗口 -> R5",
            "即使 XODR 有 signal/controller 或 meta 有灯态，也按故障灯语义覆盖 R4",
            "找不到 junction 时 R5 medium + review",
        ],
        "xml_xodr_usage": [
            "XML trigger/traffic_direction/source_dist_interval 定义故障路口窗口",
            "XODR signal/controller 在本场景中是 defect_signal evidence，而非 R4 evidence",
        ],
    },
    "highway_merge": {
        "primary_rules": [
            "actor-flow/trigger/other_actor_location 合流或驶出窗口 + ramp/merge hint -> R3",
            "进入真实信号灯路口且灯态同源时 R4 覆盖 R3",
            "合流完成、拓扑稳定后 -> R1",
        ],
        "xml_xodr_usage": [
            "XML start_actor_flow/end_actor_flow/other_actor_location 构造 R3 窗口",
            "XODR ramp/merge/split/lane-count-change 是 R3 high confidence 条件",
        ],
    },
    "interurban": {
        "primary_rules": [
            "郊区 actor-flow/merge 窗口可给 R3",
            "接近/进入 junction 后按灯态切到 R4/R5",
            "R4/R5 路口决策优先于 R3",
        ],
        "xml_xodr_usage": [
            "XML actor-flow 与 trigger 共同定义先合流后路口的序列",
            "XODR 同时检查 merge hint 与 junction/controller",
        ],
    },
    "interurban_advanced": {
        "primary_rules": [
            "主体按路口通行：有灯 -> R4，无灯/路权 -> R5",
            "只有 XODR 明确 merge/split 时给短 R3 medium/review",
        ],
        "xml_xodr_usage": [
            "XML actor-flow 只作为冲突车流证据",
            "XODR 决定是否存在可升级为 R3 的真实合流拓扑",
        ],
    },
    "parking": {
        "primary_rules": [
            "灯态/受控路口优先 -> R4，STOP/无灯路口 -> R5",
            "停车空间、遮挡和行人横穿不再生成独立 RS；非路口保持 R1，行人/遮挡进入 EVENT",
            "两侧停车压缩成有效对向单车道时才由 TwoWays/开门类规则进入 R2",
        ],
        "xml_xodr_usage": [
            "XML direction/distance/crossing_angle 定义停车侧与影响窗口，只辅助 EVENT/span",
            "XODR parking/shoulder lane 或路边空间证据只用于判断 R1/R2 与遮挡风险",
        ],
    },
    "parking_exit": {
        "primary_rules": [
            "停车位/停车带汇入主路窗口仍为 R1，道路事件用 R-E2 表达汇入/回正",
            "有灯态/受控路口时 primary=R4",
            "汇入完成且 driving lane 稳定后保持 R1/R-E1",
        ],
        "xml_xodr_usage": [
            "XML front/behind_vehicle_distance/direction 定义停车空隙与汇入侧",
            "XODR Parking/Shoulder -> Driving 的拓扑切换只增强 R-E2 汇入证据",
        ],
    },
    "vehicle_opens_door_twoways": {
        "primary_rules": [
            "R4/R5 控制源优先；两侧停车/开门压缩有效可行驶 lane 时主 RS 为 R2",
            "不再输出独立停车 RS，开门/停车风险进入 U-E2/R-E2",
        ],
        "xml_xodr_usage": [
            "XML distance/frequency/direction 定义开门风险窗口",
            "XODR 同时检查 opposite driving lane 与 parking/shoulder context",
        ],
    },
    "static_cutin": {
        "primary_rules": [
            "灯态/受控路口优先 -> R4",
            "cut-in 侧为 parking/shoulder/curbside 时仍保持 R1，切入进入 U-E3",
            "cut-in 侧为 ramp/merge/auxiliary lane -> R3，否则 -> R1",
        ],
        "xml_xodr_usage": [
            "XML distance/direction/speed 定义切入窗口与侧向",
            "XODR 用于仲裁普通 R1 与 merge-side R3",
        ],
    },
    "pedestrian_crossing": {
        "primary_rules": [
            "signalized junction/有效灯态 -> R4",
            "无灯/stop/yield junction -> R5",
            "普通路段行人横穿 -> R1",
        ],
        "xml_xodr_usage": [
            "XML trigger 定位横穿空间，不单独决定 R5",
            "XODR junction/controller/stop-yield 决定 R4/R5/R1",
        ],
    },
    "vehicle_turning": {
        "primary_rules": [
            "每个 trigger 建独立转弯窗口；受控路口 -> R4，无灯路权路口 -> R5",
            "普通弯道或路口外 -> R1",
        ],
        "xml_xodr_usage": [
            "XML 多 scenario trigger 全部保留，不能只取第一个",
            "XODR junction/controller 与 route heading change 共同确认转弯路口",
        ],
    },
    "noscenario": {
        "primary_rules": [
            "仅真实有效灯态/light_hazard + 同源受控路口可升级 -> R4",
            "禁止单靠 XODR opposite/parking/merge hint 输出 R2/R3/R5",
            "其它全部 -> R1",
        ],
        "xml_xodr_usage": [
            "XML 只提供 route/town，不能提供场景先验",
            "XODR hint 只写入 evidence/review，不改变 conservative primary",
        ],
    },
    "default_meta_map": {
        "primary_rules": [
            "有效灯态/受控路口窗口 -> R4",
            "动态横穿、急刹、失控、侧向危险等行为不改变 RS -> R1",
        ],
        "xml_xodr_usage": [
            "XML trigger 只给事件上下文与 junction 近邻窗口",
            "XODR 只用于确认是否存在受控路口",
        ],
    },
}


def print_main_menu():
    """打印主菜单"""
    print("\n" + "="*70)
    print("场景事件采集系统 - 快速启动".center(70))
    print("="*70)
    print("""
采集 + 逐帧标注模式:
  1️⃣  单场景全部采集+逐帧RS/EVENT标注   - 采集某个场景的所有routes并逐帧标注
  2️⃣  单场景指定数采集+逐帧RS/EVENT标注  - 采集某个场景的N个routes并逐帧标注
  3️⃣  多场景采集+逐帧RS/EVENT标注       - 同时采集多个指定场景并逐帧标注
  4️⃣  全部采集+逐帧RS/EVENT标注         - 采集所有43个场景并逐帧标注（可能耗时很长）

其他功能:
  5️⃣  多角度结构分析      - 分析已采集的数据
  6️⃣  启动Web应用        - 交互式可视化查看
  7️⃣  显示所有场景       - 列出所有支持的场景
  8️⃣  ROAD_STRUCTURE XML/XODR画像 - 按场景/town审计XML与地图输入
  9️⃣  逐帧RS/EVENT标注调试入口  - 小范围 smoke / 参数闭环调试
  🔟  退出
    """)
    print("="*70)


# ============================================================================
# 采集功能
# ============================================================================

def _ask_max_frames_per_route() -> Optional[int]:
    """询问每条 route 最多处理多少帧；None 表示全帧逐帧标注。"""
    try:
        value = int(input("每条 route 最多处理帧数，0 表示全帧逐帧标注 (默认0): ") or "0")
    except Exception:
        value = 0
    return value if value > 0 else None


def _print_annotation_output_contract() -> None:
    """说明采集结果中候选全集与单帧独立标签的字段区别。"""
    print("  • 输出字段:")
    print("    - road_structures: 该 scenario 的全部候选 RS（保留旧逻辑）")
    print("    - primary_road_structure: 当前帧独属主 RS")
    print("    - frame_rs_annotation: 当前帧可执行标注结果 + 证据 + 注释")
    print("    - events: 当前帧事件集合")
    print("    - primary_event: 当前帧独属主 EVENT")
    print("    - frame_event_annotation: 当前帧 EVENT 结果 + 证据 + 注释")


def _write_and_print_annotation_summary(result: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """所有采集模式统一写逐帧标注摘要。"""
    summary = _annotation_summary(result)
    summary_file = output_dir / "frame_rs_annotation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    _print_annotation_summary(summary)
    print(f"  • 逐帧标注摘要: {summary_file}")
    return summary

def collect_one_scenario_all_ui():
    """模式1: 单场景全部采集 + 逐帧 RS/EVENT 标注"""
    print("\n" + "="*70)
    print("模式1: 单场景全部采集 + 逐帧RS/EVENT标注".center(70))
    print("="*70)

    scenarios = sorted(SCENARIO_TO_ROAD_STRUCTURE.keys())
    print(f"\n支持的场景 ({len(scenarios)}个):")
    for i, scenario in enumerate(scenarios, 1):
        if i % 2 == 0:
            print(f"{scenario}")
        else:
            print(f"{scenario:<40}", end="  ")
    print()

    scenario = input("\n请输入场景名称: ").strip()
    if scenario not in SCENARIO_TO_ROAD_STRUCTURE:
        print("❌ 场景不存在")
        return

    max_frames = _ask_max_frames_per_route()

    print(f"\n开始采集并逐帧标注 {scenario} 的所有routes...")
    collector = ScenarioCollector()

    try:
        result = collector.collect_one_scenario_all(scenario, max_frames_per_route=max_frames)
        if result.get("status") != "success":
            print("\n❌ 采集失败")
            print(f"  • 场景: {scenario}")
            print(f"  • 状态: {result.get('status')}")
            print(f"  • 错误: {result.get('error', '未知错误')}")
            print(f"  • 数据根: {result.get('lead_data_root', collector.lead_data_root)}")
            return

        print(f"\n✅ 采集完成!")
        print(f"  • 场景: {scenario}")
        print(f"  • 状态: {result['status']}")
        print(f"  • Routes数: {len(result['routes'])}")
        print(f"  • 总帧数: {result.get('total_frames', 0)}")
        print(f"  • 结果: collection_output/{scenario}_result.json")
        _print_annotation_output_contract()
        _write_and_print_annotation_summary(result, collector.output_dir)
    except Exception as e:
        print(f"\n❌ 采集失败: {e}")


def collect_one_scenario_limited_ui():
    """模式2: 单场景指定数目采集 + 逐帧 RS/EVENT 标注"""
    print("\n" + "="*70)
    print("模式2: 单场景指定数采集 + 逐帧RS/EVENT标注".center(70))
    print("="*70)

    scenarios = sorted(SCENARIO_TO_ROAD_STRUCTURE.keys())
    print(f"\n支持的场景 ({len(scenarios)}个):")
    for i, scenario in enumerate(scenarios, 1):
        if i % 2 == 0:
            print(f"{scenario}")
        else:
            print(f"{scenario:<40}", end="  ")
    print()

    scenario = input("\n请输入场景名称: ").strip()
    if scenario not in SCENARIO_TO_ROAD_STRUCTURE:
        print("❌ 场景不存在")
        return

    try:
        max_routes = int(input("请输入采集的routes数量 (默认5): ") or "5")
    except:
        max_routes = 5
    max_frames = _ask_max_frames_per_route()

    print(f"\n开始采集并逐帧标注 {scenario} 的 {max_routes} 个routes...")
    collector = ScenarioCollector()

    try:
        result = collector.collect_one_scenario(
            scenario,
            max_routes=max_routes,
            max_frames_per_route=max_frames,
        )
        if result.get("status") != "success":
            print("\n❌ 采集失败")
            print(f"  • 场景: {scenario}")
            print(f"  • 状态: {result.get('status')}")
            print(f"  • 错误: {result.get('error', '未知错误')}")
            print(f"  • 数据根: {result.get('lead_data_root', collector.lead_data_root)}")
            return

        print(f"\n✅ 采集完成!")
        print(f"  • 场景: {scenario}")
        print(f"  • 状态: {result['status']}")
        print(f"  • Routes数: {len(result['routes'])}")
        print(f"  • 总帧数: {result.get('total_frames', 0)}")
        print(f"  • 结果: collection_output/{scenario}_result.json")
        _print_annotation_output_contract()
        _write_and_print_annotation_summary(result, collector.output_dir)
    except Exception as e:
        print(f"\n❌ 采集失败: {e}")


def collect_multiple_scenarios_ui():
    """模式3: 多场景采集 + 逐帧 RS/EVENT 标注"""
    print("\n" + "="*70)
    print("模式3: 多场景采集 + 逐帧RS/EVENT标注".center(70))
    print("="*70)

    scenarios = sorted(SCENARIO_TO_ROAD_STRUCTURE.keys())

    # 推荐场景
    recommended = [
        "Accident",
        "AccidentTwoWays",
        "BlockedIntersection",
        "PedestrianCrossing",
        "HighwayCutIn"
    ]

    print(f"\n推荐场景:")
    for i, scenario in enumerate(recommended, 1):
        print(f"  {i}. {scenario}")

    print(f"\n所有场景 ({len(scenarios)}个):")
    for i, scenario in enumerate(scenarios, 1):
        if i % 2 == 0:
            print(f"{scenario}")
        else:
            print(f"{scenario:<40}", end="  ")
    print()

    print("\n请输入要采集的场景 (逗号分隔, 例如: Accident,AccidentTwoWays,BlockedIntersection):")
    scenario_input = input("> ").strip()

    selected_scenarios = [s.strip() for s in scenario_input.split(",")]

    # 验证场景名称
    invalid = [s for s in selected_scenarios if s not in SCENARIO_TO_ROAD_STRUCTURE]
    if invalid:
        print(f"❌ 以下场景不存在: {invalid}")
        return

    try:
        max_routes = int(input("请输入每个场景采集的routes数量 (默认3): ") or "3")
    except:
        max_routes = 3
    max_frames = _ask_max_frames_per_route()

    print(f"\n将采集并逐帧标注 {len(selected_scenarios)} 个场景, 每个采集 {max_routes} 个routes")
    print("场景列表:")
    for i, scenario in enumerate(selected_scenarios, 1):
        print(f"  {i}. {scenario}")

    confirm = input("\n是否继续? (y/n): ").strip().lower()
    if confirm != 'y':
        print("⏭️  已取消")
        return

    collector = ScenarioCollector()

    try:
        result = collector.collect_multiple_scenarios(
            selected_scenarios,
            max_routes_per_scenario=max_routes,
            max_frames_per_route=max_frames,
        )

        print(f"\n✅ 采集完成!")
        print(f"  • 成功场景数: {result.get('scenarios_collected', 0)}")
        print(f"  • 总场景数: {result.get('total_scenarios', 0)}")
        print(f"  • 总帧数: {result.get('total_frames', 0)}")
        failed = [
            (name, item.get("error", "未知错误"))
            for name, item in result.get("results", {}).items()
            if item.get("status") != "success"
        ]
        if failed:
            print(f"  • 失败场景数: {len(failed)}")
            for name, error in failed[:10]:
                print(f"    - {name}: {error}")
        print(f"  • 结果: collection_output/multi_scenario_collection.json")
        _print_annotation_output_contract()
        _write_and_print_annotation_summary(result, collector.output_dir)
    except Exception as e:
        print(f"\n❌ 采集失败: {e}")


def collect_all_scenarios_ui():
    """模式4: 全部采集 + 逐帧 RS/EVENT 标注"""
    print("\n" + "="*70)
    print("模式4: 全部采集 + 逐帧RS/EVENT标注".center(70))
    print("="*70)

    scenarios = sorted(SCENARIO_TO_ROAD_STRUCTURE.keys())

    print(f"\n⚠️  警告: 这将采集所有 {len(scenarios)} 个场景")
    print("这可能需要很长时间和大量磁盘空间！")

    max_frames = _ask_max_frames_per_route()

    print(f"\n预计采集所有 {len(scenarios)} 个场景下的全部合法 routes")

    confirm = input("确实要继续? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("⏭️  已取消")
        return

    collector = ScenarioCollector()

    print(f"\n开始采集并逐帧标注所有 {len(scenarios)} 个场景...")

    try:
        result = collector.collect_all_scenarios(
            max_routes_per_scenario=None,
            max_frames_per_route=max_frames,
        )

        print(f"\n✅ 采集完成!")
        print(f"  • 成功场景数: {result.get('scenarios_collected', 0)}")
        print(f"  • 总场景数: {result.get('total_scenarios', 0)}")
        print(f"  • 总帧数: {result.get('total_frames', 0)}")
        failed = [
            (name, item.get("error", "未知错误"))
            for name, item in result.get("results", {}).items()
            if item.get("status") != "success"
        ]
        if failed:
            print(f"  • 失败场景数: {len(failed)}")
            for name, error in failed[:10]:
                print(f"    - {name}: {error}")
        print(f"  • 结果: collection_output/multi_scenario_collection.json")
        _print_annotation_output_contract()
        _write_and_print_annotation_summary(result, collector.output_dir)
    except Exception as e:
        print(f"\n❌ 采集失败: {e}")


# ============================================================================
# 其他功能
# ============================================================================

def run_analysis_ui():
    """运行多角度分析"""
    print("\n" + "="*70)
    print("多角度结构分析".center(70))
    print("="*70)

    output_dir = Path(__file__).resolve().parent / "collection_output"
    result_files = list(output_dir.glob("*_result.json"))

    if not result_files:
        print("\n❌ 没有找到采集结果文件")
        print("请先运行采集操作")
        return

    print(f"\n找到 {len(result_files)} 个采集结果文件")

    all_results = {}
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                scenario = result.get('scenario')
                if scenario:
                    all_results[scenario] = result
        except:
            pass

    if not all_results:
        print("❌ 没有有效的采集结果")
        return

    print(f"\n分析 {len(all_results)} 个场景的采集结果...\n")

    analyzer = StructureAnalyzer(all_results)
    analyzer.print_summary()

    report_file = output_dir / "structure_analysis_report.json"
    analyzer.generate_report(str(report_file))
    print(f"\n✓ 完整报告已保存: {report_file}")


def run_web_app_ui():
    """启动Web应用"""
    def _is_port_available(host: str, port: int) -> bool:
        """检查端口是否可用（可绑定）。"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                return False
        return True

    def _find_available_port(host: str, preferred_port: int, max_tries: int = 20) -> int:
        """从首选端口开始寻找可用端口。"""
        if _is_port_available(host, preferred_port):
            return preferred_port

        for offset in range(1, max_tries + 1):
            candidate = preferred_port + offset
            if _is_port_available(host, candidate):
                return candidate

        raise RuntimeError(f"未找到可用端口（尝试范围: {preferred_port}-{preferred_port + max_tries}）")

    host = '0.0.0.0'
    preferred_port = 5000
    port = _find_available_port(host, preferred_port)

    print("\n" + "="*70)
    print("启动Web应用".center(70))
    print("="*70)
    print("""
Web应用功能:
  ✓ 场景/Route/Frame筛选
  ✓ RGB图像预览
  ✓ 分类标签展示
  ✓ 视频播放（支持进度条）
按 Ctrl+C 停止服务
    """)

    if port == preferred_port:
        print(f"访问地址: http://localhost:{port}")
    else:
        print(f"⚠️  端口 {preferred_port} 已被占用，自动切换到端口 {port}")
        print(f"访问地址: http://localhost:{port}")

    print("="*70 + "\n")

    try:
        from web_app import app
        # 在交互式菜单中关闭 reloader，避免子进程重启后主菜单再次出现
        app.run(debug=True, use_reloader=False, host=host, port=port)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("\n请确保已安装 Flask:")
        print("  pip install flask pillow")


def list_scenarios_ui():
    """列出所有支持的场景"""
    print("\n" + "="*70)
    print("支持的场景列表".center(70))
    print("="*70)

    scenarios = sorted(SCENARIO_TO_ROAD_STRUCTURE.keys())
    print(f"\n总共 {len(scenarios)} 个场景:\n")

    for i, scenario in enumerate(scenarios, 1):
        roads = SCENARIO_TO_ROAD_STRUCTURE[scenario]
        events = SCENARIO_TO_FINE_EVENTS.get(scenario, [])

        print(f"{i:2d}. {scenario:40s} | 道路: {len(roads)} | 事件: {len(events)}")

    print("\n" + "="*70)


def _print_primary_rs_summary(result: dict, prefix: str = "") -> None:
    """打印采集结果里的逐帧专一 ROAD_STRUCTURE 分布。"""
    counter = Counter()
    total = 0
    for route in result.get("routes", []):
        for ann in route.get("annotations", []):
            primary = ann.get("primary_road_structure")
            if primary:
                counter[primary] += 1
                total += 1
    if not total:
        print(f"{prefix}主 ROAD_STRUCTURE: 无可统计帧")
        return
    summary = ", ".join(f"{rs}={count}" for rs, count in sorted(counter.items()))
    print(f"{prefix}主 ROAD_STRUCTURE 分布: {summary} (total={total})")


def _annotation_summary(result: dict) -> Dict[str, Any]:
    """汇总逐帧 RS + EVENT 标注结果，用于调参和 smoke test。"""
    primary_counter = Counter()
    event_counter = Counter()
    review_counter = Counter()
    event_review_counter = Counter()
    xodr_counter = Counter()
    rule_kind_counter = Counter()
    route_count = 0
    annotated_route_count = 0
    skipped_route_count = 0
    data_missing_skip_count = 0
    skip_reason_counter = Counter()
    frame_count = 0
    transition_count = 0
    event_transition_count = 0
    smoothing_change_count = 0
    r4_recovery_change_count = 0
    confidence_values = []
    review_frame_count = 0
    event_review_frame_count = 0
    sample_comments = []

    scenario_results = result.get("results", result)
    if isinstance(scenario_results, dict) and "routes" in scenario_results:
        scenario_results = {scenario_results.get("scenario", "UNKNOWN"): scenario_results}

    for scenario, scenario_result in scenario_results.items():
        for skipped in scenario_result.get("data_missing_skipped", []):
            skipped_route_count += 1
            data_missing_skip_count += 1
            for reason in skipped.get("skip_reasons") or [skipped.get("skip_reason", "data_missing_skip")]:
                skip_reason_counter[str(reason)] += 1
        for route in scenario_result.get("routes", []):
            route_count += 1
            if route.get("status") == "data_missing_skip":
                skipped_route_count += 1
                data_missing_skip_count += 1
                for reason in route.get("skip_reasons") or [route.get("skip_reason", "data_missing_skip")]:
                    skip_reason_counter[str(reason)] += 1
                continue
            annotated_route_count += 1
            transition_count += len(route.get("primary_rs_transitions", []))
            event_transition_count += len(route.get("primary_event_transitions", []))
            r4_recovery_change_count += len(route.get("r4_context_recovery", {}).get("changes", []))
            smoothing_change_count += len(route.get("temporal_smoothing", {}).get("changes", []))
            for ann in route.get("annotations", []):
                frame_count += 1
                primary = ann.get("primary_road_structure")
                if primary:
                    primary_counter[primary] += 1
                primary_event = ann.get("primary_event")
                if primary_event:
                    event_counter[primary_event] += 1
                if ann.get("confidence") is not None:
                    confidence_values.append(float(ann.get("confidence")))
                evidence = ann.get("evidence", {})
                if evidence.get("rule_kind"):
                    rule_kind_counter[evidence["rule_kind"]] += 1
                xodr_source = evidence.get("xodr", {}).get("xodr_source") or "unavailable"
                xodr_counter[xodr_source] += 1
                if evidence.get("review_required"):
                    review_frame_count += 1
                    for reason in evidence.get("review_reasons", ["review_required"]):
                        review_counter[reason] += 1
                event_evidence = ann.get("event_evidence", {})
                if event_evidence.get("review_required"):
                    event_review_frame_count += 1
                    for reason in event_evidence.get("review_reasons", ["event_review_required"]):
                        event_review_counter[reason] += 1
                if len(sample_comments) < 12 and ann.get("frame_rs_annotation"):
                    sample_comments.append(
                        {
                            "scenario": scenario,
                            "route_id": route.get("route_id"),
                            "frame_id": ann.get("frame_id"),
                            "label": ann.get("frame_rs_annotation", {}).get("label"),
                            "event": ann.get("frame_event_annotation", {}).get("label"),
                            "review": ann.get("frame_rs_annotation", {}).get("review_required"),
                            "comment": ann.get("frame_rs_annotation", {}).get("comment"),
                            "event_comment": ann.get("frame_event_annotation", {}).get("comment"),
                        }
                    )

    confidence_stats = {"min": None, "avg": None, "max": None}
    if confidence_values:
        confidence_stats = {
            "min": round(min(confidence_values), 4),
            "avg": round(sum(confidence_values) / len(confidence_values), 4),
            "max": round(max(confidence_values), 4),
        }

    return {
        "route_count": route_count,
        "annotated_route_count": annotated_route_count,
        "skipped_route_count": skipped_route_count,
        "data_missing_skip_count": data_missing_skip_count,
        "skip_reason_distribution": dict(sorted(skip_reason_counter.items())),
        "frame_count": frame_count,
        "road_structure_labels": ROAD_STRUCTURE_LABELS,
        "event_labels": EVENT_LABELS,
        "primary_rs_distribution": dict(sorted(primary_counter.items())),
        "primary_event_distribution": dict(sorted(event_counter.items())),
        "review_reason_distribution": dict(sorted(review_counter.items())),
        "event_review_reason_distribution": dict(sorted(event_review_counter.items())),
        "xodr_source_distribution": dict(sorted(xodr_counter.items())),
        "rule_kind_distribution": dict(sorted(rule_kind_counter.items())),
        "confidence_stats": confidence_stats,
        "review_required_frame_count": review_frame_count,
        "review_required_ratio": round(review_frame_count / frame_count, 4) if frame_count else 0.0,
        "event_review_required_frame_count": event_review_frame_count,
        "event_review_required_ratio": round(event_review_frame_count / frame_count, 4) if frame_count else 0.0,
        "transition_count": transition_count,
        "event_transition_count": event_transition_count,
        "temporal_smoothing_change_count": smoothing_change_count,
        "r4_context_recovery_change_count": r4_recovery_change_count,
        "sample_comments": sample_comments,
    }


def _print_annotation_summary(summary: Dict[str, Any]) -> None:
    """打印逐帧 RS + EVENT 标注摘要。"""
    print("\n逐帧 RS + EVENT 标注摘要:")
    print(
        f"  routes={summary['route_count']} annotated_routes={summary.get('annotated_route_count', summary['route_count'])} "
        f"skipped_routes={summary.get('skipped_route_count', 0)} frames={summary['frame_count']} "
        f"transitions={summary['transition_count']} "
        f"event_transitions={summary.get('event_transition_count', 0)} "
        f"r4_recoveries={summary.get('r4_context_recovery_change_count', 0)} "
        f"smoothing_changes={summary.get('temporal_smoothing_change_count', 0)}"
    )
    if summary.get("skipped_route_count"):
        print(f"  skipped_reasons={summary.get('skip_reason_distribution', {})}")
    print(f"  primary_rs={summary['primary_rs_distribution']}")
    print(f"  primary_event={summary.get('primary_event_distribution', {})}")
    print("  RS 代号含义:")
    for code, meaning in summary.get("road_structure_labels", {}).items():
        print(f"    {code}: {meaning}")
    print("  EVENT 代号含义:")
    for code, meaning in summary.get("event_labels", {}).items():
        print(f"    {code}: {meaning}")
    print(f"  rule_kind={summary['rule_kind_distribution']}")
    print(f"  xodr_source={summary['xodr_source_distribution']}")
    print(f"  confidence={summary['confidence_stats']} review_ratio={summary['review_required_ratio']}")
    print(f"  review_reasons={summary['review_reason_distribution']}")
    print(
        f"  event_review_ratio={summary.get('event_review_required_ratio', 0.0)} "
        f"event_review_reasons={summary.get('event_review_reason_distribution', {})}"
    )
    if summary["sample_comments"]:
        print("  示例注释:")
        for item in summary["sample_comments"][:5]:
            print(
                f"    - {item['scenario']} / {item['route_id']} / frame {item['frame_id']}: "
                f"{item['comment']} | {item.get('event_comment')}"
            )


def run_frame_rs_annotation(
    scenarios: List[str],
    max_routes_per_scenario: Optional[int] = None,
    max_frames_per_route: Optional[int] = None,
    samples_per_town: Optional[int] = None,
    lead_data_root: str = "",
    output_dir: str = "",
    xml_root: str = "",
    carla_root: str = "",
    rule_config_json: str = "",
) -> Dict[str, Any]:
    """按每个 scenario 独立规则生成逐帧 primary ROAD_STRUCTURE 和 primary EVENT 标注。"""
    collector = ScenarioCollector(
        lead_data_root=lead_data_root,
        output_dir=output_dir,
        xml_root=xml_root,
        carla_root=carla_root,
        rule_config_json=rule_config_json,
    )
    if len(scenarios) == 1:
        result = collector.collect_one_scenario(
            scenarios[0],
            max_routes=max_routes_per_scenario,
            max_frames_per_route=max_frames_per_route,
            samples_per_town=samples_per_town,
        )
    else:
        result = collector.collect_multiple_scenarios(
            scenarios,
            max_routes_per_scenario=max_routes_per_scenario,
            max_frames_per_route=max_frames_per_route,
            samples_per_town=samples_per_town,
        )
    summary = _annotation_summary(result)
    summary_file = collector.output_dir / "frame_rs_annotation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    _print_annotation_summary(summary)
    print(f"\n✓ 标注结果写入: {collector.output_dir}")
    print(f"✓ 标注摘要写入: {summary_file}")
    return {"result": result, "summary": summary, "summary_file": str(summary_file)}


def run_frame_rs_annotation_ui():
    """交互式逐帧 RS/EVENT 标注入口。"""
    print("\n" + "="*70)
    print("逐帧 ROAD_STRUCTURE + EVENT 标注生成".center(70))
    print("="*70)
    scenario_text = input("场景名，逗号分隔；输入 all 跑全部 (默认 noScenarios): ").strip() or "noScenarios"
    if scenario_text.lower() == "all":
        scenarios = sorted(SCENARIO_TO_ROAD_STRUCTURE.keys())
    else:
        scenarios = [item.strip() for item in scenario_text.split(",") if item.strip()]
    invalid = [scenario for scenario in scenarios if scenario not in SCENARIO_TO_ROAD_STRUCTURE]
    if invalid:
        print(f"❌ 未知场景: {invalid}")
        return
    try:
        max_routes_text = input("每个场景最多 route 数，留空表示全量 (默认全量): ").strip()
        max_routes = int(max_routes_text) if max_routes_text else None
    except Exception:
        max_routes = None
    try:
        samples_per_town_text = input("每个 town 抽样 route 数；留空不用 per-town 抽样: ").strip()
        samples_per_town = int(samples_per_town_text) if samples_per_town_text else None
    except Exception:
        samples_per_town = None
    try:
        max_frames = int(input("每条 route 最多帧数，0 表示全部 (默认0): ") or "0")
    except Exception:
        max_frames = 0
    lead_root = input(f"LEAD数据根目录 (默认 {_DEFAULT_LEAD_DATA_ROOT}): ").strip()
    output_dir = input("输出目录 (默认 keyframe_filter/collection_output): ").strip()
    xml_root = input(f"XML根目录 (默认 {_DEFAULT_XML_ROOT}): ").strip()
    carla_root = input(f"CARLA/XODR根目录 (默认 {_DEFAULT_CARLA_ROOT}): ").strip()
    rule_config = input("规则阈值覆盖 JSON (可空): ").strip()
    run_frame_rs_annotation(
        scenarios=scenarios,
        max_routes_per_scenario=max_routes,
        max_frames_per_route=max_frames or None,
        samples_per_town=samples_per_town,
        lead_data_root=lead_root,
        output_dir=output_dir,
        xml_root=xml_root,
        carla_root=carla_root,
        rule_config_json=rule_config,
    )


def _find_xodr_file(town: str, carla_root: Path = _DEFAULT_CARLA_ROOT) -> Optional[Path]:
    """按 CARLA 0.9.15 常规与 AdditionalMaps 路径查找 town 对应 XODR。"""
    candidates = [
        carla_root / "CarlaUE4" / "Content" / "Carla" / "Maps" / "OpenDrive" / f"{town}.xodr",
        carla_root / "CarlaUE4" / "Content" / "Carla" / "Maps" / town / "OpenDrive" / f"{town}.xodr",
        carla_root / "AdditionalMaps_0.9.15" / "CarlaUE4" / "Content" / "Carla" / "Maps" / "OpenDrive" / f"{town}.xodr",
        carla_root / "AdditionalMaps_0.9.15" / "CarlaUE4" / "Content" / "Carla" / "Maps" / town / "OpenDrive" / f"{town}.xodr",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _summarize_xodr(town: str, carla_root: Path = _DEFAULT_CARLA_ROOT) -> dict:
    """轻量解析 XODR 顶层数量，不连接 CARLA。"""
    path = _find_xodr_file(town, carla_root)
    if path is None:
        return {"available": False, "path": None}
    summary = {"available": True, "path": str(path), "junctions": 0, "signals": 0, "controllers": 0, "stop_signals": 0}
    try:
        root = ET.parse(path).getroot()
        summary["junctions"] = len(root.findall(".//junction"))
        summary["signals"] = len(root.findall(".//signal"))
        summary["controllers"] = len(root.findall(".//controller"))
        stop_count = 0
        for sig in root.findall(".//signal"):
            if sig.get("type") == "206" or "stop" in str(sig.get("name", "")).lower():
                stop_count += 1
        summary["stop_signals"] = stop_count
    except Exception as exc:
        summary["parse_error"] = str(exc)
    return summary


def _point_at_geometry_s(geom: Dict[str, float], local_s: float) -> Tuple[float, float]:
    """按 OpenDRIVE planView geometry 粗略计算 s 位置坐标。"""
    local_s = max(0.0, min(float(local_s), float(geom.get("length", 0.0))))
    x = float(geom.get("x", 0.0))
    y = float(geom.get("y", 0.0))
    hdg = float(geom.get("hdg", 0.0))
    curvature = geom.get("curvature")
    if curvature is None or abs(float(curvature)) < 1e-9:
        return (x + local_s * math.cos(hdg), y + local_s * math.sin(hdg))

    curvature = float(curvature)
    radius = 1.0 / curvature
    cx = x - radius * math.sin(hdg)
    cy = y + radius * math.cos(hdg)
    theta = hdg + local_s * curvature
    return (cx + radius * math.sin(theta), cy - radius * math.cos(theta))


def _geometry_bbox(geom: Dict[str, float]) -> Tuple[float, float, float, float]:
    """给 geometry 生成采样 bbox，用于空间近邻粗筛。"""
    length = float(geom.get("length", 0.0))
    sample_count = 2 if geom.get("curvature") is None else 8
    points = [
        _point_at_geometry_s(geom, length * idx / max(1, sample_count - 1))
        for idx in range(sample_count)
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    pad = 8.0
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _bbox_min_distance(point: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> float:
    """点到 bbox 的最小可能距离，用于跳过明显无关 road geometry。"""
    px, py = point
    min_x, min_y, max_x, max_y = bbox
    dx = max(min_x - px, 0.0, px - max_x)
    dy = max(min_y - py, 0.0, py - max_y)
    return math.hypot(dx, dy)


def _distance_to_geometry(point: Tuple[float, float], geom: Dict[str, float]) -> float:
    """用线段采样近似点到 OpenDRIVE geometry 的距离。"""
    length = float(geom.get("length", 0.0))
    if length <= 1e-6:
        gx, gy = _point_at_geometry_s(geom, 0.0)
        return math.hypot(point[0] - gx, point[1] - gy)
    sample_count = max(2, min(24, int(length / 8.0) + 2))
    best = math.inf
    prev = _point_at_geometry_s(geom, 0.0)
    for idx in range(1, sample_count):
        local_s = length * idx / (sample_count - 1)
        cur = _point_at_geometry_s(geom, local_s)
        best = min(best, _distance_point_to_segment(point, prev, cur))
        prev = cur
    return best


def _distance_point_to_segment(
    point: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """计算点到线段距离。"""
    px, py = point
    ax, ay = a
    bx, by = b
    vx = bx - ax
    vy = by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / denom))
    qx = ax + t * vx
    qy = ay + t * vy
    return math.hypot(px - qx, py - qy)


def _parse_xodr_spatial_index(town: str, carla_root: Path = _DEFAULT_CARLA_ROOT) -> Dict[str, Any]:
    """轻量解析 XODR planView/signal/junction road，用于无 CARLA API 的空间审计。"""
    path = _find_xodr_file(town, carla_root)
    if path is None:
        return {"available": False, "path": None, "roads": {}, "signals": []}
    out: Dict[str, Any] = {"available": True, "path": str(path), "roads": {}, "signals": []}
    try:
        root = ET.parse(path).getroot()
        junction_roads = set()
        for connection in root.findall(".//junction/connection"):
            incoming = connection.get("incomingRoad")
            connecting = connection.get("connectingRoad")
            if incoming:
                junction_roads.add(str(incoming))
            if connecting:
                junction_roads.add(str(connecting))

        for road in root.findall(".//road"):
            road_id = str(road.get("id", ""))
            if not road_id:
                continue
            geometries = []
            for geom in road.findall(".//planView/geometry"):
                item = {
                    "s": float(geom.get("s", 0.0)),
                    "x": float(geom.get("x", 0.0)),
                    "y": float(geom.get("y", 0.0)),
                    "hdg": float(geom.get("hdg", 0.0)),
                    "length": float(geom.get("length", 0.0)),
                }
                arc = geom.find("arc")
                if arc is not None and arc.get("curvature") is not None:
                    item["curvature"] = float(arc.get("curvature", "0"))
                item["bbox"] = _geometry_bbox(item)
                geometries.append(item)
            road_entry = {
                "id": road_id,
                "junction": road.get("junction", "-1"),
                "is_junction_road": road.get("junction", "-1") not in {"", "-1"} or road_id in junction_roads,
                "geometries": geometries,
            }
            out["roads"][road_id] = road_entry
            for signal in road.findall(".//signals/signal"):
                sig_s = float(signal.get("s", "0") or 0.0)
                geom = None
                for candidate in reversed(geometries):
                    if sig_s >= candidate.get("s", 0.0):
                        geom = candidate
                        break
                if geom is None and geometries:
                    geom = geometries[0]
                point = None
                if geom is not None:
                    point = _point_at_geometry_s(geom, sig_s - float(geom.get("s", 0.0)))
                out["signals"].append(
                    {
                        "road_id": road_id,
                        "id": signal.get("id"),
                        "name": signal.get("name"),
                        "type": signal.get("type"),
                        "subtype": signal.get("subtype"),
                        "point": point,
                    }
                )
    except Exception as exc:
        out["parse_error"] = str(exc)
    return out


def _xodr_spatial_probe_for_xml(info, spatial_index: Dict[str, Any]) -> Dict[str, Any]:
    """用 XODR 静态几何粗看 XML route/trigger 附近的 road/junction/signal。"""
    if not spatial_index.get("available"):
        return {"available": False}

    probe_points = list(info.trigger_points)
    if info.waypoints:
        probe_points.extend([info.waypoints[0], info.waypoints[len(info.waypoints) // 2], info.waypoints[-1]])
    probe_points = [point for point in probe_points if point is not None]
    if not probe_points:
        return {"available": True, "probe_points": 0}

    nearest_roads = []
    for point in probe_points[:8]:
        best = None
        for road in spatial_index.get("roads", {}).values():
            for geom in road.get("geometries", []):
                bbox = geom.get("bbox")
                if best is not None and bbox is not None and _bbox_min_distance(point, bbox) > best["distance_m"]:
                    continue
                dist = _distance_to_geometry(point, geom)
                if best is None or dist < best["distance_m"]:
                    best = {
                        "road_id": road.get("id"),
                        "distance_m": round(dist, 2),
                        "is_junction_road": bool(road.get("is_junction_road")),
                    }
        if best is not None:
            best["point"] = _tag_scalar(point)
            nearest_roads.append(best)

    signal_distances = []
    for point in probe_points[:8]:
        distances = []
        for signal in spatial_index.get("signals", []):
            sig_point = signal.get("point")
            if sig_point is None:
                continue
            distances.append(math.hypot(point[0] - sig_point[0], point[1] - sig_point[1]))
        if distances:
            signal_distances.append(min(distances))

    nearest_signal = min(signal_distances) if signal_distances else math.inf
    junction_hits = [road for road in nearest_roads if road.get("is_junction_road") and road.get("distance_m", math.inf) <= 12.0]
    return {
        "available": True,
        "probe_points": len(probe_points),
        "nearest_roads": nearest_roads[:6],
        "nearest_signal_m": round(nearest_signal, 2) if math.isfinite(nearest_signal) else None,
        "near_junction_road_points": len(junction_hits),
        "has_near_signal_60m": math.isfinite(nearest_signal) and nearest_signal <= 60.0,
        "has_near_junction_road": bool(junction_hits),
    }


def _polyline_length(points: List[Tuple[float, float]]) -> float:
    """计算 XML waypoint 折线长度。"""
    if len(points) < 2:
        return 0.0
    return sum(
        ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        for a, b in zip(points[:-1], points[1:])
    )


def _heading_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """计算二维向量航向角。"""
    import math

    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _angle_delta_deg(a: float, b: float) -> float:
    """计算两个角度的最小差值。"""
    return abs((b - a + 180.0) % 360.0 - 180.0)


def _route_heading_change(points: List[Tuple[float, float]]) -> float:
    """用首末有效线段估算 route 总转角。"""
    if len(points) < 3:
        return 0.0
    first = None
    last = None
    for a, b in zip(points[:-1], points[1:]):
        if ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 > 1e-3:
            first = _heading_deg(a, b)
            break
    for a, b in reversed(list(zip(points[:-1], points[1:]))):
        if ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 > 1e-3:
            last = _heading_deg(a, b)
            break
    if first is None or last is None:
        return 0.0
    return round(_angle_delta_deg(first, last), 2)


def _tag_scalar(value: Any) -> Any:
    """把 XML tag 值压缩成 JSON 友好的摘要。"""
    if isinstance(value, tuple):
        return [round(float(v), 3) for v in value]
    if isinstance(value, dict):
        return {k: _tag_scalar(v) for k, v in value.items()}
    if isinstance(value, float):
        return round(value, 3)
    return value


def _sample_xml_summary(info) -> Dict[str, Any]:
    """抽取单个 XML 的 route/trigger/tag 画像。"""
    tag_keys = sorted({k for tag in info.scenario_tags for k in tag.keys() if k not in {"name", "type"}})
    trigger_points = [_tag_scalar(point) for point in info.trigger_points]
    scenario_tags = []
    for tag in info.scenario_tags[:4]:
        scenario_tags.append(
            {
                key: _tag_scalar(value)
                for key, value in tag.items()
                if key not in {"name", "type"}
            }
        )
    return {
        "xml": str(info.path),
        "route_id": info.route_id,
        "town": info.town,
        "waypoint_count": len(info.waypoints),
        "route_length_m": round(_polyline_length(info.waypoints), 2),
        "route_heading_change_deg": _route_heading_change(info.waypoints),
        "scenario_tag_count": len(info.scenario_tags),
        "tag_keys": tag_keys,
        "trigger_points": trigger_points,
        "first_waypoint": info.waypoints[0] if info.waypoints else None,
        "last_waypoint": info.waypoints[-1] if info.waypoints else None,
        "sampled_scenario_tags": scenario_tags,
    }


def _index_lead_routes_for_scenario(lead_data_root: Path, scenario: str) -> Dict[str, Path]:
    """按 route 数字索引 LEAD 数据目录，缺数据时返回空索引。"""
    scenario_dir = Path(lead_data_root) / scenario
    if not scenario_dir.exists():
        return {}
    out = {}
    for route_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
        should_exclude, _abnormal_info = is_abnormal_lead_route(route_dir, scenario)
        if should_exclude:
            continue
        route_num = _extract_route_num(route_dir.name)
        if route_num:
            out.setdefault(route_num, route_dir)
    return out


def _select_meta_samples(meta_files: List[Path], max_samples: int = 3) -> List[Path]:
    """抽 first/mid/last meta，用于判断该 route 的运行时字段形态。"""
    if not meta_files:
        return []
    if len(meta_files) <= max_samples:
        return list(meta_files)
    indexes = {
        round(i * (len(meta_files) - 1) / (max_samples - 1))
        for i in range(max_samples)
    }
    return [meta_files[i] for i in sorted(indexes)]


def _meta_bool(value: Any) -> bool:
    """稳妥压缩 meta bool-like 字段。"""
    if hasattr(value, "reshape"):
        try:
            value = value.reshape(-1)[0]
        except Exception:
            return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _meta_scalar(value: Any) -> Optional[float]:
    """稳妥压缩 meta 数值字段。"""
    if hasattr(value, "reshape"):
        try:
            value = value.reshape(-1)[0]
        except Exception:
            return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _summarize_meta_probe(info, route_dir: Optional[Path]) -> Dict[str, Any]:
    """读取可用 LEAD meta，摘要运行时 RS 相关字段。"""
    if route_dir is None:
        return {"available": False, "reason": "lead_route_not_matched"}
    metas_dir = route_dir / "metas"
    if not metas_dir.exists():
        return {"available": False, "route_dir": str(route_dir), "reason": "metas_dir_missing"}
    meta_files = sorted(metas_dir.glob("*.pkl"))
    if not meta_files:
        return {"available": False, "route_dir": str(route_dir), "reason": "meta_files_missing"}

    sampled = _select_meta_samples(meta_files, max_samples=3)
    fields_counter = Counter()
    traffic_light_values = Counter()
    active_values = Counter()
    bool_counts = Counter()
    finite_distances: Dict[str, List[float]] = defaultdict(list)
    frame_ids = []
    load_errors = []

    for meta_path in sampled:
        frame_ids.append(meta_path.stem)
        try:
            meta = load_pickle_file(meta_path)
        except Exception as exc:
            load_errors.append(f"{meta_path.name}:{exc}")
            continue
        if not isinstance(meta, dict):
            load_errors.append(f"{meta_path.name}:not_dict")
            continue
        fields_counter.update(meta.keys())
        traffic_light_values[str(meta.get("traffic_light_state", None))] += 1
        active_values[str(meta.get("current_active_scenario_type", None))] += 1
        for key in ("light_hazard", "stop_sign_hazard", "stop_sign_close", "is_junction", "is_intersection"):
            if _meta_bool(meta.get(key, False)):
                bool_counts[key] += 1
        for key, value in meta.items():
            if key.startswith("dist_to_") or key in {"dist_to_junction", "distance_to_next_junction"}:
                scalar = _meta_scalar(value)
                if scalar is not None:
                    finite_distances[key].append(scalar)

    distance_summary = {
        key: {
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "count": len(values),
        }
        for key, values in sorted(finite_distances.items())
    }
    return {
        "available": not load_errors or len(load_errors) < len(sampled),
        "route_dir": str(route_dir),
        "meta_file_count": len(meta_files),
        "sampled_meta_count": len(sampled),
        "sampled_frame_ids": frame_ids,
        "fields_present_top": dict(fields_counter.most_common(40)),
        "traffic_light_values": dict(traffic_light_values),
        "active_scenario_values": dict(active_values.most_common(12)),
        "bool_true_counts": dict(bool_counts),
        "finite_distance_fields": distance_summary,
        "load_errors": load_errors[:5],
    }


def _select_diverse_xml_samples(town_infos: List[Any], samples_per_town: int) -> List[Any]:
    """每个 town 分散抽样 route，用于人工/规则审计数据形态。"""
    if not town_infos:
        return []
    if len(town_infos) <= samples_per_town:
        return list(town_infos)

    target_count = max(3, samples_per_town)
    target_count = min(target_count, len(town_infos))
    if target_count == 1:
        return [town_infos[0]]

    indexes = {
        round(i * (len(town_infos) - 1) / (target_count - 1))
        for i in range(target_count)
    }
    sampled = [town_infos[i] for i in sorted(indexes)]

    # 极少数情况下 round 会碰撞；从剩余 route 中补齐，尽量覆盖首/中/尾不同数据形态。
    if len(sampled) < target_count:
        selected_paths = {item.path for item in sampled}
        for item in town_infos:
            if item.path in selected_paths:
                continue
            sampled.append(item)
            selected_paths.add(item.path)
            if len(sampled) >= target_count:
                break
    return sampled


def _logic_validation_notes(kind: str, towns: Dict[str, Any], tag_counter: Counter) -> List[str]:
    """根据抽样 XML/XODR 形态给当前 policy 假设写审计备注。"""
    notes = []
    observed_tags = set(tag_counter)
    any_xodr = any(item.get("xodr", {}).get("available") for item in towns.values())
    any_signal = any((item.get("xodr", {}).get("signals", 0) or 0) > 0 for item in towns.values())
    any_controller = any((item.get("xodr", {}).get("controllers", 0) or 0) > 0 for item in towns.values())
    sample_spatial = [
        sample.get("xodr_spatial_probe", {})
        for item in towns.values()
        for sample in item.get("sampled_xml", [])
    ]
    near_signal_samples = sum(1 for item in sample_spatial if item.get("has_near_signal_60m"))
    near_junction_samples = sum(1 for item in sample_spatial if item.get("has_near_junction_road"))
    meta_items = [
        sample.get("lead_meta_probe", {})
        for item in towns.values()
        for sample in item.get("sampled_xml", [])
    ]
    meta_available_samples = sum(1 for item in meta_items if item.get("available"))
    meta_light_samples = sum(1 for item in meta_items if item.get("traffic_light_values"))
    meta_active_samples = sum(1 for item in meta_items if item.get("active_scenario_values"))

    if kind in {"highway_merge", "interurban", "interurban_advanced"}:
        if observed_tags & {"start_actor_flow", "end_actor_flow", "other_actor_location"}:
            notes.append("抽样 XML 含 actor-flow/other-actor 锚点，R3 窗口假设可由 XML 支撑。")
        else:
            notes.append("抽样 XML 未看到 actor-flow 锚点，R3 需要更多依赖 XODR 或 meta active window。")
    if kind in {"twoways_obstacle", "invading_turn", "vehicle_opens_door_twoways"}:
        if "distance" in observed_tags or "offset" in observed_tags:
            notes.append("抽样 XML 含 distance/offset，TwoWays/InvadingTurn 核心窗口可被 route trigger 约束。")
        else:
            notes.append("抽样 XML 缺少 distance/offset，R2 边界需依赖 trigger close 与 meta active scenario 审计。")
    if kind in {"signalized_junction", "defect_junction"}:
        if near_signal_samples:
            notes.append(f"抽样 XML trigger/route 附近有 {near_signal_samples} 个样本靠近 XODR signal，R4/故障灯窗口有空间证据。")
        elif any_signal or any_controller:
            notes.append("抽样 town 的 XODR 存在 signal/controller，信号灯/故障灯路口假设有地图侧支撑。")
        else:
            notes.append("抽样 town 的轻量 XODR 摘要未见 signal/controller，需依赖 meta 灯态确认 R4/R5。")
        if meta_light_samples:
            notes.append(f"抽样 LEAD meta 中有 {meta_light_samples} 个 route 样本含 traffic_light_state 分布，可用于运行时 R4/R5 复核。")
    if kind == "nonsignalized_junction":
        if near_junction_samples and not near_signal_samples:
            notes.append(f"抽样 XML trigger/route 附近有 {near_junction_samples} 个样本靠近 junction road 且未近邻 signal，R5 假设较一致。")
        elif any_controller:
            notes.append("抽样 town 的 XODR 存在 controller；无灯/路权标签需要运行时用 meta 灯态做冲突审计。")
        else:
            notes.append("抽样 town 未见明显 controller，R5 无灯/路权假设相对一致。")
    if kind in {"parking", "parking_exit", "static_cutin", "vehicle_opens_door_twoways"}:
        if "direction" in observed_tags or "front_vehicle_distance" in observed_tags or "behind_vehicle_distance" in observed_tags:
            notes.append("抽样 XML 含 direction/front/behind 等停车侧或停车空隙线索，可辅助 R1/R2 与 EVENT 窗口定位。")
        else:
            notes.append("抽样 XML 停车侧线索有限，停车/遮挡只作为 R1/R2 与 EVENT 弱证据。")
    if not any_xodr:
        notes.append("本地未找到对应 XODR；当前画像只能验证 XML 形态，运行时规则会自动降级。")
    if meta_available_samples:
        notes.append(f"抽样 LEAD meta 可读 route 数={meta_available_samples}；active_scenario 可读 route 数={meta_active_samples}。")
    else:
        notes.append("当前环境未读到匹配 LEAD meta；完整调研状态会标记 meta 缺失，后续需在远端数据环境复跑画像。")
    return notes or ["抽样 XML/XODR 形态未触发特殊备注；按该场景 policy 保守生成 primary RS。"]


def _town_audit_summary(kind: str, town_entry: Dict[str, Any]) -> Dict[str, Any]:
    """按 town 汇总抽样 XML/XODR 是否支撑当前场景规则假设。"""
    samples = town_entry.get("sampled_xml", [])
    spatial_items = [sample.get("xodr_spatial_probe", {}) for sample in samples]
    near_signal = sum(1 for item in spatial_items if item.get("has_near_signal_60m"))
    near_junction = sum(1 for item in spatial_items if item.get("has_near_junction_road"))
    xodr_available = bool(town_entry.get("xodr", {}).get("available"))
    meta_items = [sample.get("lead_meta_probe", {}) for sample in samples]
    meta_available_count = sum(1 for item in meta_items if item.get("available"))
    tag_counter = Counter()
    for sample in samples:
        tag_counter.update(sample.get("tag_keys", []))

    assumptions = []
    if kind in {"signalized_junction", "defect_junction"}:
        assumptions.append({
            "name": "signal_or_controlled_junction",
            "supported": near_signal > 0 or (town_entry.get("xodr", {}).get("signals", 0) or 0) > 0,
            "evidence": f"near_signal_samples={near_signal}, town_signals={town_entry.get('xodr', {}).get('signals', 0)}",
        })
    if kind == "nonsignalized_junction":
        assumptions.append({
            "name": "junction_without_near_signal",
            "supported": near_junction > 0 and near_signal == 0,
            "evidence": f"near_junction_samples={near_junction}, near_signal_samples={near_signal}",
        })
    if kind in {"twoways_obstacle", "invading_turn", "vehicle_opens_door_twoways"}:
        assumptions.append({
            "name": "two_way_window_has_xml_distance_or_offset",
            "supported": bool(set(tag_counter) & {"distance", "offset", "frequency"}),
            "evidence": f"tag_keys={sorted(set(tag_counter) & {'distance', 'offset', 'frequency'})}",
        })
    if kind in {"highway_merge", "interurban", "interurban_advanced"}:
        assumptions.append({
            "name": "merge_actor_flow_has_xml_anchor",
            "supported": bool(set(tag_counter) & {"start_actor_flow", "end_actor_flow", "other_actor_location"}),
            "evidence": f"tag_keys={sorted(set(tag_counter) & {'start_actor_flow', 'end_actor_flow', 'other_actor_location'})}",
        })
    if kind in {"parking", "parking_exit", "static_cutin", "vehicle_opens_door_twoways"}:
        assumptions.append({
            "name": "parking_side_or_gap_has_xml_anchor",
            "supported": bool(set(tag_counter) & {"direction", "front_vehicle_distance", "behind_vehicle_distance"}),
            "evidence": f"tag_keys={sorted(set(tag_counter) & {'direction', 'front_vehicle_distance', 'behind_vehicle_distance'})}",
        })
    if not assumptions:
        assumptions.append({
            "name": "conservative_default_policy",
            "supported": True,
            "evidence": "该 policy 主要依赖 meta 灯态/路口字段，XML/XODR 仅作审计。",
        })

    unsupported = [item for item in assumptions if not item["supported"]]
    readable_lead_runs = int(town_entry.get("readable_lead_run_count", 0) or 0)
    expected_samples = min(5, int(town_entry.get("xml_count", 0) or 0))
    if readable_lead_runs > 0:
        expected_samples = min(expected_samples, readable_lead_runs)
    complete_inputs = {
        "xml_sample_sufficient": len(samples) >= expected_samples,
        "xodr_available": xodr_available,
        "meta_sample_available": meta_available_count >= expected_samples if expected_samples else False,
    }
    incomplete_reasons = [key for key, ok in complete_inputs.items() if not ok]
    return {
        "xodr_available": xodr_available,
        "sample_count": len(samples),
        "expected_min_sample_count": expected_samples,
        "meta_available_sample_count": meta_available_count,
        "near_signal_sample_count": near_signal,
        "near_junction_sample_count": near_junction,
        "assumption_checks": assumptions,
        "complete_investigation_inputs": complete_inputs,
        "complete_investigation": not incomplete_reasons,
        "incomplete_investigation_reasons": incomplete_reasons,
        "needs_manual_review": bool(unsupported) or bool(incomplete_reasons),
        "manual_review_reason": [item["name"] for item in unsupported] + incomplete_reasons,
    }


def _build_scenario_policy_plan(scenario: str, scenario_entry: Dict[str, Any]) -> Dict[str, Any]:
    """根据设计文档和本地 XML/XODR 画像生成该场景的可执行 RS 规则计划。"""
    kind = scenario_entry["rule_kind"]
    template = _POLICY_LOGIC_BY_KIND.get(kind, _POLICY_LOGIC_BY_KIND["default_meta_map"])
    towns = scenario_entry.get("towns", {})
    xodr_available_towns = [
        town for town, item in towns.items()
        if item.get("xodr", {}).get("available")
    ]
    tag_counter = Counter()
    route_lengths = []
    heading_changes = []
    sampled_xml_count = 0
    town_sampled_route_ids = {}
    for item in towns.values():
        tag_counter.update(item.get("top_tag_keys", {}))
        sampled = item.get("sampled_xml", [])
        sampled_xml_count += len(sampled)
        for sample in sampled:
            route_lengths.append(sample.get("route_length_m", 0.0))
            heading_changes.append(sample.get("route_heading_change_deg", 0.0))
    for town, item in towns.items():
        town_sampled_route_ids[town] = [
            sample.get("route_id")
            for sample in item.get("sampled_xml", [])
        ]
    town_audits = {
        town: _town_audit_summary(kind, item)
        for town, item in towns.items()
    }
    complete_towns = [town for town, item in town_audits.items() if item.get("complete_investigation")]
    incomplete_towns = {
        town: item.get("incomplete_investigation_reasons", [])
        for town, item in town_audits.items()
        if not item.get("complete_investigation")
    }

    return {
        "keeps_forced_candidate_fill": True,
        "candidate_pool_from_scenario": scenario_entry["road_candidates"],
        "primary_label_output": "primary_road_structure",
        "secondary_label_output": "secondary_road_structures",
        "implemented_by": {
            "collector_engine": "RoadStructureRuleEngine.analyze",
            "rule_kind": kind,
            "rule_config": scenario_entry["rule_config"],
        },
        "xml_sampling_contract": {
            "all_towns_read": sorted(towns.keys()),
            "per_town_audit_sample_requested": scenario_entry.get("samples_per_town", 3),
            "sampled_xml_total": sampled_xml_count,
            "sampled_route_ids_by_town": town_sampled_route_ids,
            "note": "每个 town 抽至少 5 条 route/id 是为了观察该场景在这个 town 的数据形态、检查规则假设是否站得住；不是把“5 个不同 id”当成标签生成条件。",
        },
        "xodr_contract": {
            "towns_with_xodr": sorted(xodr_available_towns),
            "towns_without_xodr": sorted(set(towns) - set(xodr_available_towns)),
            "usage": template["xml_xodr_usage"],
        },
        "route_observation": {
            "sample_route_length_min_m": round(min(route_lengths), 2) if route_lengths else 0.0,
            "sample_route_length_max_m": round(max(route_lengths), 2) if route_lengths else 0.0,
            "sample_heading_change_max_deg": round(max(heading_changes), 2) if heading_changes else 0.0,
            "top_xml_tag_keys": dict(tag_counter.most_common(12)),
        },
        "logic_validation_from_samples": _logic_validation_notes(kind, towns, tag_counter),
        "town_audit_summary": town_audits,
        "complete_investigation_status": {
            "is_complete": len(complete_towns) == len(towns) and bool(towns),
            "complete_towns": sorted(complete_towns),
            "incomplete_towns": incomplete_towns,
            "definition": "每个有 XML 的 town 至少审计 min(5, xml_count, readable_lead_run_count if >0) 条分散 route/id，并且这些样本同时有 XML、XODR 静态画像、可读 LEAD meta 摘要，才算该场景调研完整。",
        },
        "frame_primary_rules": template["primary_rules"],
        "arbitration": [
            "CrossJunctionDefectTrafficLight 固定 R5 覆盖 R4",
            "普通优先级按 R4/R5 > R3 > R2 > R1，分数接近时使用全局优先级",
            "noScenarios 无灯态时强制 conservative R1",
            "只允许输出 candidate_pool_from_scenario 中已有的 RS，避免规则越界",
        ],
        "meta_fields_used_at_runtime": [
            "pos_global/ego_matrix",
            "traffic_light_state/light_hazard",
            "is_junction/is_intersection/dist_to_junction/distance_to_next_junction",
            "current_active_scenario_type",
            "stop_sign_hazard/stop_sign_close",
            "dist_to_* fields as event/window confidence only",
        ],
    }


def _scenario_town_xml_audit(
    index: RouteXmlIndex,
    carla_root: Path,
    samples_per_town: int = 3,
    lead_data_root: Path = _DEFAULT_LEAD_DATA_ROOT,
) -> dict:
    """逐场景、逐 town 抽样 XML，并附上 XODR/meta 粗画像。"""
    report = {
        "xml_root": str(index.xml_root),
        "carla_root": str(carla_root),
        "lead_data_root": str(lead_data_root),
        "lead_data_available": Path(lead_data_root).exists(),
        "samples_per_town": samples_per_town,
        "scenarios": {},
    }
    xodr_cache = {}
    xodr_spatial_cache = {}

    for scenario in sorted(SCENARIO_TO_ROAD_STRUCTURE):
        infos = index.by_scenario.get(scenario, [])
        lead_route_index = _index_lead_routes_for_scenario(Path(lead_data_root), scenario)
        by_town = defaultdict(list)
        for info in infos:
            by_town[info.town or "UNKNOWN"].append(info)

        scenario_entry = {
            "rule_kind": SCENARIO_RULE_CONFIG.get(scenario, {}).get("kind", SCENARIO_RULE_KIND.get(scenario, "default_meta_map")),
            "rule_config": SCENARIO_RULE_CONFIG.get(scenario, {}),
            "road_candidates": [rs.value for rs in SCENARIO_TO_ROAD_STRUCTURE.get(scenario, [])],
            "xml_count": len(infos),
            "samples_per_town": samples_per_town,
            "towns": {},
        }

        for town, town_infos in sorted(by_town.items()):
            if town not in xodr_cache:
                xodr_cache[town] = _summarize_xodr(town, carla_root)
            if town not in xodr_spatial_cache:
                xodr_spatial_cache[town] = _parse_xodr_spatial_index(town, carla_root)
            sampled = _select_diverse_xml_samples(town_infos, samples_per_town)
            wp_counts = [len(info.waypoints) for info in town_infos]
            tag_counter = Counter()
            samples = []
            for info in sampled:
                for tag in info.scenario_tags:
                    for key in tag:
                        if key not in {"name", "type"}:
                            tag_counter[key] += 1
                sample_summary = _sample_xml_summary(info)
                sample_summary["xodr_spatial_probe"] = _xodr_spatial_probe_for_xml(info, xodr_spatial_cache[town])
                route_num = _extract_route_num(info.route_id or info.path.stem)
                route_dir = lead_route_index.get(route_num) if route_num else None
                sample_summary["lead_meta_probe"] = _summarize_meta_probe(info, route_dir)
                samples.append(sample_summary)
            scenario_entry["towns"][town] = {
                "xml_count": len(town_infos),
                "waypoint_count_min": min(wp_counts) if wp_counts else 0,
                "waypoint_count_avg": round(sum(wp_counts) / len(wp_counts), 2) if wp_counts else 0,
                "waypoint_count_max": max(wp_counts) if wp_counts else 0,
                "xodr": xodr_cache[town],
                "top_tag_keys": dict(tag_counter.most_common(12)),
                "sampled_xml": samples,
            }
        scenario_entry["generated_frame_label_logic"] = _build_scenario_policy_plan(scenario, scenario_entry)
        report["scenarios"][scenario] = scenario_entry
    return report


def road_structure_xml_xodr_audit_ui():
    """按 ROAD_STRUCTURE 设计文档审计 XML/XODR 输入覆盖。"""
    print("\n" + "="*70)
    print("ROAD_STRUCTURE XML/XODR画像".center(70))
    print("="*70)

    xml_root_text = input(f"XML根目录 (默认 {_DEFAULT_XML_ROOT}): ").strip()
    carla_root_text = input(f"CARLA根目录 (默认 {_DEFAULT_CARLA_ROOT}): ").strip()
    lead_root_text = input(f"LEAD数据根目录 (默认 {_DEFAULT_LEAD_DATA_ROOT}): ").strip()
    try:
        samples_per_town = int(input("每个town用于验证思路的route/id抽样数 (默认3): ") or "3")
    except Exception:
        samples_per_town = 3

    xml_root = Path(xml_root_text) if xml_root_text else _DEFAULT_XML_ROOT
    carla_root = Path(carla_root_text) if carla_root_text else _DEFAULT_CARLA_ROOT
    lead_data_root = Path(lead_root_text) if lead_root_text else _DEFAULT_LEAD_DATA_ROOT
    index = RouteXmlIndex(xml_root)
    report = _scenario_town_xml_audit(
        index,
        carla_root,
        lead_data_root=lead_data_root,
        samples_per_town=max(1, samples_per_town),
    )

    output_dir = Path(__file__).resolve().parent / "collection_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "road_structure_xml_xodr_audit.json"
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)

    print(f"\n✓ 已审计 {len(report['scenarios'])} 个场景")
    for scenario, item in report["scenarios"].items():
        towns = ",".join(item["towns"].keys()) or "NONE"
        logic = item.get("generated_frame_label_logic", {})
        candidates = "/".join(logic.get("candidate_pool_from_scenario", []))
        complete = logic.get("complete_investigation_status", {}).get("is_complete", False)
        print(
            f"  - {scenario:<40} xml={item['xml_count']:<4} "
            f"rule={item['rule_kind']:<28} candidates={candidates:<11} complete={str(complete):<5} towns={towns}"
        )
    print(f"\n完整画像已保存: {output_file}")
    print("画像中 generated_frame_label_logic 字段即每个场景的帧级 primary RS 生成逻辑。")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""
    while True:
        print_main_menu()
        choice = input("请选择 (1-10): ").strip()

        if choice == '1':
            collect_one_scenario_all_ui()
        elif choice == '2':
            collect_one_scenario_limited_ui()
        elif choice == '3':
            collect_multiple_scenarios_ui()
        elif choice == '4':
            collect_all_scenarios_ui()
        elif choice == '5':
            run_analysis_ui()
        elif choice == '6':
            run_web_app_ui()
        elif choice == '7':
            list_scenarios_ui()
        elif choice == '8':
            road_structure_xml_xodr_audit_ui()
        elif choice == '9':
            run_frame_rs_annotation_ui()
        elif choice == '10':
            print("\n👋 再见！\n")
            break
        else:
            print("❌ 无效选择")

        if choice not in ['6']:  # Web应用特殊处理
            input("\n按 Enter 继续...")


def _run_cli(argv: List[str]) -> bool:
    """非交互命令入口；返回 True 表示已处理。"""
    if not argv:
        return False
    parser = argparse.ArgumentParser(description="LEAD keyframe ROAD_STRUCTURE / EVENT tools")
    subparsers = parser.add_subparsers(dest="command")

    annotate = subparsers.add_parser("annotate-rs", help="按每场景规则生成逐帧 RS + EVENT 标注")
    annotate.add_argument("--scenario", default="noScenarios", help="场景名、逗号分隔，或 all")
    annotate.add_argument("--max-routes", type=int, default=None, help="每个场景最多处理 route 数；不传表示全量")
    annotate.add_argument("--samples-per-town", type=int, default=None, help="每个 town 分散抽样 route 数；设置后优先于 --max-routes")
    annotate.add_argument("--max-frames-per-route", type=int, default=0, help="每条 route 最多处理帧数，0 表示全部")
    annotate.add_argument("--lead-data-root", default=str(_DEFAULT_LEAD_DATA_ROOT))
    annotate.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "collection_output"))
    annotate.add_argument("--xml-root", default=str(_DEFAULT_XML_ROOT))
    annotate.add_argument("--carla-root", default=str(_DEFAULT_CARLA_ROOT))
    annotate.add_argument("--rule-config-json", default="", help="可选：每场景阈值覆盖 JSON")

    args = parser.parse_args(argv)
    if args.command == "annotate-rs":
        if args.scenario.lower() == "all":
            scenarios = sorted(SCENARIO_TO_ROAD_STRUCTURE.keys())
        else:
            scenarios = [item.strip() for item in args.scenario.split(",") if item.strip()]
        invalid = [scenario for scenario in scenarios if scenario not in SCENARIO_TO_ROAD_STRUCTURE]
        if invalid:
            raise ValueError(f"未知场景: {invalid}")
        run_frame_rs_annotation(
            scenarios=scenarios,
            max_routes_per_scenario=max(1, args.max_routes) if args.max_routes is not None else None,
            max_frames_per_route=args.max_frames_per_route if args.max_frames_per_route > 0 else None,
            samples_per_town=max(1, args.samples_per_town) if args.samples_per_town is not None else None,
            lead_data_root=args.lead_data_root,
            output_dir=args.output_dir,
            xml_root=args.xml_root,
            carla_root=args.carla_root,
            rule_config_json=args.rule_config_json,
        )
        return True
    return False


if __name__ == "__main__":
    try:
        if not _run_cli(sys.argv[1:]):
            main()
    except KeyboardInterrupt:
        print("\n\n👋 已中断\n")
    except Exception as e:
        print(f"\n❌ 错误: {e}\n")
        import traceback
        traceback.print_exc()
