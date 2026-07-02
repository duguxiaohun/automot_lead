"""
快速启动脚本 - 采集、分析、Web应用

支持4种采集模式:
  1. 单场景全部采集 - 采集某场景的所有routes
  2. 单场景指定数目采集 - 采集某场景的前N个routes
  3. 多场景采集 - 同时采集多个指定场景
  4. 全部采集 - 采集所有47个场景
"""

import sys
from pathlib import Path
import socket
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Optional
from collector import (
    ScenarioCollector,
    SCENARIO_TO_ROAD_STRUCTURE,
    SCENARIO_TO_FINE_EVENTS,
    SCENARIO_RULE_KIND,
    SCENARIO_RULE_CONFIG,
    RouteXmlIndex,
    _DEFAULT_XML_ROOT,
    _DEFAULT_CARLA_ROOT,
)
from analyzer import StructureAnalyzer, quick_analysis
import json


def print_main_menu():
    """打印主菜单"""
    print("\n" + "="*70)
    print("场景事件采集系统 - 快速启动".center(70))
    print("="*70)
    print("""
采集模式:
  1️⃣  单场景全部采集      - 采集某个场景的所有routes
  2️⃣  单场景指定数采集     - 采集某个场景的前N个routes
  3️⃣  多场景采集          - 同时采集多个指定的场景
  4️⃣  全部采集            - 采集所有47个场景（可能耗时很长）

其他功能:
  5️⃣  多角度结构分析      - 分析已采集的数据
  6️⃣  启动Web应用        - 交互式可视化查看
  7️⃣  显示所有场景       - 列出所有支持的场景
  8️⃣  ROAD_STRUCTURE XML/XODR画像 - 按场景/town审计XML与地图输入
  9️⃣  退出
    """)
    print("="*70)


# ============================================================================
# 采集功能
# ============================================================================

def collect_one_scenario_all_ui():
    """模式1: 单场景全部采集"""
    print("\n" + "="*70)
    print("模式1: 单场景全部采集".center(70))
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
    
    print(f"\n开始采集 {scenario} 的所有routes...")
    collector = ScenarioCollector()
    
    try:
        result = collector.collect_one_scenario_all(scenario)
        
        print(f"\n✅ 采集完成!")
        print(f"  • 场景: {scenario}")
        print(f"  • 状态: {result['status']}")
        print(f"  • Routes数: {len(result['routes'])}")
        print(f"  • 总帧数: {result['total_frames']}")
        print(f"  • 结果: collection_output/{scenario}_result.json")
    except Exception as e:
        print(f"\n❌ 采集失败: {e}")


def collect_one_scenario_limited_ui():
    """模式2: 单场景指定数目采集"""
    print("\n" + "="*70)
    print("模式2: 单场景指定数目采集".center(70))
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
    
    print(f"\n开始采集 {scenario} 的前 {max_routes} 个routes...")
    collector = ScenarioCollector()
    
    try:
        result = collector.collect_one_scenario(scenario, max_routes=max_routes)
        
        print(f"\n✅ 采集完成!")
        print(f"  • 场景: {scenario}")
        print(f"  • 状态: {result['status']}")
        print(f"  • Routes数: {len(result['routes'])}")
        print(f"  • 总帧数: {result['total_frames']}")
        print(f"  • 结果: collection_output/{scenario}_result.json")
    except Exception as e:
        print(f"\n❌ 采集失败: {e}")


def collect_multiple_scenarios_ui():
    """模式3: 多场景采集"""
    print("\n" + "="*70)
    print("模式3: 多场景采集".center(70))
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
    
    print(f"\n将采集 {len(selected_scenarios)} 个场景, 每个采集 {max_routes} 个routes")
    print("场景列表:")
    for i, scenario in enumerate(selected_scenarios, 1):
        print(f"  {i}. {scenario}")
    
    confirm = input("\n是否继续? (y/n): ").strip().lower()
    if confirm != 'y':
        print("⏭️  已取消")
        return
    
    collector = ScenarioCollector()
    
    try:
        result = collector.collect_multiple_scenarios(selected_scenarios, max_routes)
        
        print(f"\n✅ 采集完成!")
        print(f"  • 成功场景数: {result.get('scenarios_collected', 0)}")
        print(f"  • 总场景数: {result.get('total_scenarios', 0)}")
        print(f"  • 总帧数: {result.get('total_frames', 0)}")
        print(f"  • 结果: collection_output/multi_scenario_collection.json")
    except Exception as e:
        print(f"\n❌ 采集失败: {e}")


def collect_all_scenarios_ui():
    """模式4: 全部采集"""
    print("\n" + "="*70)
    print("模式4: 全部采集".center(70))
    print("="*70)
    
    scenarios = sorted(SCENARIO_TO_ROAD_STRUCTURE.keys())
    
    print(f"\n⚠️  警告: 这将采集所有 {len(scenarios)} 个场景")
    print("这可能需要很长时间和大量磁盘空间！")
    
    try:
        max_routes = int(input("请输入每个场景采集的routes数量 (默认2): ") or "2")
    except:
        max_routes = 2
    
    print(f"\n预计采集 {len(scenarios)} × {max_routes} = {len(scenarios) * max_routes} 个routes")
    
    confirm = input("确实要继续? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("⏭️  已取消")
        return
    
    collector = ScenarioCollector()
    
    print(f"\n开始采集所有 {len(scenarios)} 个场景...")
    
    try:
        result = collector.collect_all_scenarios(max_routes)
        
        print(f"\n✅ 采集完成!")
        print(f"  • 成功场景数: {result.get('scenarios_collected', 0)}")
        print(f"  • 总场景数: {result.get('total_scenarios', 0)}")
        print(f"  • 总帧数: {result.get('total_frames', 0)}")
        print(f"  • 结果: collection_output/multi_scenario_collection.json")
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
    
    output_dir = Path("/home/cruser1/lda/AutoMoT/keyframe_filter/collection_output")
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


def _scenario_town_xml_audit(index: RouteXmlIndex, carla_root: Path, samples_per_town: int = 3) -> dict:
    """逐场景、逐 town 抽样 XML，并附上 XODR 粗画像。"""
    report = {
        "xml_root": str(index.xml_root),
        "carla_root": str(carla_root),
        "samples_per_town": samples_per_town,
        "scenarios": {},
    }
    xodr_cache: dict[str, dict] = {}

    for scenario in sorted(SCENARIO_TO_ROAD_STRUCTURE):
        infos = index.by_scenario.get(scenario, [])
        by_town = defaultdict(list)
        for info in infos:
            by_town[info.town or "UNKNOWN"].append(info)

        scenario_entry = {
            "rule_kind": SCENARIO_RULE_CONFIG.get(scenario, {}).get("kind", SCENARIO_RULE_KIND.get(scenario, "default_meta_map")),
            "rule_config": SCENARIO_RULE_CONFIG.get(scenario, {}),
            "road_candidates": [rs.value for rs in SCENARIO_TO_ROAD_STRUCTURE.get(scenario, [])],
            "xml_count": len(infos),
            "towns": {},
        }

        for town, town_infos in sorted(by_town.items()):
            if town not in xodr_cache:
                xodr_cache[town] = _summarize_xodr(town, carla_root)
            sampled = town_infos[:samples_per_town]
            wp_counts = [len(info.waypoints) for info in town_infos]
            tag_counter = Counter()
            samples = []
            for info in sampled:
                for tag in info.scenario_tags:
                    for key in tag:
                        if key not in {"name", "type"}:
                            tag_counter[key] += 1
                samples.append(
                    {
                        "xml": str(info.path),
                        "route_id": info.route_id,
                        "waypoint_count": len(info.waypoints),
                        "scenario_tag_count": len(info.scenario_tags),
                        "tag_keys": sorted({k for tag in info.scenario_tags for k in tag.keys() if k not in {"name", "type"}}),
                        "first_waypoint": info.waypoints[0] if info.waypoints else None,
                        "last_waypoint": info.waypoints[-1] if info.waypoints else None,
                    }
                )
            scenario_entry["towns"][town] = {
                "xml_count": len(town_infos),
                "waypoint_count_min": min(wp_counts) if wp_counts else 0,
                "waypoint_count_avg": round(sum(wp_counts) / len(wp_counts), 2) if wp_counts else 0,
                "waypoint_count_max": max(wp_counts) if wp_counts else 0,
                "xodr": xodr_cache[town],
                "top_tag_keys": dict(tag_counter.most_common(12)),
                "sampled_xml": samples,
            }
        report["scenarios"][scenario] = scenario_entry
    return report


def road_structure_xml_xodr_audit_ui():
    """按 ROAD_STRUCTURE 设计文档审计 XML/XODR 输入覆盖。"""
    print("\n" + "="*70)
    print("ROAD_STRUCTURE XML/XODR画像".center(70))
    print("="*70)

    xml_root_text = input(f"XML根目录 (默认 {_DEFAULT_XML_ROOT}): ").strip()
    carla_root_text = input(f"CARLA根目录 (默认 {_DEFAULT_CARLA_ROOT}): ").strip()
    try:
        samples_per_town = int(input("每个town抽样XML数量 (默认3): ") or "3")
    except Exception:
        samples_per_town = 3

    xml_root = Path(xml_root_text) if xml_root_text else _DEFAULT_XML_ROOT
    carla_root = Path(carla_root_text) if carla_root_text else _DEFAULT_CARLA_ROOT
    index = RouteXmlIndex(xml_root)
    report = _scenario_town_xml_audit(index, carla_root, samples_per_town=max(1, samples_per_town))

    output_dir = Path(__file__).resolve().parent / "collection_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "road_structure_xml_xodr_audit.json"
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)

    print(f"\n✓ 已审计 {len(report['scenarios'])} 个场景")
    for scenario, item in report["scenarios"].items():
        towns = ",".join(item["towns"].keys()) or "NONE"
        print(f"  - {scenario:<40} xml={item['xml_count']:<4} rule={item['rule_kind']:<28} towns={towns}")
    print(f"\n完整画像已保存: {output_file}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""
    while True:
        print_main_menu()
        choice = input("请选择 (1-9): ").strip()
        
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
            print("\n👋 再见！\n")
            break
        else:
            print("❌ 无效选择")
        
        if choice not in ['6']:  # Web应用特殊处理
            input("\n按 Enter 继续...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 已中断\n")
    except Exception as e:
        print(f"\n❌ 错误: {e}\n")
        import traceback
        traceback.print_exc()
