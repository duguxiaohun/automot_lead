import os
import sys
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
import imageio
import random
import sys
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

projects_root = str(pathlib.Path(__file__).parent.parent.parent)
leaderboard_root = str(os.path.join(projects_root, 'leaderboard'))
scenario_runner_root = str(os.path.join(projects_root, 'scenario_runner'))
mot_dp_root = str(os.path.join(projects_root, 'Automot'))
carla_api_root = str(os.path.join(projects_root.replace('Bench2Drive', 'carla'), 'PythonAPI', 'carla'))

for path in [projects_root, leaderboard_root, scenario_runner_root, mot_dp_root, carla_api_root]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

sys.path = [str(p) for p in sys.path]

from leaderboard.autoagents import autonomous_agent
from team_code.nav_planner import RoutePlanner, LateralPIDController  
from agents.navigation.local_planner import RoadOption
import team_code.automot_utils as t_u  
from team_code.render import render, render_self_car, render_waypoints
from preprocess.generate_lidar_bev_b2d import generate_lidar_bev_images
from scipy.optimize import fsolve
from scipy.interpolate import PchipInterpolator
import xml.etree.ElementTree as ET  
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  

# BEV 编码器主干网络
from mot.modeling.bev_encoder.backbone_extractor import BEVEncoderBackboneExtractor
from mot.modeling.bev_encoder.config import GlobalConfig as BEVEncoderConfig
import mot.modeling.bev_encoder.bev_encoder_utils as bev_encoder_t_u

# MoT 相关依赖
projects_root = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(projects_root)
mot_dp_path = str(os.path.join(os.path.dirname(projects_root), 'Automot'))
mot_path = str(os.path.join(mot_dp_path, 'mot'))
sys.path.append(mot_dp_path)
sys.path.append(mot_path)
sys.path = [str(p) for p in sys.path]

from transformers import HfArgumentParser
import json
from dataclasses import dataclass, field
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from PIL import Image
from safetensors.torch import load_file
import glob
from data.reasoning.data_utils import add_special_tokens
from mot.modeling.automot import (
    AutoMoTConfig, AutoMoT,
    Qwen3VLTextConfig, Qwen3VLTextModel, Qwen3VLForConditionalGenerationMoT
)
from mot.evaluation.inference import InterleaveInferencer
from transformers import AutoTokenizer

from team_code.bev_data_utils import lidar_to_histogram_features as lidar_to_bev_histogram

# 导入工具模块
from team_code.automot_utils import (
    ModelArguments, InferenceArguments,
    load_model_mot, build_cleaned_prompt_and_modes,
    parse_decision_sequence, split_prompt
)
from team_code.lidar_utils import lidar_to_ego_coordinate, algin_lidar
from team_code.ukf_utils import (
    bicycle_model_forward, measurement_function_hx,
    state_mean, measurement_mean,
    residual_state_x, residual_measurement_h
)
from team_code.display_interface import DisplayInterface

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
PLANNER_TYPE = os.environ.get('PLANNER_TYPE', None)
EARTH_RADIUS_EQUA = 6378137.0
USE_UKF = True  # Enable Unscented Kalman Filter for GPS/compass smoothing

# legacy AutoMoT prompt 与 LeadMoT v2 保持同一套导航分布：
# target_point / next_target_point 沿 RoutePlanner 剩余 route 按弧长前推
# max(speed * lookahead_s, min_lookahead_m)。低速时的 5m 下限避免起步目标点贴脸。
_TP_LOOKAHEAD_S = float(os.environ.get("TP_LOOKAHEAD_S", "1.0"))
_NTP_LOOKAHEAD_S = float(os.environ.get("NTP_LOOKAHEAD_S", "2.0"))
_MIN_LOOKAHEAD_M = float(os.environ.get("MIN_LOOKAHEAD_M", "5.0"))
_EMPTY_ROUTE_EXTENSION_M = 50.0


def _format_value_stats(value):
	"""返回 value 的 shape/dtype/range 描述字符串，用于调试打印。"""
	if isinstance(value, torch.Tensor):
		arr = value.detach().float().cpu().numpy()
		dtype_str = str(value.dtype)
	elif isinstance(value, np.ndarray):
		arr = value
		dtype_str = str(value.dtype)
	elif np.isscalar(value):
		arr = np.asarray(value)
		dtype_str = str(arr.dtype)
	else:
		return f"type={type(value).__name__}"

	shape_str = str(tuple(arr.shape))
	if arr.size == 0:
		return f"shape={shape_str}, dtype={dtype_str}, range=[empty]"

	if np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_:
		vmin = float(np.nanmin(arr))
		vmax = float(np.nanmax(arr))
		return f"shape={shape_str}, dtype={dtype_str}, range=[{vmin:.6g}, {vmax:.6g}]"

	return f"shape={shape_str}, dtype={dtype_str}, range=[non-numeric]"


def _print_result_stats(result, step, every=20):
	"""按固定步长打印 result 字段统计，避免每帧刷屏。"""
	if every <= 0 or (step % every != 0):
		return

	keys = [
		'rgb_front',
		'lidar_bev',
		'gps',
		'speed',
		'compass',
		'bev',
		'bev_encoder_rgb',
		'bev_encoder_lidar_bev',
		'next_command',
		'target_point',
		'next_target_point',
		'theta',
	]
	print(f"[Tick Result Stats] step={step}")
	for k in keys:
		if k not in result:
			print(f"  - {k}: <missing>")
			continue
		print(f"  - {k}: {_format_value_stats(result[k])}")


def _print_input_data_stats(input_data, step, every=20):
	"""按固定步长打印 tick 收到的原始 `input_data` 中各传感器的数据 shape/dtype/range。

	该函数会自动处理常见的传感器数据容器格式：
	- tuple/list (timestamp, data) -> 使用第二项 `data` 进行统计
	- 其它类型直接传入 `_format_value_stats`
	"""
	if every <= 0 or (step % every != 0):
		return

	print(f"[Tick InputData Stats] step={step}")
	for k, v in input_data.items():
		# 常见格式为 (timestamp, payload)
		payload = None
		if isinstance(v, (list, tuple)) and len(v) > 1:
			payload = v[1]
		else:
			payload = v

		# 对字典类型，尝试逐字段打印（如 SPEED -> {'speed': ...}）
		if isinstance(payload, dict):
			# 打印整体类型，再打印子键的数值统计（若为数组/标量）
			print(f"  - {k}: type=dict, keys={list(payload.keys())}")
			for subk, subv in payload.items():
				try:
					print(f"    - {subk}: {_format_value_stats(subv)}")
				except Exception:
					print(f"    - {subk}: type={type(subv).__name__}")
		else:
			try:
				print(f"  - {k}: {_format_value_stats(payload)}")
			except Exception:
				print(f"  - {k}: type={type(payload).__name__}")

# 入口函数
# leaderboard_evaluator 会通过该函数名反射加载 Agent 类。
def get_entry_point():
	return 'MOTAgent'


class MOTAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		# 固定使用 SENSORS 赛道模式（相机/LiDAR/IMU/GNSS 等传感器输入）。
		self.track = autonomous_agent.Track.SENSORS
		# Bench2Drive 模式下，配置字符串可能形如：config_path+save_name。
		# 这里将“配置路径”和“本次运行唯一保存名”拆开，方便后续保存可视化与日志。
		if IS_BENCH2DRIVE:
			self.save_name = path_to_conf_file.split('+')[-1]
			self.config_path = path_to_conf_file.split('+')[0]
		else:
			# 非 B2D 模式兜底：用当前时间戳生成唯一目录名，避免多次运行覆盖。
			now = datetime.datetime.now()
			self.config_path = path_to_conf_file
			self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		import gc
		
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# 加载大模型前做激进显存清理
		# 先做一次主动清理，尽量降低大模型加载阶段的 OOM 风险。
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.synchronize()

		# 加载 MoT 模型
		# 1) 读取推理参数
		# 2) 加载 AutoMoT 主模型
		# 3) 初始化 tokenizer 并注入新增 special tokens
		# 4) 构建 InterleaveInferencer 作为统一推理入口
		print("Loading MoT model...")
		parser = HfArgumentParser((ModelArguments, InferenceArguments))
		model_args, inference_args = parser.parse_args_into_dataclasses(args=[])
		self.inference_args = inference_args  
		self.AutoMoT = load_model_mot(device)
		tokenizer = AutoTokenizer.from_pretrained(model_args.qwen3vl_path)
		tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
		self.AutoMoT.language_model.tokenizer = tokenizer
		self.inferencer = InterleaveInferencer(
		model=self.AutoMoT,  # 主推理模型（包含语言模型+多模态头）
		vae_model=None,  # 图像生成用 VAE；当前驾驶推理链路不使用，置空
		tokenizer=tokenizer,  # 文本分词器（与模型词表/特殊 token 对齐）
		vae_transform=None,  # VAE 预处理；当前链路不走 VAE 图像生成
		vit_transform=None,  # ViT 预处理；Qwen3VL 内部处理图像，因此置空
		new_token_ids=new_token_ids,  # 追加 special tokens 后的新 token id 映射
		max_num_tokens=inference_args.max_num_tokens,  # 单次推理允许的最大 token 数
		visual_gen=True,  # 开启“视觉生成相关分支”，用于初始化 reasoning/action query
		visual_und=True,  # 开启“视觉理解分支”，用于文本决策与轨迹推理
    	)
		print("✓ MoT model loaded.")

		# ========== 加载 BEV 编码器主干 ==========
		# 从统一 safetensors 中裁剪出 bev_encoder.* 前缀参数，独立构建 BEV backbone。
		print("Loading BEV encoder backbone...")
		bev_encoder_config_path = os.path.join(str(pathlib.Path(__file__).parent.parent.parent), 'Automot', 'checkpoints')
		combined_ckpt_path = os.path.join(str(pathlib.Path(__file__).parent.parent.parent), 'Automot', 'checkpoints', 'model.safetensors')
		combined_sd = load_file(combined_ckpt_path)
		bev_state_dict = {k[len('bev_encoder.'):]: v for k, v in combined_sd.items() if k.startswith('bev_encoder.')}
		del combined_sd
		self.bev_encoder = BEVEncoderBackboneExtractor(
			config_path=bev_encoder_config_path,
			device='cuda:0',
			state_dict=bev_state_dict
		)
		del bev_state_dict
		# BackboneExtractor 内部已冻结参数
		self.bev_encoder.eval()
		# 转为 bfloat16，对齐主模型精度并降低显存占用
		self.bev_encoder = self.bev_encoder.to(torch.bfloat16)
		# 读取 BEV 编码器配置，用于 LiDAR 预处理
		self.bev_encoder_config = self.bev_encoder.config
		print("✓ BEV encoder backbone loaded, frozen, and converted to bfloat16.")
		
		# 初始化 BEV 编码器的 LiDAR 时序缓冲
		# 该缓存按 data_save_freq 对齐时序，用于后续多帧 LiDAR 组织与回放。
		self.bev_encoder_lidar_buffer = deque(maxlen=self.bev_encoder_config.lidar_seq_len * self.bev_encoder_config.data_save_freq)
		self.bev_encoder_lidar_last = None
		self.bev_encoder_state_log = deque(maxlen=max((self.bev_encoder_config.lidar_seq_len * self.bev_encoder_config.data_save_freq), 2))
		
		# 打印当前 GPU 显存状态
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			allocated = torch.cuda.memory_allocated() / 1024**3
			reserved = torch.cuda.memory_reserved() / 1024**3
			print(f"[GPU Memory] After BEV encoder: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

		self.turn_controller = LateralPIDController(
			inference_mode=False, 
			k_p=3.118,
			speed_offset=1.195,
			default_lookahead=24
		)
		# 纵向速度控制器：根据目标速度与当前速度差输出油门/制动趋势。
		self.speed_controller = t_u.PIDController(k_p=1.75, k_i=1.0, k_d=2.0, n=20) 
		
		# 控制相关配置
		# 这些阈值决定了：低速判定、脱困策略触发时机、油门/转向限幅等驾驶行为。
		self.carla_fps = 20
		self.wp_dilation = 1
		self.data_save_freq = 5
		self.brake_speed = 0.4
		self.brake_ratio = 1.1
		self.clip_delta = 1.0
		self.clip_throttle = 1.0
		self.stuck_threshold = 300
		self.stuck_helper_threshold = 100
		self.creep_duration = 14
		self.creep_throttle = 0.4
		
		# 卡死检测状态量
		self.stuck_detector = 0
		self.stuck_helper = 0
		self.force_move = 0

		self.steer_step = 0
		self.last_moving_status = 0
		self.last_moving_step = -1
		self.last_steers = 0
		
		self.takeover = False
		self.stop_time = 0
		self.takeover_time = 0
		self.save_path = None
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
		self.lat_ref, self.lon_ref = 42.0, 2.0
		control = carla.VehicleControl()
		control.steer = 0.0
		control.throttle = 0.0
		control.brake = 0.0	
		self.prev_control = control
		self.control = control  # Store control for UKF prediction
		
		# 初始化无迹卡尔曼滤波器（UKF）
		# UKF 用于融合 GPS/IMU/控制量，减小定位抖动对规划与控制的影响。
		self.carla_frame_rate = 1.0 / 20.0  # CARLA frame rate
		if USE_UKF:
			self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
			self.ukf = UKF(dim_x=4,
						   dim_z=4,
						   fx=bicycle_model_forward,
						   hx=measurement_function_hx,
						   dt=self.carla_frame_rate,
						   points=self.points,
						   x_mean_fn=state_mean,
						   z_mean_fn=measurement_mean,
						   residual_x=residual_state_x,
						   residual_z=residual_measurement_h)
			# 状态协方差初值（后续会由第一帧观测逐步收敛）
			self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
			# 观测噪声协方差
			self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
			self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
			# 首帧观测用于初始化滤波状态
			self.filter_initialized = False
			# 记录最近若干帧滤波后状态
			self.state_log = deque(maxlen=20)

		if SAVE_PATH is not None:
			# 评测模式下按路线名创建独立输出目录，避免不同路线互相覆盖。
			now = datetime.datetime.now()
			string = self.save_name
			print (string)

		self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
		self.save_path.mkdir(parents=True, exist_ok=False)

		(self.save_path / 'rgb_front').mkdir()
		(self.save_path / 'meta').mkdir()
		(self.save_path / 'bev').mkdir()
		(self.save_path / 'lidar_bev').mkdir()
		
		# 初始化 LiDAR 缓冲（用于双帧融合）
		self.lidar_buffer = deque(maxlen=2)
		self.lidar_step_counter = 0
		self.last_ego_transform = None
		self.last_lidar = None
		
		# 多帧观测历史缓冲（供 MoT 时序输入）
		# MoT 需要 4 帧 RGB（每 5 帧采样）和 1 帧 LiDAR BEV
		# 缓冲至少 31 帧（obs_horizon=4，LiDAR/状态按 10 步窗口）
		obs_horizon = 4
		self.obs_horizon = obs_horizon
		self.lidar_bev_history = deque(maxlen=obs_horizon*10)
		self.rgb_history = deque(maxlen=obs_horizon*10)
		self.speed_history = deque(maxlen=obs_horizon*10)
		self.theta_history = deque(maxlen=obs_horizon*10)
		self.next_command_history = deque(maxlen=obs_horizon*10)
		self.target_point_history = deque(maxlen=obs_horizon*10)
		self.next_target_point_history = deque(maxlen=obs_horizon*10)
		self.waypoint_history = deque(maxlen=obs_horizon*10)
		self.throttle_history = deque(maxlen=obs_horizon*10)
		self.brake_history = deque(maxlen=obs_horizon*10)

		# 保存预测轨迹，供 BEV 可视化使用
		self.last_pred_traj = None  # Store the last predicted trajectory (in ego frame)
		self.last_target_point = None  # Store the last target point (in ego frame)
		self.last_next_target_point = None  # Store the last next target point (in ego frame)
		self.last_route_pred = None  # Store the last route prediction (20 waypoints for lateral control)

		# ====== 车位脱困：长时位移检测相关状态 ======
		self.parking_escape_active = False
		self.parking_escape_phase = 0            # 1=lateral, 2=forward
		self.parking_escape_timer = 0
		self.parking_escape_anchor = None
		self.parking_escape_start_compass = None # Record heading at escape start
		self.parking_escape_attempt = 0
		self.parking_escape_cooldown = 0
		self.parking_escape_direction = 1.0      # +1 = escape left, -1 = escape right
		# 位置快照：每 N 帧记录一次 GPS
		self.pos_snapshot_interval = 200         # Record every 10 seconds (200 frames @ 20fps)
		self.pos_snapshots = []                  # [(step, pos), ...]
		self.parking_deadlock_window = 1500      # Check window: 1500 frames = 125 seconds (> max red light 60s)
		self.parking_deadlock_max_disp = 5.0     # Max displacement in window to be considered stuck
		# 停车起步检测：若前 N 帧位移极小，则本局禁用 force_move
		self.parking_start_check_frame = 200     # Check at frame 200 (10s @ 20fps)
		self.parking_start_disp_thresh = 6.0     # If displacement < 6m in first 200 frames -> parking start
		self.parking_start_detected = False       # Set once at check frame, never changes after
		self.parking_start_checked = False        # Whether the check has been performed
		self.parking_start_anchor = None          # GPS position at BUFFER_PHASE start

	def _init(self):
		# 直接使用 _global_plan_world_coord（已是 CARLA 坐标）
		# 避免 GPS->CARLA 反解在 fsolve 不收敛时失败
		# 优先从 CARLA 地图 OpenDRIVE 获取 lat_ref/lon_ref
		try:
			world_map = CarlaDataProvider.get_map()
			xodr = world_map.to_opendrive()
			tree = ET.ElementTree(ET.fromstring(xodr))
			
			# OpenDRIVE 未提供地理参考时使用默认值
			self.lat_ref = 42.0
			self.lon_ref = 2.0
			
			for opendrive in tree.iter('OpenDRIVE'):
				for header in opendrive.iter('header'):
					for georef in header.iter('geoReference'):
						if georef.text:
							str_list = georef.text.split(' ')
							for item in str_list:
								if '+lat_0' in item:
									self.lat_ref = float(item.split('=')[1])
								if '+lon_0' in item:
									self.lon_ref = float(item.split('=')[1])
		except Exception as e:
			# 回退方案：尝试 fsolve 估计参考经纬度（可能不收敛）
			try:
				locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
				lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
				earth_radius_equa = 6378137.0
				def equations(variables):
					x, y = variables
					eq1 = (lon * math.cos(x * math.pi / 180.0) - (locx * x * 180.0) / (math.pi * earth_radius_equa)
								 - math.cos(x * math.pi / 180.0) * y)
					eq2 = (math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * earth_radius_equa
								 * math.cos(x * math.pi / 180.0) + locy - math.cos(x * math.pi / 180.0) * earth_radius_equa
								 * math.log(math.tan((90.0 + x) * math.pi / 360.0)))
					return [eq1, eq2]
				initial_guess = [0.0, 0.0]
				solution = fsolve(equations, initial_guess)
				self.lat_ref, self.lon_ref = solution[0], solution[1]
			except Exception as e2:
				self.lat_ref, self.lon_ref = 0.0, 0.0
		

		self.route_planner_min_distance = 7.5 
		self.route_planner_max_distance = 50.0
		self._route_planner = RoutePlanner(self.route_planner_min_distance, self.route_planner_max_distance,
										   self.lat_ref, self.lon_ref)
		
		if len(self._global_plan_world_coord) > 0:
			first_wp = self._global_plan_world_coord[0]
		
		# 使用世界坐标路线，gps=False（nav_planner 中 GPS 路线已不推荐）
		self._route_planner.set_route(self._global_plan_world_coord, gps=False)
		
				
		# 初始化高层指令跟踪
		self.commands = deque(maxlen=2)
		self.commands.append(4)
		self.commands.append(4)
		self.target_point_prev = [1e5, 1e5, 1e5]
		self.last_command = -1
		self.last_command_tmp = -1
		
		self.initialized = True
		self.metric_info = {}
		self._hic = DisplayInterface()

	def sensors(self):
		"""
		向 CARLA Leaderboard 框架声明该 Agent 所需传感器列表。

		框架会在仿真环境中自动实例化这些传感器并绑定到 ego 车辆，
		每个仿真 tick 将采集到的数据通过 input_data 字典传入 run_step()。

		传感器分两类：
		1. 基础传感器（所有模式必装）：前视相机、LiDAR、IMU、GPS、速度计
		2. 条件传感器（IS_BENCH2DRIVE=True 时追加）：高空俯视 BEV 相机（仅用于可视化）
		"""
		# ===== 基础传感器：所有运行模式均挂载 =====
		sensors =  [
				# 前视 RGB 相机：安装在车身后方（x=-1.5），FOV=110° 覆盖宽视野
				# 输出 1024×512 图像，经 BGR->RGB 转换后送入 MoT 视觉分支
				{
					'type': 'sensor.camera.rgb',
					'x': -1.50, 'y': 0.0, 'z': 2.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 1024, 'height': 512, 'fov': 110,
					'id': 'CAM_FRONT'
					},
				# LiDAR：安装在车顶（z=2.5m），偏转 -90° 使点云 x 轴朝前对齐 ego 坐标系
				# 原始点云经 lidar_to_ego_coordinate 转换后用于双帧融合与 BEV 生成
				{
          			'type': 'sensor.lidar.ray_cast',
          			'x': 0.0, 'y': 0.0, 'z': 2.5,
          			'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
          			'id': 'LIDAR'
      				},
				# IMU：提供加速度/角速度/compass（航向角），采样周期 0.05s（20Hz）
				# compass 经 preprocess_compass 归一化后送入 UKF 做航向融合
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'IMU'
					},
				# GPS（GNSS）：提供经纬度坐标，采样周期 0.01s（100Hz）
				# 经 convert_gps_to_carla 转为米制世界坐标，再经 UKF 平滑后使用
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'GPS'
					},
				# 速度计：直接读取 ego 车辆当前线速度（m/s），采样 20Hz
				# 同时作为 UKF 观测量和 PID 控制的当前速度输入
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'SPEED'
					},
				]
		
		# ===== 条件传感器：仅 Bench2Drive 评测模式下追加 =====
		if IS_BENCH2DRIVE:
			sensors += [
					# 高空俯视 BEV 相机：z=50m，pitch=-90° 朝下，FOV=50° 覆盖约 93m×93m 范围
					# 仅用于可视化（tick_data['bev']），不参与网络推理
					{	
						'type': 'sensor.camera.rgb',
						'x': 0.0, 'y': 0.0, 'z': 50.0,
						'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
						'width': 512, 'height': 512, 'fov': 5 * 10.0,
						'id': 'bev'
					}]
		return sensors

	def tick(self, input_data):
		"""
		将 CARLA 原始传感器输入整理为当前决策帧所需的统一字典（tick_data/result）。

		核心职责：
		1) 读取并预处理传感器：RGB、LiDAR、GPS、IMU、速度。
		2) 执行 UKF 融合，得到更稳定的位置与航向估计。
		3) 生成两路 LiDAR BEV：
		   - 通用 lidar_bev（供主模型输入）
		   - bev_encoder_lidar_bev（供 BEV backbone 提特征）
		4) 调用 route planner 生成 target_point / next_target_point，并转换到 ego 坐标系。
		"""
		self.step += 1

		# [Tick InputData Stats] step=0
		# - GPS: shape=(3,), dtype=float64, range=[-0.0157523, 345.894]
		# - IMU: shape=(7,), dtype=float64, range=[-0.159803, 9.14938]
		# - LIDAR: shape=(13289, 4), dtype=float32, range=[-83.1503, 42.6016]
		# - bev: shape=(512, 512, 4), dtype=uint8, range=[34, 255]
		# - SPEED: type=dict, keys=['speed']
		# 	- speed: shape=(), dtype=float64, range=[0.000674771, 0.000674771]
		# - CAM_FRONT: shape=(512, 1024, 4), dtype=uint8, range=[22, 255]
		
		# _print_input_data_stats(input_data, step=self.step, every=1)
		# keys: ['IMU', 'GPS', 'LIDAR', 'bev', 'SPEED', 'CAM_FRONT']
		# 	IMU                  | type=ndarray | shape=(7,) | dtype=float64
		# 	GPS                  | type=ndarray | shape=(3,) | dtype=float64
		# 	LIDAR[x, y, z, intensity]| type=ndarray | shape=(15437, 4) | dtype=float32
		# 	bev                  | type=ndarray | shape=(512, 512, 4) | dtype=uint8
		# 	SPEED                | type=dict    | keys=['speed']
		# 	CAM_FRONT            | type=ndarray | shape=(512, 1024, 4) | dtype=uint8

		# 前视相机转为 RGB（OpenCV 默认 BGR）。
		rgb_front = cv2.cvtColor(input_data['CAM_FRONT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		# 将 LiDAR 点云从传感器坐标转换到 ego 坐标。
		lidar_ego = lidar_to_ego_coordinate(input_data['LIDAR'])
		
		gps_full = input_data['GPS'][1]  # [lat, lon, altitude]
		# GPS 转 CARLA 世界坐标（米制坐标系）。
		gps_pos = self._route_planner.convert_gps_to_carla(gps_full)
		
		# 处理 compass 为 NaN 的异常情况
		compass_raw = input_data['IMU'][1][-1]
		if math.isnan(compass_raw):
			print("compass sends nan!!!")
			compass_raw = 0.0
		
		# 将 compass 预处理到 CARLA 坐标系定义
		compass = t_u.preprocess_compass(compass_raw)
		
		# 读取速度（供 UKF 融合）
		speed = input_data['SPEED'][1]['speed']
		
		# 执行无迹卡尔曼滤波
		# 用上一时刻控制量 + 当前观测做预测/更新，降低 GPS/IMU 抖动。
		if USE_UKF:
			if not self.filter_initialized:
				self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
				self.filter_initialized = True

			self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
			self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
			filtered_state = self.ukf.x

			self.state_log.append(filtered_state)
			gps_filtered = filtered_state[0:2]
			compass_filtered = filtered_state[2]
		else:
			gps_filtered = np.array([gps_pos[0], gps_pos[1]])
			compass_filtered = compass
		
		# 使用 algin_lidar 融合双帧点云
		# 对齐时使用滤波后的 GPS/航向，提升稳定性
		if self.last_lidar is not None and self.last_ego_transform is not None:
			# 计算当前帧与上一帧的相对平移
			current_pos = np.array([gps_filtered[0], gps_filtered[1], 0.0])
			last_pos = np.array([self.last_ego_transform['gps'][0], self.last_ego_transform['gps'][1], 0.0])
			relative_translation = current_pos - last_pos
			
			# 基于滤波航向计算相对旋转
			current_yaw = compass_filtered
			last_yaw = self.last_ego_transform['compass']
			relative_rotation = current_yaw - last_yaw
			
			# 将全局位移旋转到当前车体局部坐标系
			rotation_matrix = np.array([[np.cos(current_yaw), -np.sin(current_yaw), 0.0],
										[np.sin(current_yaw), np.cos(current_yaw), 0.0], 
										[0.0, 0.0, 1.0]])
			relative_translation_local = rotation_matrix.T @ relative_translation
			
			# 将上一帧 LiDAR 对齐到当前帧 ego 坐标，再拼接成更稠密点云。
			lidar_last = algin_lidar(self.last_lidar, relative_translation_local, relative_rotation)
			# 拼接当前帧与对齐后的上一帧 LiDAR
			lidar_combined = np.concatenate((lidar_ego, lidar_last), axis=0)
		else:
			lidar_combined = lidar_ego
		
		# 保存当前帧状态供下一帧对齐使用（滤波值）
		self.last_lidar = lidar_ego
		self.last_ego_transform = {'gps': gps_filtered, 'compass': compass_filtered}
		
		# 将拼接后的点云栅格化为 BEV 图像，供主模型视觉分支使用。
		# LiDAR points: current=15656, last=15656, combined=31093
		lidar_bev_img = generate_lidar_bev_images(
			np.copy(lidar_combined), 
			saving_name=None, 
			img_height=448, 
			img_width=448
		)
		# LiDAR BEV image shape: (448, 448, 3), dtype: uint8
		# R 通道：密度图（Density）- 该 grid 内点云点数的对数归一化
		# G 通道：高度图（Height）- 该 grid 内最高点的高度
		# B 通道：强度图（Intensity）- 该 grid 内反射强度

		# 将 BEV 图转为张量格式
		lidar_bev_tensor = torch.from_numpy(lidar_bev_img).permute(2, 0, 1).float() / 255.0
		# LiDAR BEV tensor shape: torch.Size([3, 448, 448])

		# ========== BEV Encoder 路径（为 DP/MoT 提供 trans_feat） ==========
		# 处理 RGB 输入（与 bev_encoder 训练预处理保持一致）
		bev_encoder_rgb = input_data['CAM_FRONT'][1][:, :, :3]
		# Original RGB shape: (512, 1024, 3)

		# 训练集 RGB 以 jpg 存储，推理时注入相同压缩伪影以减小域偏移。
		_, compressed_image = cv2.imencode('.jpg', bev_encoder_rgb)
		bev_encoder_rgb = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
		bev_encoder_rgb = cv2.cvtColor(bev_encoder_rgb, cv2.COLOR_BGR2RGB)
		# 裁剪 RGB 到训练时使用的视野范围		
		# # Original RGB shape: (512, 1024, 3)
		bev_encoder_rgb = bev_encoder_t_u.crop_array(self.bev_encoder_config, bev_encoder_rgb)
		# Cropped RGB shape: (384, 1024, 3)

		# 转成 PyTorch 张量格式 (C, H, W) 并加 batch 维
		bev_encoder_rgb = np.transpose(bev_encoder_rgb, (2, 0, 1))
		# Transposed RGB shape: (3, 384, 1024)

		bev_encoder_rgb_tensor = torch.from_numpy(bev_encoder_rgb).float().unsqueeze(0).to('cuda')
		# BEV encoder RGB tensor shape: torch.Size([1, 3, 384, 1024])

		# 处理 LiDAR 输入（与 bev_encoder 训练预处理保持一致）
		bev_encoder_lidar = bev_encoder_t_u.lidar_to_ego_coordinate(self.bev_encoder_config, input_data['LIDAR'])
		# Original BEV encoder LiDAR shape: (15536, 3)
		
		# 记录状态用于 BEV 编码器分支的跨帧 LiDAR 对齐
		self.bev_encoder_state_log.append([gps_filtered[0], gps_filtered[1], compass_filtered, speed])
		# BEV encoder state log length: 2, latest state: 
		# 	[3128.49,      # ego_x: 世界坐标系 X 位置
		# 	6268.23,      # ego_y: 世界坐标系 Y 位置
		# 	-2.9786,      # ego_theta: 航向角（弧度）= -170.66°
		# 	0.000332]     # speed: 车速 m/s（几乎停止）
		# 每步仅覆盖半圈 LiDAR 扫描，将上一半圈对齐后拼接到当前帧
		if self.bev_encoder_lidar_last is not None and len(self.bev_encoder_state_log) >= 2:
			ego_x = self.bev_encoder_state_log[-1][0]
			ego_y = self.bev_encoder_state_log[-1][1]
			ego_theta = self.bev_encoder_state_log[-1][2]
			
			ego_x_last = self.bev_encoder_state_log[-2][0]
			ego_y_last = self.bev_encoder_state_log[-2][1]
			ego_theta_last = self.bev_encoder_state_log[-2][2]
			
			bev_encoder_lidar_last_aligned = self._align_lidar_bev_encoder(
				self.bev_encoder_lidar_last, 
				ego_x_last, ego_y_last, ego_theta_last,
				ego_x, ego_y, ego_theta
			)
			bev_encoder_lidar_full = np.concatenate((bev_encoder_lidar, bev_encoder_lidar_last_aligned), axis=0)
		else:
			bev_encoder_lidar_full = bev_encoder_lidar
		
		self.bev_encoder_lidar_last = bev_encoder_lidar.copy()
		self.bev_encoder_lidar_buffer.append(bev_encoder_lidar_full)
		# BEV encoder LiDAR shape after alignment and concatenation: (31093, 3)

		# 转为直方图式 BEV 表示
		bev_encoder_lidar_bev = lidar_to_bev_histogram(bev_encoder_lidar_full, self.bev_encoder_config)
		bev_encoder_lidar_bev_tensor = torch.from_numpy(bev_encoder_lidar_bev).float().unsqueeze(0).to('cuda')
		# BEV encoder LiDAR BEV tensor shape: torch.Size([1, 1, 256, 256])

		# 额外 BEV 相机（主要用于展示/可视化）。
		bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		
		result = {
				'rgb_front': rgb_front,
				'lidar_bev': lidar_bev_tensor,
				'gps': gps_filtered,  # Use UKF filtered CARLA coordinates
				'speed': speed,
				'compass': compass_filtered,  # Use UKF filtered compass
				'bev': bev,
				# 供 DP/MoT 推理使用的 BEV 编码器输入
				'bev_encoder_rgb': bev_encoder_rgb_tensor,  # (1, 3, H, W) on GPU
				'bev_encoder_lidar_bev': bev_encoder_lidar_bev_tensor,  # (1, C, H, W) on GPU
				}
		# _print_result_stats(result, step=self.step, every=20)
		# [Tick Result Stats] step=0
		# - rgb_front: shape=(512, 1024, 3), dtype=uint8, range=[22, 222]
		# - lidar_bev: shape=(3, 448, 448), dtype=torch.float32, range=[0, 0.996078]
		# - gps: shape=(2,), dtype=float64, range=[1753.54, 2774.21]
		# - speed: shape=(), dtype=float64, range=[0.000674771, 0.000674771]
		# - compass: shape=(), dtype=float64, range=[-1.57362, -1.57362]
		# - bev: shape=(512, 512, 3), dtype=uint8, range=[34, 201]
		# - bev_encoder_rgb: shape=(1, 3, 384, 1024), dtype=torch.float32, range=[11, 223]
		# - bev_encoder_lidar_bev: shape=(1, 1, 256, 256), dtype=torch.float32, range=[0, 1]
		# - next_command: <missing>
		# - target_point: <missing>
		# - next_target_point: <missing>
		# - theta: <missing>
		# [Target Points] TP=(28.2,-0.6), NTP=(37.1,8.3)
		
		
		# [result keys & shapes]
		# rgb_front: shape=(512, 1024, 3), dtype=uint8
		# lidar_bev: shape=torch.Size([3, 448, 448]), dtype=torch.float32
		# gps: shape=(2,), dtype=float64
		# speed: shape=(), dtype=float64
		# compass: shape=(), dtype=float64
		# bev: shape=(512, 512, 3), dtype=uint8
		# bev_encoder_rgb: shape=torch.Size([1, 3, 384, 1024]), dtype=torch.float32
		# bev_encoder_lidar_bev: shape=torch.Size([1, 1, 256, 256]), dtype=torch.float32
		# next_command: type=int, value=4
		# target_point: shape=(2,), dtype=float64
		# next_target_point: shape=(2,), dtype=float64
		# theta: shape=(), dtype=float64

		# route_planner 返回未来路点与高层指令（转向语义）。
		waypoint_route = self._route_planner.run_step(np.append(result['gps'], gps_pos[2]))
		# 用当前 GPS 去“查询”路线上在车前面的点，返回的是未来的路点序列
		
		# 统一生成 target_point 与 next_target_point：
		# 与 LeadMoT 离线训练 / eval_carla agent 同款 P1 公式。
		# tp=1.0s, ntp=2.0s；低速时至少前推 5m，避免停车起步 prompt 目标过近。
		target_point, far_command = self._lookahead_world_point(
			waypoint_route, result['speed'], _TP_LOOKAHEAD_S, result['gps'][:2], result['compass']
		)
		next_target_point, next_far_command = self._lookahead_world_point(
			waypoint_route, result['speed'], _NTP_LOOKAHEAD_S, result['gps'][:2], result['compass']
		)

		if self.last_command_tmp != far_command:
			self.last_command = self.last_command_tmp
		self.last_command_tmp = far_command
		
		if hasattr(target_point, '__iter__') and len(target_point) >= 2:
			if (target_point[:2] != self.target_point_prev[:2]).any() if isinstance(target_point, np.ndarray) else (list(target_point[:2]) != list(self.target_point_prev[:2])):
				self.target_point_prev = target_point
				self.commands.append(far_command.value)
		
		# next_command 用于 one-hot 高层指令输入（与历史逻辑对齐，取 -2）。
		result['next_command'] = self.commands[-2]
		ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass']) #result['compass'])
		ego_next_target_point = t_u.inverse_conversion_2d(next_target_point[:2], result['gps'], result['compass']) #result['compass'])
		final_goal_world = self._compute_local_final_goal_world(
			waypoint_route, result['gps'][:2], result['compass']
		)
		ego_final_goal = t_u.inverse_conversion_2d(final_goal_world[:2], result['gps'], result['compass'])


		# 	target_point (world): [3077.51879883 6259.67871094]
		#   ego position (gps): [3128.44417881 6268.18898308]
		#   compass (heading): -2.9786 rad (-170.66 deg)
		#   ego_target_point: [51.63139757  0.13357935]
		#   ego_next_target_point: [56.63139757  0.13357935]
		# 最终返回的是 ego 坐标下目标点，更适合控制与网络输入。
		result['target_point'] = ego_target_point  # numpy array (2,)
		result['next_target_point'] = ego_next_target_point  # numpy array (2,)
		result['final_goal'] = ego_final_goal  # numpy array (2,), remaining-route endpoint
		result['theta'] = compass_filtered  # Use UKF filtered compass


		# [result keys & shapes]
		# rgb_front: shape=(512, 1024, 3), dtype=uint8
		# lidar_bev: shape=torch.Size([3, 448, 448]), dtype=torch.float32
		# gps: shape=(2,), dtype=float64
		# speed: shape=(), dtype=float64
		# compass: shape=(), dtype=float64
		# bev: shape=(512, 512, 3), dtype=uint8
		# bev_encoder_rgb: shape=torch.Size([1, 3, 384, 1024]), dtype=torch.float32
		# bev_encoder_lidar_bev: shape=torch.Size([1, 1, 256, 256]), dtype=torch.float32
		# next_command: type=int, value=4
		# target_point: shape=(2,), dtype=float64
		# next_target_point: shape=(2,), dtype=float64
		# theta: shape=(), dtype=float64

		return result

	def _align_lidar_bev_encoder(self, lidar, x, y, orientation, x_target, y_target, orientation_target):
		"""
		将历史帧 LiDAR 对齐到当前帧坐标系。

		参数：
			lidar: 历史帧 LiDAR 点云，形状 (N, 3)
			x, y, orientation: 历史帧 ego 位姿
			x_target, y_target, orientation_target: 当前帧 ego 位姿

		返回：
			aligned_lidar: 对齐到当前帧坐标系的点云
		"""
		pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
		rot_diff = bev_encoder_t_u.normalize_angle(orientation_target - orientation)
		
		# 将全局位移差旋转到目标帧局部坐标系
		rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
		                            [np.sin(orientation_target), np.cos(orientation_target), 0.0], 
		                            [0.0, 0.0, 1.0]])
		pos_diff = rotation_matrix.T @ pos_diff
		
		return bev_encoder_t_u.algin_lidar(lidar, pos_diff, rot_diff)
	
	def _lookahead_world_point(self, waypoint_route, speed, lookahead_s, ego_xy, compass):
		"""
		沿 RoutePlanner 剩余 route 弧长前推 target_point / next_target_point。

		公式与 LeadMoT 训练侧 `_extract_tp_route_lookahead` 一致：
		`max(speed * lookahead_s, MIN_LOOKAHEAD_M)`。当 route 不足时 fallback
		到末端；route 为空时按当前航向构造远点占位，避免 prompt 维度/语义断裂。
		"""
		route = list(waypoint_route)
		ego_xy = np.asarray(ego_xy[:2], dtype=np.float64)
		target_dist = max(float(speed) * float(lookahead_s), _MIN_LOOKAHEAD_M)
		if len(route) == 0:
			direction = np.array([np.cos(compass), np.sin(compass)], dtype=np.float64)
			return (ego_xy + direction * _EMPTY_ROUTE_EXTENSION_M).astype(np.float32), RoadOption.LANEFOLLOW

		prev = ego_xy
		accum = 0.0
		last_cmd = route[-1][1]
		for pos, cmd in route:
			pos_xy = np.asarray(pos[:2], dtype=np.float64)
			seg = pos_xy - prev
			seg_len = float(np.linalg.norm(seg))
			if seg_len > 1e-6 and accum + seg_len >= target_dist:
				t = (target_dist - accum) / seg_len
				return (prev + t * seg).astype(np.float32), cmd
			accum += seg_len
			prev = pos_xy
			last_cmd = cmd
		return np.asarray(route[-1][0][:2], dtype=np.float32), last_cmd

	def _compute_local_final_goal_world(self, waypoint_route, ego_xy, compass):
		"""
		取 RoutePlanner 剩余 route 末端，得到 prompt 里的 final destination。

		这与 LeadMoT 训练侧 `meta["next_target_points"][-1]` 的语义一致：都是
		当前 route 剩余部分的真实末端，再统一转成 ego frame。
		"""
		route = list(waypoint_route)
		ego_xy = np.asarray(ego_xy[:2], dtype=np.float64)
		if len(route) == 0:
			direction = np.array([np.cos(compass), np.sin(compass)], dtype=np.float64)
			return (ego_xy + direction * _EMPTY_ROUTE_EXTENSION_M).astype(np.float32)
		return np.asarray(route[-1][0][:2], dtype=np.float32)

	def _truncate_route_by_target_point(self, route_waypoints_np, target_point_np):
		"""
		根据 target_point 对 route_pred 做截断。

		逻辑：
		- 将 target_point 投影到 route_pred 构成的折线上；
		- 若投影落在路线内部，则投影点之后的段可能无效，需要截断；
		- 若投影落在路线终点之后，则整段路线保持有效。

		保护机制：
		- 截断后点数过少或长度过短时，放弃截断，回退使用原路线。

		参数：
			route_waypoints_np: ego 坐标系路线点，形状 (N, 2)
			target_point_np: ego 坐标系目标点，形状 (2,)

		返回：
			truncated_route: 截断后的路线，形状 (M, 2), M<=N
			truncation_idx: 截断索引（-1 表示未截断）
		"""
		# 保护阈值
		MIN_POINTS_THRESHOLD = 5  # Minimum number of points needed for reliable control
		MIN_LENGTH_THRESHOLD = 3.0  # Minimum route length in meters for reliable lookahead
		
		if len(route_waypoints_np) < 2:
			return route_waypoints_np, -1
		
		# 查找与 target_point 最近的线段
		min_dist = float('inf')
		best_segment_idx = -1
		best_t = 0.0  # Parameter along segment [0, 1]
		best_proj_point = None
		
		for i in range(len(route_waypoints_np) - 1):
			p1 = route_waypoints_np[i]
			p2 = route_waypoints_np[i + 1]
			
			# 从 p1 指向 p2 的向量
			v = p2 - p1
			# 从 p1 指向 target_point 的向量
			w = target_point_np - p1
			
			# 线段长度平方
			l2 = np.dot(v, v)
			if l2 < 1e-10:  # Degenerate segment
				t = 0.0
				proj = p1
			else:
				# 将 target_point 投影到该线段所在直线
				t = np.dot(w, v) / l2
				proj = p1 + t * v
			
			# target_point 到投影点距离
			dist = np.linalg.norm(target_point_np - proj)
			
			# 投影参数解释：
			# t < 0：在线段起点之前
			# 0 <= t <= 1：在线段内部
			# t > 1：在线段终点之后
			
			if dist < min_dist:
				min_dist = dist
				best_segment_idx = i
				best_t = t
				best_proj_point = proj
		
		# 根据投影位置确定是否截断
		
		if best_segment_idx == -1:
			# 未找到有效线段，直接返回原路线
			return route_waypoints_np, -1
		
		# 计算投影点沿路线的弧长位置
		is_on_last_segment = (best_segment_idx == len(route_waypoints_np) - 2)
		
		if best_t > 1.0 and is_on_last_segment:
			# 投影点落在路线末端之后，不需要截断
			return route_waypoints_np, -1
		
		# 投影点在路线内部或起点附近时执行截断
		if best_t <= 0.0:
			# 投影在线段起点处或之前：保留到该线段起点
			truncation_idx = best_segment_idx
		elif best_t >= 1.0:
			# 投影在线段终点处或之后：保留到该线段终点
			truncation_idx = best_segment_idx + 1
		else:
			# 投影在线段内部：保留到线段起点并追加投影点
			truncation_idx = best_segment_idx
		
		# 组装截断后的路线
		if truncation_idx >= len(route_waypoints_np) - 1:
			# 无需截断
			return route_waypoints_np, -1
		
		# 先保留到 truncation_idx，再视情况追加投影点
		truncated = route_waypoints_np[:truncation_idx + 1].copy()
		
		# 若投影点与末点差异足够大，再追加投影点
		if best_proj_point is not None and len(truncated) > 0:
			dist_to_last = np.linalg.norm(best_proj_point - truncated[-1])
			if dist_to_last > 0.1:  # Only add if more than 0.1m away
				truncated = np.vstack([truncated, best_proj_point])
		
		# ============ 保护机制 ============
		# 计算截断路线总长度
		truncated_length = 0.0
		for i in range(len(truncated) - 1):
			truncated_length += np.linalg.norm(truncated[i + 1] - truncated[i])
		
		# 检查截断后路线是否满足最小要求
		if len(truncated) < MIN_POINTS_THRESHOLD or truncated_length < MIN_LENGTH_THRESHOLD:
			# 截断结果过短，放弃截断并使用原路线
			# 用于处理接近终点时 target_point 很近的边界情况
			print(f"[Lateral] Skip truncation: points={len(truncated)}, length={truncated_length:.2f}m "
				  f"(thresholds: {MIN_POINTS_THRESHOLD} points, {MIN_LENGTH_THRESHOLD}m)")
			return route_waypoints_np, -1
		
		return truncated, truncation_idx
	
	def control_pid(self, route_waypoints, velocity, speed_waypoints, target_point=None):
		"""
		使用 PID 将轨迹预测结果转换为车辆控制量。

		参数：
			route_waypoints: ego 坐标系路线点，形状 (1, N, 2)
			velocity: 当前速度（m/s）
			speed_waypoints: 用于速度估计的轨迹点，形状 (1, N, 2)
			target_point: 兼容保留参数，当前未启用
		"""
		assert route_waypoints.size(0) == 1
		route_waypoints_np = route_waypoints[0].data.cpu().numpy()  # (N, 2)
		speed = velocity  # Already a float
		speed_waypoints_np = speed_waypoints[0].data.cpu().numpy()  # (N, 2)
		
		# if target_point is not None:
		# 	target_point_np = target_point[0].data.cpu().numpy()  # (2,)
		# 	route_waypoints_np, _ = self._truncate_route_by_target_point(route_waypoints_np, target_point_np)
		
		# legacy AutoMoT speed head 仍输出 6 个点（0.5s 间隔）；这里只取 0.5s/1.0s
		# 两段估 desired speed。prompt 与 LeadMoT 规划视野保持 now/+1s/+2s、2s。
		mot_waypoint_interval = 0.5  # seconds between waypoints
		one_second_idx = 1 #1  # point[1] is at 1.0s
		half_second_idx = 0  # point[0] is at 0.5s
		
		if speed_waypoints_np.shape[0] >= 2:
			# 由 0.5s->1.0s 位移估计期望速度（乘 2 转 m/s）
			desired_speed = np.linalg.norm(speed_waypoints_np[one_second_idx] - speed_waypoints_np[half_second_idx]) * 2.0
			# desired_speed = np.linalg.norm(speed_waypoints_np[one_second_idx] - speed_waypoints_np[half_second_idx])
		else:
			# 兜底：仅用首点位移估计速度（假设对应 0.5 秒）
			desired_speed = np.linalg.norm(speed_waypoints_np[0]) * 2.0

		# 限速示例（当前注释掉）：35 km/h 对应 9.72 m/s
		# max_desired_speed_ms = 35.0 / 3.6  # 35 km/h in m/s
		# desired_speed = min(desired_speed, max_desired_speed_ms)

		brake = ((desired_speed < self.brake_speed) or ((speed / max(desired_speed, 1e-5)) > self.brake_ratio))
		
		delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
		throttle = self.speed_controller.step(delta)
		throttle = np.clip(throttle, 0.0, self.clip_throttle)
		throttle = throttle if not brake else 0.0
		

		route_interp = self.interpolate_waypoints(route_waypoints_np)
		
		
		steer = self.turn_controller.step(route_interp, speed)
		steer = np.clip(steer, -1.0, 1.0)
		steer = round(steer, 3)
		
		
		return steer, throttle, brake
	
	def interpolate_waypoints(self, waypoints):
		"""
		将路线点插值到约 0.1m 间距。

		参数：
			waypoints: ego 坐标系路线点，形状 (N, 2)

		返回：
			interp_points: 插值后路线点，形状 (M, 2)
		"""
		waypoints = waypoints.copy()
		# 在序列开头补原点，增强近场转向稳定性
		waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
		shift = np.roll(waypoints, 1, axis=0)
		shift[0] = shift[1]
		
		dists = np.linalg.norm(waypoints - shift, axis=1)
		dists = np.cumsum(dists)
		dists += np.arange(0, len(dists)) * 1e-4  # Prevents dists not being strictly increasing
		
		interp = PchipInterpolator(dists, waypoints, axis=0)
		
		x = np.arange(0.1, dists[-1], 0.1)
		
		interp_points = interp(x)
		
		if interp_points.shape[0] == 0:
			interp_points = waypoints[None, -1]
		
		return interp_points

	# ====== 车位脱困相关方法 ======

	def _update_pos_snapshots(self, tick_data):
		"""
		周期记录位置快照，并检测长时间“几乎不动”的死锁状态。

		规则：
		- 每隔 pos_snapshot_interval 帧记录一次 GPS；
		- 在 parking_deadlock_window 窗口内，若最大位移小于阈值，则判定死锁。

		该机制与红灯等待相容：
		- 短时等灯通常不会触发；
		- 长时不动才会进入脱困流程。
		"""
		if self.parking_escape_cooldown > 0:
			self.parking_escape_cooldown -= 1
			# 冷却期仍记录快照，加快后续再次判定，但不触发脱困
		
		if self.parking_escape_active:
			return False
		
		ego_pos = tick_data['gps'][:2]
		
		# 每 N 帧记录一次快照
		if self.step % self.pos_snapshot_interval == 0:
			self.pos_snapshots.append((self.step, ego_pos.copy()))
			# 仅保留窗口内快照
			cutoff = self.step - self.parking_deadlock_window
			self.pos_snapshots = [(s, p) for s, p in self.pos_snapshots if s >= cutoff]
		
		# 历史长度不足，暂不判定
		if len(self.pos_snapshots) < 2:
			return False
		
		oldest_step, oldest_pos = self.pos_snapshots[0]
		time_span = self.step - oldest_step
		
		# 时间窗口尚未覆盖完整，不判定
		if time_span < self.parking_deadlock_window:
			return False
		
		# 冷却期间不触发，仅持续跟踪
		if self.parking_escape_cooldown > 0:
			return False
		
		# 计算窗口内相对最早快照的最大位移
		max_displacement = 0.0
		for _, pos in self.pos_snapshots:
			d = np.linalg.norm(pos - oldest_pos)
			max_displacement = max(max_displacement, d)
		
		if max_displacement < self.parking_deadlock_max_disp:
			print(f"[ParkingDetect] === DEADLOCK === "
			      f"{time_span} frames ({time_span/20:.0f}s), "
			      f"max displacement = {max_displacement:.2f}m < {self.parking_deadlock_max_disp}m")
			return True
		
		return False

	def _activate_parking_escape(self, tick_data):
		"""
		激活车位脱困模式。

		策略：默认优先向左脱困（右侧停车位更常见）。
		Phase 1 会直接覆盖转向输出。
		"""
		self.parking_escape_active = True
		self.parking_escape_phase = 1
		self.parking_escape_timer = 40   # Phase 1: ~2.5 seconds at 20 fps (enough to turn out)
		self.parking_escape_anchor = tick_data['gps'][:2].copy()
		self.parking_escape_start_compass = tick_data['compass']  # Record heading for angle check
		self.parking_escape_attempt += 1
		
		# 固定向左脱困
		self.parking_escape_direction = 1.0   # +1 = left
		
		# 前移 route planner 队列，避免目标点过近导致持续原地打角
		n_pop = 0
		try:
			n_pop = min(5, max(0, len(self._route_planner.route) - 3))
			for _ in range(n_pop):
				if len(self._route_planner.route) > 3:
					self._route_planner.route.popleft()
					self._route_planner.route_distances.popleft()
			print(f"[ParkingEscape] Popped {n_pop} route WPs")
		except (AttributeError, IndexError) as e:
			print(f"[ParkingEscape] Skip route pop: {e}")
		
		# 清空卡死/强行前进状态，避免干扰脱困控制
		self.stuck_detector = 0
		self.force_move = 0
		
		dir_str = "LEFT" if self.parking_escape_direction > 0 else "RIGHT"
		print(f"[ParkingEscape] === ACTIVATED (attempt #{self.parking_escape_attempt}, "
		      f"dir={dir_str}, popped {n_pop} route WPs) ===")

	def _get_escape_target_points(self, target_point, next_target_point, gt_velocity):
		"""
		脱困期间生成覆盖用目标点（ego 坐标）。

		当前主要使用 Phase 1：给定侧向偏置 + 前向偏置，强制拉出车位。
		返回：
		- (override_tp, override_ntp) 张量对，或 (None, None)
		"""
		if not self.parking_escape_active:
			return None, None
		
		self.parking_escape_timer -= 1
		d = self.parking_escape_direction  # +1 left, -1 right
		
		if self.parking_escape_phase == 1:
			lat = d * (5.0 + self.parking_escape_attempt * 1.5)
			fwd = 3.0
			
			override_tp = torch.tensor([[fwd, lat]], dtype=torch.float32).to('cuda')
			override_ntp = torch.tensor([[fwd + 3.0, lat]], dtype=torch.float32).to('cuda')
			
			if self.parking_escape_timer <= 0:
				# Phase 1 超时则结束脱困，控制权回到模型输出
				self._end_parking_escape("phase 1 timeout (3s)")
				return None, None
			elif self.parking_escape_timer % 20 == 0:
				print(f"[ParkingEscape] Phase 1: timer={self.parking_escape_timer}, "
				      f"TP=({fwd:.0f}, {lat:.1f})")
			
			return override_tp, override_ntp
		
		return None, None

	def _check_escape_progress(self, ego_pos, compass=None):
		"""
		脱困过程中提前结束条件：
		- 相对锚点位移超过阈值；或
		- 航向变化角超过阈值（说明已经成功打出角度）。
		"""
		if not self.parking_escape_active or self.parking_escape_anchor is None:
			return
		
		displacement = np.linalg.norm(ego_pos - self.parking_escape_anchor)
		
		# 检查航向变化（仅 Phase 1 强制转向阶段）
		if compass is not None and self.parking_escape_start_compass is not None and self.parking_escape_phase == 1:
			heading_diff = abs(compass - self.parking_escape_start_compass)
			# 归一化到 [-pi, pi]
			if heading_diff > np.pi:
				heading_diff = 2 * np.pi - heading_diff
			heading_deg = np.degrees(heading_diff)
			if heading_deg > 25.0:
				print(f"[ParkingEscape] Turned {heading_deg:.1f}° (>{25}°), ending Phase 1 early")
				self._end_parking_escape(f"heading change {heading_deg:.1f}°, disp={displacement:.1f}m")
				return
		
		if displacement > 6.0:
			self._end_parking_escape(f"success! moved {displacement:.1f}m")

	def _end_parking_escape(self, reason=""):
		"""结束脱困并进入冷却期。"""
		print(f"[ParkingEscape] === ENDED: {reason} ===")
		self.parking_escape_active = False
		self.parking_escape_phase = 0
		self.parking_escape_timer = 0
		# 冷却期内继续保留快照，便于后续快速再次判定
		self.parking_escape_cooldown = 2400  # 2400 frames = 120 seconds (2 min) @ 20fps

	# ====== 车位脱困相关方法结束 ======

	def _build_obs_dict(self, tick_data):
		"""
		从历史缓冲构建 MoT 所需多帧观测。

		需求：
		- RGB: 每 5 帧抽样，共最多 4 帧（t0, t-5, t-10, t-15）
		- LiDAR BEV: 最新一帧

		返回：
			rgb_stacked: 形状 (1, N, C, H, W)
			lidar_last: 形状 (C, H, W)
		"""
		rgb_history_list = list(self.rgb_history)
		lidar_history_list = list(self.lidar_bev_history)

		# 从末尾按 5 帧间隔采样 RGB
		rgb_list = [rgb_history_list[-1 - i*5] for i in range(4) if -1 - i*5 >= -len(rgb_history_list)]
		rgb_list = rgb_list[::-1]  # Reverse to chronological order (oldest to newest)

		rgb_stacked = torch.stack(rgb_list, dim=0).unsqueeze(0)  # (1, N, C, H, W)

		# 使用最新一帧 LiDAR BEV
		lidar_last = lidar_history_list[-1]  # (C, H, W)

		return rgb_stacked, lidar_last

	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		"""
		每个仿真 tick 的决策主流程。

		输入：
		- input_data: CARLA 传感器原始数据（由框架注入）
		- timestamp: 当前仿真时间

		输出：
		- carla.VehicleControl（steer/throttle/brake）

		流程概览：
		1) tick() 得到结构化状态与多模态输入
		2) 维护历史缓冲并构造多帧观测
		3) 调用 MoT inferencer 输出 traj/route/text
		4) 用 control_pid 把轨迹转换为控制量
		5) 应用防卡死与限速规则，返回最终控制
		"""
		if not self.initialized:
			self._init()
		tick_data = self.tick(input_data)

		# rgb_front: shape=(512, 1024, 3), dtype=uint8
		# lidar_bev: shape=torch.Size([3, 448, 448]), dtype=torch.float32
		# gps: shape=(2,), dtype=float64
		# speed: shape=(), dtype=float64
		# compass: shape=(), dtype=float64
		# bev: shape=(512, 512, 3), dtype=uint8
		# bev_encoder_rgb: shape=torch.Size([1, 3, 384, 1024]), dtype=torch.float32
		# bev_encoder_lidar_bev: shape=torch.Size([1, 1, 256, 256]), dtype=torch.float32
		# next_command: type=int, value=4
		# target_point: shape=(2,), dtype=float64
		# next_target_point: shape=(2,), dtype=float64
		# theta: shape=(), dtype=float64

		# 当前帧基础状态打包：速度、角度、目标点、高层指令等。
		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
		# 与 agent_simlingo 保持一致：用 self.commands[-2] 做 one-hot
		one_hot_command = t_u.command_to_one_hot(self.commands[-2])
		cmd_one_hot = torch.from_numpy(one_hot_command[np.newaxis]).to('cuda', dtype=torch.float32)
		# command 仅用于元数据展示（将 1-6 映射到 0-5）
		command = tick_data['next_command']
		if command < 0:
			command = 4
		command -= 1
		speed = torch.FloatTensor([float(tick_data['speed'])]).view(1,1).to('cuda', dtype=torch.float32)
		theta = torch.FloatTensor([float(tick_data['theta'])]).view(1,1).to('cuda', dtype=torch.float32)
		lidar = tick_data['lidar_bev'].to('cuda', dtype=torch.float32)
		# torch.Size([3, 448, 448])
		
		rgb_front = torch.from_numpy(tick_data['rgb_front']).permute(2, 0, 1).float() / 255.0
		# (3, 512, 1024)

		rgb_front = rgb_front.to('cuda', dtype=torch.float32)
		waypoint = torch.from_numpy(tick_data['gps']).float().to('cuda', dtype=torch.float32)
		target_point = torch.from_numpy(tick_data['target_point']).unsqueeze(0).float().to('cuda', dtype=torch.float32)
		next_target_point = torch.from_numpy(tick_data['next_target_point']).unsqueeze(0).float().to('cuda', dtype=torch.float32)
		final_goal = torch.from_numpy(tick_data['final_goal']).unsqueeze(0).float().to('cuda', dtype=torch.float32)
		
		# 调试输出：周期打印目标点信息
		if self.step % 20 == 0:
			tp = tick_data['target_point']
			ntp = tick_data['next_target_point']
			print(f"[Target Points] TP=({tp[0]:.1f},{tp[1]:.1f}), NTP=({ntp[0]:.1f},{ntp[1]:.1f})")

		# 累积历史缓冲，后续按固定间隔抽帧构建 MoT 时序输入。
		self.lidar_bev_history.append(lidar)
		self.rgb_history.append(rgb_front)
		self.speed_history.append(speed)
		self.target_point_history.append(target_point)
		self.next_target_point_history.append(next_target_point)
		self.next_command_history.append(cmd_one_hot)
		self.theta_history.append(theta)
		self.waypoint_history.append(waypoint)

		# 记录上一帧油门/刹车（首帧补 0）
		if self.step < 1:
			self.throttle_history.append(torch.tensor(0.0).view(1, 1).to('cuda'))
			self.brake_history.append(torch.tensor(0.0).view(1, 1).to('cuda'))
		else:
			prev_control = self.prev_control if self.prev_control is not None else carla.VehicleControl()
			self.throttle_history.append(torch.tensor(prev_control.throttle).view(1, 1).to('cuda'))
			self.brake_history.append(torch.tensor(prev_control.brake).view(1, 1).to('cuda'))

		# 缓冲预热：至少累积 31 帧后再放行网络推理，避免时序输入不足。
		BUFFER_PHASE = 31

		if self.step < BUFFER_PHASE:
			# 预热阶段：强制刹车，使 UKF 初期状态更稳定
			control = carla.VehicleControl(0.0, 0.0, 1.0)
			self.control = control  # Important: UKF uses self.control for prediction
			self.pid_metadata = {}
			self.pid_metadata['agent'] = 'warmup_phase'
			self.pid_metadata['step'] = self.step
		else:
			# ====== 停车起步检测（一次性） ======
			if self.parking_start_anchor is None:
				self.parking_start_anchor = tick_data['gps'][:2].copy()
			if not self.parking_start_checked and self.step >= BUFFER_PHASE + self.parking_start_check_frame:
				disp = np.linalg.norm(tick_data['gps'][:2] - self.parking_start_anchor)
				self.parking_start_detected = (disp < self.parking_start_disp_thresh)
				self.parking_start_checked = True
				if self.parking_start_detected:
					print(f"[ParkingStart] Detected! Displacement in first {self.parking_start_check_frame} frames = {disp:.2f}m < {self.parking_start_disp_thresh}m. force_move DISABLED for this episode.")
				else:
					print(f"[ParkingStart] Normal start. Displacement = {disp:.2f}m. force_move enabled.")

			# ====== 脱困检测：基于快照进行长时卡死识别 ======
			deadlock_detected = self._update_pos_snapshots(tick_data)
			if deadlock_detected:
				self._activate_parking_escape(tick_data)
			if self.parking_escape_active:
				self._check_escape_progress(tick_data['gps'][:2], compass=tick_data['compass'])

			# 从历史缓冲抽取多帧观测（RGB 时序 + 最新 LiDAR BEV）。
			rgb_stacked, lidar_last = self._build_obs_dict(tick_data)
			# RGB torch.Size([1, 4, 3, 512, 1024]) and LiDAR torch.Size([3, 448, 448])
			# inferencer 接口使用 PIL 图像列表，这里做张量 -> PIL 转换。
			rgb_pil_list = []
			for i in range(rgb_stacked.shape[1]):
				rgb_tensor = rgb_stacked[0, i]  # (C, H, W)
				rgb_np = (rgb_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
				rgb_pil = Image.fromarray(rgb_np, mode='RGB')
				# [Observation] Converted RGB frame 0 to PIL with size (1024, 512)
				# [Observation] Converted RGB frame 1 to PIL with size (1024, 512)
				# [Observation] Converted RGB frame 2 to PIL with size (1024, 512)
				# [Observation] Converted RGB frame 3 to PIL with size (1024, 512)
				rgb_pil_list.append(rgb_pil)

			# LiDAR BEV 同样转为 PIL 形式，走统一多模态输入管线。
			lidar_tensor = lidar_last.squeeze(0) if lidar_last.dim() == 4 else lidar_last  # (C, H, W)
			lidar_np = (lidar_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
			# (448, 448, 3) 

			lidar_pil = Image.fromarray(lidar_np, mode='RGB')
			# (448, 448)
			
			lidar_pil_list = [lidar_pil]
			
			if self.stuck_helper > 0:
				target_point_speed=torch.cat([speed, target_point, next_target_point, final_goal], dim=-1)  # (1, 7)
				print("Get stucked! Trigger the stuck helper!")
			else:
				target_point_speed=torch.cat([speed, target_point, next_target_point, final_goal], dim=-1)  # (1, 7)

			# ====== 脱困期间：覆盖模型输入目标点 ======
			escape_tp, escape_ntp = self._get_escape_target_points(
				target_point, next_target_point, gt_velocity
			)
			if escape_tp is not None:
				# 仅覆盖模型输入 target_point_speed；
				# 原始 target_point 仍用于 BEV 可视化与 PID 控制
				target_point_speed = torch.cat([speed, escape_tp, escape_ntp, final_goal], dim=-1)  # (1, 7)

			# 根据速度与目标点生成干净的语言提示 + 模式开关。
			print(build_cleaned_prompt_and_modes.__code__.co_filename)
			# /home/cruser1/lda/AutoMoT/leaderboard/team_code/automot_utils.py
			
			prompt_cleaned, understanding_output, reasoning_output = build_cleaned_prompt_and_modes(target_point_speed)
			# Your current and next target point is (...), (...), your final destination is (...),
			# and your current velocity is 9.25 m/s. Predict the driving actions ( now, +1s, +2s)
			# and plan the trajectory for the next 2 seconds.
			# understanding_output = False
			# reasoning_output = True
			
			t0 = time.time()
			# if dp_vit_feat.dim() == 2:
			# 	dp_vit_feat = dp_vit_feat.unsqueeze(0)  # (1, Nvit, C)
			
			# reason_feat = predicted_answer['reasoning_feat']
			# if reason_feat.dim() == 2:
			# 	reason_feat = reason_feat.unsqueeze(0)  # (1, Nr, C)
			
			# ========== 先运行 BEV 编码器，得到 trans_feat ==========
			with torch.no_grad():
				# 输入转 bfloat16，与 BEV 编码器权重精度一致
				bev_encoder_rgb_bf16 = tick_data['bev_encoder_rgb'].to(torch.bfloat16)
				bev_encoder_lidar_bev_bf16 = tick_data['bev_encoder_lidar_bev'].to(torch.bfloat16)
				
				# bev_encoder_rgb_bf16 -> shape=(1, 3, 384, 1024), dtype=torch.bfloat16, range=[0, 254]
				# bev_encoder_lidar_bev_bf16 -> shape=(1, 1, 256, 256), dtype=torch.bfloat16, range=[0, 1]
				bev_encoder_output = self.bev_encoder(
					rgb=bev_encoder_rgb_bf16,  # (1, 3, H, W) on GPU, bfloat16
					lidar_bev=bev_encoder_lidar_bev_bf16  # (1, C, H, W) on GPU, bfloat16
				)
				# [BEV Encoder] Output keys: ['bev_feature', 'bev_feature_upscale', 'fused_features', 'image_feature_grid']
			
			# 提取 BEV 编码器特征
			# bev_feature: (B, 1512, 8, 8)
			bev_encoder_bev_feature = bev_encoder_output['bev_feature']  # (1, 1512, 8, 8)
			# bev_encoder_output keys: ['bev_feature', 'bev_feature_upscale', 'fused_features', 'image_feature_grid']
			# bev_feature: tensor, shape=(1, 1512, 8, 8), dtype=torch.bfloat16, min=-1.015625, max=4.406250
			# bev_feature_upscale: tensor, shape=(1, 64, 64, 64), dtype=torch.bfloat16, min=0.000000, max=25.750000
			# fused_features: tensor, shape=(1, 1512, 8, 8), dtype=torch.bfloat16, min=-1.015625, max=4.406250
			# image_feature_grid: tensor, shape=(1, 1512, 12, 32), dtype=torch.bfloat16, min=-10.250000, max=18.875000



			# 主推理：输出文本决策、速度轨迹 traj、转向路径 route。
			output = self.inferencer(
				image=rgb_pil_list,
				front=[rgb_pil_list[-1]],
				lidar=lidar_pil_list,
				text=prompt_cleaned,
				understanding_output=understanding_output,
				reasoning_output=reasoning_output,
				max_think_token_n=self.inference_args.max_num_tokens,
				v_target_point=target_point_speed,
				trans_feat=bev_encoder_bev_feature,
				do_sample=False,
				text_temperature=0.0,
				frame_idx=self.step,
			)
			# gen_text: <|im_start|> accelerate, slow, slow<|im_end|>
            # gen_traj (after cumsum).shape: torch.Size([1, 6, 2])
            # route.shape: torch.Size([1, 20, 2])
            # reasoning_hidden_states.shape: torch.Size([8, 2560])

            
			pred_traj = output['traj']
			pred_route = output['route']
			pred_decision = output.get('text', '')
			
			# 缓存预测结果供渲染使用
			self.last_pred_traj = pred_traj.squeeze(0).float().cpu().numpy()  # (6, 2) in [x, y] format
			self.last_target_point = target_point.squeeze(0).float().cpu().numpy()  # (2,) in [x, y] format
			self.last_next_target_point = next_target_point.squeeze(0).float().cpu().numpy()  # (2,) in [x, y] format
			
			t1 = time.time()
			print(f"[Inference] MoT inference time: {t1 - t0:.2f} seconds")

			# ================== 轨迹 -> 控制 ==================
			# - speed_waypoints: 用 pred_traj 做纵向（油门/刹车）
			# - route_waypoints: 用 pred_route 做横向（转向）
			speed_waypoints = pred_traj.float()  # (1, 6, 2) for speed control
			route_pred = pred_route
			if isinstance(route_pred, torch.Tensor):
				route_waypoints = route_pred.float().cpu()  # (1, 20, 2) for steering control
			else:
				route_waypoints = torch.from_numpy(route_pred).float()
			self.last_route_pred = route_waypoints.squeeze(0).numpy().copy()  # (20, 2) for visualization

			gt_velocity = tick_data['speed']
			
			# 使用 target_point 参与横向控制（当前接口保留，截断逻辑可选）
			steer, throttle, brake = self.control_pid(route_waypoints, gt_velocity, speed_waypoints, target_point=target_point)
			
			# 常规卡死检测（低速持续时间）。
			if gt_velocity < 0.1:
				self.stuck_detector += 1
			elif gt_velocity > 0.2:
				self.stuck_detector = 0
			
			# 停车脱困逻辑优先级高于常规 force_move，必要时覆盖控制输出。
			if self.parking_escape_active:
				# 脱困阶段关闭 force_move，避免互相干扰
				self.stuck_detector = 0
				self.force_move = 0
				d = self.parking_escape_direction  # +1=left, -1=right
				# Phase 1：强制大转角 + 中等油门，优先驶离车位
				if self.parking_escape_phase == 1:
					# CARLA 约定：steer>0 向右，steer<0 向左
					steer = -d * 0.65  # Strong left turn
					throttle = 0.45
					brake = False
					if self.parking_escape_timer % 20 == 0:
						print(f"[ParkingEscape] Phase1 OVERRIDE: steer={steer:.2f}, throttle={throttle:.2f}")
			else:
				# 持续低速过久触发 force_move
				# 但停车起步场景下禁用，避免误触
				if self.stuck_detector > self.stuck_threshold and not self.parking_start_detected:
					self.force_move = self.creep_duration
				
				# force_move：覆盖油门/刹车，尝试脱离低速僵局
				if self.force_move > 0:
					throttle = max(self.creep_throttle, throttle)
					brake = False
					self.force_move -= 1
					print(f"force_move: {self.force_move}")

			print(f"stuck_detector: {self.stuck_detector}")

			
			control = carla.VehicleControl()
			control.steer = float(steer)
			control.throttle = float(throttle)
			control.brake = float(brake)

			# 限速保护：超过 35 km/h 强制制动（安全兜底）。
			# gt_velocity 单位 m/s，乘 3.6 转换为 km/h
			if gt_velocity * 3.6 > 35:
				control.throttle = 0.0
				control.brake = 1.0

			# 保存本帧控制与状态元数据，供日志/回放/分析。
			self.pid_metadata = {
				'agent': 'mot',
				'steer': control.steer,
				'throttle': control.throttle,
				'brake': control.brake,
				'speed': gt_velocity,
				'command': command,
			}

			self.prev_control = control
			self.control = control  # Update control for UKF prediction in next tick
			metric_info = self.get_metric_info()
			self.metric_info[self.step] = metric_info

			if SAVE_PATH is not None:
				self.save(tick_data)

			##### 渲染与可视化 ####
			ego_car_map = render_self_car(
				loc=np.array([0, 0]),
				ori=np.array([0, -1]),
				box=np.array([2.45, 1.0]),
				color=[1, 1, 0], pixels_per_meter=10, max_distance=30,
			)

			# 为渲染准备目标点
			tp_for_render = target_point.cpu().float().numpy().copy()
			if tp_for_render.ndim == 2:
				tp_for_render = tp_for_render.squeeze(0)
			tp_for_render[1] = -tp_for_render[1]  # Negate y: left -> right

			# 渲染 MoT 速度轨迹（绿色）
			mot_traj_for_render = pred_traj.squeeze(0).cpu().float().numpy().copy()  # (6, 2)
			mot_traj_for_render[:, 1] = -mot_traj_for_render[:, 1]  # Negate y: left -> right
			mot_traj_trajectory = np.concatenate((mot_traj_for_render, tp_for_render.reshape(1, 2)), axis=0)
			mot_traj_trajectory = mot_traj_trajectory[:, [1, 0]]
			mot_traj_trajectory[:, 0] = -mot_traj_trajectory[:, 0]  # y (now in col 0) 
			mot_traj_trajectory[:, 1] = -mot_traj_trajectory[:, 1]  # x (now in col 1)
			render_mot_traj = render_waypoints(mot_traj_trajectory, pixels_per_meter=30, max_distance=20, color=(0, 255, 0))
			
			# 渲染 MoT 路线轨迹（红色）
			mot_route_for_render = pred_route.squeeze(0).cpu().float().numpy().copy()  # (6, 2)
			mot_route_for_render[:, 1] = -mot_route_for_render[:, 1]  # Negate y: left -> right
			mot_route_trajectory = np.concatenate((mot_route_for_render, tp_for_render.reshape(1, 2)), axis=0)
			mot_route_trajectory = mot_route_trajectory[:, [1, 0]]
			mot_route_trajectory[:, 0] = -mot_route_trajectory[:, 0]  # y (now in col 0) 
			mot_route_trajectory[:, 1] = -mot_route_trajectory[:, 1]  # x (now in col 1)
			render_mot_route = render_waypoints(mot_route_trajectory, pixels_per_meter=30, max_distance=20, color=(255, 0, 0))

			ego_car_map = cv2.resize(ego_car_map, (200, 200))
			render_mot_traj = cv2.resize(render_mot_traj, (200, 200))
			render_mot_route = cv2.resize(render_mot_route, (200, 200))

			surround_map = np.clip(
				(
					ego_car_map.astype(np.float32)
					+ render_mot_traj.astype(np.float32)
					+ render_mot_route.astype(np.float32)
				),
				0,
				255,
			).astype(np.uint8)
			tick_data["predicted_trajectory"] = surround_map
			decision_now, decision_1s, decision_2s = parse_decision_sequence(pred_decision)
			tick_data["decision_now"] = decision_now
			tick_data["decision_1s"] = decision_1s
			tick_data["decision_2s"] = decision_2s

			tick_data["rgb_raw"] = tick_data["rgb_front"]

			tick_data["rgb"] = cv2.resize(tick_data["rgb_front"], (800, 600))

			# 保存前先在 BEV 图上绘制轨迹，生成 bev_traj
			bev_img_for_display = tick_data['bev'].copy()
			if self.last_pred_traj is not None:
				bev_img_for_display = self._draw_trajectory_on_bev(
					bev_img_for_display, self.last_pred_traj, self.last_target_point,
					self.last_next_target_point, self.last_route_pred)
			tick_data["bev_traj"] = cv2.resize(bev_img_for_display, (400, 400))

			tick_data["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
				control.throttle,
				control.steer,
				control.brake,
			)
			tick_data["speed"] = "speed: %.2f Km/h, target point x: %.2f m, target point y: %.2f m" % (gt_velocity*3.6, target_point.squeeze(0).cpu().float().numpy()[0], target_point.squeeze(0).cpu().float().numpy()[1])
			
			sentence1, sentence2 = split_prompt(prompt_cleaned)
			tick_data["language_1"] = "Instruction: " + sentence1
			tick_data["language_2"] = sentence2

			tick_data["mes"] = "speed: %.2f" % gt_velocity
			tick_data["time"] = "time: %.3f" % timestamp

			surface = self._hic.run_interface(tick_data)
			tick_data["surface"] = surface

		return control

	def save(self, tick_data):
		frame = self.step 
		Image.fromarray(tick_data['rgb_front']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
		
		# 若有预测轨迹，则先绘制到 BEV 后再保存
		bev_img = tick_data['bev'].copy()
		if self.last_pred_traj is not None:
			# 传入 last_route_pred（横向控制点）与双目标点用于可视化
			bev_img = self._draw_trajectory_on_bev(bev_img, self.last_pred_traj, self.last_target_point,
			                                        self.last_next_target_point, self.last_route_pred)
		tick_data['bev_traj'] = bev_img
		Image.fromarray(bev_img).save(self.save_path / 'bev' / ('%04d.png' % frame))
		
		if 'lidar_bev' in tick_data:
			lidar_bev_tensor = tick_data['lidar_bev']
			if isinstance(lidar_bev_tensor, torch.Tensor):
				lidar_bev_tensor = lidar_bev_tensor.cpu().numpy()
			lidar_bev_img = (lidar_bev_tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
			imageio.imwrite(str(self.save_path / 'lidar_bev' / (f'{frame:04d}.png')), lidar_bev_img)

		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

		# 写出 metric 信息
		outfile = open(self.save_path / 'metric_info.json', 'w')
		json.dump(self.metric_info, outfile, indent=4)
		outfile.close()

	def _draw_trajectory_on_bev(self, bev_img, traj, target_point=None, next_target_point=None, route_pred=None):
		"""
		在 BEV 图像上绘制预测轨迹与关键点。

		BEV 相机参数：
		- 位置: (x=0, y=0, z=50)
		- FOV: 50 度
		- 图像尺寸: 512x512

		坐标约定：
		- 模型 ego 坐标：x 前向，y 左向
		- 图像坐标：中心为 ego，向上为前向（行号减小）
		- 可视化时会对 y 取反以适配右向为正的屏幕方向

		参数：
			bev_img: 原始 BEV 图像
			traj: 预测轨迹点 (6,2)
			target_point: 目标点（可选）
			next_target_point: 下一目标点（可选）
			route_pred: 横向控制路线点（可选）

		返回：
			bev_img: 绘制后的图像
		"""
		img_h, img_w = bev_img.shape[:2]  # 512, 512
		
		# 根据 BEV 相机参数换算每像素对应米数
		fov_rad = np.deg2rad(50.0)
		ground_size = 2 * 50.0 * np.tan(fov_rad / 2)  # meters covered by the image
		meters_per_pixel = ground_size / img_w  # ~0.093 m/pixel
		
		# 图像中心即 ego 位置
		cx, cy = img_w // 2, img_h // 2
		
		# 轨迹点从 ego 坐标投影到像素坐标
		
		pixels = []
		for i in range(len(traj)):
			x, y = traj[i]  # x: forward, y: left (model convention)
			# 可视化时对 y 取反：左正方向 -> 右正方向
			pixel_col = int(cx + y / meters_per_pixel)  # y_left negated: +y_left -> -col, so use + to flip
			pixel_row = int(cy - x / meters_per_pixel)
			pixels.append((pixel_col, pixel_row))
		
		# 先绘制轨迹连线
		for i in range(len(pixels) - 1):
			pt1 = pixels[i]
			pt2 = pixels[i + 1]
			# 绘制前检查坐标是否在图像边界内
			if (0 <= pt1[0] < img_w and 0 <= pt1[1] < img_h and
				0 <= pt2[0] < img_w and 0 <= pt2[1] < img_h):
				cv2.line(bev_img, pt1, pt2, (0, 255, 0), 2)  # Green line
		
		# 再绘制轨迹点
		for i, (col, row) in enumerate(pixels):
			if 0 <= col < img_w and 0 <= row < img_h:
				# 使用颜色渐变区分轨迹先后（起点偏红，终点偏蓝）
				color_r = int(255 * (1 - i / (len(pixels) - 1)))
				color_b = int(255 * (i / (len(pixels) - 1)))
				cv2.circle(bev_img, (col, row), 5, (color_r, 0, color_b), -1)
		
		# 若提供 route_pred，额外绘制横向控制路线点
		if route_pred is not None:
			route_pixels = []
			for i in range(len(route_pred)):
				x, y = route_pred[i]  # x: forward, y: left (model convention)
				pixel_col = int(cx + y / meters_per_pixel)
				pixel_row = int(cy - x / meters_per_pixel)
				route_pixels.append((pixel_col, pixel_row))
			
			# 绘制 route_pred 连线
			for i in range(len(route_pixels) - 1):
				pt1 = route_pixels[i]
				pt2 = route_pixels[i + 1]
				if (0 <= pt1[0] < img_w and 0 <= pt1[1] < img_h and
					0 <= pt2[0] < img_w and 0 <= pt2[1] < img_h):
					cv2.line(bev_img, pt1, pt2, (255, 165, 0), 1)  # Orange line (thinner)
			
			# 绘制 route_pred 点
			for i, (col, row) in enumerate(route_pixels):
				if 0 <= col < img_w and 0 <= row < img_h:
					# route_pred 使用实心蓝点
					cv2.circle(bev_img, (col, row), 3, (0, 0, 255), -1)  # Blue circles (smaller)
		
		# 若提供 target_point，绘制青色大圆
		if target_point is not None:
			x, y = target_point[0], target_point[1]  # x: forward, y: left (model convention)
			# 可视化坐标系转换（y 取反）
			tp_col = int(cx + y / meters_per_pixel)  # Negate y: +y_left -> -col, use + to flip
			tp_row = int(cy - x / meters_per_pixel)
			if 0 <= tp_col < img_w and 0 <= tp_row < img_h:
				cv2.circle(bev_img, (tp_col, tp_row), 10, (0, 255, 255), -1)  # Cyan circle for target point
				cv2.circle(bev_img, (tp_col, tp_row), 12, (255, 255, 255), 2)  # White border
		
		# 若提供 next_target_point，绘制品红色大圆
		if next_target_point is not None:
			x, y = next_target_point[0], next_target_point[1]  # x: forward, y: left (model convention)
			# 可视化坐标系转换（y 取反）
			ntp_col = int(cx + y / meters_per_pixel)
			ntp_row = int(cy - x / meters_per_pixel)
			if 0 <= ntp_col < img_w and 0 <= ntp_row < img_h:
				cv2.circle(bev_img, (ntp_col, ntp_row), 10, (255, 0, 255), -1)  # Magenta circle for next target point
				cv2.circle(bev_img, (ntp_col, ntp_row), 12, (255, 255, 255), 2)  # White border
		
		# 绘制 ego 位置（中心点）
		cv2.circle(bev_img, (cx, cy), 8, (255, 255, 0), -1)  # Yellow circle for ego
		
		return bev_img

	def destroy(self):
		torch.cuda.empty_cache()

	def gps_to_location(self, gps):
		# gps 内容格式：numpy 数组 [lat, lon, alt]
		lat, lon = gps
		scale = math.cos(self.lat_ref * math.pi / 180.0)
		my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
		mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
		y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
		x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
		return np.array([x, y])


