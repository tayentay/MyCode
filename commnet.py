from ns import ns
import math
import random
from typing import List, Dict
import numpy as np
import sys
import collections
from config import CC_CONFIGS, Config
from channel_utils import G2AChannel, A2SChannel, G2AScheduler, LEOSatelliteMobility
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils import parse_args

ns.cppyy.cppdef("""
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/quic-module.h"
#include <map>

using namespace ns3;

EventImpl* pythonMakeEvent(void (*f)(std::vector<std::string>), std::vector<std::string> l)
{
    return MakeEvent(f, l);
}

class OooMonitor {
public:
    uint64_t m_oooCount;
    uint64_t m_totalRx;
    uint64_t m_oooMagnitudeSum;
    std::map<uint32_t, uint64_t> m_maxUid;
    std::map<uint32_t, uint64_t> m_lastUid;
    uint32_t m_numUes;

    OooMonitor(uint32_t numUes) : m_oooCount(0), m_totalRx(0), m_oooMagnitudeSum(0), m_numUes(numUes) {}

    void RxCallback(Ptr<const Packet> p, Ptr<Ipv4> ipv4, uint32_t interface) {
        Ipv4Header header;
        if (p->PeekHeader(header) == 0) return;
        Process(header, p);
    }

    void LocalDeliverCallback(const Ipv4Header &header, Ptr<const Packet> p, uint32_t interface) {
        Process(header, p);
    }

    void Process(const Ipv4Header &header, Ptr<const Packet> p) {
        Ipv4Address src = header.GetSource();
        uint32_t addr = src.Get(); 
        
        // Check IP format 10.{ue_id+1}.x.x
        uint8_t firstOctet = (addr >> 24) & 0xFF;
        if (firstOctet != 10) return;
        
        uint8_t secondOctet = (addr >> 16) & 0xFF;
        if (secondOctet < 1 || secondOctet > m_numUes) return;
        
        uint32_t ue_id = secondOctet - 1;

        // Try to extract QUIC Packet Number
        uint64_t seq = 0;
        bool found = false;
        uint8_t proto = header.GetProtocol();
        
        Ptr<Packet> pCopy = p->Copy();
        if (proto == 17) { // UDP
            UdpHeader udp;
            if (pCopy->RemoveHeader(udp) > 0) {
                // Check for valid QUIC header to avoid crash in GetPacketNumLen
                if (pCopy->GetSize() >= 1) {
                    uint8_t buf[1];
                    pCopy->CopyData(buf, 1);
                     // Short Header Check (Bit 7 is 0)
                    if ((buf[0] & 0x80) == 0) {
                         // ns-3 implementation limitation: Type (lower 5 bits) must be 0, 1, or 2
                         if ((buf[0] & 0x1F) > 2) goto skip_quic; 
                    }
                    
                    QuicHeader quic;
                    if (pCopy->PeekHeader(quic) > 0) {
                        seq = quic.GetPacketNumber().GetValue();
                        found = true;
                    }
                }
            }
        } else if (proto == 143) { // QUIC
             QuicHeader quic;
             if (pCopy->PeekHeader(quic) > 0) {
                 seq = quic.GetPacketNumber().GetValue();
                 found = true;
             }
        }
        
        skip_quic:
        if (!found) return;

        m_totalRx++;
        uint64_t uid = seq;
        
        // Calculate OOO magnitude based on previous packet arrival
        auto itLast = m_lastUid.find(ue_id);
        if (itLast != m_lastUid.end()) {
            // Handle wrap-around or reordering
            // SequenceNumber32 logic approx: if diff is large positive, it might be old packet (OOO)
            if (itLast->second > uid) {
                 // Check if it's not just a wrap around (heuristic for 32-bit seq)
                 // If diff is huge (e.g. > 2^31), it might be forward wrap
                 m_oooMagnitudeSum += (itLast->second - uid);
            }
        }
        m_lastUid[ue_id] = uid;
        
        auto it = m_maxUid.find(ue_id);
        if (it == m_maxUid.end()) {
            m_maxUid[ue_id] = uid;
        } else {
            if (uid < it->second) {
                m_oooCount++;
            } else {
                it->second = uid;
            }
        }
    }
};

void ConnectOooRx(std::string path, OooMonitor* monitor) {
    Config::ConnectWithoutContext(path, MakeCallback(&OooMonitor::RxCallback, monitor));
}

void ConnectOooLocalDeliver(std::string path, OooMonitor* monitor) {
    Config::ConnectWithoutContext(path, MakeCallback(&OooMonitor::LocalDeliverCallback, monitor));
}
"""
)


g_sim = None
def scheduling_callback(args):
    global g_sim
    if g_sim:
        g_sim.do_scheduling()

# ==================== 主仿真类 ====================
class UAVNetworkSimulation:
    def __init__(self, config):
        self.config = config
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        ns.SeedManager.SetSeed(self.config.seed)
        self.G2A_channel = G2AChannel(config)
        self.A2S_channel = A2SChannel(config)
        self.G2A_scheduler = G2AScheduler(config, self.G2A_channel)
        self.num_ues = config.num_ues
        self.num_uavs = config.num_uavs
        self.port = config.port
        self.n_rbs = config.n_rbs
        # self.ue_app_sending = np.zeros(self.num_ues, dtype=bool)  # 记录UE应用是否在发数据
        self.time_slot_duration = self.config.time_slot_duration
        self.sat_mobility = LEOSatelliteMobility(config)
        self.selected_cc_config = CC_CONFIGS.get(config.cc, CC_CONFIGS['OLIA'])
        factory = ns.ObjectFactory("ns3::RateErrorModel")
        self.em = factory.Create()
        ns.Config.SetDefault("ns3::QuicSocketBase::IdleTimeout", ns.TimeValue(ns.Seconds(10)))
    
        # 配置MPQUIC参数
        # 参考wns3-mpquic-four-path.cc中的配置
        # 显式关闭 0-RTT 握手，并固定初始版本，避免 ZRTT 触发版本断言
        # ns.Config.SetDefault("ns3::QuicL4Protocol::0RTT-Handshake", ns.BooleanValue(False))
        # 使用 ns-3 实现的稳定版本 0xf1f1f1f1 (QUIC_VERSION_NS3_IMPL)
        ns.Config.SetDefault("ns3::QuicSocketBase::InitialVersion", ns.UintegerValue(0xf1f1f1f1))
        ns.Config.SetDefault("ns3::QuicSocketBase::EnableMultipath", ns.BooleanValue(True))
        
        # 使用 RateBased (2) 拥塞控制
        ns.Config.SetDefault("ns3::QuicSocketBase::CcType", ns.IntegerValue(self.selected_cc_config.enum_value))
        
        # 根据 cc 类型设置 SocketType
        if config.cc == 'RateB':
            ns.Config.SetDefault("ns3::QuicL4Protocol::SocketType", ns.TypeIdValue(ns.RateBasedCongestionOps.GetTypeId()))
            # RateB parameters
            ns.Config.SetDefault("ns3::RateBasedCongestionOps::Gamma", ns.DoubleValue(config.rate_b_gamma))
            ns.Config.SetDefault("ns3::RateBasedCongestionOps::Rho", ns.DoubleValue(config.rate_b_rho))
            ns.Config.SetDefault("ns3::RateBasedCongestionOps::A", ns.DoubleValue(config.rate_b_a))
            ns.Config.SetDefault("ns3::RateBasedCongestionOps::B", ns.DoubleValue(config.rate_b_b))
            ns.Config.SetDefault("ns3::RateBasedCongestionOps::MMax", ns.DoubleValue(config.rate_b_m_max))
        else:
            # 对于 OLIA 和 QuicNewReno，使用默认的 MpQuicCongestionOps
            # 注意：QuicL4Protocol::SocketType 默认为 QuicCongestionOps，但为了支持多路径，通常使用 MpQuicCongestionOps
            # 如果 ns-3 实现中，MpQuicCongestionOps 是用于 OLIA 等的类，则需要设置它
            # 从 C++ 代码看，OLIA 分支会 cast 为 MpQuicCongestionOps，所以必须是这个类型
             ns.Config.SetDefault("ns3::QuicL4Protocol::SocketType", ns.TypeIdValue(ns.MpQuicCongestionOps.GetTypeId()))
        
        ns.Config.SetDefault("ns3::MpQuicScheduler::SchedulerType", ns.IntegerValue(6))  # 0=ROUND_ROBIN, 1=MIN_RTT, 2=BLEST, 3=ECF, 4=PEEKABOO, 6=SELECT
        # 缓冲区设置
        # Reduced buffer to avoids stalling simulation at t=1.0s
        ns.Config.SetDefault("ns3::QuicSocketBase::SocketSndBufSize", ns.UintegerValue(256*1024))  # 256*1024
        ns.Config.SetDefault("ns3::QuicSocketBase::SocketRcvBufSize", ns.UintegerValue(256*1024))  # 256*1024
        ns.Config.SetDefault("ns3::QuicStreamBase::StreamSndBufSize", ns.UintegerValue(256*1024))  # 256*1024
        ns.Config.SetDefault("ns3::QuicStreamBase::StreamRcvBufSize", ns.UintegerValue(256*1024))  # 256*1024
        
        # 统计乱序用的变量

    def init_states(self):
        self.ue_positions = self.config.ue_positions.copy()
        self.uav_positions = np.empty((self.num_uavs, 3), dtype=np.float32)
        self.G2A_rates = np.zeros(self.num_ues, dtype=np.float32)
        self.A2S_rates = np.zeros(self.num_uavs, dtype=np.float32)
        self.G2A_gain = np.empty((self.num_uavs, self.num_ues), dtype=np.float32)  # 信道增益矩阵 
        self.assignment = np.zeros((self.num_uavs, self.num_ues), dtype=bool) 
        self.ue_uav_devices = [[None for _ in range(self.num_uavs)] for _ in range(self.num_ues)]
        self.ue_uav_interfaces = [[None for _ in range(self.num_uavs)] for _ in range(self.num_ues)]
        self.uav_sat_devices = [None for _ in range(self.num_uavs)]
        self.uav_sat_interfaces = [None  for _ in range(self.num_uavs)]
        self.G2A_sum_rates = np.zeros(self.num_ues, dtype=np.float32)
        self.A2S_sum_rates = np.zeros(self.num_uavs, dtype=np.float32)
        self.ue_apps = []
        self.ue_sockets = []
        self.ue_send_buf_remain = np.zeros(self.num_ues, dtype=np.float32)
        # per-UAV A2S queue occupancy (normalized)
        self.A2S_queue_occupancy = np.zeros(self.num_uavs, dtype=np.float32)
        # 随机噪声状态（维度由 config.noise_dim 决定）
        self.noise_state = np.zeros(int(getattr(self.config, 'noise_dim', 0)), dtype=np.float32)
        self.last_total_tx_bytes = 0
        self.trans_per_step = 0
        self.last_total_rx_bytes = 0
        self.recei_per_step = 0
        self.loss_per_step = 0
        self.flow_loss_counts = collections.defaultdict(int) # Track lost packets per flow
        
        # 使用C++ Monitor
        self.ooo_monitor = ns.cppyy.gbl.OooMonitor(self.num_ues)
        self.G2A_scheduler.history:  List[Dict] = []
        
        # Rate averaging state for BDP calculation
        self.ue_last_connected_uav = np.full(self.num_ues, -1, dtype=int)
        self.ue_last_rate = np.zeros(self.num_ues, dtype=np.float32)
        self.ue_rate_ema = np.zeros(self.num_ues, dtype=np.float32)
        self.ue_rate_queue = [collections.deque(maxlen=10) for _ in range(self.num_ues)]

        # Congestion Avoidance States
        self.ue_ca_cwnd = np.zeros(self.num_ues, dtype=np.float32)
        self.ue_last_rtt = np.zeros(self.num_ues, dtype=np.float32)
        self.ue_last_loss_count = {} # Map ue_id -> lost_packets
        self.ue_cwnd_history = [[] for _ in range(self.num_ues)]

        # CA State
        self.ue_ca_cwnd = np.zeros(self.num_ues, dtype=np.float32) # Track current cwnd
        self.ue_last_rtt = np.zeros(self.num_ues, dtype=np.float32) # D(t-1)
        self.ue_cwnd_history = [collections.deque(maxlen=self.config.ca_w_prime) for _ in range(self.num_ues)]

        # Stats tracking for RewardM
        self.last_flow_stats = {} # flow_id -> (tx, rx, lost, delay_sum)

    def get_last_interval_stats(self):
        """
        Get link statistics for the last interval.
        Returns:
            stats: dict[ue_id][link_id] -> {
                'acks': int,
                'losses': int,
                'rtt_sum': float (seconds),
                'tx_bytes': int
            }
        """
        # stats_res[ue_id][uav_id] -> dict
        stats_res = collections.defaultdict(lambda: collections.defaultdict(lambda: {'acks': 0, 'losses': 0, 'rtt_sum': 0.0, 'tx_bytes': 0}))
        
        if not hasattr(self, 'flow_monitor'):
            return stats_res

        self.flow_monitor.CheckForLostPackets()
        monitor_stats = self.flow_monitor.GetFlowStats()
        classifier = self.flow_helper.GetClassifier()

        for flow_id, flow_stats in monitor_stats:
            # Get 5-tuple
            t = classifier.FindFlow(flow_id)
            src_addr = t.sourceAddress
            
            # Check if source is a UE (10.ue.uav.1)
            # Address format: 10.{ue_id+1}.{uav_id+1}.1
            try:
                # Convert Ipv4Address to string "10.1.1.1"
                ip_str = str(src_addr) 
                parts = ip_str.split('.')
                if len(parts) != 4 or parts[0] != '10':
                    continue
                
                ue_id = int(parts[1]) - 1
                uav_id = int(parts[2]) - 1
                
                if ue_id < 0 or ue_id >= self.num_ues or uav_id < 0 or uav_id >= self.num_uavs:
                    continue

                # Calculate deltas
                prev = self.last_flow_stats.get(flow_id, {'tx': 0, 'rx': 0, 'lost': 0, 'delay': 0.0, 'bytes': 0})
                
                curr_tx = flow_stats.txPackets
                curr_rx = flow_stats.rxPackets
                curr_lost = flow_stats.lostPackets
                curr_delay = flow_stats.delaySum.GetSeconds()
                curr_tx_bytes = flow_stats.txBytes

                delta_rx = max(0, curr_rx - prev['rx']) # ACKs (approx)
                delta_lost = max(0, curr_lost - prev['lost']) # Timeouts/Losses
                delta_delay = max(0.0, curr_delay - prev['delay']) # Sum RTT
                delta_bytes = max(0, curr_tx_bytes - prev['bytes'])

                # Update history
                self.last_flow_stats[flow_id] = {
                    'tx': curr_tx, 
                    'rx': curr_rx, 
                    'lost': curr_lost, 
                    'delay': curr_delay,
                    'bytes': curr_tx_bytes
                }

                if delta_rx > 0 or delta_lost > 0:
                    stats_res[ue_id][uav_id]['acks'] += delta_rx
                    stats_res[ue_id][uav_id]['losses'] += delta_lost
                    stats_res[ue_id][uav_id]['rtt_sum'] += delta_delay
                    stats_res[ue_id][uav_id]['tx_bytes'] += delta_bytes
            
            except Exception:
                continue

        return stats_res

    def setup_nodes(self):
        """创建节点"""
        self.ue_nodes = ns.NodeContainer()
        self.ue_nodes.Create(self.num_ues)
        self.uav_nodes = ns.NodeContainer()
        self.uav_nodes.Create(self.num_uavs)
        self.sat_nodes = ns.NodeContainer()
        self.sat_nodes.Create(1)
        print(f"[节点] 创建 {self.num_ues} UE + {self.num_uavs} UAV + 1 SAT")

    def setup_mobility(self):
        """设置移动模型"""

        num_ues = self.num_ues
        area_size = self.config.area_size
        z = 0.0  # 地面高度

        grid_n = int(math.ceil(num_ues ** 0.5))
        spacing = area_size / (grid_n + 1)

        # 初始化位置容器
        ue_pos_alloc = ns.ListPositionAllocator()  # 根据ns-3实际API调整
        self.ue_positions = np.zeros((num_ues, 3))  # 创建 (num_ues, 3) 的数组

        ue_idx = 0
        for i in range(grid_n):
            for j in range(grid_n):
                if ue_idx >= num_ues:
                    break
                x = spacing * (i + 1)
                y = spacing * (j + 1)
                ue_pos_alloc.Add(ns.Vector(x, y, z))
                self.ue_positions[ue_idx] = [x, y, z]  # 使用ue_idx作为索引
                ue_idx += 1


        # ue_pos_alloc = ns.ListPositionAllocator()
        # for i, pos in enumerate(self.ue_positions):
        #     x, y, z = pos[0], pos[1], pos[2]
        #     ue_pos_alloc.Add(ns.Vector(x, y, z))

        ue_mobility = ns.MobilityHelper()
        ue_mobility.SetPositionAllocator(ue_pos_alloc)
        ue_mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
        ue_mobility.Install(self.ue_nodes)

        # UAV位置
        uav_pos_alloc = ns.ListPositionAllocator()
        center = self.config.area_size / 2

        for i in range(self.num_uavs):
            # angle = 2 * math.pi * i / self.num_uavs
            angle = self.config.uav_angles[i]
            x = center + self.config.uav_radius * math.cos(angle)
            y = center + self.config.uav_radius * math.sin(angle)
            z = self.config.uav_height
            uav_pos_alloc.Add(ns.Vector(x, y, z))
            self.uav_positions[i] = (x, y, z)
            print(f"[初始化] UAV{i} 位置: ({x:.1f}, {y:.1f}, {z:.1f})")

        uav_mobility = ns.MobilityHelper()
        uav_mobility.SetPositionAllocator(uav_pos_alloc)
        uav_mobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel")
        uav_mobility.Install(self.uav_nodes)

        for i in range(self.num_uavs):
            # angle = 2 * math.pi * i / self.num_uavs + math.pi/2
            angle = self.config.uav_angles[i] + math.pi/2
            speed = self.config.uav_speed
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            vz = 0
            uav_node = self.uav_nodes.Get(i)
            mob = uav_node.GetObject[ns.MobilityModel]()
            mob.SetVelocity(ns.Vector(vx, vy, vz))

        print("[移动] 模型配置完成")

    def setup_network(self):
        """设置网络"""
        # Configure default SocketType based on config
        selected_cc = self.selected_cc_config.ns3_class
        print(f"[Network] Setting Default Congestion Control to: {selected_cc} (config.cc={self.config.cc})")
        ns.Config.SetDefault("ns3::QuicL4Protocol::SocketType", ns.TypeIdValue(ns.TypeId.LookupByName(selected_cc)))
        # Ensure Multipath is enabled on sockets by default
        ns.Config.SetDefault("ns3::QuicSocketBase::EnableMultipath", ns.BooleanValue(True))

        # 安装协议栈
        quic_helper = ns.QuicHelper()
        internet_stack = ns.InternetStackHelper()
        # 为每个节点安装协议栈
        for i in range(self.num_ues):
            quic_helper.InstallQuic(self.ue_nodes.Get(i))
        for i in range(self.num_uavs):
            internet_stack.Install(self.uav_nodes.Get(i))
        quic_helper.InstallQuic(self.sat_nodes.Get(0))
        # 创建点对点链路
        # UE ↔ UAV 链路（多路径）
        p2p_ue_uav = ns.PointToPointHelper()
        p2p_ue_uav.SetDeviceAttribute("DataRate", ns.StringValue("10Mbps")) 
        p2p_ue_uav.SetChannelAttribute("Delay", ns.StringValue("5ms"))
        p2p_ue_uav.SetDeviceAttribute("Mtu", ns.UintegerValue(1400))
        self.p2p_ue_uav = p2p_ue_uav
        # UAV ↔ SAT 链路
        p2p_uav_sat = ns.PointToPointHelper()
        p2p_uav_sat.SetDeviceAttribute("DataRate", ns.StringValue("100Mbps")) 
        p2p_uav_sat.SetChannelAttribute("Delay", ns.StringValue("50ms"))   
        p2p_uav_sat.SetDeviceAttribute("Mtu", ns.UintegerValue(1400))
        ipv4 = ns.Ipv4AddressHelper()
        self.em.SetAttribute("ErrorRate", ns.DoubleValue(self.config.error_rate))
        # --- A. UE <-> UAV Links (Full Mesh for dynamic assignment) ---
        for ue_idx in range(self.num_ues):
            for uav_idx in range(self.num_uavs):
                ue_node = self.ue_nodes.Get(ue_idx)
                uav_node = self.uav_nodes.Get(uav_idx)
                devices = p2p_ue_uav.Install(ue_node, uav_node)
                self.ue_uav_devices[ue_idx][uav_idx] = devices
                base_ip = f"10.{ue_idx+1}.{uav_idx+1}.0"
                ipv4.SetBase(ns.Ipv4Address(base_ip), ns.Ipv4Mask("255.255.255.0"))
                interfaces = ipv4.Assign(devices)
                self.ue_uav_interfaces[ue_idx][uav_idx] = interfaces
                devices.Get(1).SetAttribute("ReceiveErrorModel", ns.PointerValue(self.em))
        # --- B. UAV <-> SAT Links ---
        for uav_idx in range(self.num_uavs):
            uav_node = self.uav_nodes.Get(uav_idx)
            sat_node = self.sat_nodes.Get(0)
            devices = p2p_uav_sat.Install(uav_node, sat_node)
            self.uav_sat_devices[uav_idx] = devices
            base_ip = f"10.{100+uav_idx+1}.1.0"
            ipv4.SetBase(ns.Ipv4Address(base_ip), ns.Ipv4Mask("255.255.255.0"))
            interfaces = ipv4.Assign(devices)
            self.uav_sat_interfaces[uav_idx] = interfaces
            devices.Get(1).SetAttribute("ReceiveErrorModel", ns.PointerValue(self.em))
        ns.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

        # 连接乱序统计 Monitor (C++ Implementation)
        sat_node = self.sat_nodes.Get(0)
        
        try:
            path_rx = f"/NodeList/{sat_node.GetId()}/$ns3::Ipv4L3Protocol/Rx"
            ns.cppyy.gbl.ConnectOooRx(path_rx, self.ooo_monitor)
        except Exception as e:
             print(f"Warn: Failed to connect Rx trace: {e}")

        try:
            path_ld = f"/NodeList/{sat_node.GetId()}/$ns3::Ipv4L3Protocol/LocalDeliver"
            ns.cppyy.gbl.ConnectOooLocalDeliver(path_ld, self.ooo_monitor)
        except Exception as e:
             print(f"Warn: Failed to connect LocalDeliver trace: {e}")

        self.assignment = self.G2A_scheduler.get_default_assignment(self.ue_positions, self.uav_positions)
        self.G2A_gain = self.G2A_channel.estimate_chan_gain(self.ue_positions, self.uav_positions)

        # Initial Link Setup:
        # Calculate rates and apply assignments immediately so that at t=1.0 when Apps start,
        # the links are already configured correctly (ErrorRate low for active links).
        # Otherwise, they stay at 100% loss until scheduler runs.
        self.G2A_rates = self.G2A_scheduler.get_rates(self.ue_positions, self.uav_positions, 
                                              self.assignment, self.G2A_gain, 0, self.config.tx_power)
        # rates = self.G2A_rates.copy()
        # non_zero_mask = rates != 0
        # non_zero_elements = rates[non_zero_mask]
        # mean_value = non_zero_elements.mean()
        # rates[~non_zero_mask] = mean_value

        # distances = np.sqrt(np.sum((self.uav_positions - self.ue_positions) ** 2))
        # rates = self.
        # delay_seconds = 2*self._calculate_delay_from_distance(distances)
        # Config::SetDefault("ns3::RateBasedCongestionOps::AssignedBdp", ns.UintegerValue(your_bdp))
        # Config::SetDefault("ns3::RateBasedCongestionOps::AssignedCwnd", ns.UintegerValue(your_bdp))
        
        # # Optional pcap capture for debugging (enable with DEBUG_PCAP=1)
        # if self.debug_pcap:
        #     try:
        #         # Capture both directions of the first UE-UAV link to confirm packet emission
        #         self.p2p_ue_uav.EnablePcap("debug-ue0-uav0-ue-side", self.ue_uav_devices[0][0].Get(0), False, True)
        #         self.p2p_ue_uav.EnablePcap("debug-ue0-uav0-uav-side", self.ue_uav_devices[0][0].Get(1), False, True)
        #     except Exception as e:
        #         print(f"[PCAP] enable failed: {e}")

        # print("[Network] MPQUIC Topology configured.")

    def setup_applications(self):
        # 1. Server (Sink) on Satellites
        # All satellites listen on port
        port = self.port
        addr =  ns.InetSocketAddress(ns.Ipv4Address.GetAny(), port)
        # QuicSocketFactory is required for MPQUIC
        sink = ns.PacketSinkHelper("ns3::QuicSocketFactory", addr.ConvertTo())
        apps = sink.Install(self.sat_nodes.Get(0))
        apps.Start(ns.Seconds(0.0))
        apps.Stop(ns.Seconds(self.config.simulation_time))

        # Use the IP of the Satellite (on the link to UAV 0) as the stable destination address.
        # GlobalRouting will ensure packets routed via other UAVs still reach the Satellite node.
        fixed_sat_addr = self.uav_sat_interfaces[0].GetAddress(1)

        for ue_i in range(self.num_ues):
            addr = ns.InetSocketAddress(fixed_sat_addr, port)
            source = ns.MpquicBulkSendHelper("ns3::QuicSocketFactory", addr.ConvertTo())
            # Set MaxBytes to a finite but large value (500MB) to prevent stalling
            # MaxBytes=0 (Unlimited) causes stalling with MPQUIC in this setup
            source.SetAttribute("MaxBytes", ns.UintegerValue(0))  # 500 * 1024 * 1024
            source.SetAttribute("SendSize", ns.UintegerValue(self.config.packet_size))
            # Bind the app to the UE node
            apps = source.Install(self.ue_nodes.Get(ue_i))
            self.ue_apps.append(apps)
            # 保存 application 对象关联的 socket 以便后续读取发送缓冲区信息
            try:
                app_obj = apps.Get(0)
                sock = app_obj.GetSocket()
            except Exception:
                sock = None
            self.ue_sockets.append(sock)
            apps.Start(ns.Seconds(1.0))
            # Keep the sender alive for the full simulation (short runs would otherwise stop immediately)
            apps.Stop(ns.Seconds(self.config.simulation_time))
        
        self.update_G2A_Connection()
        print("[应用] 配置完成")

    def update_ue_send_bufs(self):
        """读取每个 UE 的 Quic socket 的发送缓冲可用字节并归一化到 [0,1]。"""
        # 首先尝试使用新的 GetAllSocketTxAvailable 方法
        all_tx_avail_per_ue = {}
        for i in range(self.num_ues):
            ue_node = self.ue_nodes.Get(i)
            quic_obj = ue_node.GetObject[ns.QuicL4Protocol]()
            tx_avail = quic_obj.GetAllSocketTxAvailable()
            all_tx_avail_per_ue[i] = tx_avail

        for i in range(self.num_ues):
            sock = self.ue_sockets[i]

            # 如果之前未能拿到 socket，尝试从 application 在运行时获取它（app 在 t=1.0 后创建 socket）
            if sock is None:
                apps = self.ue_apps[i]
                app_obj = apps.Get(0)
                sock_try = app_obj.GetSocket()
                if sock_try is not None:
                    sock = sock_try
                    self.ue_sockets[i] = sock_try

            # 如果仍然没有 socket，尝试通过节点的 QuicL4Protocol 的绑定列表遍历查找 QuicSocket
            if sock is None:
                ue_node = self.ue_nodes.Get(i)
                # 获取 QuicL4Protocol 对象
                quic_obj = ue_node.GetObject[ns.QuicL4Protocol]()
                # 尝试访问内部的绑定列表属性（m_quicUdpBindingList）
                bindings = quic_obj.m_quicUdpBindingList

                if bindings:
                    # Print binding list summary for debugging
                    for idx, b in enumerate(bindings):
                        candidate = b.m_quicSocket
                        if candidate is not None:
                            sock = candidate
                            self.ue_sockets[i] = candidate
                            break

            # 尝试使用 GetAllSocketTxAvailable 的结果
            if i in all_tx_avail_per_ue and len(all_tx_avail_per_ue[i]) > 0:
                # 假设每个 UE 只有一个套接字，取第一个值
                remain = all_tx_avail_per_ue[i][0]
                # 假设最大缓冲区大小为 128KB（与默认值匹配）
                max_sz = 128 * 1024
                if max_sz > 0:
                    self.ue_send_buf_remain[i] = float(remain) / float(max_sz)
                    print(f"[DEBUG] UE{i}: using GetAllSocketTxAvailable, remain={remain}, normalized={self.ue_send_buf_remain[i]}")
                    continue

            if sock is None:
                self.ue_send_buf_remain[i] = 1.0
                continue

            # GetTxAvailable 返回可继续写入的字节数（剩余容量）
            remain = sock.GetTxAvailable()
            # GetSocketSndBufSize 返回 socket 的最大发送缓冲尺寸
            max_sz = sock.GetSocketSndBufSize()
            if max_sz > 0:
                self.ue_send_buf_remain[i] = float(remain) / float(max_sz)
            else:
                self.ue_send_buf_remain[i] = 1.0

    def setup_flow_monitor(self):
        """设置流监控"""
        self.flow_helper = ns.FlowMonitorHelper()
        self.flow_monitor = self.flow_helper.InstallAll()

    def setup_all(self):
        self.init_states()
        self.setup_nodes()
        self.setup_mobility()
        self.setup_network()
        self.setup_applications()
        self.setup_flow_monitor()

    def set_uav_positions(self):
        current_time = ns.Simulator.Now().GetSeconds()
        center = self.config.area_size / 2
        
        for i in range(self.num_uavs):
            # 计算当前角度（匀速圆周运动）
            angular_velocity = self.config.uav_speed / self.config.uav_radius  # 角速度 ω = v/r
            angle = self.config.uav_angles[i] + angular_velocity * current_time
            
            # 计算新位置
            x = center + self.config.uav_radius * math.cos(angle)
            y = center + self.config.uav_radius * math.sin(angle)
            z = self.config.uav_height
            
            # 直接设置新位置
            uav_node = self.uav_nodes.Get(i)
            mob = uav_node.GetObject[ns.MobilityModel]()
            mob.SetPosition(ns.Vector(x, y, z))
            
            # 计算并设置当前切向速度（正确的圆周运动速度方向）
            # 速度方向垂直于半径方向，大小为uav_speed
            vx = -self.config.uav_speed * math.sin(angle)  # v_x = -v * sin(θ)
            vy = self.config.uav_speed * math.cos(angle)   # v_y = v * cos(θ)
            vz = 0
            mob.SetVelocity(ns.Vector(vx, vy, vz))
            self.uav_positions[i] = [x, y, z]
            # """计算UAV位置（圆周运动）"""
            # center = self.config.area_size / 2
            # period = 2 * math.pi * self.config.uav_radius / self.config.uav_speed
            # angle = self.initial_angle[uav_idx] + (2 * math.pi * t / period)

            # x = center + self.config.uav_radius * math.cos(angle)
            # y = center + self.config.uav_radius * math.sin(angle)
            # z = self.config.uav_height
        # for uav_idx in range(self.num_uavs):
            # mobility = self.uav_nodes.Get(uav_idx).GetObject[ns.MobilityModel]()
            # pos = mobility.GetPosition()
            # self.uav_positions[uav_idx] = [pos.x, pos.y, pos.z]


    def _calculate_delay_from_distance(self, distance):
        """
        根据距离计算传输时延
        :param distance: UE到UAV的距离（米）
        :return: 传输时延（秒）
        """
        speed_of_light = 3e8  # 光速 (m/s)
        # 考虑光纤/电缆的折射率（约1.5），实际信号传播速度约为光速的2/3
        propagation_speed = speed_of_light / 1.5
        delay_seconds = distance / propagation_speed * self.config.d_loss_factor
        return delay_seconds

    def _update_link_rate(self, device, rate_bps, distance, interface_pair=None, ue_id=None):
        """
        辅助函数：更新P2P链路速率
        :param device: ns.NetDevice 对象 (PointToPointNetDevice)
        :param rate_bps: 目标速率 (bps)，如果为0则设置为极低速率(1bps)模拟断开
        :param distance: 距离
        :param ue_flag: 是否是UE侧的设备
        :param ue_id: UE编号
        """
        # 将通用 NetDevice 转换为 PointToPointNetDevice
        p2p_dev = device.GetObject[ns.PointToPointNetDevice]()
        
        if rate_bps <= 0:
            if ue_id is not None:
                # app = self.ue_apps[ue_id].Get(0)
                # app.SetAttribute("MaxBytes", ns.UintegerValue(1)) 
                # app.SetAttribute("SendSize", ns.UintegerValue(1))
                # ipv4 = interface_pair.first
                # idx = interface_pair.second
                # ipv4.SetDown(idx)
                pass
            else:
                # pass
                self.em.SetAttribute("ErrorRate", ns.DoubleValue(1))
                p2p_dev.SetAttribute("ReceiveErrorModel", ns.PointerValue(self.em))
        else:
            if ue_id is not None:
                # app = self.ue_apps[ue_id].Get(0)
                # app.SetAttribute("MaxBytes", ns.UintegerValue(0))
                # app.SetAttribute("SendSize", ns.UintegerValue(self.config.packet_size))
                # ipv4 = interface_pair.first
                # idx = interface_pair.second
                # ipv4.SetUp(idx)
                pass
            else:
                # pass
                self.em.SetAttribute("ErrorRate", ns.DoubleValue(self.config.error_rate))
                p2p_dev.SetAttribute("ReceiveErrorModel", ns.PointerValue(self.em))
            
            # 根据距离动态计算时延
            delay_seconds = self._calculate_delay_from_distance(distance)
            delay = ns.Seconds(delay_seconds)
            rate = f"{int(rate_bps)}bps"
            p2p_dev.SetDataRate(ns.DataRate(rate))
            p2p_dev.GetChannel().SetAttribute("Delay", ns.TimeValue(delay)) 

    def update_G2A_Connection(self):
        current_time = ns.Simulator.Now().GetSeconds()
        time_slot = int(current_time / self.config.time_slot_duration)
        self.G2A_gain = self.G2A_channel.estimate_chan_gain(self.ue_positions, self.uav_positions)
        self.G2A_rates = self.G2A_scheduler.get_rates(self.ue_positions, self.uav_positions, 
                                              self.assignment, self.G2A_gain, time_slot, self.config.tx_power)
        self.G2A_sum_rates += self.G2A_rates
        # 优化：仅更新状态改变的链路，避免频繁 Down/Up 导致传输中断
        for ue_id in range(self.num_ues):
            for uav_id in range(self.num_uavs):
                devices = self.ue_uav_devices[ue_id][uav_id]
                ue_pos = self.ue_positions[ue_id]
                uav_pos = self.uav_positions[uav_id]
                # 计算欧几里得距离
                distance = np.sqrt(np.sum((uav_pos - ue_pos) ** 2))
                is_active = self.assignment[uav_id][ue_id]
                interfaces = self.ue_uav_interfaces[ue_id][uav_id]

                if is_active:
                    rate = self.G2A_rates[ue_id]
                    self._update_link_rate(devices.Get(0), rate, distance, interfaces.Get(0), ue_id)
                    self._update_link_rate(devices.Get(1), rate, distance, interfaces.Get(1))
                    node_id = self.ue_nodes.Get(int(ue_id)).GetId()
                    if self.config.cc == 'RateB':
                        # Check for handover and update EMA
                        last_uav = self.ue_last_connected_uav[ue_id]
                        if last_uav != -1 and last_uav != uav_id:
                            # Handover occurred: store last rate
                            prev_rate = self.ue_last_rate[ue_id]
                            self.ue_rate_queue[ue_id].append(prev_rate)
                            
                            # Update EMA using values in the queue
                            queue_len = len(self.ue_rate_queue[ue_id])
                            current_avg = sum(self.ue_rate_queue[ue_id]) / queue_len
                            
                            # Apply EMA smoothing to the average
                            if self.ue_rate_ema[ue_id] == 0:
                                self.ue_rate_ema[ue_id] = current_avg
                            else:
                                self.ue_rate_ema[ue_id] = self.config.alpha * current_avg + (1 - self.config.alpha) * self.ue_rate_ema[ue_id]

                        # Use EMA for BDP calculation if available, otherwise use current rate
                        rate_for_bdp = self.ue_rate_ema[ue_id] if self.ue_rate_ema[ue_id] > 0 else rate

                        # Update state for next check
                        self.ue_last_connected_uav[ue_id] = uav_id
                        self.ue_last_rate[ue_id] = rate

                        # 路径: /NodeList/{node_id}/$ns3::QuicL4Protocol/SocketList/*/CongestionControl/$ns3::RateBasedCongestionOps/AssignedRate
                        config_path = f"/NodeList/{node_id}/$ns3::QuicL4Protocol/SocketList/*/QuicSocketBase/CongestionControl/$ns3::RateBasedCongestionOps/AssignedRate"
                        ns.Config.Set(config_path, ns.DataRateValue(ns.DataRate(f"{int(rate)}bps")))

                        # Calculate BDP and set AssignedBdp (SST) and AssignedCwnd
                        delay_seconds = self._calculate_delay_from_distance(distance)
                        rtt_estimated = delay_seconds * 2
                        bdp = int(rate_for_bdp * rtt_estimated / 8 * self.config.b_scale) 
                        
                        # === Congestion Avoidance Logic ===
                        current_time = ns.Simulator.Now().GetSeconds()
                        
                        # 1. Packet Loss Detection (Simplified Global Heuristic)
                        loss_detected = False
                        
                        # Update stats cache once per second or step to avoid overhead
                        if hasattr(self, 'flow_monitor'):
                            if not hasattr(self, '_current_flow_stats_cache_time') or self._current_flow_stats_cache_time != current_time:
                                self.flow_monitor.CheckForLostPackets()
                                self._current_flow_stats_cache = self.flow_monitor.GetFlowStats()
                                self._current_flow_stats_cache_time = current_time
                            
                            # Heuristic: Flow ID = UE ID + 1
                            flow_id_guess = ue_id + 1
                            current_lost = 0
                            
                            # Find flow stats
                            # Note: _current_flow_stats_cache is a list of (flowId, FlowStats)
                            # We can optimize this by converting to dict once, but list is short.
                            for fid, fstats in self._current_flow_stats_cache:
                                if fid == flow_id_guess:
                                    current_lost = fstats.lostPackets
                                    break
                            
                            # Check for new loss
                            if current_lost > self.ue_last_loss_count.get(ue_id, 0):
                                loss_detected = True
                                self.ue_last_loss_count[ue_id] = current_lost
                        else:
                            # Flow monitor not initialized (e.g., during setup)
                            loss_detected = False

                        # 2. Get/Init State
                        current_cwnd = self.ue_ca_cwnd[ue_id]
                        if current_cwnd == 0: 
                            current_cwnd = bdp # Initialize to BDP if 0
                        
                        # Variables
                        tp_t = rate # Current Rate (bps)
                        c_t_1 = self.ue_last_rate[ue_id] # Previous Rate
                        tau_t = rtt_estimated # Current RTT
                        d_t_1 = self.ue_last_rtt[ue_id] # Previous RTT
                        dist_t_1 = distance # Current Distance
                        dist_t_2, elevation = self.sat_mobility.get_distance_and_elevation(ns.Simulator.Now().GetSeconds())
                        # Update RTT History
                        self.ue_last_rtt[ue_id] = tau_t

                        # 3. Evaluation Conditions
                        con1 = tp_t < c_t_1
                        con2 = tau_t > d_t_1 + self.config.ca_delta_r
                        con3 = dist_t_1 < self.config.ca_d_th_1 or dist_t_2 < self.config.ca_d_th_2
                        
                        new_cwnd = current_cwnd
                        
                        if loss_detected:
                            if con1 and con2:
                                # Congestion Loss: Multiplicative Decrease
                                new_cwnd = current_cwnd / 2.0
                            elif con3:
                                # Wireless Loss (Close distance/Interference): Reduce by Gamma
                                new_cwnd = current_cwnd * self.config.ca_gamma
                            else:
                                # Fallback for other loss (e.g. random error, far distance)
                                # Default to strictly less aggressive than congestion loss
                                new_cwnd = current_cwnd * 0.8
                        else:
                            # Congestion Avoidance: Rate-Based Increase
                            # Calculate Lambda
                            # Gain approx 1/dist^2 if data unavailable, but we have self.G2A_gain?
                            # config.py doesn't show G2A_gain, likely internal or calculated.
                            # Fallback to Friis-like approx
                            gain_linear = 1.0 / (dist_t_1**2) if dist_t_1 > 0 else 1.0
                            snr_linear = gain_linear * self.config.tx_power / self.config.N
                            snr_db = 10 * np.log10(max(snr_linear, 1e-9))
                            
                            rho = 0.0 # Packet loss rate assumed low in CA
                            lambda_val = self.config.ca_sigma * (snr_db / self.config.ca_snr_max) + \
                                         (1 - self.config.ca_sigma) * (1 - rho)
                            
                            # Global Summation Term
                            sum_term = 0.0
                            max_tau_history = 0.0 
                            
                            # Iterate all UEs assigned to this UAV
                            for k in range(self.num_ues):
                                if self.assignment[uav_id][k]:
                                    w_k = self.ue_ca_cwnd[k] if self.ue_ca_cwnd[k] > 0 else bdp
                                    
                                    # Calc tau_k
                                    k_uav_dist = np.linalg.norm(self.uav_positions[uav_id] - self.ue_positions[k])
                                    tau_k = 2 * self._calculate_delay_from_distance(k_uav_dist)
                                    
                                    if tau_k > 0:
                                        sum_term += w_k / tau_k
                                        if tau_k > max_tau_history:
                                            max_tau_history = tau_k
                                            
                            if sum_term > 0 and tau_t > 0:
                                # Increment Formula
                                # Ensure max_term is defined. Using T_nm = max_tau_history
                                increment = (3 * lambda_val * math.sqrt(max_tau_history)) / (2 * tau_t * sum_term)
                                # Apply increment (assuming additive to Window)
                                new_cwnd = current_cwnd + increment
                            else:
                                new_cwnd = current_cwnd # No increment
                        
                        # Apply Limits
                        if new_cwnd < 2000: new_cwnd = 2000 # Min window
                        
                        # Save State
                        self.ue_ca_cwnd[ue_id] = new_cwnd
                        
                        # Apply to NS-3 Config
                        config_path_bdp = f"/NodeList/{node_id}/$ns3::QuicL4Protocol/SocketList/*/QuicSocketBase/CongestionControl/$ns3::RateBasedCongestionOps/AssignedBdp"
                        ns.Config.Set(config_path_bdp, ns.UintegerValue(int(bdp)))

                        config_path_cwnd = f"/NodeList/{node_id}/$ns3::QuicL4Protocol/SocketList/*/QuicSocketBase/CongestionControl/$ns3::RateBasedCongestionOps/AssignedCwnd"
                        ns.Config.Set(config_path_cwnd, ns.UintegerValue(int(new_cwnd)))
                                           
                    config_path_select = f"/NodeList/{node_id}/$ns3::QuicL4Protocol/SocketList/*/QuicSocketBase/Scheduler/Select"
                    ns.Config.Set(config_path_select, ns.UintegerValue(int(uav_id)))    
                else:
                    self._update_link_rate(devices.Get(0), 0, distance, interfaces.Get(0), ue_id)
                    self._update_link_rate(devices.Get(1), 0, distance, interfaces.Get(1))
                    
    
    def update_A2S_Connection(self):
        distance, elevation = self.sat_mobility.get_distance_and_elevation(ns.Simulator.Now().GetSeconds())
        p_t = np.full(self.num_uavs, self.config.tx_power)
        bw = np.full(self.num_uavs, self.config.A2S_bandwidth)
        self.A2S_rates = self.A2S_channel.calculate_transmission_rate(p_t, bw, distance, elevation) # shape (num_uavs,)
        self.A2S_sum_rates += self.A2S_rates
        
        # DEBUG: Print rates occasionally
        # if int(ns.Simulator.Now().GetSeconds()) % 5 == 0:
        #      print(f"[DEBUG] T={ns.Simulator.Now().GetSeconds():.1f}")
        #      print(f"  G2A Rates (UE0-UE{min(3, self.num_ues)-1}): {self.G2A_rates[:3]}")
        #      print(f"  A2S Rates (UAV0-UAV{min(3, self.num_uavs)-1}): {A2S_rates[:3]}")
        #      # Check what SELECT is set to for the first UE
        #      try:
        #          # Check config value for first socket of first UE
        #          # Note: This path might be tricky to get right without iterating, but let's try to verify assignment logic
        #          uav_indices, ue_indices = np.where(self.assignment)
        #          for i in range(min(3, len(ue_indices))):
        #              ue_id_check = ue_indices[i]
        #              uav_id_check = uav_indices[i]
        #              print(f"  Assignment Check: UE{ue_id_check} -> UAV{uav_id_check}")
        #      except Exception as e:
        #          print(f"  Debug check failed: {e}")

        for uav_id in range(self.num_uavs):
            rate = self.A2S_rates[uav_id]
            dev_uav_side = self.uav_sat_devices[uav_id].Get(0)
            dev_sat_side = self.uav_sat_devices[uav_id].Get(1)
            self._update_link_rate(dev_uav_side, rate, distance)
            self._update_link_rate(dev_sat_side, rate, distance)

        # 更新每个 UE 的发送缓冲剩余（归一化）以供 DRL 状态使用
        try:
            self.update_ue_send_bufs()
        except Exception as e:
            # 保守容错：若读取失败，不中断仿真
            # print(f"Warn: update_ue_send_bufs failed: {e}")
            pass
        # 更新每个 UAV -> SAT 链路的队列占用（bytes / max_bytes -> [0,1]）
        for uav_id in range(self.num_uavs):
            try:
                dev = self.uav_sat_devices[uav_id].Get(0)
                p2p_dev = dev.GetObject[ns.PointToPointNetDevice]()
                q = p2p_dev.GetQueue()
                bytes_in_queue = q.GetNBytes()
                # 尝试读取队列最大尺寸（单位可能为 bytes 或 packets）
                try:
                    max_q_raw = float(q.GetMaxSize().GetValue())
                except Exception:
                    # 退回到一个保守常量（128KB）
                    max_q_raw = 128 * 1024

                # 如果 max_q_raw 看起来较小（例如 <= 100000），很可能是以“包数”为单位。
                # 在这种情况下，用设备 MTU 估算字节数；否则直接认为是字节数。
                try:
                    if max_q_raw <= 100000:
                        try:
                            mtu = float(p2p_dev.GetMtu())
                        except Exception:
                            mtu = 1400.0
                        max_q_bytes = max_q_raw * mtu
                    else:
                        max_q_bytes = max_q_raw
                except Exception:
                    max_q_bytes = max_q_raw

                if max_q_bytes > 0:
                    occ = float(bytes_in_queue) / float(max_q_bytes)
                else:
                    occ = 0.0
                # 截断到 [0,1]
                self.A2S_queue_occupancy[uav_id] = min(max(occ, 0.0), 1.0)
            except Exception:
                # 读取失败则置为 0（空）
                self.A2S_queue_occupancy[uav_id] = 0.0

        # 更新噪声状态（用于自预测鲁棒性测试）
        try:
            self.update_noise_state()
        except Exception:
            pass

        self.flow_monitor.CheckForLostPackets()
        stats = self.flow_monitor.GetFlowStats()
        current_total_tx_bytes = 0
        current_total_rx_bytes = 0
        for flow_id, flow_stats in stats:
            current_total_tx_bytes += flow_stats.txBytes
            current_total_rx_bytes += flow_stats.rxBytes
        delta_tx_bytes = current_total_tx_bytes - self.last_total_tx_bytes
        delta_rx_bytes = current_total_rx_bytes - self.last_total_rx_bytes
        self.last_total_tx_bytes = current_total_tx_bytes
        self.last_total_rx_bytes = current_total_rx_bytes
        self.trans_per_step = delta_tx_bytes
        self.recei_per_step = delta_rx_bytes
        

    def mask_assignment(self, assignment):
        num_ues = self.num_ues
        rbs = self.n_rbs   
        """
        将action动作转换为assignment矩阵，并确保每个UAV最多连接rbs个UE
        
        Args:
            action: 形状为(num_ues,)的数组，表示每个UE连接的UAV编号
            num_uavs: UAV数量
            num_ues: UE数量
            rbs: 每个UAV最大连接数
        
        Returns:
            assignment: 形状为(num_uavs, num_ues)的bool矩阵
        """
        # # 初始化assignment矩阵
        # assignment = np.zeros((num_uavs, num_ues), dtype=bool)
        
        # # 创建UE索引数组
        # ue_indices = np.arange(num_ues)
        
        # # 将action转换为one-hot编码（使用数组广播）
        # # 这一步将每个UE与其对应的UAV建立连接
        # assignment[action, ue_indices] = True
        
        # 检查每个UAV的连接数
        uav_connections = np.sum(assignment, axis=1)  # 每个UAV的连接数
        
        # 找出连接数超过rbs的UAV
        overloaded_uavs = np.where(uav_connections > rbs)[0]
        
        # 对于每个超载的UAV，随机选择rbs个UE，屏蔽其他UE
        for uav in overloaded_uavs:
            # 获取该UAV当前连接的所有UE索引
            connected_ues = np.where(assignment[uav])[0]
            
            # 随机选择要保留的rbs个UE
            keep_ues = np.random.choice(connected_ues, size=rbs, replace=False)
            
            # 创建掩码：保留选择的UE，屏蔽其他UE
            mask = np.zeros(num_ues, dtype=bool)
            mask[keep_ues] = True
            
            # 更新该UAV的assignment行
            assignment[uav] = assignment[uav] & mask
        return assignment

    def do_scheduling(self):
        current_time = ns.Simulator.Now().GetSeconds()
        """执行调度"""
        self.assignment = self.G2A_scheduler.scheduler(self.ue_positions, self.uav_positions)
        self.assignment = self.mask_assignment(self.assignment)
        self.update_G2A_Connection()
        self.update_A2S_Connection()
        ns.Ipv4GlobalRoutingHelper.RecomputeRoutingTables()
        self.set_uav_positions()
        # print("uav positions:", self.uav_positions)
        # 每5秒输出
        # if time_slot % 50 == 0:
        #     print(f"\n=== 时隙 {time_slot} (t={current_time:.1f}s) ===")
        #     total_capacity = 0
        #     uav_indices, ue_indices = np.where(self.assignment)
        #     for uav_id in uav_indices:
        #         ue_list = ue_indices[uav_indices == uav_id]
        #         if len(ue_list) > 0:
        #             rates = self.G2A_scheduler.history[-1]['rates']
        #             ue_info = []
        #             for ue in ue_list:
        #                 rate = rates[ue]
        #                 cap = rate / 1e6
        #                 total_capacity += cap
        #                 ue_info.append(f"UE{ue}({rate:.1f}bps)")
        #             print(f"  UAV{uav_id}: {', '.join(ue_info)}")
        #     print(f"  总系统容量: {total_capacity:.1f} Mbps")

        print("step:", current_time, "trans_per_step:", self.trans_per_step)

        # 下一次调度
        if current_time + self.config.time_slot_duration < self.config.simulation_time:
            event = ns.cppyy.gbl.pythonMakeEvent(scheduling_callback, sys.argv)
            ns.Simulator.Schedule(ns.Seconds(self.config.time_slot_duration), event)
            
    
    def do_scheduling_action(self, action):
        """执行调度"""
        # self.assignment = np.zeros((self.num_uavs, self.num_ues), dtype=bool)
        # self.assignment[action, np.arange(self.num_ues)] = True
        self.assignment = np.zeros((self.num_uavs, self.num_ues), dtype=bool)
        ue_indices = np.arange(self.num_ues)
        # 将action转换为one-hot编码（使用数组广播）
        # 这一步将每个UE与其对应的UAV建立连接
        self.assignment[action, ue_indices] = True
        self.assignment = self.mask_assignment(self.assignment)
        
        
        self.update_G2A_Connection()
        self.update_A2S_Connection()
        # 确保在动作执行后更新 UE 发送缓冲状态
        try:
            self.update_ue_send_bufs()
        except Exception:
            pass
        ns.Ipv4GlobalRoutingHelper.RecomputeRoutingTables()
        self.set_uav_positions()
        # print("assgn:",self.assignment)
        # print("rates:",self.G2A_rates)
        # # 每5秒输出
        # if time_slot % 50 == 0:
        #     print(f"\n=== 时隙 {time_slot} (t={current_time:.1f}s) ===")
        #     total_capacity = 0
        #     uav_indices, ue_indices = np.where(self.assignment)
        #     for uav_id in uav_indices:
        #         ue_list = ue_indices[uav_indices == uav_id]
        #         if len(ue_list) > 0:
        #             rates = self.G2A_scheduler.history[-1]['rates']
        #             ue_info = []
        #             for ue in ue_list:
        #                 rate = rates[ue]
        #                 cap = rate / 1e6
        #                 total_capacity += cap
        #                 ue_info.append(f"UE{ue}({rate:.1f}bps)")
        #             print(f"  UAV{uav_id}: {', '.join(ue_info)}")
        #     print(f"  总系统容量: {total_capacity:.1f} Mbps")
        
    def get_observation(self) -> np.ndarray:
       obs = self.G2A_gain.flatten()*1e9  # shape=(num_uavs*num_ues,)
       obs = np.concatenate((obs, self.uav_positions.flatten()/1000))  # shape=(num_uavs*num_ues+num_uavs*3,)
       # Append per-UE normalized send-buffer-remaining (shape=num_ues, values in [0,1])
       try:
           buf = self.ue_send_buf_remain.flatten()
           obs = np.concatenate((obs, buf))
       except Exception:
           # 如果未初始化，则补 1.0（表示充足）
           obs = np.concatenate((obs, np.ones(self.num_ues, dtype=np.float32)))
       # Append per-UAV A2S queue occupancy (shape=num_uavs)
       try:
           obs = np.concatenate((obs, self.A2S_queue_occupancy.flatten()))
       except Exception:
           obs = np.concatenate((obs, np.zeros(self.num_uavs, dtype=np.float32)))
       # Append random noise state (shape=config.noise_dim)
       try:
           noise = self.noise_state.flatten()
           obs = np.concatenate((obs, noise))
       except Exception:
           obs = np.concatenate((obs, np.zeros(int(getattr(self.config, 'noise_dim', 0)), dtype=np.float32)))
       return obs 
    
    def update_noise_state(self):
        """生成当前时刻的噪声向量，服从 N(0, noise_scale^2)。"""
        dim = int(getattr(self.config, 'noise_dim', 0))
        if dim <= 0:
            self.noise_state = np.zeros(0, dtype=np.float32)
            return
        scale = float(getattr(self.config, 'noise_scale', 0.1))
        self.noise_state = np.random.normal(loc=0.0, scale=scale, size=dim).astype(np.float32)

    def collect_results(self):
        """收集并输出结果"""
        print("\n" + "=" * 60)
        print("                 性能统计结果")
        print("=" * 60)

        self.flow_monitor.CheckForLostPackets()
        stats = self.flow_monitor.GetFlowStats()

        total_tx = 0
        total_rx = 0
        total_tx_bytes = 0
        total_rx_bytes = 0
        total_delay = 0.0
        total_jitter = 0.0
        delay_count = 0
        lost_packets = 0

        print("\n[流量详情]")
        flow_num = 0
        for flow_id, flow_stats in stats:
            flow_num += 1
            total_tx += flow_stats.txPackets
            total_rx += flow_stats.rxPackets
            total_tx_bytes += flow_stats.txBytes
            total_rx_bytes += flow_stats.rxBytes
            lost_packets += flow_stats.lostPackets

            if flow_stats.rxPackets > 0:
                total_delay += flow_stats.delaySum.GetSeconds()
                total_jitter += flow_stats.jitterSum.GetSeconds()
                delay_count += flow_stats.rxPackets

            # if flow_num <= 5:
            pdr = flow_stats.rxPackets / max(1, flow_stats.txPackets) * 100
            print(f"  Flow {flow_id}: TX={flow_stats.txPackets}, "
                    f"RX={flow_stats.rxPackets}, Lost={flow_stats.lostPackets}, PDR={pdr:.1f}%")

        if flow_num > 5:
            print(f"  ...  共 {flow_num} 个流")
        print("---------------------------------")
        print('lost:',lost_packets,'lost_cal:',total_tx-total_rx)
        lost_packets = total_tx-total_rx
        # 计算指标
        pdr = total_rx / max(1, total_tx)
        plr = 1 - pdr
        avg_delay = (total_delay / delay_count * 1000) if delay_count > 0 else 0
        avg_jitter = (total_jitter / delay_count * 1000) if delay_count > 0 else 0
        throughput = total_rx_bytes * 8 / self.config.simulation_time / 1e6
        G2A_sum_rates = self.G2A_sum_rates.sum() / self.config.simulation_time / 1e6
        A2S_sum_rates = self.A2S_sum_rates.sum() / self.config.simulation_time / 1e6
        # 从 C++ Monitor 读取乱序数据
        self.ooo_count = self.ooo_monitor.m_oooCount
        total_rx_monitored = self.ooo_monitor.m_totalRx
        ooo_magnitude_sum = self.ooo_monitor.m_oooMagnitudeSum

        # 乱序率 (基于 Monitor 统计)
        ooo_rate = (self.ooo_count+lost_packets) / total_rx_monitored if total_rx_monitored > 0 else 0
        # ooo_rate = self.ooo_count/total_tx
        ooo_magnitude_avg = ooo_magnitude_sum / total_rx_monitored if total_rx_monitored > 0 else 0
        
        print(f"  (Monitor Stats: OOO={self.ooo_count}, Rx={total_rx_monitored}, AvgDegree={ooo_magnitude_avg:.2f})")

        print("\n" + "-" * 40)
        print("[传输统计]")
        print(f"  发送数据包总数: {total_tx}")
        print(f"  接收数据包总数: {total_rx}")
        print(f"  丢失数据包总数: {lost_packets}")
        print(f"  发送字节总数: {total_tx_bytes:,}")
        print(f"  接收字节总数:  {total_rx_bytes:,}")

        print("\n" + "-" * 40)
        print("[核心性能指标]")
        print(f"  ✓ 数据包投递率 (PDR):    {pdr*100:.2f}%")
        print(f"  ✓ 数据包丢失率 (PLR):    {plr*100:.2f}%")
        print(f"  ✓ 平均端到端延迟:         {avg_delay:.2f} ms")
        print(f"  ✓ 平均抖动 (Jitter):     {avg_jitter:.2f} ms")
        print(f"  ✓ G2A平均吞吐量:            {G2A_sum_rates:.2f} Mbps")
        print(f"  ✓ A2S平均吞吐量:            {A2S_sum_rates:.2f} Mbps")
        print(f"  ✓ 乱序率:            {ooo_rate*100:.2f}%")
        print(f"  ✓ 平均乱序程度:       {ooo_magnitude_avg:.2f}")

        print("\n" + "-" * 40)
        print("[调度统计]")
        print(f"  总推进次数: {len(self.G2A_scheduler.history)}")

        if self.G2A_scheduler.history:
            all_rates = []
            uav_load = {i: 0 for i in range(self.num_uavs)}
            ue_service_count = {}

            for record in self.G2A_scheduler.history:
                assignment = record['assignment']
                for uav_id in range(assignment.shape[0]):
                    ue_list = np.where(assignment[uav_id])[0].tolist()
                    uav_load[uav_id] += len(ue_list)
                    for ue in ue_list:
                        all_rates.append(record['rates'][ue])
                        ue_service_count[ue] = ue_service_count.get(ue, 0) + 1
            if all_rates:
                print(f"  平均调度速率:  {sum(all_rates)/len(all_rates):.2f} bps")
                print(f"  速率范围: [{min(all_rates):.2f}, {max(all_rates):.2f}] bps")

            print(f"\n[UAV负载分布]")
            for uav_id, load in uav_load.items():
                avg_load = load / len(self.G2A_scheduler.history)
                print(f"  UAV{uav_id}: 平均服务 {avg_load:.2f} UE/时隙")

            served_ues = len(ue_service_count)
            print(f"\n  被服务UE数: {served_ues}/{self.num_ues}")

        metrics = {  # 关键：字典打包所有核心指标，方便外部调用
        'pdr': pdr,                           # 投递率 [0,1]
        'plr': plr,                           # 丢失率 [0,1]
        'avg_delay': avg_delay,               # ms
        'avg_jitter': avg_jitter,             # ms
        'G2A_rate': G2A_sum_rates,             # Mbps
        'A2S_rate': A2S_sum_rates,             # Mbps
        'throughput': throughput,             # Mbps
        'ooo_rate': ooo_rate,                 # 乱序率 [0,1]
        'ooo_magnitude_avg': ooo_magnitude_avg,   # 平均乱序程度
        'total_tx': total_tx,
        'total_rx': total_rx,
        'lost_packets': lost_packets,
        'total_tx_bytes': total_tx_bytes,
        'total_rx_bytes': total_rx_bytes,
        'schedule_count': len(self.G2A_scheduler.history),
        }
        # metrics['served_ues'] = served_ues
        return metrics


class Main:
    def __init__(self,args, config):
        self.args = args
        self.config = config
        self.sim = UAVNetworkSimulation(self.config)
        self.metrics = None


    def run(self):
        """运行仿真"""
        global g_sim
        g_sim = self.sim
        args = self.args
        print(f"[Config] simulation_time={self.config.simulation_time}s, num_updates={args.num_updates}")
        run_name = f"{self.config.cpscheme}_{self.config.cc}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        # Disable wandb unless explicitly requested and not blocked via WANDB_DISABLED
        track_enabled = args.track
        if track_enabled:
            import wandb
            wandb.init(
                project=args.project_name,
                # entity=args.wandb_entity,
                # sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        # logdir = f'./{args.project_name}/{args.out_subdir}/'
        # if os.path.exists(logdir):



        
        #     shutil.rmtree(logdir)
        # os.makedirs(logdir, exist_ok=True)
        # 使用子目录区分不同的 Run，这样可以在 TensorBoard 中对比，而不需要手动更换文件夹
        logdir = f'./{args.project_name}/{args.out_subdir}/{run_name}'
        tsboard = SummaryWriter(logdir)
        tsboard.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

        for update in range(1, args.num_updates + 1):
            print("=" * 60)
            print("       UAV-地面网络仿真系统 (ns-3.46.1)")
            print("=" * 60)
            print(f"  场景:  {self.config.area_size}m × {self.config.area_size}m")
            print(f"  UE数量: {self.sim.num_ues}")
            print(f"  UAV数量: {self.sim.num_uavs}")
            print(f"  UAV高度: {self.config.uav_height}m")
            print(f"  UAV轨道半径: {self.config.uav_radius}m")
            print(f"  UAV速度: {self.config.uav_speed}m/s")
            print(f"  每UAV最大接入:  {self.config.n_rbs}")
            print(f"  时隙长度: {self.config.time_slot_duration}s")
            print(f"  仿真时长: {self.config.simulation_time}s")
            print("=" * 60)
            self.sim.setup_all()
            # 启动调度
            event = ns.cppyy.gbl.pythonMakeEvent(scheduling_callback, sys.argv)
            ns.Simulator.Schedule(ns.Seconds(1.0), event)
            print("\n>>> 开始仿真...")
            ns.Simulator.Stop(ns.Seconds(self.config.simulation_time))
            ns.Simulator.Run()
            metrics = self.sim.collect_results()

            if self.metrics is None:
                self.metrics = metrics
            else:
                for key, value in metrics.items():
                    self.metrics[key] += value

            print("num_updates: ",args.num_updates," current update: ",update)

            if update % args.log_interval == 0:
                averaged = {}
                for key, value in self.metrics.items():
                    averaged[key] = value / args.log_interval
                    tsboard.add_scalar(f'metrics/{key}', averaged[key], update)
                if args.track:
                    wandb.log(averaged, step=update)
                self.metrics = None 

            ns.Simulator.Destroy()

        tsboard.close()
        print("\n>>> 仿真完成!")

if __name__ == "__main__":
    # ns.LogComponentEnable("QuicSocketBase", ns.LOG_LEVEL_INFO)
    # ns.LogComponentEnable("MpQuicSubflow", ns.LOG_LEVEL_INFO)
    args = parse_args()
    config = Config()
    config.simulation_time = args.num_steps*config.time_slot_duration
    if getattr(args, 'simulation_time', None) is not None:
        config.simulation_time = args.simulation_time
    main = Main(args, config)
    main.run()