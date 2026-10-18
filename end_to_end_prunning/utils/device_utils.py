"""
设备检测工具模块 - 支持 NVIDIA CUDA 和华为昇腾 CANN
"""
import torch
import os
import logging

logger = logging.getLogger(__name__)

# 🔧 尝试导入 torch_npu (华为昇腾NPU支持)
# 必须显式导入才能使用 torch.npu 功能
try:
    import torch_npu
    _TORCH_NPU_AVAILABLE = True
except ImportError:
    _TORCH_NPU_AVAILABLE = False


class DeviceManager:
    """统一的设备管理器，支持 CUDA 和 CANN"""
    
    _device_type = None  # 缓存设备类型
    _device = None  # 缓存设备对象
    
    @staticmethod
    def get_device_type():
        """
        检测可用的加速器类型
        
        Returns:
            str: 'cuda', 'npu', 或 'cpu'
        """
        if DeviceManager._device_type is not None:
            return DeviceManager._device_type
        
        # 1. 检查是否有华为昇腾 NPU (CANN)
        try:
            logger.info("🔍 开始NPU设备检测...")
            logger.info(f"   PyTorch版本: {torch.__version__}")
            logger.info(f"   torch_npu包可用: {_TORCH_NPU_AVAILABLE}")
            logger.info(f"   是否有torch.npu属性: {hasattr(torch, 'npu')}")
            
            if _TORCH_NPU_AVAILABLE and hasattr(torch, 'npu'):
                try:
                    npu_available = torch.npu.is_available()
                    logger.info(f"   torch.npu.is_available(): {npu_available}")
                    
                    if npu_available:
                        npu_count = torch.npu.device_count()
                        DeviceManager._device_type = 'npu'
                        logger.info("✅ 检测到华为昇腾 NPU (CANN)")
                        logger.info(f"   NPU数量: {npu_count}")
                        return 'npu'
                    else:
                        logger.warning("⚠️ torch_npu已导入但torch.npu.is_available()返回False")
                        logger.warning("   可能原因:")
                        logger.warning("   1. CANN驱动未安装或未正确配置")
                        logger.warning("   2. NPU设备不可用或被占用")
                        logger.warning("   3. 环境变量未正确设置 (ASCEND_HOME_PATH等)")
                        logger.warning("   4. NPU设备被占用或无权限访问")
                except Exception as npu_check_err:
                    logger.warning(f"⚠️ NPU可用性检查失败: {npu_check_err}")
            else:
                if not _TORCH_NPU_AVAILABLE:
                    logger.info("   torch_npu包未安装")
                    logger.info("   如需使用NPU，请安装: pip install torch_npu")
                else:
                    logger.warning("   torch_npu已安装但torch.npu属性不存在")
                    logger.warning("   可能版本不兼容，请检查torch和torch_npu版本匹配")
        except Exception as e:
            logger.debug(f"NPU 检测失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # 2. 检查是否有 NVIDIA CUDA
        try:
            if torch.cuda.is_available():
                DeviceManager._device_type = 'cuda'
                logger.info("✅ 检测到 NVIDIA CUDA")
                return 'cuda'
        except Exception as e:
            logger.debug(f"CUDA 检测失败: {e}")
        
        # 3. 回退到 CPU
        DeviceManager._device_type = 'cpu'
        logger.info("⚠️ 未检测到 GPU/NPU，使用 CPU")
        return 'cpu'
    
    @staticmethod
    def get_device(device_id=0):
        """
        获取 torch.device 对象
        
        Args:
            device_id: 设备ID，默认为0
            
        Returns:
            torch.device: 设备对象
        """
        device_type = DeviceManager.get_device_type()
        
        if device_type == 'npu':
            return torch.device(f'npu:{device_id}')
        elif device_type == 'cuda':
            return torch.device(f'cuda:{device_id}')
        else:
            return torch.device('cpu')
    
    @staticmethod
    def is_available():
        """检查是否有可用的加速器 (GPU/NPU)"""
        device_type = DeviceManager.get_device_type()
        return device_type in ['cuda', 'npu']
    
    @staticmethod
    def device_count():
        """获取可用的加速器数量"""
        device_type = DeviceManager.get_device_type()
        
        if device_type == 'npu':
            try:
                return torch.npu.device_count()
            except:
                return 0
        elif device_type == 'cuda':
            try:
                return torch.cuda.device_count()
            except:
                return 0
        else:
            return 0
    
    # ⚠️ current_device() 方法已移除
    # 不要使用 torch.npu.current_device() 或 torch.cuda.current_device()
    # 在多卡训练中，每个进程应该直接使用 local_rank 来确定设备ID
    # 调用 current_device() 可能在设备未初始化时返回-1，导致stream初始化失败
    
    @staticmethod
    def set_device(device_id):
        """设置当前使用的设备（简化版，避免初始化冲突）"""
        device_type = DeviceManager.get_device_type()
        
        if device_type == 'npu':
            try:
                # 🔧 NPU修复：简化设置流程，避免stream初始化冲突
                torch.npu.set_device(device_id)
                logger.debug(f"✅ 设置NPU设备: npu:{device_id}")
            except Exception as e:
                logger.error(f"❌ 设置NPU设备失败 npu:{device_id}: {e}")
                raise
                
        elif device_type == 'cuda':
            try:
                torch.cuda.set_device(device_id)
                logger.debug(f"✅ 设置CUDA设备: cuda:{device_id}")
            except Exception as e:
                logger.error(f"❌ 设置CUDA设备失败 cuda:{device_id}: {e}")
                raise
    
    @staticmethod
    def get_device_name(device_id=0):
        """获取设备名称"""
        device_type = DeviceManager.get_device_type()
        
        if device_type == 'npu':
            try:
                return torch.npu.get_device_name(device_id)
            except:
                return f"NPU:{device_id}"
        elif device_type == 'cuda':
            try:
                return torch.cuda.get_device_name(device_id)
            except:
                return f"CUDA:{device_id}"
        else:
            return "CPU"
    
    @staticmethod
    def synchronize(device=None):
        """同步设备操作"""
        device_type = DeviceManager.get_device_type()
        
        if device_type == 'npu':
            torch.npu.synchronize(device)
        elif device_type == 'cuda':
            torch.cuda.synchronize(device)
    
    @staticmethod
    def empty_cache():
        """清空设备缓存，带错误处理"""
        device_type = DeviceManager.get_device_type()
        
        try:
            if device_type == 'npu':
                torch.npu.empty_cache()
                logger.debug("✅ NPU缓存已清空")
            elif device_type == 'cuda':
                torch.cuda.empty_cache()
                logger.debug("✅ CUDA缓存已清空")
        except Exception as e:
            logger.warning(f"⚠️ 清空{device_type.upper()}缓存失败: {e}")
    
    @staticmethod
    def memory_allocated(device=None):
        """获取已分配的内存（字节）"""
        device_type = DeviceManager.get_device_type()
        
        if device_type == 'npu':
            try:
                return torch.npu.memory_allocated(device)
            except:
                return 0
        elif device_type == 'cuda':
            try:
                return torch.cuda.memory_allocated(device)
            except:
                return 0
        else:
            return 0
    
    @staticmethod
    def memory_reserved(device=None):
        """获取预留的内存（字节）"""
        device_type = DeviceManager.get_device_type()
        
        if device_type == 'npu':
            try:
                return torch.npu.memory_reserved(device)
            except:
                return 0
        elif device_type == 'cuda':
            try:
                return torch.cuda.memory_reserved(device)
            except:
                return 0
        else:
            return 0
    
    @staticmethod
    def get_device_properties(device_id=0):
        """获取设备属性"""
        device_type = DeviceManager.get_device_type()
        
        if device_type == 'npu':
            try:
                return torch.npu.get_device_properties(device_id)
            except:
                return None
        elif device_type == 'cuda':
            try:
                return torch.cuda.get_device_properties(device_id)
            except:
                return None
        else:
            return None
    
    @staticmethod
    def get_backend_name():
        """获取分布式训练后端名称"""
        device_type = DeviceManager.get_device_type()
        
        if device_type == 'npu':
            return 'hccl'  # 华为昇腾使用 HCCL
        elif device_type == 'cuda':
            return 'nccl'  # NVIDIA 使用 NCCL
        else:
            return 'gloo'  # CPU 使用 Gloo
    
    @staticmethod
    def init_device_environment(device_id=0):
        """
        初始化设备环境，确保设备可用
        
        Args:
            device_id: 设备ID
            
        Returns:
            bool: 初始化是否成功
        """
        device_type = DeviceManager.get_device_type()
        
        try:
            logger.info(f"🔧 [Rank {device_id}] 初始化{device_type.upper()}环境...")
            
            if device_type == 'npu':
                # 🔧 NPU多进程修复：按正确顺序初始化
                # 1. 先清理可能存在的缓存
                try:
                    torch.npu.empty_cache()
                except:
                    pass
                
                # 2. 直接设置设备（不使用包装函数，避免循环）
                logger.info(f"   [Rank {device_id}] 设置NPU设备: npu:{device_id}")
                torch.npu.set_device(device_id)
                
                # 3. 设备设置完成，不再验证current_device以避免初始化问题
                logger.info(f"   [Rank {device_id}] NPU设备已设置: npu:{device_id}")
                
                # 4. 延迟测试 - 在实际使用时再创建张量，这里只验证设备设置
                # 不在初始化阶段创建张量，避免stream初始化冲突
                
                logger.info(f"✅ [Rank {device_id}] NPU环境初始化成功: npu:{device_id}")
                return True
                
            elif device_type == 'cuda':
                # CUDA环境初始化
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
                
                logger.info(f"   [Rank {device_id}] 设置CUDA设备: cuda:{device_id}")
                torch.cuda.set_device(device_id)
                
                # 验证设备设置
                current = torch.cuda.current_device()
                if current != device_id:
                    raise RuntimeError(f"CUDA设备设置失败: 期望 {device_id}, 实际 {current}")
                
                logger.info(f"✅ [Rank {device_id}] CUDA环境初始化成功: cuda:{device_id}")
                return True
                
            else:
                logger.info(f"✅ [Rank {device_id}] CPU环境，无需特殊初始化")
                return True
                
        except Exception as e:
            logger.error(f"❌ [Rank {device_id}] {device_type.upper()}环境初始化失败: {e}")
            logger.error(f"   设备ID: {device_id}")
            logger.error(f"   设备类型: {device_type}")
            
            if device_type == 'npu':
                logger.error(f"   建议:")
                logger.error(f"   1. 运行 npu-smi info 检查NPU状态")
                logger.error(f"   2. 检查是否有其他进程占用NPU")
                logger.error(f"   3. 尝试重启NPU驱动: systemctl restart ascend-driver")
                logger.error(f"   4. 设置环境变量: export ASCEND_RT_VISIBLE_DEVICES={device_id}")
                logger.error(f"   5. 强制单进程: export FORCE_SINGLE_PROCESS=1")
            
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def setup_environment():
        """设置环境变量"""
        device_type = DeviceManager.get_device_type()
        
        if device_type == 'npu':
            # 华为昇腾环境变量设置
            if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
                logger.info("💡 提示: 可以通过设置 ASCEND_RT_VISIBLE_DEVICES 来指定使用的 NPU")
        elif device_type == 'cuda':
            # NVIDIA CUDA 环境变量设置
            if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
                logger.info("✅ 设置 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
            
            if "CUDA_VISIBLE_DEVICES" not in os.environ:
                logger.info("💡 提示: 可以通过设置 CUDA_VISIBLE_DEVICES 来指定使用的 GPU")
    
    @staticmethod
    def print_device_info():
        """打印设备信息"""
        device_type = DeviceManager.get_device_type()
        
        logger.info("=" * 60)
        logger.info("🔍 设备信息检查")
        logger.info("=" * 60)
        
        if device_type == 'npu':
            try:
                num_devices = torch.npu.device_count()
                logger.info(f"   - 加速器类型: 华为昇腾 NPU (CANN)")
                logger.info(f"   - NPU可用: True")
                logger.info(f"   - NPU数量: {num_devices}")
                
                try:
                    # 列出所有NPU设备（不查询当前设备，避免多进程冲突）
                    for i in range(num_devices):
                        device_name = DeviceManager.get_device_name(i)
                        logger.info(f"   - NPU {i}: {device_name}")
                except Exception as e:
                    logger.warning(f"   ⚠️ 获取NPU详细信息失败: {e}")
            except Exception as e:
                logger.error(f"   ❌ NPU信息获取失败: {e}")
        
        elif device_type == 'cuda':
            try:
                num_devices = torch.cuda.device_count()
                logger.info(f"   - 加速器类型: NVIDIA CUDA")
                logger.info(f"   - CUDA可用: True")
                logger.info(f"   - GPU数量: {num_devices}")
                
                try:
                    current = torch.cuda.current_device()
                    logger.info(f"   - 当前GPU: {current}")
                    logger.info(f"   - CUDA版本: {torch.version.cuda}")
                    
                    for i in range(num_devices):
                        device_name = torch.cuda.get_device_name(i)
                        props = torch.cuda.get_device_properties(i)
                        logger.info(f"   - GPU {i}: {device_name} ({props.total_memory / 1024**3:.1f} GB)")
                except Exception as e:
                    logger.warning(f"   ⚠️ 获取GPU详细信息失败: {e}")
            except Exception as e:
                logger.error(f"   ❌ CUDA信息获取失败: {e}")
        
        else:
            logger.info("   - 加速器类型: CPU")
            logger.info("   - 未检测到 GPU/NPU")
        
        logger.info("=" * 60)


# 便捷函数
def get_device(device_id=0):
    """获取设备对象"""
    return DeviceManager.get_device(device_id)


def is_accelerator_available():
    """检查是否有可用的加速器"""
    return DeviceManager.is_available()


def get_device_type():
    """获取设备类型"""
    return DeviceManager.get_device_type()


