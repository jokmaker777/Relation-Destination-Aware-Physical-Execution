import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class PathConfig:
    """统一路径配置管理"""
    
    def __init__(self, base_project_dir: str = None):
        if base_project_dir is None:
            # The artifact deliberately does not contain an author-specific
            # server path.  Set SIGMOD_ARTIFACT_ROOT before running, or pass
            # base_project_dir explicitly.
            self.base_project_dir = Path(
                os.environ.get("SIGMOD_ARTIFACT_ROOT", "/path/to/artifact_workspace")
            )
        else:
            self.base_project_dir = Path(base_project_dir)
        
        self.data_dir = Path(
            os.environ.get(
                "SIGMOD_PRIVATE_DATA_ROOT",
                str(self.base_project_dir / "private_data"),
            )
        )
        
        self.cache_dir = Path(
            os.environ.get(
                "SIGMOD_GRAPH_CACHE_ROOT",
                str(self.base_project_dir / "cache"),
            )
        )
        
        self.result_dir = Path(
            os.environ.get(
                "SIGMOD_OUTPUT_ROOT",
                str(self.base_project_dir / "outputs"),
            )
        )
        
        # 配置文件路径
        self.config_dir = Path(__file__).resolve().parent
        
        # 模型保存路径
        self.model_dir = self.base_project_dir / "models"
        
        # 日志路径
        self.log_dir = self.result_dir / "logs"
        
        # 分布式训练临时目录（固定路径，不依赖时间戳）
        self.dist_temp_dir = self.result_dir / "dist_temp"
        
        # 创建必要的目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        dirs_to_create = [
            self.cache_dir,
            self.result_dir,
            self.model_dir,
            self.log_dir,
            self.dist_temp_dir
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_data_path(self, date_str: str) -> Path:
        """获取指定日期的数据路径"""
        folder_name = f"{date_str}_index"
        return self.data_dir / folder_name
    
    def get_cache_path(self, filename: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / filename
    
    def get_model_path(self, model_name: str) -> Path:
        """获取模型路径"""
        return self.model_dir / model_name
    
    def get_result_path(self, filename: str) -> Path:
        """获取结果文件路径"""
        return self.result_dir / filename
    
    
    def get_log_path(self, filename: str) -> Path:
        """获取日志文件路径"""
        return self.log_dir / filename
    
    def get_dist_temp_dir(self) -> Path:
        """获取分布式训练临时目录（固定路径，所有rank共享）"""
        return self.dist_temp_dir


# Global configuration.  All author/server-specific locations have been
# replaced by environment-controlled placeholders.
path_config = PathConfig()
