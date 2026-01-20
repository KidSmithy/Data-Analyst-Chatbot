# mcp_playground/start_mcp.py
"""
MCP 服务器启动脚本
用于启动 mcp-server-data-exploration 中的数据探索服务器
"""

import sys
import os
import asyncio
import io

# ================================================
# 1. 关键：强制无缓冲输出（对 Windows stdio 通信至关重要）
# ================================================
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    line_buffering=True,      # 每行立即刷新
    write_through=True        # 写入立即生效
)

sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer,
    encoding='utf-8',
    line_buffering=True,
    write_through=True
)

os.environ['PYTHONUNBUFFERED'] = '1'

# ================================================
# 2. 设置 Python 路径，确保能正确导入服务器模块
# ================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
server_project_dir = os.path.join(current_dir, "mcp-server-data-exploration")
sys.path.insert(0, server_project_dir)
src_dir = os.path.join(server_project_dir, "src")
sys.path.insert(0, src_dir)

# ================================================
# 3. 导入并运行服务器
# ================================================
if __name__ == "__main__":
    try:
        print(f"[启动器] 工作目录: {os.getcwd()}", file=sys.stderr)
        print(f"[启动器] Python 路径: {sys.path}", file=sys.stderr)
        print(f"[启动器] 服务器项目目录: {server_project_dir}", file=sys.stderr)
        print("[启动器] 正在导入服务器模块...", file=sys.stderr)

        try:
            from src.mcp_server_ds.server import main
            print("[启动器] 使用方式2导入成功", file=sys.stderr)
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "server",
                os.path.join(src_dir, "mcp_server_ds", "server.py")
            )
            server_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(server_module)
            main = server_module.main
            print("[启动器] 使用方式3导入成功", file=sys.stderr)
        
        print("=" * 50, file=sys.stderr)
        print("🚀 MCP 数据探索服务器启动中...", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        
        sys.stdout.flush()
        sys.stderr.flush()
        
        asyncio.run(main())
        
    except ImportError as e:
        print(f"❌ [启动器] 导入失败！", file=sys.stderr)
        print(f"   错误详情: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[启动器] 服务器被用户中断", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"❌ [启动器] 服务器运行出错: {type(e).__name__}", file=sys.stderr)
        print(f"   错误信息: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)