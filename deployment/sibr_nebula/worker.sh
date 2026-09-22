#!/usr/bin/env bash
set -euo pipefail
set +x
unset PYTHONHOME PYTHONPATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
work=/tmp/sibr_viewer
mkdir -p "$work"
exec > >(tee -a "$work/worker.log") 2>&1
children=()
cleanup() { for pid in "${children[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
fail() { echo "[失败] $*"; exit 1; }
echo '[1/5] 检测 GPU 与系统'
nvidia-smi --query-gpu=name,memory.total,pci.bus_id --format=csv
cat /etc/os-release
[[ $(id -u) == 0 ]] || fail '自动安装依赖需要容器 root 权限。'
echo '[2/5] 安装最小图形依赖，先验证 GPU 显示能力'
if command -v apt-get >/dev/null; then
 package_family=apt
 export DEBIAN_FRONTEND=noninteractive
 apt-get update
 apt-get install -y --no-install-recommends xserver-xorg-core mesa-utils xauth openssl
elif command -v dnf >/dev/null || command -v yum >/dev/null; then
 package_family=rpm
 package_manager=$(command -v dnf || command -v yum)
 "$package_manager" install -y xorg-x11-server-Xorg glx-utils xorg-x11-xauth openssl || fail '安装最小图形依赖失败，请检查上方软件仓库错误。'
else
 fail '找不到 apt-get/dnf/yum，无法安装图形依赖。'
fi
# 不安装或替换宿主机 NVIDIA 驱动；图形驱动必须由平台注入。
driver=$(find /usr/lib64 /usr/lib /usr/local/nvidia /usr/local/lib /lib64 -name nvidia_drv.so -print -quit 2>/dev/null || true)
[[ -n "$driver" ]] || fail '未找到 NVIDIA Xorg 驱动 nvidia_drv.so。当前容器可能仅注入计算驱动；需平台提供 NVIDIA graphics/display 驱动挂载及设备权限。更换 Python/conda 环境无法解决。'
module_root=$(dirname "$(dirname "$driver")")
echo "[图形驱动] $driver"
echo '[3/5] 创建 NVIDIA 硬件显示会话'
# 使用 CUDA 可见 GPU 的 PCI 地址，避免多卡节点选错物理卡。
bus=$(python3 - <<'PY'
import ctypes
c=ctypes.CDLL('libcuda.so.1')
assert c.cuInit(0)==0
x=ctypes.c_int(); assert c.cuDeviceGet(ctypes.byref(x),0)==0
b=ctypes.create_string_buffer(32); assert c.cuDeviceGetPCIBusId(b,32,x)==0
s=b.value.decode(); domain,bus,df=s.split(':'); dev,func=df.split('.')
print('PCI:%d@%d:%d:%d' % (int(bus,16),int(domain,16),int(dev,16),int(func,16)))
PY
)
cat > "$work/xorg.conf" <<XORG
Section "Files"
 ModulePath "$module_root"
 ModulePath "/usr/lib64/xorg/modules"
 ModulePath "/usr/lib/xorg/modules"
EndSection
Section "Device"
 Identifier "GPU"
 Driver "nvidia"
 BusID "$bus"
 Option "AllowEmptyInitialConfiguration" "True"
EndSection
Section "Screen"
 Identifier "Screen0"
 Device "GPU"
 DefaultDepth 24
 SubSection "Display"
  Depth 24
  Virtual 1920 1080
 EndSubSection
EndSection
XORG
export DISPLAY=:99
export XAUTHORITY="$work/Xauthority"
touch "$XAUTHORITY"
xauth -f "$XAUTHORITY" add "$DISPLAY" . "$(openssl rand -hex 16)"
Xorg "$DISPLAY" -config "$work/xorg.conf" -auth "$XAUTHORITY" -noreset -nolisten tcp -logfile "$work/Xorg.log" &
children+=("$!")
ready=false
for ((i=0;i<30;i++)); do
 if glxinfo -B > "$work/glxinfo.txt" 2>&1; then ready=true; break; fi
 sleep 1
done
if [[ $ready != true ]]; then tail -60 "$work/Xorg.log"; fail '硬件 Xorg 启动失败，检查 NVIDIA graphics/display 驱动挂载及设备权限。'; fi
cat "$work/glxinfo.txt"
grep -q NVIDIA "$work/glxinfo.txt" || fail '检测到非 NVIDIA OpenGL，停止以避免软件渲染。'
echo '[依赖] GPU 图形会话通过，安装远程桌面和 SIBR 构建依赖'
if [[ $package_family == apt ]]; then
 apt-get install -y --no-install-recommends x11vnc novnc websockify git ca-certificates cmake build-essential libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
else
 # 使用镜像已配置的仓库；如缺包则列出具体包，避免静默跳过。
 "$package_manager" install -y x11vnc novnc python3-websockify git ca-certificates || fail '远程桌面依赖安装失败。请平台在该镜像配置可用的 EPEL/兼容仓库，或预装 x11vnc、novnc、python3-websockify。'
 if [[ ${SIBR_BIN:-auto} == auto ]]; then
  "$package_manager" install -y gcc gcc-c++ make cmake glew-devel assimp-devel boost-devel gtk3-devel opencv-devel glfw-devel ffmpeg-devel eigen3-devel libXxf86vm-devel embree-devel || fail 'SIBR 构建依赖不全；请根据上方缺包信息配置兼容仓库，或通过 SIBR_BIN 指定与此系统兼容的预编译程序。'
 fi
fi
novnc_web=''
for candidate in /usr/share/novnc /usr/share/novnc/www /usr/share/noVNC; do
 if [[ -f "$candidate/vnc.html" ]]; then novnc_web=$candidate; break; fi
done
[[ -n "$novnc_web" ]] || fail '未找到 noVNC vnc.html 页面。'
if [[ ${SIBR_BIN:-auto} == auto ]]; then
 command -v nvcc >/dev/null || fail '镜像缺少 CUDA 编译器 nvcc，请使用 CUDA devel 镜像或提供 SIBR_BIN。'
 echo '[构建] 下载官方 SIBR 源码并编译；输出保留在 worker.log'
 if [[ ! -d "$work/source/.git" ]]; then
  git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git "$work/source"
 fi
 cmake -S "$work/source/SIBR_viewers" -B "$work/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$work/install"
 cmake --build "$work/build" -j 8 --target install
 SIBR_BIN="$work/install/bin/SIBR_gaussianViewer_app"
fi
[[ -x "$SIBR_BIN" ]] || fail "找不到可执行查看器: $SIBR_BIN"
echo '[4/5] 选择并缓存模型'
python3 - <<'PY'
import os, shutil
from pathlib import Path
root=Path('/data/oss_bucket_0/beifen/data/3dgs_bushu')
model=os.environ.get('MODEL_PATH','auto')
if model=='auto':
 candidates=sorted(p for p in root.glob('huanliufa_*') if list(p.glob('point_cloud/iteration_*/point_cloud.ply')))
 if not candidates: raise RuntimeError('OSS 下没有已保存的模型；请等保存完成或设置 MODEL_PATH')
 src=candidates[-1]
else:
 src=Path(model.replace('oss://fmc/','/data/oss_bucket_0/',1))
clouds=list(src.glob('point_cloud/iteration_*/point_cloud.ply'))
if not clouds: raise RuntimeError('指定目录没有 point_cloud/iteration_*/point_cloud.ply')
cloud=max(clouds,key=lambda p:int(p.parent.name.split('_')[-1]))
dst=Path('/tmp/sibr_viewer/model'); target=dst/'point_cloud'/cloud.parent.name/'point_cloud.ply'
target.parent.mkdir(parents=True,exist_ok=True)
print('模型:',src,'迭代:',cloud.parent.name,'字节:',cloud.stat().st_size,flush=True)
shutil.copy2(str(cloud),str(target))
for name in ['cfg_args','cameras.json','input.ply']:
 if (src/name).exists(): shutil.copy2(str(src/name),str(dst/name))
PY
echo '[5/5] 启动远程画面服务（仅监听节点本地地址）'
x11vnc -display "$DISPLAY" -auth "$XAUTHORITY" -localhost -rfbport 5900 -nopw -forever -shared > "$work/vnc.log" 2>&1 &
children+=("$!")
websockify --web="$novnc_web" 127.0.0.1:6080 127.0.0.1:5900 > "$work/web.log" 2>&1 &
children+=("$!")
"$SIBR_BIN" -m "$work/model" --rendering-size 1280 720 > "$work/viewer.log" 2>&1 &
children+=("$!")
echo '[访问] 需通过星云端口代理或 SSH 隧道连接到此 worker 的 127.0.0.1:6080。'
echo '[SSH 示例] ssh -N -L 6080:127.0.0.1:6080 <可登录该 worker 的地址>'
echo '[浏览器] 隧道建立后打开 http://127.0.0.1:6080/vnc.html?autoconnect=true'
echo '[日志] /tmp/sibr_viewer/{worker,Xorg,vnc,web,viewer}.log'
echo '[说明] 查看器持续占用 GPU，结束试用后在星云停止此任务。'
while true; do
 for pid in "${children[@]}"; do
  if ! kill -0 "$pid" 2>/dev/null; then
   tail -30 "$work/viewer.log" "$work/vnc.log" "$work/web.log"
   fail "子进程 $pid 已退出"
  fi
 done
 echo "[运行] $(date -Is) SIBR/远程桌面存活"
 sleep 30
done
