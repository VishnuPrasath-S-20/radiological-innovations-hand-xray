import subprocess
import json
import psutil
import torch
import time

class AutoResourceManager:
    """
    Reads GPU sensors (via nvidia-smi) and CPU/RAM sensors (via psutil).
    Automatically adjusts batch size AND num_workers to prevent crashes.
    """
    
    def __init__(self, 
                 gpu_temp_limit=82,
                 cpu_temp_limit=85,
                 vram_limit_percent=0.92,
                 ram_limit_percent=90,
                 verbose=True):
        self.gpu_temp_limit = gpu_temp_limit
        self.cpu_temp_limit = cpu_temp_limit
        self.vram_limit_percent = vram_limit_percent
        self.ram_limit_percent = ram_limit_percent
        self.verbose = verbose
        self.batch_size = 64
        self.num_workers = 4
    
    def get_gpu_metrics(self):
        """Read GPU data using nvidia-smi"""
        try:
            # Use nvidia-smi to get GPU stats in XML format
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 5:
                    return {
                        'temperature': float(parts[0]),
                        'vram_used': float(parts[1]),
                        'vram_total': float(parts[2]),
                        'utilization': float(parts[3]),
                        'power_draw': float(parts[4]) if parts[4] else 0
                    }
        except Exception:
            pass
        return None
    
    def get_cpu_ram_metrics(self):
        """Read CPU and RAM data using psutil"""
        try:
            # Get CPU temperature
            cpu_temp = None
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    cpu_temp = temps['coretemp'][0].current
                elif 'cpu-thermal' in temps:
                    cpu_temp = temps['cpu-thermal'][0].current
            
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.5),
                'cpu_temp': cpu_temp,
                'ram_used_gb': psutil.virtual_memory().used / (1024**3),
                'ram_total_gb': psutil.virtual_memory().total / (1024**3),
                'ram_percent': psutil.virtual_memory().percent
            }
        except Exception:
            return None
    
    def manage_resources(self, current_batch_size, current_num_workers):
        """
        Read all sensors and automatically take action.
        Returns: (new_batch_size, new_num_workers)
        """
        gpu = self.get_gpu_metrics()
        sys_metrics = self.get_cpu_ram_metrics()
        
        new_batch_size = current_batch_size
        new_num_workers = current_num_workers
        action_taken = False
        
        # --- 1. GPU Temperature Check ---
        if gpu:
            temp = gpu['temperature']
            if temp > self.gpu_temp_limit:
                if self.verbose:
                    print(f"\n🔥 GPU TEMP: {temp}°C (Limit: {self.gpu_temp_limit}°C)")
                
                if current_batch_size > 16:
                    new_batch_size = max(16, current_batch_size // 2)
                    print(f"   🔄 Auto-reducing batch size: {current_batch_size} → {new_batch_size}")
                    action_taken = True
                else:
                    print(f"   ⏸️  Pausing for 30 seconds to cool GPU...")
                    time.sleep(30)
                    action_taken = True
            
            # --- 2. VRAM Check ---
            if gpu['vram_total'] > 0:
                vram_percent = gpu['vram_used'] / gpu['vram_total']
                if vram_percent > self.vram_limit_percent:
                    if self.verbose:
                        print(f"   💾 VRAM: {vram_percent*100:.1f}% - Clearing cache...")
                    torch.cuda.empty_cache()
                    action_taken = True
        
        # --- 3. CPU Temperature Check ---
        if sys_metrics and sys_metrics.get('cpu_temp'):
            cpu_temp = sys_metrics['cpu_temp']
            if cpu_temp > self.cpu_temp_limit:
                if self.verbose:
                    print(f"\n🔥 CPU TEMP: {cpu_temp:.0f}°C (Limit: {self.cpu_temp_limit}°C)")
                
                if current_num_workers > 1:
                    new_num_workers = max(1, current_num_workers // 2)
                    print(f"   🔄 Auto-reducing num_workers: {current_num_workers} → {new_num_workers}")
                    action_taken = True
                else:
                    print(f"   ⏸️  CPU too hot! Pausing for 15 seconds...")
                    time.sleep(15)
                    action_taken = True
        
        # --- 4. CPU Usage Check (Bottleneck detection) ---
        if sys_metrics:
            cpu_percent = sys_metrics['cpu_percent']
            ram_percent = sys_metrics['ram_percent']
            
            if cpu_percent > 90 and gpu and gpu.get('utilization', 100) < 70:
                if self.verbose:
                    print(f"   ⚠️  CPU at {cpu_percent}% but GPU at {gpu['utilization']}% - CPU bottleneck!")
                    if current_num_workers > 1:
                        new_num_workers = max(1, current_num_workers // 2)
                        print(f"   🔄 Reducing num_workers: {current_num_workers} → {new_num_workers}")
                        action_taken = True
            
            if ram_percent > self.ram_limit_percent:
                if self.verbose:
                    print(f"   ⚠️  RAM: {ram_percent:.1f}% - High usage!")
                    if current_num_workers > 1:
                        new_num_workers = max(1, current_num_workers // 2)
                        print(f"   🔄 Reducing num_workers to free RAM: {current_num_workers} → {new_num_workers}")
                        action_taken = True
        
        if not action_taken and self.verbose:
            temp_str = f"{gpu['temperature']}°C" if gpu else "N/A"
            cpu_temp_str = f"{sys_metrics['cpu_temp']:.0f}°C" if (sys_metrics and sys_metrics.get('cpu_temp')) else "N/A"
            ram_str = f"{sys_metrics['ram_percent']:.1f}%" if sys_metrics else "N/A"
            print(f"   ✅ GPU: {temp_str}, CPU: {cpu_temp_str}, RAM: {ram_str} - All good.")
        
        return new_batch_size, new_num_workers
    
    def get_display_string(self):
        """Get a one-line status string for logging"""
        gpu = self.get_gpu_metrics()
        sys_metrics = self.get_cpu_ram_metrics()
        
        parts = []
        if gpu:
            parts.append(f"GPU:{gpu['temperature']:.0f}°C")
            if gpu['vram_total'] > 0:
                vram_pct = (gpu['vram_used'] / gpu['vram_total']) * 100
                parts.append(f"VRAM:{vram_pct:.1f}%")
        if sys_metrics:
            if sys_metrics.get('cpu_temp'):
                parts.append(f"CPU:{sys_metrics['cpu_temp']:.0f}°C")
            parts.append(f"RAM:{sys_metrics['ram_percent']:.1f}%")
        
        return " | ".join(parts)
