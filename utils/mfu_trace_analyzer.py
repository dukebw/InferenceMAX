#!/usr/bin/env python3
"""
MFU (Model FLOPS Utilization) Trace Analyzer for PyTorch Profiler Traces

This script analyzes PyTorch profiler traces and adds MFU metrics to matmul operations.
Designed for H200 SXM GPUs running SGLang with DeepSeek models.

Usage:
    python mfu_trace_analyzer.py input_trace.json output_trace.json [--gpu H200]
"""

import json
import re
import argparse
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class GPUSpecs:
    """GPU specifications for MFU calculation"""
    name: str
    fp16_tflops: float  # BF16/FP16 peak TFLOPS
    fp8_tflops: float   # FP8 peak TFLOPS (if supported)
    num_sms: int
    
# GPU specifications database
GPU_SPECS = {
    "H200": GPUSpecs(
        name="NVIDIA H200 SXM",
        fp16_tflops=989.4,    # BF16 Tensor Core peak
        fp8_tflops=1978.9,    # FP8 Tensor Core peak (2x BF16)
        num_sms=132
    ),
    "H100": GPUSpecs(
        name="NVIDIA H100 SXM", 
        fp16_tflops=989.4,
        fp8_tflops=1978.9,
        num_sms=132
    ),
    "A100": GPUSpecs(
        name="NVIDIA A100 SXM",
        fp16_tflops=312.0,
        fp8_tflops=312.0,  # A100 doesn't have FP8
        num_sms=108
    ),
}


@dataclass
class GemmInfo:
    """Information about a GEMM operation"""
    m: int
    n: int
    k: int
    dtype: str
    duration_us: float
    flops: int
    tflops: float
    mfu: float
    kernel_name: str = ""
    external_id: int = 0


def parse_deep_gemm_kernel_name(kernel_name: str) -> Optional[Tuple[int, int]]:
    """
    Parse deep_gemm kernel name to extract N and K dimensions.
    Example: void deep_gemm::sm90_fp8_gemm_1d2d_impl<0u, 2112u, 7168u, 1u, ...>
    Returns (N, K) tuple
    """
    match = re.search(r'deep_gemm::sm90_fp8_gemm_1d2d_impl<(\d+)u,\s*(\d+)u,\s*(\d+)u', kernel_name)
    if match:
        # Template params: <?, N, K, ...>
        n = int(match.group(2))
        k = int(match.group(3))
        return (n, k)
    return None


def parse_deep_gemm_kernel_full(kernel_name: str, grid: List[int]) -> Optional[Tuple[int, int, int, str]]:
    """
    Parse deep_gemm kernel name to extract M, N, K dimensions.
    For decode kernels without External ID correlation.
    
    Template: <0u, N, K, 1u, M_tile, N_tile, ...>
    M is typically inferred from grid[0] * M_tile / num_warps or is the batch size
    
    Returns (M, N, K, dtype) tuple
    """
    match = re.search(r'deep_gemm::sm90_fp8_gemm_1d2d_impl<(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u', kernel_name)
    if match:
        n = int(match.group(2))
        k = int(match.group(3))
        m_tile = int(match.group(5))
        
        # For decode, use the global batch size hint if available
        global DECODE_BATCH_SIZE
        try:
            m = DECODE_BATCH_SIZE
        except NameError:
            m = m_tile  # Fallback to tile size
        
        dtype = 'fp8' if 'fp8' in kernel_name.lower() else 'bf16'
        return (m, n, k, dtype)
    
    return None


# Global batch size for decode traces
DECODE_BATCH_SIZE = 64


def parse_nvjet_kernel_name(kernel_name: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse nvjet (cuBLAS) kernel name to extract dimensions.
    Example: nvjet_tst_128x8_64x12_2x1_v_bz_TNT
    """
    # nvjet kernels are harder to parse - dimensions encoded differently
    # For now, we'll rely on CPU op correlation
    return None


def calculate_gemm_flops(m: int, n: int, k: int) -> int:
    """Calculate FLOPs for a GEMM operation: C = A @ B where A is MxK and B is KxN"""
    return 2 * m * n * k


def calculate_mfu(flops: int, duration_us: float, peak_tflops: float) -> float:
    """Calculate MFU (Model FLOPS Utilization)"""
    if duration_us <= 0:
        return 0.0
    
    duration_s = duration_us / 1e6
    achieved_tflops = (flops / 1e12) / duration_s
    mfu = (achieved_tflops / peak_tflops) * 100.0
    return mfu


def get_dtype_peak_tflops(dtype: str, gpu_specs: GPUSpecs) -> float:
    """Get peak TFLOPS based on data type"""
    dtype_lower = dtype.lower()
    if 'float8' in dtype_lower or 'fp8' in dtype_lower or 'e4m3' in dtype_lower or 'e5m2' in dtype_lower:
        return gpu_specs.fp8_tflops
    else:
        # BF16, FP16, or fallback
        return gpu_specs.fp16_tflops


def build_external_id_map(events: List[Dict]) -> Dict[int, List[Dict]]:
    """Build a map from External ID to events"""
    ext_id_map = defaultdict(list)
    for event in events:
        ext_id = event.get('args', {}).get('External id')
        if ext_id is not None:
            ext_id_map[ext_id].append(event)
    return ext_id_map


def get_total_kernel_time_for_ext_id(events: List[Dict], ext_id: int) -> float:
    """Get total kernel execution time for all kernels with the same External ID.
    This handles splitK and multi-kernel ops correctly."""
    total_time = 0.0
    for event in events:
        if event.get('cat') == 'kernel':
            if event.get('args', {}).get('External id') == ext_id:
                total_time += event.get('dur', 0)
    return total_time


def extract_dimensions_from_cpu_op(event: Dict) -> Optional[Tuple[int, int, int, str]]:
    """
    Extract M, N, K dimensions and dtype from CPU op.
    For sglang::deep_gemm_fp8_fp8_bf16_nt:
      Input Dims: [[M, K], [M, ?], [N, K], [?, ?], [M, N]]
      A = [M, K], B = [N, K].T, C = [M, N]
    """
    args = event.get('args', {})
    input_dims = args.get('Input Dims', [])
    input_types = args.get('Input type', [])
    name = event.get('name', '')
    
    if not input_dims:
        return None
    
    # Get dtype from first input type
    dtype = 'bf16'  # default
    if input_types:
        for t in input_types:
            if t and isinstance(t, str):
                if 'float8' in t.lower() or 'e4m3' in t.lower() or 'e5m2' in t.lower():
                    dtype = 'fp8'
                    break
                elif 'bfloat16' in t.lower():
                    dtype = 'bf16'
                    break
                elif 'float16' in t.lower():
                    dtype = 'fp16'
                    break
    
    # sglang::deep_gemm_fp8_fp8_bf16_nt format:
    # Input Dims: [[M, K], [M, scale_groups], [N, K], [scale_groups_n, scale_groups], [M, N]]
    if 'deep_gemm' in name and len(input_dims) >= 5:
        a_dims = input_dims[0]  # [M, K]
        b_dims = input_dims[2]  # [N, K]
        c_dims = input_dims[4]  # [M, N]
        
        if isinstance(a_dims, list) and len(a_dims) >= 2:
            m = a_dims[0]
            k = a_dims[1]
        else:
            return None
            
        if isinstance(b_dims, list) and len(b_dims) >= 1:
            n = b_dims[0]
        else:
            return None
            
        return (m, n, k, dtype)
    
    # aten::mm format: Input Dims: [[M, K], [K, N]]
    if 'aten::mm' in name and len(input_dims) >= 2:
        a_dims = input_dims[0]
        b_dims = input_dims[1]
        
        if isinstance(a_dims, list) and len(a_dims) >= 2:
            m = a_dims[0]
            k = a_dims[1]
        else:
            return None
            
        if isinstance(b_dims, list) and len(b_dims) >= 2:
            n = b_dims[1]
        else:
            return None
            
        return (m, n, k, dtype)
    
    # aten::linear format: Input Dims: [[M, K], [N, K], bias]
    # A @ B.T = C where A is [M, K] and B is [N, K]
    if 'aten::linear' in name and len(input_dims) >= 2:
        a_dims = input_dims[0]
        b_dims = input_dims[1]
        
        if isinstance(a_dims, list) and len(a_dims) >= 2:
            m = a_dims[0]
            k = a_dims[1]
        else:
            return None
            
        if isinstance(b_dims, list) and len(b_dims) >= 2:
            n = b_dims[0]
        else:
            return None
            
        return (m, n, k, dtype)
    
    # aten::matmul format: Input Dims: [[M, K], [K, N]] or batched
    if 'aten::matmul' in name and len(input_dims) >= 2:
        a_dims = input_dims[0]
        b_dims = input_dims[1]
        
        if isinstance(a_dims, list) and len(a_dims) >= 2:
            m = a_dims[0] if len(a_dims) == 2 else a_dims[-2]
            k = a_dims[1] if len(a_dims) == 2 else a_dims[-1]
        else:
            return None
            
        if isinstance(b_dims, list) and len(b_dims) >= 2:
            n = b_dims[1] if len(b_dims) == 2 else b_dims[-1]
        else:
            return None
            
        return (m, n, k, dtype)
    
    return None


def analyze_matmul_ops(events: List[Dict], gpu_specs: GPUSpecs) -> List[GemmInfo]:
    """Analyze all matmul operations and calculate MFU"""
    gemm_infos = []
    
    # Build External ID to dimensions map from CPU ops
    ext_id_map = build_external_id_map(events)
    
    # Pre-compute total kernel time per external ID
    kernel_times_by_ext_id = defaultdict(float)
    for event in events:
        if event.get('cat') == 'kernel':
            ext_id = event.get('args', {}).get('External id')
            if ext_id is not None:
                kernel_times_by_ext_id[ext_id] += event.get('dur', 0)
    
    # Build map of CPU op dims
    cpu_op_dims = {}
    for event in events:
        if event.get('cat') != 'cpu_op':
            continue
        dims = extract_dimensions_from_cpu_op(event)
        if dims:
            ext_id = event.get('args', {}).get('External id')
            if ext_id is not None:
                cpu_op_dims[ext_id] = dims
    
    # Find GPU kernels (first kernel per ext_id for naming)
    kernel_events = {}
    for e in events:
        if e.get('cat') == 'kernel':
            ext_id = e.get('args', {}).get('External id')
            if ext_id is not None and ext_id not in kernel_events:
                kernel_events[ext_id] = e
    
    # Process CPU ops (prefill/extend traces with External IDs)
    matmul_ops = ['deep_gemm', 'fp8_gemm', 'mm', 'matmul', 'linear', 'gemm']
    processed_ext_ids = set()
    
    for event in events:
        if event.get('cat') != 'cpu_op':
            continue
            
        name = event.get('name', '')
        name_lower = name.lower()
        
        if not any(op in name_lower for op in matmul_ops):
            continue
        
        if name == 'aten::linear' or name == 'aten::matmul':
            continue
            
        dims = extract_dimensions_from_cpu_op(event)
        if dims is None:
            continue
            
        m, n, k, dtype = dims
        if m <= 0 or n <= 0 or k <= 0:
            continue
        
        ext_id = event.get('args', {}).get('External id')
        
        if ext_id in processed_ext_ids:
            continue
        if ext_id is not None:
            processed_ext_ids.add(ext_id)
        
        if ext_id is not None and ext_id in kernel_times_by_ext_id:
            duration_us = kernel_times_by_ext_id[ext_id]
            kernel_event = kernel_events.get(ext_id)
            kernel_name = kernel_event.get('name', '') if kernel_event else name
        else:
            duration_us = event.get('dur', 0)
            kernel_name = name
        
        if duration_us <= 0:
            continue
        
        flops = calculate_gemm_flops(m, n, k)
        peak_tflops = get_dtype_peak_tflops(dtype, gpu_specs)
        duration_s = duration_us / 1e6
        achieved_tflops = (flops / 1e12) / duration_s
        mfu = calculate_mfu(flops, duration_us, peak_tflops)
        
        gemm_info = GemmInfo(
            m=m, n=n, k=k,
            dtype=dtype,
            duration_us=duration_us,
            flops=flops,
            tflops=achieved_tflops,
            mfu=mfu,
            kernel_name=kernel_name,
            external_id=ext_id if ext_id else 0
        )
        gemm_infos.append(gemm_info)
    
    # If no CPU ops found with dims, analyze kernels directly (decode traces)
    if len(gemm_infos) == 0:
        print("No CPU ops with dims found, analyzing kernels directly (decode mode)...")
        
        for event in events:
            if event.get('cat') != 'kernel':
                continue
            
            name = event.get('name', '')
            if 'deep_gemm' not in name.lower():
                continue
            
            duration_us = event.get('dur', 0)
            if duration_us <= 0:
                continue
            
            grid = event.get('args', {}).get('grid', [1, 1, 1])
            parsed = parse_deep_gemm_kernel_full(name, grid)
            
            if parsed is None:
                continue
            
            m, n, k, dtype = parsed
            
            flops = calculate_gemm_flops(m, n, k)
            peak_tflops = get_dtype_peak_tflops(dtype, gpu_specs)
            duration_s = duration_us / 1e6
            achieved_tflops = (flops / 1e12) / duration_s
            mfu = calculate_mfu(flops, duration_us, peak_tflops)
            
            gemm_info = GemmInfo(
                m=m, n=n, k=k,
                dtype=dtype,
                duration_us=duration_us,
                flops=flops,
                tflops=achieved_tflops,
                mfu=mfu,
                kernel_name=name,
                external_id=0
            )
            gemm_infos.append(gemm_info)
    
    return gemm_infos


def extract_tp_rank(pid) -> Optional[str]:
    """Extract TP rank from PID like '[TP06] 6' or '[TP06] 729201'"""
    if pid is None:
        return None
    pid_str = str(pid)
    match = re.search(r'\[TP(\d+)\]', pid_str)
    if match:
        return match.group(1)
    return pid_str  # Fallback to full pid


def add_mfu_to_trace(trace_data: Dict, gpu_specs: GPUSpecs) -> Dict:
    """Add MFU metrics to trace events"""
    events = trace_data.get('traceEvents', [])
    
    # Build External ID to dimensions map from CPU ops
    # Key by (tp_rank, ext_id) to handle multi-GPU traces with different PID formats
    cpu_op_dims = {}
    matmul_ops = ['deep_gemm', 'fp8_gemm', 'sglang::deep_gemm', 'aten::mm', 'aten::linear', 'aten::matmul']
    
    for event in events:
        if event.get('cat') != 'cpu_op':
            continue
        
        name = event.get('name', '')
        if not any(op in name.lower() for op in matmul_ops):
            continue
            
        dims = extract_dimensions_from_cpu_op(event)
        if dims:
            ext_id = event.get('args', {}).get('External id')
            tp_rank = extract_tp_rank(event.get('pid'))
            if ext_id is not None:
                cpu_op_dims[(tp_rank, ext_id)] = dims
    
    # Also map child kernel External IDs to parent CPU op dims
    ext_id_to_parent = {}
    for event in events:
        if event.get('cat') == 'cpu_op':
            ext_id = event.get('args', {}).get('External id')
            tp_rank = extract_tp_rank(event.get('pid'))
            name = event.get('name', '')
            if ext_id is not None and 'sglang::deep_gemm' in name and (tp_rank, ext_id) in cpu_op_dims:
                ext_id_to_parent[(tp_rank, ext_id + 1)] = (tp_rank, ext_id)
    
    # Propagate dimensions to child External IDs
    for child_key, parent_key in ext_id_to_parent.items():
        if parent_key in cpu_op_dims and child_key not in cpu_op_dims:
            cpu_op_dims[child_key] = cpu_op_dims[parent_key]
    
    # Pre-compute total kernel time per (tp_rank, external ID)
    # This handles splitK kernels that share the same ext_id
    kernel_times_by_key = defaultdict(float)
    for event in events:
        if event.get('cat') == 'kernel':
            ext_id = event.get('args', {}).get('External id')
            tp_rank = extract_tp_rank(event.get('pid'))
            if ext_id is not None:
                kernel_times_by_key[(tp_rank, ext_id)] += event.get('dur', 0)
    
    # Process kernel events and add MFU
    modified_count = 0
    
    for event in events:
        if event.get('cat') == 'kernel':
            name = event.get('name', '')
            
            # Check if this is a GEMM kernel
            is_gemm = any(x in name.lower() for x in ['gemm', 'matmul', 'nvjet', 'splitk'])
            if not is_gemm:
                continue
            
            ext_id = event.get('args', {}).get('External id')
            tp_rank = extract_tp_rank(event.get('pid'))
            this_kernel_duration = event.get('dur', 0)
            
            if this_kernel_duration <= 0:
                continue
            
            # Get grid info for M estimation
            grid = event.get('args', {}).get('grid', [1, 1, 1])
            
            dims = None
            key = (tp_rank, ext_id) if ext_id is not None else None
            
            # For kernels with External ID, use this kernel's own duration
            # (don't sum across splitK unless it's actually a splitK operation)
            total_duration_us = this_kernel_duration
            
            # Try to get dimensions from CPU op (if External ID available)
            if key is not None:
                if key in cpu_op_dims:
                    dims = cpu_op_dims[key]
                elif (tp_rank, ext_id - 1) in cpu_op_dims:  # Try parent
                    dims = cpu_op_dims[(tp_rank, ext_id - 1)]
                
                # Only use aggregated time for splitK kernels (nvjet)
                if 'nvjet' in name.lower() or 'splitk' in name.lower():
                    total_duration_us = kernel_times_by_key.get(key, this_kernel_duration)
            
            if dims:
                m, n, k, dtype = dims
            else:
                # Parse dimensions from kernel name (for decode traces without External ID)
                parsed = parse_deep_gemm_kernel_full(name, grid)
                if parsed:
                    m, n, k, dtype = parsed
                else:
                    continue
            
            # Calculate metrics
            flops = calculate_gemm_flops(m, n, k)
            peak_tflops = get_dtype_peak_tflops(dtype, gpu_specs)
            
            if total_duration_us > 0:
                total_duration_s = total_duration_us / 1e6
                achieved_tflops = (flops / 1e12) / total_duration_s
                mfu = calculate_mfu(flops, total_duration_us, peak_tflops)
            else:
                achieved_tflops = 0
                mfu = 0
            
            # Add MFU info to event args
            if 'args' not in event:
                event['args'] = {}
            
            event['args']['MFU (%)'] = round(mfu, 2)
            event['args']['Achieved TFLOPS'] = round(achieved_tflops, 2)
            event['args']['Peak TFLOPS'] = round(peak_tflops, 2)
            event['args']['FLOPs'] = flops
            event['args']['GEMM M'] = m
            event['args']['GEMM N'] = n
            event['args']['GEMM K'] = k
            event['args']['GEMM dtype'] = dtype
            event['args']['Total kernel time (us)'] = round(total_duration_us, 2)
            
            modified_count += 1
        
        elif event.get('cat') == 'cpu_op':
            name = event.get('name', '')
            
            # Add MFU to CPU matmul ops too
            if any(x in name.lower() for x in ['deep_gemm', 'fp8_gemm', 'aten::mm']):
                dims = extract_dimensions_from_cpu_op(event)
                if dims:
                    m, n, k, dtype = dims
                    ext_id = event.get('args', {}).get('External id')
                    tp_rank = extract_tp_rank(event.get('pid'))
                    key = (tp_rank, ext_id) if ext_id is not None else None
                    
                    # Use kernel time if available
                    if key is not None and key in kernel_times_by_key:
                        duration_us = kernel_times_by_key[key]
                    else:
                        duration_us = event.get('dur', 0)
                    
                    if duration_us > 0:
                        flops = calculate_gemm_flops(m, n, k)
                        peak_tflops = get_dtype_peak_tflops(dtype, gpu_specs)
                        duration_s = duration_us / 1e6
                        achieved_tflops = (flops / 1e12) / duration_s
                        mfu = calculate_mfu(flops, duration_us, peak_tflops)
                        
                        if 'args' not in event:
                            event['args'] = {}
                        
                        event['args']['MFU (%)'] = round(mfu, 2)
                        event['args']['Achieved TFLOPS'] = round(achieved_tflops, 2)
                        event['args']['Peak TFLOPS'] = round(peak_tflops, 2)
                        event['args']['FLOPs'] = flops
                        event['args']['GEMM M'] = m
                        event['args']['GEMM N'] = n
                        event['args']['GEMM K'] = k
                        event['args']['GEMM dtype'] = dtype
                        event['args']['Kernel time (us)'] = round(duration_us, 2)
                        
                        modified_count += 1
    
    print(f"Added MFU metrics to {modified_count} events")
    return trace_data


def print_summary(gemm_infos: List[GemmInfo], gpu_specs: GPUSpecs):
    """Print summary statistics"""
    if not gemm_infos:
        print("No GEMM operations found")
        return
    
    total_flops = sum(g.flops for g in gemm_infos)
    total_time_us = sum(g.duration_us for g in gemm_infos)
    total_time_s = total_time_us / 1e6
    
    avg_mfu = sum(g.mfu * g.duration_us for g in gemm_infos) / total_time_us if total_time_us > 0 else 0
    overall_tflops = (total_flops / 1e12) / total_time_s if total_time_s > 0 else 0
    
    print("\n" + "="*70)
    print("GEMM/MatMul MFU Analysis Summary")
    print("="*70)
    print(f"GPU: {gpu_specs.name}")
    print(f"Peak FP8 TFLOPS: {gpu_specs.fp8_tflops}")
    print(f"Peak BF16 TFLOPS: {gpu_specs.fp16_tflops}")
    print("-"*70)
    print(f"Total GEMM operations analyzed: {len(gemm_infos)}")
    print(f"Total GEMM FLOPs: {total_flops / 1e12:.2f} TFLOPs")
    print(f"Total GEMM time: {total_time_us:.2f} us ({total_time_s*1000:.2f} ms)")
    print(f"Average TFLOPS: {overall_tflops:.2f}")
    print(f"Weighted Average MFU: {avg_mfu:.2f}%")
    print("-"*70)
    
    # Group by dtype
    fp8_ops = [g for g in gemm_infos if g.dtype == 'fp8']
    bf16_ops = [g for g in gemm_infos if g.dtype != 'fp8']
    
    if fp8_ops:
        fp8_time = sum(g.duration_us for g in fp8_ops)
        fp8_flops = sum(g.flops for g in fp8_ops)
        fp8_avg_mfu = sum(g.mfu * g.duration_us for g in fp8_ops) / fp8_time if fp8_time > 0 else 0
        print(f"\nFP8 GEMMs: {len(fp8_ops)} ops, {fp8_time:.2f} us, avg MFU: {fp8_avg_mfu:.2f}%")
    
    if bf16_ops:
        bf16_time = sum(g.duration_us for g in bf16_ops)
        bf16_flops = sum(g.flops for g in bf16_ops)
        bf16_avg_mfu = sum(g.mfu * g.duration_us for g in bf16_ops) / bf16_time if bf16_time > 0 else 0
        print(f"BF16 GEMMs: {len(bf16_ops)} ops, {bf16_time:.2f} us, avg MFU: {bf16_avg_mfu:.2f}%")
    
    # Top 10 by MFU
    print("\n" + "-"*70)
    print("Top 10 GEMMs by MFU:")
    sorted_by_mfu = sorted(gemm_infos, key=lambda g: g.mfu, reverse=True)[:10]
    for i, g in enumerate(sorted_by_mfu):
        print(f"  {i+1}. M={g.m}, N={g.n}, K={g.k}, {g.dtype}: "
              f"MFU={g.mfu:.2f}%, {g.tflops:.2f} TFLOPS, {g.duration_us:.2f}us")
    
    # Bottom 10 by MFU (excluding very short ops)
    print("\nBottom 10 GEMMs by MFU (duration > 5us):")
    significant_ops = [g for g in gemm_infos if g.duration_us > 5]
    sorted_by_mfu_asc = sorted(significant_ops, key=lambda g: g.mfu)[:10]
    for i, g in enumerate(sorted_by_mfu_asc):
        print(f"  {i+1}. M={g.m}, N={g.n}, K={g.k}, {g.dtype}: "
              f"MFU={g.mfu:.2f}%, {g.tflops:.2f} TFLOPS, {g.duration_us:.2f}us")
    
    # Top 10 by time
    print("\n" + "-"*70)
    print("Top 10 GEMMs by time:")
    sorted_by_time = sorted(gemm_infos, key=lambda g: g.duration_us, reverse=True)[:10]
    for i, g in enumerate(sorted_by_time):
        print(f"  {i+1}. M={g.m}, N={g.n}, K={g.k}, {g.dtype}: "
              f"{g.duration_us:.2f}us, MFU={g.mfu:.2f}%, {g.tflops:.2f} TFLOPS")
    
    print("="*70)


def load_trace(input_path: str) -> Dict:
    """Load trace file (supports both .json and .json.gz)"""
    path = Path(input_path)
    
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


def save_trace(trace_data: Dict, output_path: str, compress: bool = False):
    """Save trace file"""
    path = Path(output_path)
    
    if compress or path.suffix == '.gz':
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(trace_data, f)
    else:
        with open(path, 'w') as f:
            json.dump(trace_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Add MFU metrics to PyTorch profiler traces'
    )
    parser.add_argument('input_trace', help='Input trace file (.json or .json.gz)')
    parser.add_argument('output_trace', help='Output trace file')
    parser.add_argument('--gpu', default='H200', choices=list(GPU_SPECS.keys()),
                        help='GPU model for peak FLOPS calculation')
    parser.add_argument('--compress', action='store_true',
                        help='Compress output with gzip')
    parser.add_argument('--summary-only', action='store_true',
                        help='Only print summary, do not modify trace')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for decode traces (used when M cannot be inferred)')
    
    args = parser.parse_args()
    
    # Store batch_size globally for decode M estimation
    global DECODE_BATCH_SIZE
    DECODE_BATCH_SIZE = args.batch_size
    
    gpu_specs = GPU_SPECS[args.gpu]
    print(f"Using GPU specs: {gpu_specs.name}")
    print(f"  FP8 Peak: {gpu_specs.fp8_tflops} TFLOPS")
    print(f"  BF16 Peak: {gpu_specs.fp16_tflops} TFLOPS")
    print(f"  Decode batch size hint: {args.batch_size}")
    
    # Load trace
    print(f"\nLoading trace from {args.input_trace}...")
    trace_data = load_trace(args.input_trace)
    
    events = trace_data.get('traceEvents', [])
    print(f"Loaded {len(events)} events")
    
    # Analyze GEMM operations
    print("\nAnalyzing GEMM operations...")
    gemm_infos = analyze_matmul_ops(events, gpu_specs)
    
    # Print summary
    print_summary(gemm_infos, gpu_specs)
    
    if not args.summary_only:
        # Add MFU to trace
        print("\nAdding MFU metrics to trace events...")
        trace_data = add_mfu_to_trace(trace_data, gpu_specs)
        
        # Save output
        print(f"\nSaving modified trace to {args.output_trace}...")
        save_trace(trace_data, args.output_trace, args.compress)
        print("Done!")


if __name__ == '__main__':
    main()
