import sys
import json
from pathlib import Path


results = []
results_dir = Path(sys.argv[1])
for result_path in results_dir.rglob(f'*.json'):
    with open(result_path) as f:
        result = json.load(f)
    results.append(result)

is_multinode = results[0]['is_multinode']

if is_multinode:
    results.sort(key=lambda r: (r['model'], r['hw'], r['framework'], r['precision'], r['isl'], r['osl'], r['prefill_tp'], r['prefill_ep'], r['decode_tp'], r['decode_ep'], r['conc']))

    summary_header = '''\
| Model | Hardware | Framework | Precision | ISL | OSL | Prefill TP | Prefill EP | Prefill DP Attn | Prefill Workers | Prefill GPUs | Decode TP | Decode EP | Decode DP Attn | Decode Workers | Decode GPUs | Conc | TTFT (ms) | TPOT (ms) | Interactivity (tok/s/user) | E2EL (s) | TPUT per GPU | Output TPUT per GPU | Input TPUT per GPU |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |\
'''
    print(summary_header)

    for result in results:
        print(
            f"| {result['model']} "
            f"| {result['hw'].upper()} "
            f"| {result['framework'].upper()} "
            f"| {result['precision'].upper()} "
            f"| {result['isl']} "
            f"| {result['osl']} "
            f"| {result['prefill_tp']} "
            f"| {result['prefill_ep']} "
            f"| {result['prefill_dp_attention']} "
            f"| {result['prefill_num_workers']} "
            f"| {result['num_prefill_gpu']} "
            f"| {result['decode_tp']} "
            f"| {result['decode_ep']} "
            f"| {result['decode_dp_attention']} "
            f"| {result['decode_num_workers']} "
            f"| {result['num_decode_gpu']} "
            f"| {result['conc']} "
            f"| {(result['median_ttft'] * 1000):.4f} "
            f"| {(result['median_tpot'] * 1000):.4f} "
            f"| {result['median_intvty']:.4f} "
            f"| {result['median_e2el']:.4f} "
            f"| {result['tput_per_gpu']:.4f} "
            f"| {result['output_tput_per_gpu']:.4f} "
            f"| {result['input_tput_per_gpu']:.4f} |"
        )
else:
    results.sort(key=lambda r: (r['model'], r['hw'], r['framework'], r['precision'], r['isl'], r['osl'], r['tp'], r['ep'], r['conc']))

    summary_header = '''\
| Model | Hardware | Framework | Precision | ISL | OSL | TP | EP | DP Attention | Conc | TTFT (ms) | TPOT (ms) | Interactivity (tok/s/user) | E2EL (s) | TPUT per GPU | Output TPUT per GPU | Input TPUT per GPU |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |\
'''
    print(summary_header)

    for result in results:
        print(
            f"| {result['model']} "
            f"| {result['hw'].upper()} "
            f"| {result['framework'].upper()} "
            f"| {result['precision'].upper()} "
            f"| {result['isl']} "
            f"| {result['osl']} "
            f"| {result['tp']} "
            f"| {result['ep']} "
            f"| {result['dp_attention']} "
            f"| {result['conc']} "
            f"| {(result['median_ttft'] * 1000):.4f} "
            f"| {(result['median_tpot'] * 1000):.4f} "
            f"| {result['median_intvty']:.4f} "
            f"| {result['median_e2el']:.4f} "
            f"| {result['tput_per_gpu']:.4f} "
            f"| {result['output_tput_per_gpu']:.4f} "
            f"| {result['input_tput_per_gpu']:.4f} |"
        )
