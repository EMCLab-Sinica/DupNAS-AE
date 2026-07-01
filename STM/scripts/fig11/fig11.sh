set -euo pipefail

case "$1" in
    tflm_f7)
        bash utils/run_tflm_f7.sh scripts/fig11/tflite.txt ram,latency
        ;;
    tflm_h7)
        bash utils/run_tflm_h7.sh scripts/fig11/tflite.txt ram,latency
        ;;
    cubeai_f7)
        bash utils/run_cubeai_f7.sh scripts/fig11/onnx.txt ram,latency
        ;;
    cubeai_h7)
        bash utils/run_cubeai_h7.sh scripts/fig11/onnx.txt ram,latency
        ;;
    *)
        echo "unknown target: $1"
        exit 1
        ;;
esac
