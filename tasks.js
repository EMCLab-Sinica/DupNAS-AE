window.workflowTasks = {
  fig6: {
    label: "Sec. 5.B (Fig. 6: Peak Memory)",
    option: [
      { value: "shufflenet-vm96", label: "shufflenet-vm96 (Est. 1 hr 40 mins)" },
      { value: "shufflenet-vm128", label: "shufflenet-vm128 (Est. 1 hr 40 mins)" },
      { value: "shufflenet-vm256", label: "shufflenet-vm256 (Est. 1 hr 40 mins)" },
      { value: "mobilenet-vm96", label: "mobilenet-vm96 (Est. 2 hrs 20 mins)" },
      { value: "mobilenet-vm128", label: "mobilenet-vm128 (Est. 2 hrs 20 mins)" },
      { value: "mobilenet-vm256", label: "mobilenet-vm256 (Est. 2 hrs 20 mins)" },
      { value: "inception-vm96", label: "inception-vm96 (Est. 2 hrs 20 mins)" },
      { value: "inception-vm128", label: "inception-vm128 (Est. 2 hrs 20 mins)" },
      { value: "inception-vm256", label: "inception-vm256 (Est. 2 hrs 20 mins)" }
    ]
  },
  fig7: {
    label: "Sec. 5.C (Fig. 7: Model Accuracy)",
    option: [
      { value: "shuffle", label: "shufflenet (Est. 1 mins)" },
      { value: "mbv2", label: "mobilenet (Est. 1 mins)" },
      { value: "incept", label: "inception (Est. 1 mins)" }
    ]
  },
  fig8: {
    label: "Sec. 5.D (Fig. 8: Inference Latency)",
    option: [
      { value: "shufflenet-vm96", label: "shufflenet-vm96 (Est. 2 mins)" },
      { value: "shufflenet-vm128", label: "shufflenet-vm128 (Est. 2 mins)" },
      { value: "mobilenet-vm96", label: "mobilenet-vm96 (Est. 2 mins)" },
      { value: "mobilenet-vm128", label: "mobilenet-vm128 (Est. 2 mins)" },
      { value: "inception-vm96", label: "inception-vm96 (Est. 2 mins)" },
      { value: "inception-vm128", label: "inception-vm128 (Est. 2 mins)" }
    ]
  },
  fig9: {
    label: "Sec. 5.E (Fig. 9: Search Time)",
    option: [
      { value: "shufflenet-samples", label: "shufflenet-samples (Est. 10 mins)" },
      { value: "mobilenet-samples", label: "mobilenet-samples (Est. 10 mins)" },
      { value: "inception-samples", label: "inception-samples (Est. 10 mins)" },
      { value: "shufflenet-full", label: "shufflenet-full (Est.  5 hrs)" },
      { value: "mobilenet-full", label: "mobilenet-full (Est.  more than 6 hrs)" },
      { value: "inception-full", label: "inception-full (Est.  more than 6 hrs)" }
    ]
  },
  fig10: {
    label: "Sec. 5.F (Fig. 10: Heuristic Effectiveness)",
    option: [
      { value: "neither-shufflenet-vm96", label: "neither-shufflenet-vm96 (Est. 1 hr 20 mins)" },
      { value: "neither-shufflenet-vm128", label: "neither-shufflenet-vm128 (Est. 1 hr 10 mins)" },
      { value: "neither-shufflenet-vm256", label: "neither-shufflenet-vm256 (Est. 30 mins)" },
      { value: "neither-mobilenet-vm96", label: "neither-mobilenet-vm96 (Est. more than 6 hrs)" },
      { value: "neither-mobilenet-vm128", label: "neither-mobilenet-vm128 (Est. more than 6 hrs)" },
      { value: "neither-mobilenet-vm256", label: "neither-mobilenet-vm256 (Est. 1 hr 20 mins)" },
      { value: "neither-inception-vm96", label: "neither-inception-vm96 (Est. more than 6 hrs)" },
      { value: "neither-inception-vm128", label: "neither-inception-vm128 (Est. 5 hrs 30 mins)" },
      { value: "neither-inception-vm256", label: "neither-inception-vm256 (Est. 3 hrs 30 mins)" },
      { value: "BPonly-shufflenet-vm96", label: "BPonly-shufflenet-vm96 (Est. 50 mins)" },
      { value: "BPonly-shufflenet-vm128", label: "BPonly-shufflenet-vm128 (Est. 40 mins)" },
      { value: "BPonly-shufflenet-vm256", label: "BPonly-shufflenet-vm256 (Est. 20 mins)" },
      { value: "BPonly-mobilenet-vm96", label: "BPonly-mobilenet-vm96 (Est. 3 hrs 10 mins)" },
      { value: "BPonly-mobilenet-vm128", label: "BPonly-mobilenet-vm128 (Est. 2 hrs)" },
      { value: "BPonly-mobilenet-vm256", label: "BPonly-mobilenet-vm256 (Est. 30 mins)" },
      { value: "BPonly-inception-vm96", label: "BPonly-inception-vm96 (Est. 2 hrs 40 mins)" },
      { value: "BPonly-inception-vm128", label: "BPonly-inception-vm128 (Est. 1 hr 50 mins)" },
      { value: "BPonly-inception-vm256", label: "BPonly-inception-vm256 (Est. 1 hr)" },
      { value: "PConly-shufflenet-vm96", label: "PConly-shufflenet-vm96 (Est. 1 hr)" },
      { value: "PConly-shufflenet-vm128", label: "PConly-shufflenet-vm128 (Est. 50 mins)" },
      { value: "PConly-shufflenet-vm256", label: "PConly-shufflenet-vm256 (Est. 20 mins)" },
      { value: "PConly-mobilenet-vm96", label: "PConly-mobilenet-vm96 (Est. more than 6 hrs)" },
      { value: "PConly-mobilenet-vm128", label: "PConly-mobilenet-vm128 (Est. 4 hrs 30 mins)" },
      { value: "PConly-mobilenet-vm256", label: "PConly-mobilenet-vm256 (Est. 1 hr)" },
      { value: "PConly-inception-vm96", label: "PConly-inception-vm96 (Est. 4 hrs 50 mins)" },
      { value: "PConly-inception-vm128", label: "PConly-inception-vm128 (Est. 3 hrs 30 mins)" },
      { value: "PConly-inception-vm256", label: "PConly-inception-vm256 (Est. 2 hrs)" }
    ]
  },
  fig11: {
    label: "Sec. 5.G (Fig. 11: Solution Portability)",
    option: [
      { value: "tflm-f7", label: "F7-TFLM (Est. 10 mins)" },
      { value: "cubeai-f7", label: "F7-Cube.AI (Est. 1 hr 10 mins)" },
      { value: "tflm-h7", label: "H7-TFLM (Est. 10 mins)" },
      { value: "cubeai-h7", label: "H7-Cube.AI (Est. 20 mins)" }
    ]
  },
  auto_dupnas: {
    label: "Auto-run DupNAS stages",
    option: [
      { value: "stage1+2", label: "Stage 1+2: search-space optimization + supernet training (more than 1 day)" },
      { value: "stage3+4", label: "Stage 3+4: evolutionary search + fine-tuning (more than 8 hrs)" },
      { value: "full-stage", label: "Full run (Stages 1–4 sequentially)" }
    ]
  }
  
};
