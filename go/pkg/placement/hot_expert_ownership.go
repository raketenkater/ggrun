package placement

import (
	"fmt"
	"math"
	"strconv"
)

// emittedLayerDeviceAssignments mirrors the backend's full layer-mode ownership:
// argv is rounded to two decimals, parsed as float32 and compared with strict
// upper boundaries. Unknown/default splits are not allocation evidence.
func emittedLayerDeviceAssignments(split []float64, numLayers int) ([]int, int, bool) {
	if numLayers <= 0 {
		return nil, -1, false
	}
	values := make([]float32, len(split))
	var sum float32
	for i, v := range split {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, -1, false
		}
		// The backend sees the formatted argv, not the planner's extra precision.
		emitted, err := strconv.ParseFloat(fmt.Sprintf("%.2f", v), 32)
		if err != nil {
			return nil, -1, false
		}
		values[i] = float32(emitted)
		sum += values[i]
	}
	if sum <= 0 || math.IsNaN(float64(sum)) || math.IsInf(float64(sum), 0) {
		return nil, -1, false
	}
	cum := make([]float32, len(values))
	var running float32
	for i, v := range values {
		running += v
		cum[i] = running / sum
	}
	layerDevices := make([]int, numLayers)
	outputDev := len(values) - 1
	slots := numLayers + 1
	for slot := 0; slot < slots; slot++ {
		f := float32(slot) / float32(slots)
		dev := len(values) - 1
		for i, boundary := range cum {
			if boundary > f {
				dev = i
				break
			}
		}
		if slot == numLayers {
			outputDev = dev
		} else {
			layerDevices[slot] = dev
		}
	}
	return layerDevices, outputDev, true
}
