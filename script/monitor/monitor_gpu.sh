#!/bin/bash
# Unified GPU memory sampler. GPU_SMI: nvidia-smi / xpu-smi / mx-smi / mthreads-gmi.
# Emits: Temp Power Memory-Usage Memory-ALL Util%
SMI_BIN=${GPU_SMI:-nvidia-smi}
sleep 10
echo "Temp Power Memory-Usage Memory-ALL Util% " > "${MEMORY_LOG_FILE}"
while true; do
    echo "$(date '+%Y.%m.%d-%H.%M.%S')" >> "${MEMORY_LOG_FILE}"
    smi_out="$("${SMI_BIN}" 2>/dev/null || true)"
    lines=$(printf '%s\n' "${smi_out}" | grep 'Default' | awk '{print $3,$5,$9,$11,$13}')
    if [ -z "${lines}" ]; then
        lines=$(printf '%s\n' "${smi_out}" | awk '
            /%[ \t]+[0-9]+MiB\([0-9]+MiB\)/ {
                util = "0%"; used = 0; total = 0
                for (i = 1; i <= NF; i++) {
                    f = $i; sub(/^\|/, "", f)
                    if (f ~ /^[0-9]+%$/) util = f
                    if (f ~ /^[0-9]+MiB\([0-9]+MiB\)$/) {
                        x = f; gsub(/MiB/, "", x)
                        split(x, m, /[()]/)
                        used = m[1]; total = m[2]
                    }
                }
                pending = 1; next
            }
            pending {
                temp = "0C"; pwr = "0W"
                for (j = 1; j <= NF; j++) {
                    f = $j; sub(/^\|/, "", f)
                    if (f ~ /^[0-9]+C$/) temp = f
                    if (f ~ /^[0-9]+W$/) pwr = f
                }
                print temp, pwr, used "MiB", total "MiB", util
                pending = 0
            }
        ')
    fi
    if [ -z "${lines}" ]; then
        lines=$(printf '%s\n' "${smi_out}" | awk '
            { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+%$/) util = $i }
            /MiB/ {
                for (i = 1; i <= NF; i++) if ($i == "MiB") {
                    split($(i - 1), m, "/")
                    if (m[2] == "") { m[1] = $(i - 3); m[2] = $(i - 1) }
                    temp = "0C"; pwr = "0W"
                    for (j = 1; j <= NF; j++) {
                        if ($j ~ /^[0-9]+C$/) temp = $j
                        if ($j ~ /^[0-9]+W$/) { pwr = $j; break }
                    }
                    print temp, pwr, m[1] "MiB", m[2] "MiB", (util == "" ? "0%" : util)
                }
            }
        ')
    fi
    if [ -n "${lines}" ]; then
        echo "${lines}" >> "${MEMORY_LOG_FILE}"
    else
        echo "0 0 0MiB 0MiB 0%" >> "${MEMORY_LOG_FILE}"
    fi
    echo -e >> "${MEMORY_LOG_FILE}"
    sleep 5
done
