#!/bin/bash

LOG_FILE=${1:-"/workspace/code/OptiCo/output/megatron_log/1.txt"}
CURRENT_ERROR_LOG_FILE=${2:-"/workspace/code/OptiCo/output/megatron_error.txt"}

{
    grep -i "error" "$LOG_FILE" | \
        sed -E 's/\[rank[0-9]+\]: //g' | \
        sort -u


    echo "--- Tracebacks ---"

    awk '
    BEGIN {
        in_traceback = 0
        tb_buffer = ""
    }

    { gsub(/\[rank[0-9]+\]: /, "") }

    /[Tt]raceback/ {
        if (in_traceback) {
            if (found_error) {
                standardize_and_print()
            }
        }
        in_traceback = 1
        tb_buffer = $0
        found_error = 0
        next
    }

    in_traceback {
        tb_buffer = tb_buffer "\n" $0
        if (/.*[Ee]rror/) {
            found_error = 1
        }
    }

    in_traceback && !/^$/ && /^[^[:space:]]/ && !/[Tt]raceback/ {
        if (found_error) {
            standardize_and_print()
        }
        in_traceback = 0
        tb_buffer = ""
        found_error = 0
        next
    }

    END {
        if (in_traceback && found_error) {
            standardize_and_print()
        }
    }

    function standardize_and_print() {
        gsub_buffer = tb_buffer

        gsub(/GPU [0-9]+/, "GPU X", gsub_buffer)

        gsub(/Process [0-9]+/, "Process X", gsub_buffer)
        gsub(/process [0-9]+/, "process X", gsub_buffer)

        gsub(/[0-9]+\.[0-9]+ (GiB|MiB)/, "X.XX UNIT", gsub_buffer)

        gsub(/[0-9]+ (GiB|MiB)/, "X UNIT", gsub_buffer)

        gsub(/\^[~\^]+$/, "^...", gsub_buffer)

        # gsub(/\/[^ \n"]+\/[^ \n"]+\//, "/path/to/", gsub_buffer)

        gsub(/[ \t]+$/, "", gsub_buffer)
        gsub(/^[ \t]+/, "", gsub_buffer)

        if (!seen[gsub_buffer]++) {
            print gsub_buffer
        }
    }
    ' "$LOG_FILE"

} >> "$CURRENT_ERROR_LOG_FILE"

echo "extract error finish"