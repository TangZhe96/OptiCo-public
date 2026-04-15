#!/bin/bash
sleep 10
echo "Temp Power Memory-Usage Memory-ALL Util% " > ${MEMORY_LOG_FILE}
while true; do
    echo "$(date -d '+%Y.%m.%d-%H.%M.%S')" >> ${MEMORY_LOG_FILE}
    echo "$(nvidia-smi | grep 'Default'| awk '{print $3,$5,$9,$11,$13}')" >> ${MEMORY_LOG_FILE}
    echo -e >> ${MEMORY_LOG_FILE}
    sleep 5
done
