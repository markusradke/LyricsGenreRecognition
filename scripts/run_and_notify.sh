#!/bin/bash

# Check if a script path is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <path_to_script> [email_recipient] [update_interval_minutes] [max_log_size_mb] [max_restarts]"
    echo "  path_to_script: Path to R script (.R) or Python script (.py)"
    echo "  email_recipient: Email address for notifications (default: mail@markus-radke.de)"
    echo "  update_interval_minutes: Minutes between progress updates (default: 60)"
    echo "  max_restarts: Maximum number of restart attempts (default: 10)"
    echo "  max_log_size_mb: Maximum log file size in MB before rotation (default: 100)"
    exit 1
fi

# Script path is the first argument
SCRIPT_PATH="$1"

# Optional email recipient (default if not provided)
RECIPIENT="${2:-mail@markus-radke.de}"

# Update interval in minutes (default: 60)
UPDATE_INTERVAL="${3:-60}"

# Maximum restart attempts (default: 10)
MAX_RESTARTS="${4:-10}"

# Maximum log size in MB (default: 100)
MAX_LOG_SIZE_MB="${5:-100}"



SUBJECT_SUCCESS="$SCRIPT_TYPE script finished: $(basename "$SCRIPT_PATH")"
SUBJECT_FAIL="$SCRIPT_TYPE script interrupted or failed: $(basename "$SCRIPT_PATH")"
SUBJECT_UPDATE="$SCRIPT_TYPE script progress update: $(basename "$SCRIPT_PATH")"
SUBJECT_RESTART="$SCRIPT_TYPE script restarting after interrupt: $(basename "$SCRIPT_PATH")"
SUBJECT_MAX_RESTARTS="$SCRIPT_TYPE script failed after max restarts: $(basename "$SCRIPT_PATH")"
BODY_SUCCESS="$SCRIPT_TYPE script $(basename "$SCRIPT_PATH") completed successfully on $(hostname) at $(date)."

# Validate script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found at $SCRIPT_PATH"
    exit 1
fi

# Detect script type and set appropriate command
SCRIPT_EXT="${SCRIPT_PATH##*.}"
case "$SCRIPT_EXT" in
    R|r)
        INTERPRETER="Rscript"
        SCRIPT_TYPE="R"
        PROCESS_NAME="Rscript"
        ;;
    py)
        INTERPRETER="python -u" # unbuffered output
        SCRIPT_TYPE="Python"
        PROCESS_NAME="python"
        echo "ATTENTION: PLEASE MAKE SURE THE CORRECT PYTHON ENVIRONMENT IS ACTIVATED BEFORE RUNNING THIS SCRIPT."
        ;;
    *)
        echo "Error: Unsupported script type. Use .R or .py files"
        exit 1
        ;;
esac

echo "Detected $SCRIPT_TYPE script: $(basename "$SCRIPT_PATH")"

# Create temp files for tracking state
LOG_FILE=$(mktemp)
PID_FILE=$(mktemp)
RESTART_FILE=$(mktemp)
RUNNING_FLAG=$(mktemp)
STATS_FILE=$(mktemp)

# Initialize files
echo "1" > "$RUNNING_FLAG"
echo "0" > "$RESTART_FILE"

# Track script start time
SCRIPT_START_TIME=$(date +%s)
echo "$SCRIPT_START_TIME" > "$STATS_FILE"

# PID of the update timer
TIMER_PID=""

# PID of the resource monitor
MONITOR_PID=""

# Store the final exit code
FINAL_EXIT_CODE=0

# Number of rotated log files to keep
MAX_LOG_ARCHIVES=5

# Function to get elapsed time in human-readable format
_format_duration() {
    local total_seconds=$1
    local days=$((total_seconds / 86400))
    local hours=$(((total_seconds % 86400) / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    
    if [ $days -gt 0 ]; then
        echo "${days}d ${hours}h ${minutes}m ${seconds}s"
    elif [ $hours -gt 0 ]; then
        echo "${hours}h ${minutes}m ${seconds}s"
    elif [ $minutes -gt 0 ]; then
        echo "${minutes}m ${seconds}s"
    else
        echo "${seconds}s"
    fi
}

# Function to get system-wide resource usage
_get_system_usage() {
    local cpu_used="N/A"
    local mem_used="N/A"
    
    # Method 1: Use vmstat (locale-independent, most reliable)
    # vmstat output: r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
    # We want the 'id' (idle) column, which is typically column 15
    local vmstat_line=$(vmstat 1 2 2>/dev/null | tail -1)
    if [ -n "$vmstat_line" ]; then
        local vmstat_idle=$(echo "$vmstat_line" | awk '{print $15}')
        # Validate it's a number between 0 and 100
        if [ -n "$vmstat_idle" ] && echo "$vmstat_idle" | grep -qE '^[0-9]+$' && [ "$vmstat_idle" -ge 0 ] && [ "$vmstat_idle" -le 100 ]; then
            cpu_used=$(awk "BEGIN {printf \"%.1f\", 100 - $vmstat_idle}")
        fi
    fi
    
    # Method 2: Parse /proc/stat (completely locale-independent)
    if [ "$cpu_used" = "N/A" ] && [ -r /proc/stat ]; then
        local cpu_line1=$(grep "^cpu " /proc/stat)
        sleep 0.5
        local cpu_line2=$(grep "^cpu " /proc/stat)
        
        if [ -n "$cpu_line1" ] && [ -n "$cpu_line2" ]; then
            # Extract values: user nice system idle iowait irq softirq
            local vals1=($(echo $cpu_line1 | awk '{print $2,$3,$4,$5,$6,$7,$8}'))
            local vals2=($(echo $cpu_line2 | awk '{print $2,$3,$4,$5,$6,$7,$8}'))
            
            # Calculate deltas
            local user_d=$((${vals2[0]} - ${vals1[0]}))
            local nice_d=$((${vals2[1]} - ${vals1[1]}))
            local system_d=$((${vals2[2]} - ${vals1[2]}))
            local idle_d=$((${vals2[3]} - ${vals1[3]}))
            local iowait_d=$((${vals2[4]} - ${vals1[4]}))
            local irq_d=$((${vals2[5]} - ${vals1[5]}))
            local softirq_d=$((${vals2[6]} - ${vals1[6]}))
            
            local total_d=$((user_d + nice_d + system_d + idle_d + iowait_d + irq_d + softirq_d))
            
            if [ $total_d -gt 0 ]; then
                cpu_used=$(awk "BEGIN {printf \"%.1f\", 100 - (($idle_d / $total_d) * 100)}")
            fi
        fi
    fi
    
    # Get memory usage using free command (works with any locale - use numeric columns)
    # free -m output has consistent column positions regardless of headers
    # Line 2 is always memory stats: column 2=total, column 3=used
    local mem_line=$(LC_ALL=C free -m 2>/dev/null | sed -n '2p')
    if [ -n "$mem_line" ]; then
        local mem_total=$(echo "$mem_line" | awk '{print $2}')
        local mem_used_val=$(echo "$mem_line" | awk '{print $3}')
        
        if [ -n "$mem_total" ] && [ "$mem_total" -gt 0 ] 2>/dev/null && [ -n "$mem_used_val" ]; then
            mem_used=$(awk "BEGIN {printf \"%.1f\", ($mem_used_val / $mem_total) * 100}")
        fi
    fi
    
    # Fallback for memory: parse /proc/meminfo (locale-independent)
    if [ "$mem_used" = "N/A" ] && [ -r /proc/meminfo ]; then
        local mem_total_kb=$(grep "^MemTotal:" /proc/meminfo | awk '{print $2}')
        local mem_avail_kb=$(grep "^MemAvailable:" /proc/meminfo | awk '{print $2}')
        
        if [ -n "$mem_total_kb" ] && [ "$mem_total_kb" -gt 0 ] && [ -n "$mem_avail_kb" ]; then
            local mem_used_kb=$((mem_total_kb - mem_avail_kb))
            mem_used=$(awk "BEGIN {printf \"%.1f\", ($mem_used_kb / $mem_total_kb) * 100}")
        fi
    fi
    
    echo "$cpu_used $mem_used"
}

# Function to get current resource usage for specific process
_get_resource_usage() {
    local pid=$1
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        # Try to find all child Rscript processes and sum their usage
        local child_pids=$(pgrep -P "$pid" 2>/dev/null)
        local all_pids="$pid"
        if [ -n "$child_pids" ]; then
            all_pids="$pid $child_pids"
        fi
        
        # Get CPU and memory usage using ps for all relevant PIDs
        local stats=$(ps -p $(echo $all_pids | tr ' ' ',') -o %cpu,%mem,rss --no-headers 2>/dev/null | awk '{cpu+=$1; mem+=$2; rss+=$3} END {printf "%.1f %.1f %d", cpu, mem, rss}')
        if [ -n "$stats" ]; then
            echo "$stats"
        else
            echo "N/A N/A N/A"
        fi
    else
        echo "N/A N/A N/A"
    fi
}

# Function to rotate log file
_rotate_log() {
    local log_size_bytes=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
    local log_size_mb=$((log_size_bytes / 1048576))
    
    if [ $log_size_mb -ge $MAX_LOG_SIZE_MB ]; then
        echo "===== Log rotation at $(date) (size: ${log_size_mb}MB) =====" >> "$LOG_FILE"
        
        # Rotate old archives
        for i in $(seq $((MAX_LOG_ARCHIVES - 1)) -1 1); do
            if [ -f "${LOG_FILE}.$i" ]; then
                mv "${LOG_FILE}.$i" "${LOG_FILE}.$((i + 1))"
            fi
        done
        
        # Move current log to .1 and start fresh
        cp "$LOG_FILE" "${LOG_FILE}.1"
        echo "===== New log file started at $(date) =====" > "$LOG_FILE"
        echo "===== Previous log archived to ${LOG_FILE}.1 =====" >> "$LOG_FILE"
        
        # Remove oldest archive if exceeding limit
        if [ -f "${LOG_FILE}.$((MAX_LOG_ARCHIVES + 1))" ]; then
            rm -f "${LOG_FILE}.$((MAX_LOG_ARCHIVES + 1))"
        fi
        
        return 0  # Indicate rotation occurred
    fi
    return 1  # No rotation needed
}

# Monitor resource usage periodically
_monitor_resources() {
    while [ "$(cat "$RUNNING_FLAG" 2>/dev/null)" = "1" ]; do
        sleep 30  # Check every 30 seconds
        
        # Get system-wide usage
        local sys_usage=$(_get_system_usage)
        local sys_cpu=$(echo "$sys_usage" | awk '{print $1}')
        local sys_mem=$(echo "$sys_usage" | awk '{print $2}')
        
        # Log system-wide resource usage
        echo "$(date +%s) $sys_cpu $sys_mem" >> "${STATS_FILE}.system"
        
        # Check if log rotation is needed
        _rotate_log
    done
}

# Send periodic updates
_send_update() {
    local next_update=$((UPDATE_INTERVAL * 60))
    local elapsed=0
    
    while [ "$(cat "$RUNNING_FLAG" 2>/dev/null)" = "1" ]; do
        sleep 10
        elapsed=$((elapsed + 10))
        
        if [ $elapsed -ge $next_update ]; then
            local current_pid=$(cat "$PID_FILE" 2>/dev/null)
            local restart_count=$(cat "$RESTART_FILE" 2>/dev/null)
            
            if [ "$(cat "$RUNNING_FLAG" 2>/dev/null)" = "1" ] && [ -n "$current_pid" ] && kill -0 "$current_pid" 2>/dev/null; then
                # Calculate runtime statistics
                local script_start=$(cat "$STATS_FILE" 2>/dev/null)
                local current_time=$(date +%s)
                local total_runtime=$((current_time - script_start))
                local runtime_formatted=$(_format_duration $total_runtime)
                
                # Get current system-wide usage
                local sys_usage=$(_get_system_usage)
                local sys_cpu=$(echo "$sys_usage" | awk '{print $1}')
                local sys_mem=$(echo "$sys_usage" | awk '{print $2}')
                
                # Calculate average system-wide resource usage
                local avg_sys_cpu="N/A"
                local avg_sys_mem="N/A"
                local max_sys_mem="N/A"
                if [ -f "${STATS_FILE}.system" ]; then
                    avg_sys_cpu=$(awk '$2 != "N/A" {sum+=$2; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}' "${STATS_FILE}.system")
                    avg_sys_mem=$(awk '$3 != "N/A" {sum+=$3; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}' "${STATS_FILE}.system")
                    max_sys_mem=$(awk 'BEGIN{max=0} $3 != "N/A" && $3>max {max=$3} END {if(max>0) printf "%.1f", max; else print "N/A"}' "${STATS_FILE}.system")
                fi
                
                BODY_UPDATE="Progress update for $(basename "$SCRIPT_PATH") on $(hostname) at $(date).
Restart count: $restart_count
Total runtime: $runtime_formatted

--- System-Wide Resource Usage ---
Current: CPU ${sys_cpu}%, Memory ${sys_mem}%
Average: CPU ${avg_sys_cpu}%, Memory ${avg_sys_mem}%
Peak Memory: ${max_sys_mem}%

--- Last 100 lines of output ---
$(tail -n 100 "$LOG_FILE")"
                echo "$BODY_UPDATE" | mail -r markus@hendrix.ak.tu-berlin.de -s "$SUBJECT_UPDATE" "$RECIPIENT"
                echo "Update email sent at $(date)" >&2
            fi
            next_update=$((next_update + UPDATE_INTERVAL * 60))
        fi
    done
}

# cleanup function called on exit
_on_exit() {
    echo "0" > "$RUNNING_FLAG"
    
    # Kill the update timer if running
    if [ -n "$TIMER_PID" ]; then
        kill "$TIMER_PID" 2>/dev/null
        wait "$TIMER_PID" 2>/dev/null
    fi
    
    # Kill the resource monitor if running
    if [ -n "$MONITOR_PID" ]; then
        kill "$MONITOR_PID" 2>/dev/null
        wait "$MONITOR_PID" 2>/dev/null
    fi
    
    # Calculate final statistics
    local script_start=$(cat "$STATS_FILE" 2>/dev/null)
    local script_end=$(date +%s)
    local total_runtime=$((script_end - script_start))
    local runtime_formatted=$(_format_duration $total_runtime)
    
    # Read final restart count
    local final_restart_count=$(cat "$RESTART_FILE" 2>/dev/null)
    
    # Calculate system-wide resource statistics
    local avg_sys_cpu="N/A"
    local avg_sys_mem="N/A"
    local peak_sys_mem="N/A"
    if [ -f "${STATS_FILE}.system" ]; then
        avg_sys_cpu=$(awk '$2 != "N/A" {sum+=$2; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}' "${STATS_FILE}.system")
        avg_sys_mem=$(awk '$3 != "N/A" {sum+=$3; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}' "${STATS_FILE}.system")
        peak_sys_mem=$(awk 'BEGIN{max=0} $3 != "N/A" && $3>max {max=$3} END {if(max>0) printf "%.1f", max; else print "N/A"}' "${STATS_FILE}.system")
    fi
    
    # Send final notification using stored exit code
    if [ $FINAL_EXIT_CODE -eq 0 ]; then
        BODY_SUCCESS="$SCRIPT_TYPE script $(basename "$SCRIPT_PATH") completed successfully on $(hostname) at $(date).
Total runtime: $runtime_formatted
Total restarts: $final_restart_count

--- System-Wide Resource Statistics ---
Average CPU: ${avg_sys_cpu}%
Average Memory: ${avg_sys_mem}%
Peak Memory: ${peak_sys_mem}%

--- Last 100 lines of output ---
$(tail -n 100 "$LOG_FILE")"
        echo "$BODY_SUCCESS" | mail -r markus@hendrix.ak.tu-berlin.de -s "$SUBJECT_SUCCESS" "$RECIPIENT"
    else
        BODY_FAIL="$SCRIPT_TYPE script $(basename "$SCRIPT_PATH") stopped (exit code $FINAL_EXIT_CODE) on $(hostname) at $(date).
Total runtime: $runtime_formatted
Total restarts: $final_restart_count

--- System-Wide Resource Statistics ---
Average CPU: ${avg_sys_cpu}%
Average Memory: ${avg_sys_mem}%
Peak Memory: ${peak_sys_mem}%

--- Last 100 lines of output ---
$(tail -n 100 "$LOG_FILE")"
        echo "$BODY_FAIL" | mail -r markus@hendrix.ak.tu-berlin.de -s "$SUBJECT_FAIL" "$RECIPIENT"
    fi
    
    # Cleanup temp files
    rm -f "$LOG_FILE" "$LOG_FILE".* "$PID_FILE" "$RESTART_FILE" "$RUNNING_FLAG" "$STATS_FILE" "${STATS_FILE}.system"
}
trap _on_exit EXIT

# Start the update timer in background
_send_update &
TIMER_PID=$!

# Start the resource monitor in background
_monitor_resources &
MONITOR_PID=$!

# Counter for restart attempts
RESTART_COUNT=0

# Log script start
echo "===== Script started at $(date) =====" | tee -a "$LOG_FILE"
echo "===== Command: $0 $@ =====" | tee -a "$LOG_FILE"
echo "===== Update interval: ${UPDATE_INTERVAL} minutes =====" | tee -a "$LOG_FILE"
echo "===== Max log size: ${MAX_LOG_SIZE_MB}MB =====" | tee -a "$LOG_FILE"
echo "===== Max restarts: ${MAX_RESTARTS} =====" | tee -a "$LOG_FILE"

# Main loop to run and restart script
while [ "$(cat "$RUNNING_FLAG" 2>/dev/null)" = "1" ]; do
    # Log restart attempt with timestamp
    ATTEMPT_START=$(date +%s)
    echo "===== Starting $SCRIPT_TYPE script (attempt $((RESTART_COUNT + 1))) at $(date) =====" | tee -a "$LOG_FILE"
    
    # Run the script with tee to show output AND save to log
    set -o pipefail
    $INTERPRETER "$SCRIPT_PATH" 2>&1 | tee -a "$LOG_FILE" &
    TEE_PID=$!
    
    # Wait a bit and find the actual script process
    sleep 0.5
    
    # Try multiple methods to find the script PID
    SCRIPT_PID=""
    
    # Method 1: Look for child of tee
    SCRIPT_PID=$(pgrep -P $TEE_PID 2>/dev/null | head -1)
    
    # Method 2: Look for process with our script name
    if [ -z "$SCRIPT_PID" ]; then
        SCRIPT_PID=$(pgrep -f "$PROCESS_NAME.*$(basename "$SCRIPT_PATH")" 2>/dev/null | head -1)
    fi
    
    # Method 3: Use the tee PID as fallback
    if [ -z "$SCRIPT_PID" ]; then
        SCRIPT_PID=$TEE_PID
    fi
    
    echo "$SCRIPT_PID" > "$PID_FILE"
    echo "Monitoring PID: $SCRIPT_PID (tee PID: $TEE_PID)" >&2
    
    # Wait for the pipe to complete
    wait $TEE_PID
    EXIT_CODE=$?
    set +o pipefail
    
    # Clear PID file
    echo "" > "$PID_FILE"
    
    # Calculate attempt duration
    ATTEMPT_END=$(date +%s)
    ATTEMPT_DURATION=$((ATTEMPT_END - ATTEMPT_START))
    ATTEMPT_DURATION_FORMATTED=$(_format_duration $ATTEMPT_DURATION)
    
    echo "===== $SCRIPT_TYPE script finished with exit code $EXIT_CODE at $(date) (duration: $ATTEMPT_DURATION_FORMATTED) =====" | tee -a "$LOG_FILE"
    
    # If script completed successfully, exit the loop
    if [ $EXIT_CODE -eq 0 ]; then
        echo "===== $SCRIPT_TYPE script completed successfully at $(date) ====="
        FINAL_EXIT_CODE=0
        echo "0" > "$RUNNING_FLAG"
        exit 0
    fi
    
    # Script was interrupted or failed
    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "$RESTART_COUNT" > "$RESTART_FILE"
    echo "===== $SCRIPT_TYPE script interrupted (exit code $EXIT_CODE) at $(date) =====" | tee -a "$LOG_FILE"
    
    # Check if max restarts exceeded
    if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
        # Calculate statistics for max restarts notification
        local script_start=$(cat "$STATS_FILE" 2>/dev/null)
        local current_time=$(date +%s)
        local total_runtime=$((current_time - script_start))
        local runtime_formatted=$(_format_duration $total_runtime)
        
        # Send max restarts notification
        BODY_MAX_RESTARTS="$SCRIPT_TYPE script $(basename "$SCRIPT_PATH") failed after $MAX_RESTARTS restart attempts on $(hostname) at $(date).
Last exit code: $EXIT_CODE
Total runtime: $runtime_formatted

--- Last 100 lines of output ---
$(tail -n 100 "$LOG_FILE")"
        echo "$BODY_MAX_RESTARTS" | mail -r markus@hendrix.ak.tu-berlin.de -s "$SUBJECT_MAX_RESTARTS" "$RECIPIENT"
        
        echo "===== Maximum restart attempts ($MAX_RESTARTS) exceeded. Stopping. =====" | tee -a "$LOG_FILE"
        FINAL_EXIT_CODE=$EXIT_CODE
        echo "0" > "$RUNNING_FLAG"
        exit 1
    fi
    
    # Calculate statistics for restart notification
    local script_start=$(cat "$STATS_FILE" 2>/dev/null)
    local current_time=$(date +%s)
    local total_runtime=$((current_time - script_start))
    local runtime_formatted=$(_format_duration $total_runtime)
    
    # Send restart notification
    BODY_RESTART="$SCRIPT_TYPE script $(basename "$SCRIPT_PATH") was interrupted on $(hostname) at $(date).
Exit code: $EXIT_CODE
Attempt duration: $ATTEMPT_DURATION_FORMATTED
Total runtime so far: $runtime_formatted
Restarting automatically (restart #$RESTART_COUNT of $MAX_RESTARTS)...

--- Last 100 lines of output ---
$(tail -n 100 "$LOG_FILE")"
    echo "$BODY_RESTART" | mail -r markus@hendrix.ak.tu-berlin.de -s "$SUBJECT_RESTART" "$RECIPIENT"
    
    # Store the exit code in case we exit the loop
    FINAL_EXIT_CODE=$EXIT_CODE
    
    # Brief pause before restart
    echo "===== Waiting 5 seconds before restart... =====" | tee -a "$LOG_FILE"
    sleep 5
done
