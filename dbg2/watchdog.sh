while true; do
  {
    echo "=== $(date -u +%H:%M:%S) ==="
    echo "total_procs=$(ps aux | wc -l) python_procs=$(ps aux | grep -c python)"
    ps aux --sort=-%cpu | head -3
  } >> /dbg2/watchdog.log
  sleep 20
done
