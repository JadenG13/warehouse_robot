import re
from collections import defaultdict
import math  # For isnan, isinf if needed

# Uncomment below if you want to use pandas for tabular display
# import pandas as pd


def parse_log_line(line):
    """Parses a single [PERF_LOG] line into a dictionary."""
    if not line.startswith("[PERF_LOG]"):
        return None

    original_line_for_debug = line.strip()
    line = original_line_for_debug.replace("[PERF_LOG] ", "")
    # Regex to split by comma, but not commas inside single quotes
    parts = re.split(r",\s*(?=(?:[^']*'[^']*')*[^']*$)", line)
    log_data = {}

    # Keys that should always be parsed as numbers
    numeric_keys = [
        "timestamp", "duration_sec", "astar_duration_sec",
        "planning_duration_sec", "exec_duration_sec", "wait_duration_sec",
        "path_length_meters", "conceptual_cell_size", "res",
        "num_initial_tasks", "num_pending_tasks", "num_waypoints", "num_actions",
        "waypoint_idx", "width", "height", "world_x", "world_y", "grid_x", "grid_y",
        "actual_dist", "expected_conceptual_dist", "expected_dist",
        "num_safety_checks", "num_replans",
        "get_model_state_call_duration",
        "teleport_call_duration_sec",  # From your log example
        "astar_call_duration_sec",     # From your log example
        "pat_call_duration",           # From your log example (manager)
        "planner_call_duration",       # From your log example (manager)
        "executor_call_duration"       # From your log example (manager)
    ]
    boolean_keys = ["is_safe", "success", "exists"]

    for part in parts:
        try:
            key, value_str = part.split("=", 1)
            key = key.strip()
            value_str_original = value_str.strip()

            parsed_value = None

            if key in numeric_keys:
                try:
                    parsed_value = float(value_str_original)
                except ValueError:
                    if value_str_original == "''" or value_str_original.lower() == 'n/a' or not value_str_original:
                        parsed_value = None
                    else:
                        parsed_value = value_str_original
                # (If parsing fails, keep as string for debugging)
            elif key in boolean_keys:
                if value_str_original.lower() == 'true':
                    parsed_value = True
                elif value_str_original.lower() == 'false':
                    parsed_value = False
                else:
                    parsed_value = value_str_original
            elif value_str_original.startswith("'") and value_str_original.endswith("'"):
                parsed_value = value_str_original[1:-1]
            else:
                parsed_value = value_str_original
                # Fallback numeric parsing
                if key not in numeric_keys and key not in boolean_keys:
                    try:
                        if '.' in value_str_original:
                            parsed_value = float(value_str_original)
                        elif value_str_original.isdigit() or \
                                (value_str_original.startswith('-') and value_str_original[1:].isdigit()):
                            parsed_value = int(value_str_original)
                    except ValueError:
                        pass

            log_data[key] = parsed_value

        except ValueError:
            # Malformed part without '=', skip
            pass
        except Exception:
            # Generic parsing error, skip
            pass

    # Final check for 'timestamp'
    if "timestamp" in log_data and not isinstance(log_data.get("timestamp"), (int, float)):
        print(
            f"CRITICAL_PARSE_WARNING: 'timestamp' parsed as non-numeric: "
            f"{log_data.get('timestamp')} (type {type(log_data.get('timestamp'))}). "
            f"Line: {original_line_for_debug[:150]}..."
        )

    return log_data


def calculate_metrics(log_file_path):
    all_parsed_logs = []
    try:
        with open(log_file_path, 'r') as f:
            for line_content in f:
                parsed = parse_log_line(line_content)
                if parsed:
                    all_parsed_logs.append(parsed)
    except FileNotFoundError:
        print(f"ERROR: Log file not found at {log_file_path}")
        return
    except Exception as e:
        print(f"ERROR: Could not read log file {log_file_path}: {e}")
        return

    if not all_parsed_logs:
        print("No performance logs found or parsed from the file.")
        return

    # --- Data Structures for Metrics ---
    planner_requests = 0
    planner_successes = 0
    planner_total_planning_time = 0.0
    planner_successful_path_lengths_meters = []
    planner_successful_num_waypoints = []
    planner_successful_num_actions = []

    executor_paths_attempted = 0
    executor_paths_completed_successfully = 0
    executor_paths_ended_in_replan_signal = 0
    executor_total_execution_time_successful = 0.0
    executor_safety_checks_total_duration = 0.0
    executor_total_replans_triggered_by_executor = 0

    manager_llm_requests = 0
    manager_total_llm_time = 0.0
    manager_pat_validations = 0
    manager_total_pat_time = 0.0

    tasks_assigned_by_manager = {}
    tasks_completed_by_manager = {}

    # For storing (pat_exists, planner_succeeded) per task
    pat_agreement_temp_store = {}

    # --- Process Logs ---
    for log in all_parsed_logs:
        event = log.get("event")
        task_id = log.get("task_id")
        timestamp = log.get("timestamp")

        # Try to get a generic duration from common keys
        duration_keys = [
            "duration_sec", "planning_duration_sec", "exec_duration_sec",
            "pat_call_duration", "astar_call_duration_sec",
            "teleport_call_duration_sec", "get_model_state_call_duration"
        ]
        duration = 0.0
        for d_key in duration_keys:
            val = log.get(d_key)
            if isinstance(val, (int, float)):
                duration = val
                break

        # == Planner Metrics ==
        if event == "PLANNER_GET_PATH_REQUEST_RECEIVED":
            planner_requests += 1

        elif event == "PLANNER_GET_PATH_SUCCESS":
            # Collect planning time, path length, waypoints, actions
            if isinstance(log.get("planning_duration_sec"), (int, float)):
                planner_total_planning_time += log.get("planning_duration_sec")

            path_len = log.get("path_length_meters")
            if isinstance(path_len, (int, float)):
                planner_successful_path_lengths_meters.append(path_len)

            num_wp = log.get("num_waypoints")
            if isinstance(num_wp, (int, float)):
                planner_successful_num_waypoints.append(int(num_wp))

            num_act = log.get("num_actions")
            if isinstance(num_act, (int, float)):
                planner_successful_num_actions.append(int(num_act))

        elif event == "PLANNER_GET_PATH_FAIL":
            # Even on failure, collect planning duration
            if isinstance(log.get("planning_duration_sec"), (int, float)):
                planner_total_planning_time += log.get("planning_duration_sec")

        # == Executor Metrics ==
        if event == "EXEC_PATH_REQUEST_RECEIVED":
            executor_paths_attempted += 1
        elif event == "EXEC_PATH_SUCCESS_COMPLETE":
            executor_paths_completed_successfully += 1
            if isinstance(log.get("exec_duration_sec"), (int, float)):
                executor_total_execution_time_successful += log.get("exec_duration_sec")
        elif event == "EXEC_PATH_SUCCESS_REPLAN_INITIATED":
            executor_paths_ended_in_replan_signal += 1
        elif event == "EXEC_REPLAN_TRIGGERED":
            executor_total_replans_triggered_by_executor += 1
        elif event == "EXEC_SAFETY_CHECK_END":
            safety_dur = log.get("duration_sec")
            if isinstance(safety_dur, (int, float)):
                executor_safety_checks_total_duration += safety_dur

        # == Manager LLM Metrics ==
        if event == "MANAGER_LLM_RESPONSE_RECEIVED":
            manager_llm_requests += 1
            if isinstance(log.get("duration_sec"), (int, float)):
                manager_total_llm_time += log.get("duration_sec")

        # == Manager PAT (validator) Metrics ==
        if event == "MANAGER_PAT_RESPONSE_RECEIVED":
            manager_pat_validations += 1
            if isinstance(log.get("duration_sec"), (int, float)):
                manager_total_pat_time += log.get("duration_sec")
            if task_id:
                # Initialize entry in pat_agreement_temp_store
                pat_agreement_temp_store[task_id] = {
                    'pat_exists': bool(log.get('exists', False)),
                    'planner_succeeded': None
                }

        # == Manager Planner Response: record planner_succeeded with task_id ==
        if event == "MANAGER_PLANNER_RESPONSE_RECEIVED":
            # The manager log line DOES include task_id and success=True/False
            if isinstance(log.get("success"), bool):
                if task_id and task_id in pat_agreement_temp_store:
                    pat_agreement_temp_store[task_id]['planner_succeeded'] = log.get("success", False)

            # Count planner success for statistics
            if log.get("success", False):
                planner_successes += 1

        # == Manager Task Assignment/Completion Metrics ==
        if event == "MANAGER_TASK_ASSIGNED":
            if task_id and isinstance(timestamp, (int, float)):
                tasks_assigned_by_manager[task_id] = {'assign_time': timestamp}

        if event in [
            "MANAGER_TASK_COMPLETED_POST_EXEC_SUCCESS",
            "MANAGER_TASK_COMPLETED_VIA_STATUS",
            "MANAGER_TASK_ALREADY_AT_GOAL",
            "MANAGER_TASK_ALREADY_REMOVED_POST_EXEC"
        ]:
            if task_id and task_id in tasks_assigned_by_manager and isinstance(timestamp, (int, float)):
                if task_id not in tasks_completed_by_manager:
                    tasks_completed_by_manager[task_id] = {'complete_time': timestamp}

    # --- Calculate Decide to Assign Latency ---
    decide_to_assign_times = []
    temp_last_decide_start_time = None
    for log in all_parsed_logs:
        log_event = log.get("event")
        log_timestamp = log.get("timestamp")
        if not isinstance(log_timestamp, (int, float)):
            continue

        if log_event == "MANAGER_DECIDE_CB_START":
            temp_last_decide_start_time = log_timestamp
        elif log_event == "MANAGER_TASK_ASSIGNED" and temp_last_decide_start_time is not None:
            try:
                latency = log_timestamp - temp_last_decide_start_time
                decide_to_assign_times.append(latency)
            except TypeError:
                pass
            temp_last_decide_start_time = None

    # --- Print Metrics ---
    print("\n--- Planner Agent Metrics ---")
    if planner_requests > 0:
        planning_success_rate = (planner_successes / planner_requests) * 100
        avg_planning_time = planner_total_planning_time / planner_requests if planner_requests > 0 else 0.0
        print(f"Planning Success Rate: {planning_success_rate:.2f}% ({planner_successes}/{planner_requests})")
        print(f"Average Planning Time (from logged durations): {avg_planning_time:.4f} sec")
    else:
        print("No planner requests logged.")

    valid_lengths = [l for l in planner_successful_path_lengths_meters if isinstance(l, (int, float))]
    avg_path_length = (sum(valid_lengths) / len(valid_lengths)) if valid_lengths else 0.0
    print(f"Average Successful Path Length: {avg_path_length:.2f} meters")

    valid_waypoints = [wp for wp in planner_successful_num_waypoints if isinstance(wp, (int, float))]
    avg_waypoints = (sum(valid_waypoints) / len(valid_waypoints)) if valid_waypoints else 0.0
    print(f"Average Waypoints per Successful Path: {avg_waypoints:.2f}")

    valid_actions = [act for act in planner_successful_num_actions if isinstance(act, (int, float))]
    avg_actions = (sum(valid_actions) / len(valid_actions)) if valid_actions else 0.0
    print(f"Average Actions per Successful Path: {avg_actions:.2f}")

    print("\n--- Executor Agent Metrics ---")
    if executor_paths_attempted > 0:
        path_execution_success_rate = (executor_paths_completed_successfully / executor_paths_attempted) * 100
        print(
            f"Path Execution Success Rate (full completion): "
            f"{path_execution_success_rate:.2f}% ({executor_paths_completed_successfully}/{executor_paths_attempted})"
        )
        if executor_paths_completed_successfully > 0:
            avg_execution_time_successful = (
                executor_total_execution_time_successful / executor_paths_completed_successfully
            )
            print(
                f"Average Execution Time (for fully successful paths, from durations): "
                f"{avg_execution_time_successful:.4f} sec"
            )
    else:
        print("No executor paths attempted.")

    # Safety check overhead
    individual_safety_check_events_count = sum(
        1 for log in all_parsed_logs
        if log.get("event") == "EXEC_SAFETY_CHECK_END" and isinstance(log.get("duration_sec"), (int, float))
    )
    if individual_safety_check_events_count > 0:
        avg_safety_check_overhead = (
            executor_safety_checks_total_duration / individual_safety_check_events_count
            if executor_safety_checks_total_duration > 0 else 0.0
        )
        print(
            f"Average Safety Check Overhead (per individual check event): "
            f"{avg_safety_check_overhead:.4f} sec "
            f"(total individual checks: {individual_safety_check_events_count})"
        )

        replanning_rate_per_individual_check = (
            (executor_total_replans_triggered_by_executor / individual_safety_check_events_count) * 100
        )
        print(
            f"Replanning Rate (EXEC_REPLAN_TRIGGERED events / individual safety checks): "
            f"{replanning_rate_per_individual_check:.2f}% "
            f"({executor_total_replans_triggered_by_executor}/{individual_safety_check_events_count})"
        )
    else:
        print(
            f"Not enough data for average safety check overhead. "
            f"Individual check events: {individual_safety_check_events_count}"
        )

    print("\n--- Manager Agent Metrics ---")
    if manager_llm_requests > 0:
        avg_llm_time = manager_total_llm_time / manager_llm_requests
        print(
            f"Average LLM Decision Time (from logged durations): "
            f"{avg_llm_time:.4f} sec (over {manager_llm_requests} requests)"
        )
    else:
        print("No LLM requests logged.")

    task_orchestration_times = []
    successful_orchestrations = 0
    for tid, assign_data in tasks_assigned_by_manager.items():
        if tid in tasks_completed_by_manager:
            assign_time = assign_data.get('assign_time')
            complete_time = tasks_completed_by_manager[tid].get('complete_time')
            if isinstance(assign_time, (int, float)) and isinstance(complete_time, (int, float)):
                successful_orchestrations += 1
                task_orchestration_times.append(complete_time - assign_time)

    if tasks_assigned_by_manager:
        success_orch_rate = (successful_orchestrations / len(tasks_assigned_by_manager)) * 100
        print(
            f"Rate of Successful Task Orchestration: "
            f"{success_orch_rate:.2f}% ({successful_orchestrations}/{len(tasks_assigned_by_manager)})"
        )
    else:
        print("No tasks assigned by manager.")

    if task_orchestration_times:
        avg_task_orch_time = sum(task_orchestration_times) / len(task_orchestration_times)
        print(f"Average End-to-End Task Orchestration Time (successful): {avg_task_orch_time:.4f} sec")
    else:
        print("No successful task orchestrations to average time for.")

    if decide_to_assign_times:
        avg_decide_to_assign = sum(decide_to_assign_times) / len(decide_to_assign_times)
        print(
            f"Average Manager Decision to Assignment Time: "
            f"{avg_decide_to_assign:.4f} sec (approx, count: {len(decide_to_assign_times)})"
        )
    else:
        print("No Decide-to-Assign latencies calculated.")

    print("\n--- Validator (PAT) Metrics ---")
    if manager_pat_validations > 0:
        avg_pat_time = manager_total_pat_time / manager_pat_validations
        print(f"Average PAT Validation Time (from manager's perspective, logged duration): {avg_pat_time:.4f} sec")
    else:
        print("No PAT validations logged by manager.")

    # Build final PAT vs Planner agreement list
    final_pat_planner_agreements = []
    for task_id_key, data_val in pat_agreement_temp_store.items():
        if data_val.get('planner_succeeded') is not None:
            final_pat_planner_agreements.append(
                (task_id_key, data_val.get('pat_exists', False), data_val['planner_succeeded'])
            )

    tp_val = tn_val = fp_val = fn_val = 0
    for (_tid, pat_existed_val, planner_succeeded_val) in final_pat_planner_agreements:
        if pat_existed_val and planner_succeeded_val:
            tp_val += 1
        elif (not pat_existed_val) and (not planner_succeeded_val):
            tn_val += 1
        elif pat_existed_val and (not planner_succeeded_val):
            fp_val += 1
        elif (not pat_existed_val) and planner_succeeded_val:
            fn_val += 1

    total_validations_for_agreement = tp_val + tn_val + fp_val + fn_val
    if total_validations_for_agreement > 0:
        validator_accuracy = (tp_val + tn_val) / total_validations_for_agreement * 100
        validator_precision = tp_val / (tp_val + fp_val) * 100 if (tp_val + fp_val) > 0 else 0.0
        validator_recall = tp_val / (tp_val + fn_val) * 100 if (tp_val + fn_val) > 0 else 0.0
        print(f"Validator Accuracy (vs Planner): {validator_accuracy:.2f}%")
        print(f"Validator Precision (for 'path exists'): {validator_precision:.2f}%")
        print(f"Validator Recall (for 'path exists'): {validator_recall:.2f}%")
        print(
            f"  TP={tp_val}, TN={tn_val}, FP={fp_val}, FN={fn_val} "
            f"(Total compared: {total_validations_for_agreement})"
        )

        # --- Print confusion matrix ---
        print("\n--- Confusion Matrix (PAT vs Planner) ---")
        print("                  PAT predicts →  EXISTS     NOT_EXISTS")
        print("Actual Planner →  -----------------------------------")
        print(f"   SUCCEEDED            {tp_val:>5d}            {fn_val:>5d}    ")
        print(f"   FAILED               {fp_val:>5d}            {tn_val:>5d}    ")
        print("")

        # (Optional) Display as pandas DataFrame:
        # df_cm = pd.DataFrame(
        #     [[tp_val, fn_val],
        #      [fp_val, tn_val]],
        #     index=["Planner: SUCCEEDED", "Planner: FAILED"],
        #     columns=["PAT: EXISTS", "PAT: NOT_EXISTS"]
        # )
        # print("--- Confusion Matrix (Pandas DataFrame) ---")
        # print(df_cm)

    else:
        print(
            f"Not enough data for Validator/Planner agreement. "
            f"Temp store size: {len(pat_agreement_temp_store)}, "
            f"Final agreements processed: {len(final_pat_planner_agreements)}"
        )

    print("\n--- Overall System Metrics ---")
    # Compute throughput without relying on num_initial_tasks
    if all_parsed_logs:
        first_log_ts_val = all_parsed_logs[0].get("timestamp")
        last_log_ts_val = all_parsed_logs[-1].get("timestamp")

        if isinstance(first_log_ts_val, (int, float)) and isinstance(last_log_ts_val, (int, float)):
            total_operational_time = last_log_ts_val - first_log_ts_val
            if total_operational_time > 0:
                # Use completed tasks count and total time directly
                completed_count_for_throughput = len(tasks_completed_by_manager)
                throughput_per_sec = completed_count_for_throughput / total_operational_time
                print(
                    f"System Throughput (completed tasks ÷ total time): "
                    f"{throughput_per_sec * 60:.2f} tasks/min "
                    f"({completed_count_for_throughput} tasks over {total_operational_time:.2f}s)"
                )
            else:
                print("System Throughput: N/A (total operational time is zero or negative)")
        else:
            print("Could not determine total operational time due to non-numeric start/end timestamps.")

        # Calculate idle time
        total_idle_time = 0.0
        last_status_time_val = None
        last_status_state_val = None
        for log_entry_idle in all_parsed_logs:
            log_event_idle = log_entry_idle.get("event")
            log_ts_idle = log_entry_idle.get("timestamp")
            if not isinstance(log_ts_idle, (int, float)):
                continue

            if log_event_idle in ["MANAGER_ROBOT_STATUS_RECEIVED", "EXECUTOR_ROBOT_STATUS_PUBLISHED"]:
                current_state_idle = log_entry_idle.get("state")
                if last_status_time_val is not None and last_status_state_val == "idle":
                    try:
                        total_idle_time += (log_ts_idle - last_status_time_val)
                    except TypeError:
                        pass

                last_status_time_val = log_ts_idle
                last_status_state_val = current_state_idle

        # If last state was idle, account for idle until last log
        if (
            last_status_state_val == "idle"
            and last_status_time_val is not None
            and isinstance(last_log_ts_val, (int, float))
            and last_log_ts_val > last_status_time_val
        ):
            try:
                total_idle_time += (last_log_ts_val - last_status_time_val)
            except TypeError:
                pass

        if isinstance(first_log_ts_val, (int, float)) and isinstance(last_log_ts_val, (int, float)):
            total_operational_time = last_log_ts_val - first_log_ts_val
            if total_operational_time > 0:
                idle_percentage = (total_idle_time / total_operational_time) * 100
                print(
                    f"Robot Idle Time Percentage: {idle_percentage:.2f}% "
                    f"(approx, total idle: {total_idle_time:.2f}s / total op: {total_operational_time:.2f}s)"
                )
            else:
                print("Total operational time is zero or negative; cannot compute idle percentage.")
        else:
            print("Could not compute idle time due to non-numeric timestamps.")

    print("\n--- End of Metrics ---")


if __name__ == "__main__":
    # Update this path to point at your log file
    log_file = "/home/eflinspy/ros_ws/src/warehouse_robot/logging/performance_capture.bag.log"
    calculate_metrics(log_file)
