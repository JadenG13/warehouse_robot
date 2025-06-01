import re
from collections import defaultdict


# import pandas as pd # Optional, but very helpful for analysis. Keep commented if not used.

def parse_log_line(line):
    """Parses a single [PERF_LOG] line into a dictionary."""
    if not line.startswith("[PERF_LOG]"):
        return None

    line = line.strip().replace("[PERF_LOG] ", "")
    # Regex to split by comma, but not commas inside single quotes
    parts = re.split(r",\s*(?=(?:[^']*'[^']*')*[^']*$)", line)
    log_data = {}

    # Define keys that should always be numeric (float or int) if possible
    # This helps prioritize their conversion.
    # !!! REVIEW AND EXPAND THIS LIST BASED ON YOUR ACTUAL LOGS !!!
    numeric_keys = [
        "timestamp", "duration_sec", "astar_duration_sec",
        "planning_duration_sec", "exec_duration_sec", "wait_duration_sec",
        "path_length_meters", "conceptual_cell_size", "res",
        "num_initial_tasks", "num_pending_tasks", "num_waypoints", "num_actions",
        "waypoint_idx", "width", "height", "world_x", "world_y", "grid_x", "grid_y",
        "actual_dist", "expected_conceptual_dist", "expected_dist",
        "num_safety_checks", "num_replans",  # From executor final log for EXEC_PATH_SUCCESS_COMPLETE etc.
        "get_model_state_call_duration", "teleport_call_duration_sec",  # Added based on potential new log keys
        "astar_call_duration_sec",  # Ensure consistency if used
        "pat_call_duration", "planner_call_duration", "executor_call_duration"  # from manager logs
    ]

    for part in parts:
        try:
            key, value_str = part.split("=", 1)
            key = key.strip()
            value_str_original = value_str.strip()  # Keep original string for debugging

            parsed_value = None

            if key in numeric_keys:
                try:
                    # For numeric keys, always try to parse as float
                    parsed_value = float(value_str_original)
                except ValueError:
                    # If float conversion fails for a designated numeric key,
                    # it might be a quoted string or an error string like 'N/A'.
                    if value_str_original.startswith("'") and value_str_original.endswith("'"):
                        parsed_value = value_str_original[1:-1]
                    else:
                        parsed_value = value_str_original  # Store as is (e.g., 'N/A')

                    # --- ADDED DEBUG FOR NUMERIC KEYS THAT FAIL FLOAT CONVERSION ---
                    # This helps identify why a numeric key isn't becoming a float
                    # print(f"DEBUG_PARSE_NUMERIC_FAIL: Key '{key}' (expected numeric) failed float conversion. Original_str: '{value_str_original}', Parsed_as: '{parsed_value}' (Type: {type(parsed_value)}), Line: {line[:120]}...")
                    # --- END ADDED DEBUG ---
            elif value_str_original.lower() == 'true':
                parsed_value = True
            elif value_str_original.lower() == 'false':
                parsed_value = False
            elif value_str_original.startswith("'") and value_str_original.endswith("'"):
                parsed_value = value_str_original[1:-1]
            else:
                # For non-numeric_keys, default to string but attempt conversion if it looks numeric
                parsed_value = value_str_original
                try:
                    if '.' in value_str_original:
                        parsed_value = float(value_str_original)
                    elif value_str_original.isdigit() or \
                            (value_str_original.startswith('-') and value_str_original[1:].isdigit()):
                        parsed_value = int(value_str_original)
                except ValueError:
                    pass  # Keep as original string if these fallbacks fail

            log_data[key] = parsed_value

        except ValueError:
            # This happens if a part doesn't contain '='
            # print(f"Skipping malformed part (no '='): '{part}' in line: {line.strip()}")
            pass
        except Exception as e:
            # print(f"Generic error parsing part '{part}': {e} in line: {line.strip()}")
            pass

    # --- DEBUG FOR TIMESTAMP TYPE AFTER ALL PARSING FOR THE LINE ---
    # if "timestamp" in log_data and not isinstance(log_data["timestamp"], (int, float)):
    #    print(f"DEBUG_POST_PARSE: Line '{line[:100]}...' resulted in timestamp of type {type(log_data['timestamp'])} with value '{log_data['timestamp']}'")
    # --- END ADDED DEBUG ---
    return log_data


def calculate_metrics(log_file_path):
    all_parsed_logs = []
    with open(log_file_path, 'r') as f:
        for line_num, line_content in enumerate(f):
            parsed = parse_log_line(line_content)
            if parsed:
                all_parsed_logs.append(parsed)
            # else:
            # print(f"Warning: Could not parse line {line_num + 1}: {line_content.strip()}")

    if not all_parsed_logs:
        print("No performance logs found or parsed.")
        return

    # --- Data Structures for Metrics ---
    planner_requests = 0
    planner_successes = 0
    planner_total_planning_time_from_duration = 0.0  # Using logged duration
    planner_successful_path_lengths_meters = []
    planner_successful_num_waypoints = []
    planner_successful_num_actions = []

    executor_paths_attempted = 0
    executor_paths_completed_successfully = 0
    executor_paths_ended_in_replan_signal = 0
    executor_total_execution_time_successful_from_duration = 0.0  # Using logged duration
    executor_safety_checks_total_duration = 0.0
    executor_total_safety_checks = 0
    executor_total_replans_triggered_in_path = 0  # From EXEC_REPLAN_TRIGGERED

    manager_llm_requests = 0
    manager_total_llm_time_from_duration = 0.0  # Using logged duration
    manager_pat_validations = 0
    manager_total_pat_time_from_duration = 0.0  # Using logged duration

    tasks_assigned_by_manager = {}
    tasks_completed_by_manager = {}

    pat_planner_agreements = []
    # For PAT/Planner agreement, we need to track state across log lines
    # Example: {task_id_X: {'pat_exists': True, 'planner_succeeded': None}}
    # Then when planner log comes for task_id_X, update 'planner_succeeded'
    # For simplicity, the current script uses a single 'current_task_being_planned_for_pat_agreement'
    # This might miss agreements if logs are interleaved for different tasks before planner responds.
    # A dictionary keyed by task_id would be more robust for this specific metric.
    pat_agreement_temp_store = {}

    # --- Process Logs ---
    last_decide_start_time = None  # For manager decide to assign latency

    for log_idx, log in enumerate(all_parsed_logs):
        event = log.get("event")
        task_id = log.get("task_id")
        timestamp = log.get("timestamp")  # Should be float now

        # == Planner Metrics ==
        if event == "PLANNER_GET_PATH_REQUEST_RECEIVED":
            planner_requests += 1
        elif event == "PLANNER_GET_PATH_SUCCESS":
            planner_successes += 1
            planner_total_planning_time_from_duration += log.get("planning_duration_sec", 0.0)
            planner_successful_path_lengths_meters.append(log.get("path_length_meters", 0.0))
            planner_successful_num_waypoints.append(log.get("num_waypoints", 0))
            planner_successful_num_actions.append(log.get("num_actions", 0))
            if task_id and task_id in pat_agreement_temp_store:
                pat_agreement_temp_store[task_id]['planner_succeeded'] = True
        elif event == "PLANNER_GET_PATH_FAIL":
            planner_total_planning_time_from_duration += log.get("planning_duration_sec", 0.0)
            if task_id and task_id in pat_agreement_temp_store:
                pat_agreement_temp_store[task_id]['planner_succeeded'] = False

        # == Executor Metrics ==
        if event == "EXEC_PATH_REQUEST_RECEIVED":
            executor_paths_attempted += 1
        elif event == "EXEC_PATH_SUCCESS_COMPLETE":
            executor_paths_completed_successfully += 1
            executor_total_execution_time_successful_from_duration += log.get("exec_duration_sec", 0.0)
            # The num_replans is for this specific path execution
            executor_total_replans_triggered_in_path += log.get("num_replans", 0)
            executor_total_safety_checks += log.get("num_safety_checks", 0)  # Add from successful paths
        elif event == "EXEC_PATH_SUCCESS_REPLAN_INITIATED":
            executor_paths_ended_in_replan_signal += 1
            executor_total_replans_triggered_in_path += log.get("num_replans", 0)  # Or assume 1 if not logged per path
            executor_total_safety_checks += log.get("num_safety_checks", 0)  # Add from replan paths
        elif event == "EXEC_PATH_FAIL":
            executor_total_replans_triggered_in_path += log.get("num_replans", 0)  # Add from failed paths too
            executor_total_safety_checks += log.get("num_safety_checks", 0)  # Add from failed paths

        # Safety check overhead is summed from individual checks if that log exists,
        # or can be taken from EXEC_SAFETY_CHECK_END events.
        # The provided sample log has EXEC_SAFETY_CHECK_END.
        if event == "EXEC_SAFETY_CHECK_END":  # Individual check timing
            # This check will sum up all safety check durations regardless of path outcome
            # which is correct for "average overhead per check".
            safety_duration = log.get("duration_sec")
            if isinstance(safety_duration, (int, float)):  # Ensure it's a number
                executor_safety_checks_total_duration += safety_duration
                # executor_total_safety_checks count is now taken from EXEC_PATH_..._COMPLETE/FAIL/REPLAN logs
                # as they provide the count *per path execution attempt*.
                # If we want to count individual EXEC_SAFETY_CHECK_START/END, we can do that too.
                # For "Replanning Rate", we need "Total waypoints checked for safety".
                # This is logged as num_safety_checks in EXEC_PATH_... logs.
            # else:
            # print(f"Warning: EXEC_SAFETY_CHECK_END duration_sec is not a number: {safety_duration} in log: {log}")

        # == Manager Metrics ==
        if event == "MANAGER_LLM_RESPONSE_RECEIVED":
            manager_llm_requests += 1
            manager_total_llm_time_from_duration += log.get("duration_sec", 0.0)

        if event == "MANAGER_PAT_RESPONSE_RECEIVED":
            manager_pat_validations += 1
            manager_total_pat_time_from_duration += log.get("duration_sec", 0.0)
            if task_id:  # For PAT/Planner agreement
                pat_agreement_temp_store[task_id] = {
                    'pat_exists': log.get('exists', False),
                    'planner_succeeded': None  # To be filled by planner logs
                }

        if event == "MANAGER_TASK_ASSIGNED":
            if task_id:
                tasks_assigned_by_manager[task_id] = {'assign_time': timestamp}

        if event == "MANAGER_TASK_COMPLETED_POST_EXEC_SUCCESS" or \
                event == "MANAGER_TASK_COMPLETED_VIA_STATUS" or \
                event == "MANAGER_TASK_ALREADY_AT_GOAL":  # Consider "already at goal" as a completion
            if task_id and task_id in tasks_assigned_by_manager:
                if task_id not in tasks_completed_by_manager:
                    tasks_completed_by_manager[task_id] = {'complete_time': timestamp}

        # For Manager Decide to Assign Latency
        if log.get("event") == "MANAGER_DECIDE_CB_START":
            # Ensure timestamp is float before assigning
            if isinstance(timestamp, (int, float)):
                last_decide_start_time = timestamp
            # else:
            # print(f"Warning: Timestamp for MANAGER_DECIDE_CB_START is not float: {timestamp}. Log: {log}")

    # --- Calculate and Print Metrics ---
    print("\n--- Planner Agent Metrics ---")
    if planner_requests > 0:
        planning_success_rate = (planner_successes / planner_requests) * 100
        avg_planning_time = planner_total_planning_time_from_duration / planner_requests if planner_requests > 0 else 0
        print(f"Planning Success Rate: {planning_success_rate:.2f}% ({planner_successes}/{planner_requests})")
        print(f"Average Planning Time (from logged durations): {avg_planning_time:.4f} sec")
    else:
        print("No planner requests logged.")

    if planner_successful_path_lengths_meters:
        avg_path_length = sum(filter(None, planner_successful_path_lengths_meters)) / len(
            planner_successful_path_lengths_meters) if len(planner_successful_path_lengths_meters) > 0 else 0
        print(f"Average Successful Path Length: {avg_path_length:.2f} meters")
    if planner_successful_num_waypoints:
        avg_waypoints = sum(planner_successful_num_waypoints) / len(planner_successful_num_waypoints) if len(
            planner_successful_num_waypoints) > 0 else 0
        print(f"Average Waypoints per Successful Path: {avg_waypoints:.2f}")
    if planner_successful_num_actions:
        avg_actions = sum(planner_successful_num_actions) / len(planner_successful_num_actions) if len(
            planner_successful_num_actions) > 0 else 0
        print(f"Average Actions per Successful Path: {avg_actions:.2f}")

    print("\n--- Executor Agent Metrics ---")
    if executor_paths_attempted > 0:
        path_execution_success_rate = (executor_paths_completed_successfully / executor_paths_attempted) * 100
        print(
            f"Path Execution Success Rate (full completion): {path_execution_success_rate:.2f}% ({executor_paths_completed_successfully}/{executor_paths_attempted})")
        if executor_paths_completed_successfully > 0:
            avg_execution_time_successful = executor_total_execution_time_successful_from_duration / executor_paths_completed_successfully
            print(
                f"Average Execution Time (for fully successful paths, from logged durations): {avg_execution_time_successful:.4f} sec")
    else:
        print("No executor paths attempted.")

    # Sum of num_safety_checks from all path execution outcomes
    total_safety_checks_across_all_paths = 0
    for log in all_parsed_logs:
        if log.get("event") in ["EXEC_PATH_SUCCESS_COMPLETE", "EXEC_PATH_SUCCESS_REPLAN_INITIATED", "EXEC_PATH_FAIL"]:
            num_sc = log.get("num_safety_checks")
            if isinstance(num_sc, (int, float)):  # Check if num_safety_checks is a number
                total_safety_checks_across_all_paths += num_sc

    if total_safety_checks_across_all_paths > 0:  # Use this total count
        # Average Safety Check Overhead is sum of individual check durations / total number of individual checks
        # The variable 'executor_total_safety_checks' was intended for individual checks but got mixed.
        # Let's count individual EXEC_SAFETY_CHECK_END events for this:
        individual_safety_check_count = sum(1 for log in all_parsed_logs if
                                            log.get("event") == "EXEC_SAFETY_CHECK_END" and isinstance(
                                                log.get("duration_sec"), (int, float)))
        if individual_safety_check_count > 0 and executor_safety_checks_total_duration > 0:
            avg_safety_check_overhead = executor_safety_checks_total_duration / individual_safety_check_count
            print(
                f"Average Safety Check Overhead (per individual check): {avg_safety_check_overhead:.4f} sec (total checks: {individual_safety_check_count})")
        else:
            print("Not enough data for average safety check overhead (no individual EXEC_SAFETY_CHECK_END durations).")

        # Replanning Rate: (Total replans triggered during path execution / Total safety checks performed during path execution)
        # executor_total_replans_triggered_in_path should be sum of 'num_replans' from EXEC_PATH_... logs
        # total_safety_checks_across_all_paths is sum of 'num_safety_checks' from EXEC_PATH_... logs
        replanning_rate_per_check = (
                                                executor_total_replans_triggered_in_path / total_safety_checks_across_all_paths) * 100
        print(
            f"Replanning Rate (per safety check during paths): {replanning_rate_per_check:.2f}% ({executor_total_replans_triggered_in_path}/{total_safety_checks_across_all_paths})")
    else:
        print("No safety checks logged from path execution summaries.")

    print("\n--- Manager Agent Metrics ---")
    if manager_llm_requests > 0:
        avg_llm_time = manager_total_llm_time_from_duration / manager_llm_requests
        print(
            f"Average LLM Decision Time (from logged durations): {avg_llm_time:.4f} sec (over {manager_llm_requests} requests)")

    task_orchestration_times = []
    successful_orchestrations = 0
    for tid, assign_data in tasks_assigned_by_manager.items():
        if tid in tasks_completed_by_manager:
            # Ensure both timestamps are numbers
            assign_time = assign_data.get('assign_time')
            complete_time = tasks_completed_by_manager[tid].get('complete_time')
            if isinstance(assign_time, (int, float)) and isinstance(complete_time, (int, float)):
                successful_orchestrations += 1
                duration = complete_time - assign_time
                task_orchestration_times.append(duration)
            # else:
            # print(f"Warning: Timestamps for task '{tid}' not numeric for orchestration time. Assign: {assign_time}, Complete: {complete_time}")

    if tasks_assigned_by_manager:
        success_orch_rate = (successful_orchestrations / len(tasks_assigned_by_manager)) * 100 if len(
            tasks_assigned_by_manager) > 0 else 0
        print(
            f"Rate of Successful Task Orchestration: {success_orch_rate:.2f}% ({successful_orchestrations}/{len(tasks_assigned_by_manager)})")
    if task_orchestration_times:
        avg_task_orch_time = sum(task_orchestration_times) / len(
            task_orchestration_times) if task_orchestration_times else 0
        print(f"Average End-to-End Task Orchestration Time (successful): {avg_task_orch_time:.4f} sec")

    decide_to_assign_times = []
    # Re-loop to correctly pair DECIDE_CB_START with subsequent TASK_ASSIGNED
    # This simple sequential pairing might be incorrect if multiple decide_cb calls happen before an assignment
    # or if a decide_cb doesn't lead to an assignment.
    # A better way would be to link them via task_id if decide_cb immediately focuses on one.
    # Given the current logs, this sequential approach is an approximation.

    temp_last_decide_start_time = None
    for log in all_parsed_logs:
        log_event = log.get("event")
        log_timestamp = log.get("timestamp")

        if not isinstance(log_timestamp, (int, float)):  # Skip if timestamp isn't parsed as number
            # print(f"Skipping log for decide_to_assign due to non-numeric timestamp: {log}")
            continue

        if log_event == "MANAGER_DECIDE_CB_START":
            temp_last_decide_start_time = log_timestamp
        elif log_event == "MANAGER_TASK_ASSIGNED" and temp_last_decide_start_time is not None:
            # --- ADDED DEBUG PRINTS FOR THE PROBLEMATIC SECTION ---
            # print(f"\nDEBUG: Calculating decide_to_assign_time:")
            # print(f"  MANAGER_TASK_ASSIGNED timestamp: '{log_timestamp}' (Type: {type(log_timestamp)})")
            # print(f"  temp_last_decide_start_time: '{temp_last_decide_start_time}' (Type: {type(temp_last_decide_start_time)})")
            # --- END DEBUG ---
            try:
                latency = log_timestamp - temp_last_decide_start_time
                decide_to_assign_times.append(latency)
            except TypeError:
                print(
                    f"  ERROR: TypeError during subtraction for decide_to_assign_times. MANAGER_TASK_ASSIGNED ts: {log_timestamp} (type {type(log_timestamp)}), last_decide_start ts: {temp_last_decide_start_time} (type {type(temp_last_decide_start_time)})")
            temp_last_decide_start_time = None  # Reset after use

    if decide_to_assign_times:
        avg_decide_to_assign = sum(decide_to_assign_times) / len(
            decide_to_assign_times) if decide_to_assign_times else 0
        print(
            f"Average Manager Decision to Assignment Time: {avg_decide_to_assign:.4f} sec (approx, based on sequential events)")

    print("\n--- Validator (PAT) Metrics ---")
    if manager_pat_validations > 0:
        avg_pat_time = manager_total_pat_time_from_duration / manager_pat_validations
        print(f"Average PAT Validation Time (from manager's perspective, logged duration): {avg_pat_time:.4f} sec")

    # Finalize PAT/Planner agreements
    for task_id, data in pat_agreement_temp_store.items():
        if data['planner_succeeded'] is not None:  # Ensure planner part was processed
            pat_planner_agreements.append((task_id, data['pat_exists'], data['planner_succeeded']))

    tp_val, tn_val, fp_val, fn_val = 0, 0, 0, 0
    for _, pat_existed, planner_succeeded_val in pat_planner_agreements:  # Unpack correctly
        if pat_existed and planner_succeeded_val:
            tp_val += 1
        elif not pat_existed and not planner_succeeded_val:
            tn_val += 1
        elif pat_existed and not planner_succeeded_val:
            fp_val += 1
        elif not pat_existed and planner_succeeded_val:
            fn_val += 1

    total_validations_for_agreement = tp_val + tn_val + fp_val + fn_val
    if total_validations_for_agreement > 0:
        validator_accuracy = (tp_val + tn_val) / total_validations_for_agreement * 100
        validator_precision = tp_val / (tp_val + fp_val) * 100 if (tp_val + fp_val) > 0 else 0
        validator_recall = tp_val / (tp_val + fn_val) * 100 if (tp_val + fn_val) > 0 else 0
        print(f"Validator Accuracy (vs Planner): {validator_accuracy:.2f}%")
        print(f"Validator Precision (for 'path exists'): {validator_precision:.2f}%")
        print(f"Validator Recall (for 'path exists'): {validator_recall:.2f}%")
        print(
            f"  TP={tp_val}, TN={tn_val}, FP={fp_val}, FN={fn_val} (Total compared: {total_validations_for_agreement})")
    else:
        print("Not enough data for Validator/Planner agreement metrics.")

    print("\n--- Overall System Metrics ---")
    num_initial_tasks = 0
    for log in all_parsed_logs:
        if log.get("event") == "MANAGER_TASK_LIST_INIT":
            init_tasks_val = log.get("num_initial_tasks")
            if isinstance(init_tasks_val, (int, float)):  # Check if it's a number
                num_initial_tasks = int(init_tasks_val)
            break

    if num_initial_tasks > 0:
        completed_task_ids_count = len(tasks_completed_by_manager.keys())
        overall_task_completion_rate = (completed_task_ids_count / num_initial_tasks) * 100
        print(
            f"Overall Task Completion Rate: {overall_task_completion_rate:.2f}% ({completed_task_ids_count}/{num_initial_tasks})")

    if task_orchestration_times:
        avg_task_orch_time = sum(task_orchestration_times) / len(
            task_orchestration_times) if task_orchestration_times else 0
        print(f"Average End-to-End Task Completion Time (successful): {avg_task_orch_time:.4f} sec")

    if all_parsed_logs:
        first_log_ts_val = all_parsed_logs[0].get("timestamp")
        last_log_ts_val = all_parsed_logs[-1].get("timestamp")

        if isinstance(first_log_ts_val, (int, float)) and isinstance(last_log_ts_val, (int, float)):
            total_operational_time = last_log_ts_val - first_log_ts_val
            if total_operational_time > 0 and num_initial_tasks > 0:
                system_throughput_tasks_per_sec = len(tasks_completed_by_manager.keys()) / total_operational_time
                print(
                    f"System Throughput: {system_throughput_tasks_per_sec * 60:.2f} tasks/min (approx, over {total_operational_time:.2f}s)")

            total_idle_time = 0
            last_status_time = None
            last_status_state = None
            for log in all_parsed_logs:
                log_event = log.get("event")
                log_ts = log.get("timestamp")
                if not isinstance(log_ts, (int, float)): continue  # Skip if no valid timestamp

                if log_event == "MANAGER_ROBOT_STATUS_RECEIVED" or log_event == "EXECUTOR_ROBOT_STATUS_PUBLISHED":
                    current_state = log.get("state")
                    if last_status_time is not None and last_status_state == "idle":
                        total_idle_time += (log_ts - last_status_time)
                    last_status_time = log_ts
                    last_status_state = current_state

            if last_status_state == "idle" and last_status_time is not None and \
                    isinstance(last_log_ts_val, (int, float)) and last_log_ts_val > last_status_time:
                total_idle_time += (last_log_ts_val - last_status_time)

            if total_operational_time > 0:
                idle_percentage = (total_idle_time / total_operational_time) * 100
                print(f"Robot Idle Time Percentage: {idle_percentage:.2f}% (approx)")
        else:
            print("Could not determine total operational time due to non-numeric start/end timestamps.")

    print("\n--- End of Metrics ---")


if __name__ == "__main__":
    log_file = "/home/eflinspy/ros_ws/performance_only.log"  # Path to your log file
    # import sys
    # if len(sys.argv) > 1:
    #    log_file = sys.argv[1]
    # else:
    #    print("Usage: python parse_perf_logs.py <path_to_log_file>")
    #    sys.exit(1)
    calculate_metrics(log_file)