import rosbag
import sys


def extract_logs(bag_file_path, output_txt_file):
    count = 0
    with open(output_txt_file, 'w') as outfile:
        try:
            with rosbag.Bag(bag_file_path, 'r') as bag:
                for topic, msg, t in bag.read_messages(topics=['/rosout_agg']):
                    # msg is of type rosgraph_msgs/Log
                    if "[PERF_LOG]" in msg.msg:
                        # Write the raw message string from msg.msg
                        # It should not be truncated here.
                        outfile.write(msg.msg + '\n')
                        count += 1
        except Exception as e:
            print(f"Error processing bag file: {e}")
    print(f"Extracted {count} [PERF_LOG] lines to {output_txt_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_perf_logs_from_bag.py <input_bag_file> <output_txt_file>")
        sys.exit(1)

    bag_file = sys.argv[1]
    out_file = sys.argv[2]
    extract_logs(bag_file, out_file)